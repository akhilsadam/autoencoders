from __future__ import annotations

import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'

from dataclasses import dataclass
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from . import vlm_diffusion, llm
from autoencoders.metrics import vlm_image_diffusion as MX
from autoencoders.metrics import text as TMX

# Convention: model class ends with 'Diffusion', config is 'Config' or endswith 'Config'
@dataclass
class Config(vlm_diffusion.Config, llm.Config):
    pass


def safe_load_state_dict(model, ckpt):
    model_dict = model.state_dict()

    filtered = {}
    for k, v in ckpt.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered[k] = v
        else:
            print(f"Skipping {k}: ckpt={getattr(v, 'shape', None)} model={model_dict.get(k).shape if k in model_dict else 'missing'}")

    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=True)

class OptVLMDiffusion(pl.LightningModule):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        
        lpath = config['pretrain_llm']
        self.freeze_llm = os.path.exists(lpath)
        if self.freeze_llm:
            self.llm = llm.CRPNAutoencoder.load_from_checkpoint(lpath)
            print(f'Loaded from checkpoint: {lpath}')
        else:
            self.llm = llm.CRPNAutoencoder(config)
            
        self.opt = vlm_diffusion.Diffusion(config)
        
        self.use_llm = config['use_llm']

        # additive fusion doesn't work
        # nn.init.zeros_(self.proj_latent[-1].weight)
        # nn.init.zeros_(self.proj_latent[-1].bias)
        
        self.learning_rate = config['learning_rate']
        
        cond_dim = config['sem_dim']
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4),
            nn.GELU(),
            nn.Linear(cond_dim*4, cond_dim * 4),
            nn.GELU(),
            nn.Linear(cond_dim*4, cond_dim)
        )
        self.cond_dim = cond_dim

        checkpoint_path = config.get('pretrain_vm','')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            state_dict = checkpoint["state_dict"]
            
            # Load the weights into the current instance
            # self.load_state_dict(state_dict, strict=False)
            safe_load_state_dict(self, state_dict)
            print(f'Loaded VM weights from checkpoint: {checkpoint_path}')

    def compute_latent(self, rpns):
        encoding = self.llm.encode(rpns)
        semantic = self.llm.crpn.gen.semantic(encoding)
        semantic = self.cond_net(semantic) + semantic
        
        if not self.use_llm:
            semantic = semantic * 0.0        
        return semantic
    
    def encode_LLM(self, rpns):
        encoding = self.llm.encode(rpns)
        return encoding

    def compute_from_LLM(self, encoding):
        semantic = self.llm.crpn.gen.semantic(encoding)
        semantic = self.cond_net(semantic) + semantic
        if not self.use_llm:
            semantic = semantic * 0.0        
        return semantic
    
    def export_from_LLM(self, encoding):
        encoding = self.llm.sample(encoding) # get structural part too
        return self.llm.decode(encoding)
    
    def gen(self, *args, **kwargs):
        return self.opt.gen(*args, **kwargs)
    # ── Lightning ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_id) -> torch.Tensor:
        rpn_batch, fused_batch = batch
        rpn_loss = self.llm.training_step(rpn_batch, batch_id, logger=self)
        
        rpns, seq = fused_batch
        x = seq[:,0]
        y = seq[:,1] # one timestep only
        latent = self.compute_latent(rpns)
        diffusion_loss = self.opt.loss(y, x, latent)
        self.log('v_diffusion_loss', diffusion_loss, prog_bar=True)
        
        loss = 0.1 * rpn_loss + diffusion_loss
        return loss

    def validation_step(self, batch, batch_id) -> None:
        rpn_batch, fused_batch = batch
        self.llm.validation_step(rpn_batch, batch_id, logger=self)
                
        rpns, seq = fused_batch
        x = seq[:,0]
        y = seq[:,1] # one timestep only
        latent = self.compute_latent(rpns)
        diffusion_loss = self.opt.loss(y, x, latent)
        self.log('val_v_diffusion_loss', diffusion_loss, prog_bar=True)
        
        latent_jumbled = latent[torch.randperm(latent.shape[0], device=latent.device)]
        
        MX.quick_reconstruction(self, rpns, seq, self.dirs, '', latent=latent)
        MX.quick_reconstruction(self, rpns, seq, self.dirs, 'jumbled', latent=latent_jumbled)
        TMX.metrics(self.llm, batch_id, rpns, self.dirs)
            
    def metrics(self, assistant):
        # pass
        val_loader, pred_loader = assistant #
        # TMX.inverse_metrics_all(self, pred_loader, self.dirs)
        MX.final_reco(self, val_loader, self.dirs)
        # MX.reconstruction(self, pred_loader, self.dirs)
        # MX.generation(self, val_loader, dirs)
        

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def inverse_solver(self, seq):
        rpn = "q psi jacobian neg"
        encoding_init = self.encode_LLM([rpn,]).detach().to('cuda')
        
        T = seq.shape[1] -1
        seq = seq.to('cuda')
        
        # encoding_init = torch.randn_like(encoding_init) * torch.std(encoding_init) * 0.1 + encoding_init

        encoding = encoding_init.repeat(seq.shape[0], 1).requires_grad_(True)
        optimizer = torch.optim.Adam([encoding], lr=1e-3)        
        print('DEVICE:', encoding.device)

        
        x = seq[:,:-1,...].reshape(-1, *seq.shape[2:]) # B T C H W
        y = seq[:,1:,...].reshape(-1, *seq.shape[2:]) # B T C H W
        
        # print(x.shape, encoding.shape)


        forward = lambda encoding: self.opt.loss(y, x, 
            self.compute_from_LLM(encoding)[:,None,:].repeat(1,T,1).reshape(-1, self.cond_dim)
            )
        percep = lambda x: self.llm.crpn.head.forward(self.llm.crpn.head.reverse(x))

        # optim loop
        losses = {'d_loss': [], 'p_loss': []}
        for j in range(500):
            
            optimizer.zero_grad()
            n_encoding = encoding + 0.05 * torch.randn_like(encoding) * torch.std(encoding)  # jitter for stability

            e_percep = percep(n_encoding)
            percep_loss = F.mse_loss(e_percep, encoding) / (torch.mean(e_percep**2) + 1e-8)


            diffusion_loss = (forward(n_encoding) + forward(e_percep))/2
            # self.log('optimization_loss', diffusion_loss, prog_bar=True)
            losses['d_loss'].append(diffusion_loss.item())
            losses['p_loss'].append(percep_loss.item())

            init_target = self.llm.crpn.head.reverse(encoding_init.repeat(seq.shape[0], 1))
            decoding = self.llm.crpn.head.reverse(encoding)
            cosine_sim = F.cosine_similarity(decoding[:,:4], init_target[:,:4], dim=-1) # match initial part
            sem_loss = torch.mean(1.0 - cosine_sim)

            lz = diffusion_loss + percep_loss + sem_loss

            lz.backward()
            optimizer.step()

        encoding = percep(encoding)

        torch.cuda.empty_cache()
            
        return encoding, self.export_from_LLM(encoding), losses, forward
        