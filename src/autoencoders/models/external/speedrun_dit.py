import torch
import sys
import os
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model

from .SpeedrunDiT.sit import SiT_B_1
from .SpeedrunDiT.invae import INVAE 

class LoRASpeedrunDiT(torch.nn.Module):
    def __init__(self, resolution=512, r=16, alpha=32):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
              # 1. Load VAE (Essential for Latent Diffusion)
        # SR-DiT uses a specific INVAE; ensure the local path is correct
        self.vae = INVAE().to(self.device)
        vae_ckpt = hf_hub_download(repo_id="SwayStar123/SpeedrunDiT", filename="invae.pt")
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location=self.device))
        self.vae.eval() # Keep VAE in eval mode to act as a fixed feature extractor
        for param in self.vae.parameters():
            param.requires_grad = False

        # 2. Load Base SiT Model
        model_file = f"model_{resolution}.pt"
        checkpoint_path = hf_hub_download(repo_id="SwayStar123/SpeedrunDiT", filename=model_file)
        self.base_model = SiT_B_1().to(self.device)
        self.base_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        # 3. Inject LoRA into the Transformer only
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["qkv", "proj"],
            lora_dropout=0.05,
            bias="none"
        )
        self.model = get_peft_model(self.base_model, config)
        self.model.print_trainable_parameters()

        self.latent_dim = 768 # from SiT

    def encode(self, x):
        """Converts pixels to latents."""
        with torch.no_grad():
            return self.vae.encode(x)

    def decode(self, z):
        """Converts latents back to pixels."""
        with torch.no_grad():
            return self.vae.decode(z)

    def forward(self, x, t, y):
        """Predicts velocity v_t."""
        return self.model(x, t, y)
    
    def denoise(self, x, t, c, y):
        if y is None:
            y = torch.zeros((x.shape[0], self.latent_dim), device=x.device)
        
        im = torch.cat([x, c, c], dim=1)
        denoised = self.model(im, t, y) + im # denoised image
        return denoised.chunk(3, dim=1)[0]