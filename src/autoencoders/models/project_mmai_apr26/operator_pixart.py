from diffusers import PixArtAlphaPipeline
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F

from autoencoders.models.project_mmai_apr26.operator_diffusion import Diffusion

class PixArtDiffusion(Diffusion):
    def __init__(self, config):
        super().__init__(config)

        # ---- Load correct PixArt model ----
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-512x512",
            torch_dtype=torch.float32
        )

        self.vae = pipe.vae
        self.transformer = pipe.transformer
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

        # ---- Freeze backbone ----
        
        for p in self.transformer.parameters():
            p.requires_grad = False
        # for p in self.tokenizer.parameters():
        #     p.requires_grad = False

        # ---- Lightweight LoRA (~1–10M params) ----
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["to_q", "to_k", "to_v", "proj_out"],
        )
        self.transformer = get_peft_model(self.transformer, lora_config)

    # -------------------------------------------------
    # Correct PixArt forward (1-step training)
    # -------------------------------------------------
    def denoise(self, x, t, text_prompt=None, c=None):

        t_ = t.view(-1, 1, 1, 1)
        noise = torch.randn_like(x)  

        # ---- Flow-style interpolation ----
        z_t = x * t_ + noise * (1 - t_)
        
        # ---- Encode image to latent ----
        x_complete = torch.cat([z_t, c, c], dim=1) #bchw
        latents = self.vae.encode(x_complete).latent_dist.sample() * 0.18215

        # ---- Text conditioning ----
        if text_prompt is not None:
            tokens = self.tokenizer(
                text_prompt,
                padding="max_length",
                truncation=True,
                max_length=120,
                return_tensors="pt"
            ).to(x.device)

            text_emb = self.text_encoder(**tokens).last_hidden_state
        else:
            text_emb = None

        # ---- PixArt forward (DiT) ----
        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=(t * 1000).long(),
            encoder_hidden_states=text_emb,
        ).sample

        # ---- 1-step inversion (critical) ----
        z_pred = (latents - (1 - t_) * noise_pred) / (t_ + 1e-6)

        # ---- Decode ----
        x_pred = self.vae.decode(z_pred / 0.18215).sample

        return x_pred[:,:1,...] # select one color for grayscale image


    # ── Lightning ─────────────────────────────────────────────────────────

    # def training_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:
    #     loss = self.loss(batch[:, 1], batch[:, 0]) 
    #     self.log('train_loss', loss, prog_bar=True)
    #     return loss

    # def validation_step(self, batch: torch.Tensor, _: int) -> None:
    #     self.log('val_loss', self.loss(batch[:, 1], batch[:, 0]), prog_bar=True)
        
    # def metrics(self, assistant, dirs):
    #     val_loader = assistant #
    #     MX.reconstruction(self, val_loader, dirs)
    #     # MX.generation(self, val_loader, dirs)

    # def configure_optimizers(self) -> torch.optim.Optimizer:
    #     return torch.optim.Adam(self.parameters(), lr=self.learning_rate)