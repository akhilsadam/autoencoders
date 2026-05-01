import torch
import sys
import os
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model

from .SpeedrunDiT.sit import SiT_B_1
from .SpeedrunDiT.invae import VAE_F16D32

class LoRASpeedrunDiT(torch.nn.Module):
    def __init__(self, resolution=512, latent_size=16, num_classes=1000, r=16, alpha=32, 
                 cfg_prob=0.1, z_dims=[768], block_kwargs=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.latent_scale = 0.3099  # Scaling factor from official implementation
        
        if block_kwargs is None:
            block_kwargs = {"fused_attn": True, "qk_norm": False}
        self.block_kwargs = block_kwargs
        
        # 1. Load VAE (Essential for Latent Diffusion)
        # VAE_F16D32 outputs [B, 32, 16, 16] from 256x256 images
        self.vae = VAE_F16D32().to(self.device)
        vae_ckpt = hf_hub_download(repo_id="SwayStar123/SpeedrunDiT", filename="invae.pt")
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location=self.device))
        self.vae.eval()  # Keep VAE in eval mode as fixed feature extractor
        for param in self.vae.parameters():
            param.requires_grad = False

        # 2. Load Base SiT Model with proper parameters
        # SiT_B_1 returns SiT(depth=12, hidden_size=768, decoder_hidden_size=768, 
        #                     patch_size=1, num_heads=12, encoder_depth=4)
        model_file = f"model_{resolution}.pt"
        checkpoint_path = hf_hub_download(repo_id="SwayStar123/SpeedrunDiT", filename=model_file)
        
        # Initialize SiT with required parameters
        self.base_model = SiT_B_1(
            input_size=latent_size,          # latent spatial size (256 image -> 256//16 = 16)
            in_channels=32,                  # VAE output channels
            num_classes=num_classes,
            use_cfg=(cfg_prob > 0),         # Classifier-free guidance
            z_dims=z_dims,                  # Projector embedding dimensions
            **block_kwargs
        ).to(self.device)
        
        self.base_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        # 3. Inject LoRA into the Transformer
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["qkv", "proj"],
            lora_dropout=0.05,
            bias="none"
        )
        self.model = get_peft_model(self.base_model, config)
        self.model.print_trainable_parameters()

        self.latent_dim = 768  # from SiT hidden_size

    def encode(self, x):
        """Converts pixels to latents using VAE encoder.
        
        Args:
            x: Image tensor [B, 3, H, W]
            
        Returns:
            Scaled latent tensor [B, 32, H//16, W//16]
        """
        with torch.no_grad():
            posterior = self.vae.encode(x)  # Returns DiagonalGaussianDistribution
            z = posterior.sample()  # Sample from posterior
            z = (z - 0.5) / self.latent_scale  # Apply scaling factor
        return z

    def decode(self, z):
        """Converts latents back to pixels using VAE decoder.
        
        Args:
            z: Latent tensor [B, 32, H//16, W//16]
            
        Returns:
            Reconstructed image tensor [B, 3, H, W]
        """
        with torch.no_grad():
            z_scaled = z * self.latent_scale + 0.5  # Reverse scaling
            output = self.vae.decode(z_scaled)  # Returns dict-like object
            x_recon = output.sample  # Access .sample attribute
        return x_recon

    def forward(self, x, t, y, cls_token=None, return_logvar=False, uncond=False):
        """Forward pass through SiT model with SPRINT.
        
        Args:
            x: Latent tensor [B, 32, H//16, W//16]
            t: Timestep tensor [B,]
            y: Class label tensor [B,]
            cls_token: Optional class token embedding
            return_logvar: Whether to return log variance (unused)
            uncond: Unconditional guidance flag
            
        Returns:
            Tuple of (x_out, zs, cls_token_out):
                - x_out: Denoised latents [B, 32, H//16, W//16]
                - zs: List of projected embeddings for auxiliary losses
                - cls_token_out: Output class token
        """
        return self.model(x, t, y, cls_token=cls_token, uncond=uncond)
    
    def denoise(self, x, t, c, y):
        """Denoise images with conditional guidance using diffusion.
        
        Args:
            x: Image tensor [B, C, H, W] to denoise (grayscale: [B, 1, H, W])
            t: Timestep tensor [B,] (diffusion timesteps)
            c: Conditional image tensor [B, C, H, W] (same shape as x)
            y: Class label tensor [B,] or None
            
        Returns:
            Denoised image tensor [B, C, H, W]
            
        Notes:
            - Works with any image size (256x256, 512x512, etc.)
            - Concatenates x and c before encoding: [B, 1, H, W] + [B, 1, H, W] -> [B, 2, H, W]
            - Encoded concatenation -> [B, 32, H//16, W//16] latent space
        """
        if y is None:
            y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        # Concatenate grayscale image with conditioning
        # [B, 1, H, W] + [B, 1, H, W] -> [B, 3, H, W]
        x_cond = torch.cat([x, c, c], dim=1)
        
        # Encode concatenated input to latent space
        # For 512x512 images: [B, 3, 512, 512] -> [B, 32, 32, 32]
        with torch.no_grad():
            z = self.encode(x_cond)  # [B, 32, latent_h, latent_w]
        
        # Forward pass through diffusion model
        z_out, _, _ = self.forward(z, t, y)
        
        # Decode latent representation back to image space
        x_denoised = self.decode(z_out)
        
        return x_denoised.chunk(3, dim=1)[0]