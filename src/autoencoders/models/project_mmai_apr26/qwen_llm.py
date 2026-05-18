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

from qg.solver.opt.operator.rpn import QwenContrastiveRPN

from autoencoders.metrics import text as MX

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
 
@dataclass
class QwenCRPNConfig:
    proj_dim: int = 64
    sem_dim: int = 64
    struct_dim: int = 64
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_rules: bool = False
    temperature: float = 0.1
    max_rpn_len: int = 128
    qwen_model_id: str = QWEN_MODEL_ID
    learning_rate: float = 1e-4
 
 
class QwenCRPNAutoencoder(pl.LightningModule):
    """
    Drop-in Lightning replacement for CRPNAutoencoder (llm.py).
 
    Config keys match QwenCRPNConfig fields.
    """
 
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        cfg = config
 
        self.crpn = QwenContrastiveRPN(
            proj_dim=cfg.get("proj_dim", 96),
            sem_dim=cfg.get("sem_dim", 64),
            struct_dim=cfg.get("struct_dim", 64),
            lora_r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            use_rules=cfg.get("use_rules", False),
            temperature=cfg.get("temperature", 0.1),
            max_rpn_len=cfg.get("max_rpn_len", 50),
            qwen_model_id=cfg.get("qwen_model_id", QWEN_MODEL_ID),
        )
        self.learning_rate = cfg.get("learning_rate", 1e-4)
 
        # Expose for text.py compatibility
        self.tokenize = self._tokenize_compat
        self.detokenize = self._detokenize_compat
 
    # ── text.py compatibility shims ───────────────────────────────────────
 
    def _tokenize_compat(self, rpns):
        """Returns (input_ids, attention_mask) tensors."""
        return self.crpn.tokenize(rpns)
 
    def _detokenize_compat(self, input_ids, attention_mask=None):
        return self.crpn.detokenize(input_ids, attention_mask)
 
    def encode(self, rpns):
        return self.crpn.encode(rpns)
 
    def decode(self, z):
        """
        z : (B, proj_dim) → list of RPN strings.
        Also returns dummy amp tensor for text.py compatibility.
        """
        strings = self.crpn.decode(z)
        return strings
 
    def sample(self, z):
        return self.crpn.sample(z)
 
    # ── Lightning ─────────────────────────────────────────────────────────
 
    def training_step(self, batch, _, logger=None):
        logger = logger or self
        (
            loss, token_acc, masked_supcon_loss,
            denoise_distortion_loss_tk, denoise_distortion_loss_sc,
            denoise_perception_loss, syntax_loss, denoise_loss, rule_loss,
        ) = self.crpn.loss(batch)
        logger.log_dict({
            'llm_token_acc': token_acc,
            'llm_masked_supcon_loss': masked_supcon_loss,
            'llm_ae_distortion_loss_token': denoise_distortion_loss_tk,
            'llm_ae_distortion_loss_scalar': denoise_distortion_loss_sc,
            'llm_ae_perception_loss': denoise_perception_loss,
            'llm_syntax_loss': syntax_loss,
            'llm_semantic_loss': rule_loss,
            'llm_denoise_loss': denoise_loss,
            'train_llm_loss': loss,
        }, batch_size=len(batch))
        self.log('train_loss', loss, prog_bar=True, batch_size=len(batch))
        return loss
 
    def validation_step(self, batch, _, logger=None):
        logger = logger or self
        (
            loss, token_acc, masked_supcon_loss,
            denoise_distortion_loss_tk, denoise_distortion_loss_sc,
            denoise_perception_loss, syntax_loss, denoise_loss, rule_loss,
        ) = self.crpn.loss(batch)
        logger.log_dict({
            'val_llm_token_acc': token_acc,
            'val_llm_ae_distortion_loss_token': denoise_distortion_loss_tk,
            'val_llm_ae_distortion_loss_scalar': denoise_distortion_loss_sc,
            'val_llm_ae_perception_loss': denoise_perception_loss,
            'val_llm_syntax_loss': syntax_loss,
            'val_llm_semantic_loss': rule_loss,
            'val_llm_denoise_loss': denoise_loss,
            'val_llm_loss': loss,
        }, batch_size=len(batch))
        self.log('val_loss', loss, prog_bar=True, batch_size=len(batch))
 
    def metrics(self, assistant):
        # pass
        val_loader = assistant[0] #
        # MX.reconstruction(self, val_loader, dirs)
        MX.generation(self, val_loader, self.dirs)

    def configure_optimizers(self):
        # Only train LoRA adapters, projection heads, and RPN_GEN
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=self.learning_rate)
 