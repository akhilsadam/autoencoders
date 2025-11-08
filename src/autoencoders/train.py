"""Hydra-driven training entrypoint for autoencoders."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict


from git import Repo
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import save_image
import wandb

from . import get_default_config, get_model
from .data import build_dataloaders
os.environ.setdefault("WANDB_MODE", "online")

from .trainer import create_trainer

torch.set_float32_matmul_precision('high')

def _prepare_model(cfg: DictConfig) -> pl.LightningModule:
    default_cfg = get_default_config(cfg.model.name)
    params: Dict[str, Any] = asdict(default_cfg)

    if cfg.model.get("params"):
        overrides = OmegaConf.to_container(cfg.model.params, resolve=True)
        params.update(overrides)  # type: ignore[arg-type]

    return get_model(cfg.model.name, params)


def _create_logger(cfg: DictConfig) -> WandbLogger:
    wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
    kwargs = {k: v for k, v in wandb_cfg.items() if v not in (None, "")}
    
    # add data and model name to run_name and tags
    cfg.run.name = f"{cfg.data.name}-{cfg.model.name}-{cfg.run.get('name','')}"
    if "tags" in cfg.run and isinstance(cfg.run.tags, list):
        cfg.run.tags = cfg.run.tags + [cfg.data.name, cfg.model.name]
    
    kwargs['name'] = cfg.run.name
    kwargs['tags'] = cfg.run.tags

    logger = WandbLogger(**kwargs, log_model=False)
    logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    
    repo = Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    dirty = repo.is_dirty()
    logger.experiment.config["git_sha"] = sha
    logger.experiment.config["git_dirty"] = dirty
    print(f"Git SHA: {sha}, Dirty: {dirty}")
    cfg.git.sha = sha
    cfg.git.dirty = dirty
    
    return logger


def _artifact_dirs(cfg: DictConfig) -> Dict[str, str]:
    # Build directories using os.path.join for portability across platforms
    artifacts_root = str(cfg.paths.artifacts_root)
    checkpoints = os.path.join(artifacts_root, "checkpoints")
    reconstructions = os.path.join(artifacts_root, "reconstructions")
    for path in (artifacts_root, checkpoints, reconstructions):
        os.makedirs(path, exist_ok=True)
    return {"root": artifacts_root, "checkpoints": checkpoints, "reconstructions": reconstructions}


def _save_reconstructions(model: pl.LightningModule, dataloader: torch.utils.data.DataLoader, output_dir: str) -> None:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            recon = model(inputs).cpu()
            save_image(inputs[:8], os.path.join(output_dir, "inputs.png"), nrow=4)
            save_image(recon[:8], os.path.join(output_dir, "reconstructions.png"), nrow=4)
            break

def _save_info_files(cfg: DictConfig, output_dir: str) -> None:
    """Save config and git info to output directory."""
    cfg_path = os.path.join(output_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
        
def _save_diff(cfg: DictConfig, output_dir: str) -> None:
    """Calculate and save git changes to output directory."""
    repo = Repo(search_parent_directories=True)
    if repo.is_dirty():
        diff = repo.git.diff()
        diff_path = os.path.join(output_dir, "unstaged_diff.patch")
        with open(diff_path, "w") as f:
            f.write(diff)   
    else:
        # diff to last commit
        diff = repo.git.diff(f"{cfg.git.sha}~1", cfg.git.sha)

    try:
        from .util import llm
        summary = llm.summarize_diff(diff)
        cfg.git.changelog = summary
    except Exception as e:
        print(f"Warning: failed to summarize diff: {e}")

def _log_wandb_artifacts(cfg: DictConfig, logger: WandbLogger, dirs: Dict[str, str]) -> None:
    """Upload checkpoints and reconstruction images to Weights & Biases as artifacts."""
    run = logger.experiment  # wandb.sdk.wandb_run.Run
    try:
        base_name = getattr(run, "name", None) or getattr(run, "id", "run")

        # Log checkpoints as a model artifact
        ckpt_dir = dirs.get("checkpoints", "")
        if ckpt_dir and os.path.isdir(ckpt_dir) and any(os.scandir(ckpt_dir)):
            model_art = wandb.Artifact(name=f"{base_name}-checkpoints", type="model")
            model_art.add_dir(ckpt_dir)
            run.log_artifact(model_art, aliases=["latest"])  # add alias for convenience

        # Log reconstructions as an evaluation artifact
        rec_dir = dirs.get("reconstructions", "")
        if rec_dir and os.path.isdir(rec_dir) and any(os.scandir(rec_dir)):
            eval_art = wandb.Artifact(name=f"{base_name}-reconstructions", type="evaluation")
            eval_art.add_dir(rec_dir)
            run.log_artifact(eval_art)
    except Exception as e:
        # Do not fail training if artifact upload encounters a transient error
        print(f"Warning: failed to log artifacts to W&B: {e}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    train_loader, val_loader = build_dataloaders(cfg.data)

    model = _prepare_model(cfg)
    logger = _create_logger(cfg)
    dirs = _artifact_dirs(cfg)
    
    # Determine rank for distributed setups
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    
    if rank == 0:
        _save_info_files(cfg, dirs["root"])
        _save_diff(cfg, dirs["root"])
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(dirs["checkpoints"]),
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = create_trainer(cfg.trainer, logger=logger, callbacks=[checkpoint_cb, lr_cb])

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(os.path.join(dirs["checkpoints"], "last.ckpt"))

    device = trainer.strategy.root_device
    if rank == 0:
        model.to(device)
        _save_reconstructions(model, val_loader, dirs["reconstructions"])
        _log_wandb_artifacts(cfg, logger, dirs)



if __name__ == "__main__":
    main()
