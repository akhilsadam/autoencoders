# Complete Function & Class Inventory (Detailed)

This document lists every function and class in the autoencoders codebase, organized by destination package. Use this during refactoring to verify nothing is missed.

---

## ORCHESTRATOR (Stays in autoencoders/)

### `__init__.py`
- **def** `list_autoencoders()` — Returns tuple of registered model keys
- **def** `get_model(name, config)` — Instantiate model from registry
- **def** `get_default_config(name)` — Get default config for model

### `data.py`
- **def** `_to_plain_dict(cfg)` — Convert DictConfig to plain dict
- **def** `build_dataloaders(cfg)` — Instantiate dataloaders from config

### `train.py`
- **def** `_prepare_model(cfg)` — Load model + config
- **def** `_create_logger(cfg)` — Create WandB logger
- **def** `_artifact_dirs(cfg)` — Set up artifact directories
- **def** `_save_reconstructions(model, dataloader, output_dir)` — Save model outputs
- **def** `_save_info_files(cfg, output_dir)` — Save config + git info
- **def** `_log_wandb_artifacts(cfg, logger, dirs)` — Log to WandB
- **def** `main(cfg)` @hydra.main — Entry point

### `trainer.py`
- **def** `create_trainer(cfg, logger, callbacks)` — Factory for Lightning trainer

### `util/llm.py` (stays in orchestrator)
- **def** `_extract_diff_content(diff_text)` — Parse git diff
- **def** `_chunk_text(text, max_chars)` — Split text into chunks
- **def** `summarize_diff(diff_text, quality)` — Summarize git diff with LLM

---

## ORCHESTRATOR - DATAMODULES (All Stay)

### `datamodules/__init__.py`
- **class** `DatasetEntry` — Registry entry (config_cls, builder)
- **def** `list_datasets()` — List all dataset keys

### `datamodules/aesthetic4k.py`
- **class** `Aesthetic4KConfig` — Config dataclass
- **class** `HFDatasetWrapper` — Torch dataset wrapper
- **def** `_build_transform(cfg)` — Build image transforms
- **def** `build_dataloaders(cfg)` — Create train/val dataloaders

### `datamodules/cache.py`
- **def** `_get_version_hash(cfg, extra_keys)` — Compute cache key
- **def** `get_cache(cfg, extra_keys)` — Get cached dataset

### `datamodules/fashion_mnist.py`
- **class** `FashionMNISTConfig` — Config
- **def** `build_dataloaders(cfg)` — Create dataloaders

### `datamodules/forced_turbulence.py`
- **class** `ForcedTurbulenceConfig` — Config
- **class** `_TensorDatasetNoTuple` — Torch dataset for tensors
- **def** `get_dataset(cfg, name, mmap)` — Load or generate
- **def** `build_dataloaders(cfg)` — Create dataloaders
- **def** `generate_data()` — Generate synthetic data

### `datamodules/load_timeseries_small.py`
- **class** `TimeSeriesDataset` — Timeseries torch dataset
- **class** `Normalize` — Normalization transform
- **def** `filter_kwargs(dict_to_filter, thing_with_kwargs)` — Filter dict keys
- **def** `collate_fn(batch)` — Custom collate
- **def** `load_npy(filepath)` — Load numpy file
- **def** `load_data(folder)` — Load all data from folder

### `datamodules/qg_turbulence.py`
- **class** `QGDatasetConfig` — Config
- **def** `_generate_qg_data(config, cache_path)` — Generate QG data
- **def** `build_dataloaders(cfg)` — Create dataloaders

### `datamodules/rpn_encoder.py`
- **class** `RPNEncoderConfig` — Config
- **class** `TextDataset` — Text torch dataset
- **def** `get_dataset(cfg, name)` — Load or generate
- **def** `build_dataloaders(cfg)` — Create dataloaders
- **def** `generate_data()` — Generate data

### `datamodules/rpn_textvision.py`
- **class** `RPNETConfig` — Config
- **class** `CombinedLoader` — Combine two dataloaders
- **def** `build_dataloaders(cfg)` — Create combined dataloaders

### `datamodules/rpn_turbulence.py`
- **class** `RPNTurbulenceConfig` — Config
- **def** `build_dataloaders(cfg)` — Create dataloaders
- **def** `generate_data()` — Generate data

### `datamodules/singlestep_forced_turbulence.py`
- **def** `build_dataloaders(cfg)` — Single-step variant

### `datamodules/timeseries.py`
- (No functions/classes — just imports)

### `datamodules/timeseries_decaying_qg_turbulence.py`
- **class** `TimeseriesDecayingQGTurbulenceConfig` — Config with __post_init__
- **def** `_get_version_hash(cfg)` — Cache key
- **def** `build_dataloaders(cfg)` — Create dataloaders
- **def** `generate_data()` — Generate data

### `datamodules/timeseries_delay_2d.py`
- **class** `TimeseriesDelay2DConfig` — Config
- **def** `_get_version_hash(cfg)` — Cache key
- **def** `build_dataloaders(cfg)` — Create dataloaders
- **def** `generate_data()` — Generate data
- **def** `f(t, u_stack)` — ODE function @jax.jit

### `datamodules/timeseries_viscous_burgers_1d.py`
- **class** `TimeseriesViscousBurgers1DConfig` — Config
- **def** `_get_version_hash(cfg)` — Cache key
- **def** `build_dataloaders(cfg)` — Create dataloaders
- **def** `f(t, u)` — ODE function
- **def** `generate_data()` — Generate data

---

## metrics-core (NEW PACKAGE)

### `metrics/image_diffusion.py`
- **def** `plot(recon, output_dir, name, nrow)` — Visualize images
- **def** `rplot(recon, output_dir, name)` — Reduced plot
- **def** `reconstruction_step(x, net, level)` — Single step
- **def** `reconstruction(net, loader, dirs, level)` — Full reconstruction
- **def** `generation(net, loader, dirs, level, warmup, n_samples)` — Generate samples

### `metrics/conditional_image_diffusion.py`
- **def** `plot(recon, output_dir, name, nrow)` — Visualize
- **def** `rplot(recon, output_dir, name)` — Reduced plot
- **def** `reconstruction(net, loader, dirs, level)` — Conditional reconstruction
- **def** `quick_reconstruction(net, batch, dirs, info)` — Quick version

### `metrics/text.py`
- **def** `token_accuracy(tks, rpns)` — Compute accuracy
- **def** `metrics(net, i, batch, dirs)` — Compute text metrics
- **def** `generation(net, loader, dirs)` — Generate text
- **def** `inverse_metrics(net, i, batch, d, dirs)` — Inverse metrics
- **def** `inverse_metrics_all(net, loader, dirs)` — Batch inverse metrics

---

## ae-core (EXTRACTED PACKAGE)

### `models/project_develop/mnist.py`
- **class** `Config` — Config dataclass
- **class** `MNISTAutoencoder(pl.LightningModule)` — AE model
  - `__init__(config)` — Initialize
  - `forward(x)` — Forward pass
  - `training_step(batch, _)` — Training step
  - `validation_step(batch, _)` — Validation step
  - `configure_optimizers()` — Optimizer config

### `models/project_develop/spatial.py`
- **class** `Config` — Config
- **class** `SpatialAutoencoder(pl.LightningModule)` — Spatial AE
  - `__init__(config)` — Initialize
  - `forward(x)` — Forward pass
  - `training_step(batch, _)` — Training
  - `validation_step(batch, _)` — Validation
  - `configure_optimizers()` — Optimizer config

### `models/modules/act.py`
- **class** `Swish(nn.Module)` — Swish activation
- **class** `Tri(nn.Module)` — Triangular activation
- **def** `saw(x)` — Sawtooth function
- **class** `Tri2(nn.Module)` — Tri variant 2
- **class** `Sharp(nn.Module)` — Sharp activation
- **class** `Sinc(nn.Module)` — Sinc activation
- **class** `Finer(nn.Module)` — Finer activation
- **class** `FFT(nn.Module)` — FFT layer
- **class** `IFFT(nn.Module)` — Inverse FFT layer

### `models/modules/ae.py`
- **class** `ConvBlock` — Conv + norm + activation
- **class** `TransposeConvBlock` — Transpose conv block
- **class** `Encoder(nn.Module)` — Encoder
- **class** `Decoder(nn.Module)` — Decoder

### `models/modules/patch_att.py`
- **class** `PatchAttention(nn.Module)` — Patch-wise attention
- **def** `forward(x)` — Forward pass

### `models/modules/shuffle.py`
- **def** `shuffle_groups(x, num_groups)` — Shuffle tensor groups

### `models/modules/siren.py`
- **class** `Sine(nn.Module)` — Sine activation (SIREN)
- **class** `SirenNet(nn.Module)` — SIREN network

### `models/modules/skip.py`
- **class** `Skip(nn.Module)` — Skip connection

### `models/modules/spatial.py`
- **class** `Spatial(nn.Module)` — Spatial operations
- **def** `forward(x)` — Forward pass

### `models/modules/math/derivative.py`
- **def** `compute_jacobian(model, x)` — Jacobian computation
- **class** `DerivativeNet(nn.Module)` — Network with derivatives

### `models/project_cudafused/cu/tiny_cu.py`
- **class** `CUDALinear` — CUDA-fused linear
- **class** `CUDAConv2d` — CUDA-fused conv
- **def** `compile_kernels()` — Compile CUDA kernels

### `models/project_cudafused/hl/tiny_hl.py`
- **class** `FusedLinear(nn.Module)` — High-level fused linear
- **class** `FusedConv2d(nn.Module)` — High-level fused conv

---

## diffusion (EXTRACTED + NEW PACKAGE)

### `models/project_develop/spatial_diffusion.py`
- **class** `Config` — Config
- **class** `SpatialDiffusion(pl.LightningModule)` — Spatial diffusion model
  - `__init__(config)` — Initialize
  - `forward(x, t)` — Denoise
  - `training_step(batch, _)` — Training
  - `configure_optimizers()` — Optimizer

### `models/modules/diffusion/embeddings.py`
- **class** `TimestepEmbedding(nn.Module)` — Timestep embeddings
- **class** `Fourier(nn.Module)` — Fourier embeddings

### `models/modules/diffusion/samplers/cache.py`
- **class** `SamplerCache` — Caching for sampling
- **def** `get_cached(key)` — Get from cache
- **def** `cache_put(key, value)` — Store in cache

### `models/modules/diffusion/samplers/flow_matching.py`
- **def** `flow_matching_sample(model, x_init, steps)` — Flow matching sampler
- **class** `FlowMatcher(nn.Module)` — Flow matching core

### `models/external/speedrun_dit.py`
- **class** `LoRASpeedrunDiT(nn.Module)` — LoRA-finetuned DiT
  - `__init__(...)` — Initialize with config
  - `forward(x, t, c)` — Forward pass

---

## pde-cond-diffusion (EXTRACTED PACKAGE)

### `models/project_mmai_apr26/diffusion.py`
- **class** `Config` — Config
- **class** `Diffusion(pl.LightningModule)` — Base diffusion
  - `__init__(config)` — Initialize
  - `forward(x)` — Forward
  - `training_step(batch, batch_id)` — Training
  - `validation_step(batch, batch_id)` — Validation
  - `configure_optimizers()` — Optimizer

### `models/project_mmai_apr26/vlm_diffusion.py`
- **class** `Config` — Config
- **class** `Diffusion(pl.LightningModule)` — VLM-conditioned diffusion
  - `__init__(config)` — Initialize
  - `denoise(x, t, c, latent)` — Denoise with VLM conditioning
  - `loss(x, c, latent)` — Compute loss
  - `training_step(batch, _, logger)` — Training
  - `validate_step(batch, _, logger)` — Validation
  - `metrics(assistant)` — Compute metrics
  - `configure_optimizers()` — Optimizer

### `models/project_mmai_apr26/vlm_diffusion_srdit.py`
- **class** `Config` — Config
- **class** `Diffusion(pl.LightningModule)` — SRDIT variant with VLM
  - (Similar methods to vlm_diffusion.py)

### `models/project_mmai_apr26/vlm.py`
- **class** `Config` — Config
- **class** `OptVLMDiffusion(pl.LightningModule)` — Optimized VLM diffusion
  - `__init__(config)` — Initialize
  - `compute_latent(rpns)` — Compute latent from RPNs
  - `encode_LLM(rpns)` — Encode with LLM
  - `compute_from_LLM(encoding)` — Compute from LLM output
  - `export_from_LLM(encoding)` — Export learned representations
  - `gen()` — Generate
  - `training_step(batch, batch_id)` — Training
  - `validation_step(batch, batch_id)` — Validation
  - `inverse_solver(seq)` — Solve inverse problem
- **def** `safe_load_state_dict(model, ckpt)` — Safe checkpoint loading

### `models/project_mmai_apr26/operator_diffusion.py`
- **class** `Config` — Config
- **class** `Diffusion(pl.LightningModule)` — Operator-conditioned diffusion
  - `__init__(config)` — Initialize
  - `denoise(x, t, c)` — Denoise with operator
  - `forward(x)` — Forward
  - `training_step(batch, batch_id)` — Training
  - `validation_step(batch, batch_id)` — Validation
  - `configure_optimizers()` — Optimizer

### `models/project_mmai_apr26/operator_diffusion_latent.py`
- **class** `Config` — Config
- **class** `Diffusion(pl.LightningModule)` — Operator conditioning in latent space
  - (Similar to operator_diffusion.py)

### `models/project_mmai_apr26/operator_pixart.py`
- **class** `Config` — Config
- **class** `Diffusion(pl.LightningModule)` — PixArt operator variant

### `models/project_mmai_apr26/operator_srdit.py`
- **class** `Config` — Config
- **class** `Diffusion(pl.LightningModule)` — SRDIT operator variant
  - (Similar methods, SRDIT-specific implementation)

### `models/project_mmai_apr26/llm.py`
- **class** `LLMComponent(nn.Module)` — LLM integration
- **def** `encode_text(text)` — Encode text with LLM
- **def** `generate_text(encoding)` — Generate from encoding

### `models/project_mmai_apr26/qwen_llm.py`
- **class** `QwenCRPNConfig` — Config for Qwen
- **class** `QwenCRPNAutoencoder(pl.LightningModule)` — Qwen LLM AE
  - `__init__(config)` — Initialize with Qwen
  - `encode(rpns)` — Encode RPNs
  - `decode(z)` — Decode latent
  - `sample(z)` — Sample from latent
  - `training_step(batch, _, logger)` — Training
  - `validation_step(batch, _, logger)` — Validation
  - `metrics(assistant)` — Compute metrics
  - `configure_optimizers()` — Optimizer

### `metrics/vlm_image_diffusion.py`
- **def** `plot_vlm(recon, output_dir, name)` — VLM-specific plots
- **def** `reconstruction_vlm(net, loader, dirs)` — VLM reconstruction
- **def** `metrics_vlm(net, loader, dirs)` — VLM metrics

---

## mura (ENHANCED — Add to Existing)

### `util/gitinfo.py` (ALREADY THERE)
- **def** `_compute_diff()` — Compute git diff
- **def** `_gitinfo_resolver(key)` — Hydra resolver for git info

### `util/sec_id.py` (ALREADY THERE)
- **def** `_compact_sec_id()` — Generate compact ID
- **def** `_register_resolver()` — Register Hydra resolver

### NEW: `registry.py`
- **class** `Registry[T]` — Generic registry
  - `register(key, config_cls, builder)` — Register item
  - `get(key)` — Retrieve by key
  - `list_all()` — List all keys

### NEW: `experiment.py`
- **class** `TaskConfig` — Task configuration
- **def** `run_task(cfg)` — Dispatcher based on cfg.task

### NEW: `checkpointing.py`
- **def** `save_checkpoint(model, path, cfg, metadata)` — Save with metadata
- **def** `load_checkpoint(path)` — Load checkpoint + config

### NEW: `metrics.py`
- **def** `compute_psnr(pred, target)` — PSNR
- **def** `compute_ssim(pred, target)` — SSIM
- **def** `aggregate_metrics(preds, targets, metric_names)` — Batch metrics

---

## diffusion-information-studies (NEW PACKAGE)

### NEW: `operators/base.py`
- **class** `Operator` — Abstract operator base
  - `forward(x)` — Apply operator
  - `adjoint(y)` — Adjoint operator

### NEW: `operators/camera.py`
- **def** `camera_projection(x, height, width)` — Camera operator function
- **class** `CameraOperator(Operator)` — Wrapper for Hydra

### NEW: `operators/inpainting.py`
- **def** `inpainting_mask(x, mask)` — Inpainting operator function
- **class** `InpaintingOperator(Operator)` — Wrapper

### NEW: `solvers/dps.py`
- **def** `dps_step(x_t, y, measurement_op, diffusion_model, t)` — Single DPS step
- **class** `DPSSolver` — Full DPS solver
  - `solve(y, measurement_op)` — Solve inverse problem

### NEW: `run.py`
- **def** `main(cfg)` @hydra.main — Study entry point
- **def** `run_dps_study(cfg)` — Run DPS experiment
- **def** `run_conditioning_study(cfg)` — Run conditioning experiment

### NEW: `config.py`
- **class** `OperatorConfig` — Operator config
- **class** `SolverConfig` — Solver config
- **class** `StudyConfig` — Study config

---

## Summary Statistics

| Destination | Files | Classes | Functions | Total |
|-------------|-------|---------|-----------|-------|
| Orchestrator (keep) | 4 | 1 | 12 | 13 |
| Orchestrator (datamodules) | 15 | 15 | 45 | 60 |
| metrics-core | 3 | 0 | 12 | 12 |
| ae-core (models) | 2 | 2 | 10 | 12 |
| ae-core (modules) | 10 | 30 | 15 | 45 |
| ae-core (cudafused) | 12 | 15 | 20 | 35 |
| diffusion (existing) | 5 | 8 | 8 | 16 |
| diffusion (external) | 2 | 1 | 5 | 6 |
| pde-cond-diffusion | 11 | 40 | 80 | 120 |
| mura (existing) | 2 | 0 | 2 | 2 |
| mura (new) | 4 | 5 | 15 | 20 |
| diffusion-information-studies (new) | 7 | 8 | 25 | 33 |
| **TOTAL** | **78** | **135** | **251** | **374** |

---

**Last updated**: Auto-generated from AST parsing of all Python files in `src/autoencoders/`
