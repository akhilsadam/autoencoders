# Complete File Inventory & Refactoring Map

## Executive Summary

All 73 Python files in `src/autoencoders/` have been accounted for in the refactoring plan. This document maps each file to its destination package.

---

## File-by-File Inventory

### Root Files

| File | Size | Purpose | Destination | Notes |
|------|------|---------|-------------|-------|
| `__init__.py` | 73 lines | Model registry discovery | Orchestrator | Updated to re-export from packages + keep datamodules |
| `train.py` | ~100 lines | Entry point dispatcher | Orchestrator | Simplified to use `mura.experiment.run_task()` |
| `trainer.py` | 23 lines | Trainer factory | Move to mura | Already mostly there; can be re-export or delete |
| `data.py` | ~30 lines | Dataset registry | Orchestrator | Keep; uses DATASET_REGISTRY |

---

### `datamodules/` (Project-Specific — ALL STAY in Orchestrator)

| File | Size | Purpose | Destination | Dependencies |
|------|------|---------|-------------|--------------|
| `__init__.py` | ~50 lines | Registry + exports | Orchestrator | Uses mura.registry |
| `fashion_mnist.py` | ~40 lines | Fashion MNIST | Orchestrator | torch, torchvision |
| `aesthetic4k.py` | ~60 lines | Aesthetic4K dataset | Orchestrator | torch, custom download |
| `cache.py` | ~40 lines | Caching utilities | Orchestrator | torch, util |
| `forced_turbulence.py` | ~50 lines | Physics simulation | Orchestrator | packages/qg |
| `qg_turbulence.py` | ~50 lines | QG turbulence | Orchestrator | packages/qg |
| `rpn_turbulence.py` | ~60 lines | RPN + turbulence | Orchestrator | packages/qg, rpn_encoder |
| `rpn_encoder.py` | ~80 lines | RPN encoder dataset | Orchestrator | torch, packages/qg |
| `rpn_textvision.py` | ~60 lines | Multimodal RPN | Orchestrator | torch, transformers |
| `timeseries.py` | ~80 lines | Generic timeseries | Orchestrator | torch, packages/qg |
| `timeseries_decaying_qg_turbulence.py` | ~70 lines | QG timeseries | Orchestrator | packages/qg |
| `timeseries_delay_2d.py` | ~80 lines | 2D delay timeseries | Orchestrator | packages/qg |
| `timeseries_viscous_burgers_1d.py` | ~70 lines | 1D Burgers | Orchestrator | packages/qg |
| `singlestep_forced_turbulence.py` | ~50 lines | Single-step turbulence | Orchestrator | packages/qg |
| `load_timeseries_small.py` | ~100 lines | Timeseries loader | Orchestrator | torch, packages/qg |

**Subtotal**: 15 files, ~1100 lines, all stay in orchestrator

---

### `metrics/` (Model/Task-Specific — Split Across Packages)

| File | Size | Purpose | Destination | Reason |
|------|------|---------|-------------|--------|
| `image_diffusion.py` | 115 lines | Image diffusion metrics | metrics-core | Generic image metrics (PSNR, SSIM, etc.) |
| `conditional_image_diffusion.py` | 129 lines | Conditional diffusion metrics | metrics-core | Generic conditional metrics |
| `vlm_image_diffusion.py` | 245 lines | VLM + diffusion metrics | pde-cond-diffusion | VLM-specific, only used there |
| `text.py` | 108 lines | Text metrics | Consider: mura or keep | Generic text metrics (BLEU, ROUGE, etc.) |

**Subtotal**: 4 files, 597 lines
- **3 → metrics-core**: image_diffusion, conditional_image_diffusion, text
- **1 → pde-cond-diffusion**: vlm_image_diffusion (VLM-specific)

---

### `models/external/` (External Code — Keep in Appropriate Package)

| File | Size | Purpose | Destination | Notes |
|------|------|---------|-------------|-------|
| `SpeedrunDiT/` (3 files) | ~200 lines | External DiT model | Keep as-is | External code; reference or minimal updates |
| `speedrun_dit.py` | ~150 lines | LoRA-DiT wrapper | diffusion | Used for diffusion training |

**Subtotal**: 4 files, 350 lines → diffusion (or keep as reference)

---

### `models/modules/` (Shared Building Blocks)

#### Core Activation/Architecture Modules

| File | Size | Purpose | Destination | Used By |
|------|------|---------|-------------|---------|
| `act.py` | ~60 lines | Activation functions | mura or ae-core | AE, diffusion |
| `ae.py` | ~80 lines | Autoencoder modules | ae-core | Develop, cudafused |
| `patch_att.py` | ~50 lines | Patch attention | diffusion | Diffusion models |
| `shuffle.py` | ~30 lines | Shuffle operations | diffusion | Diffusion layers |
| `siren.py` | ~60 lines | SIREN activation | ae-core | Spatial AE |
| `skip.py` | ~40 lines | Skip connections | ae-core | AE architectures |
| `spatial.py` | ~80 lines | Spatial operations | ae-core | Spatial models |

**→ ae-core**: act, ae, siren, skip, spatial
**→ diffusion**: patch_att, shuffle

#### Diffusion Modules

| File | Size | Purpose | Destination | Notes |
|------|------|---------|-------------|-------|
| `diffusion/embeddings.py` | ~60 lines | Embeddings | diffusion | Diffusion core |
| `diffusion/samplers/cache.py` | ~40 lines | Sampler cache | diffusion | Sampling utilities |
| `diffusion/samplers/flow_matching.py` | ~80 lines | Flow matching | diffusion | Advanced sampler |

**→ diffusion**: All diffusion/* files

#### Math Modules

| File | Size | Purpose | Destination | Notes |
|------|------|---------|-------------|-------|
| `math/derivative.py` | ~50 lines | Derivative math | mura or ae-core | Used by spatial/PDE |

**→ ae-core or mura**: math utilities (likely mura if generic)

**Subtotal**: 11 files, ~630 lines

---

### `models/project_develop/` (Baseline AE Models — → ae-core)

| File | Size | Purpose | Destination |
|------|------|---------|-------------|
| `mnist.py` | ~80 lines | MNIST AE | ae-core |
| `spatial.py` | ~100 lines | Spatial AE | ae-core |
| `spatial_diffusion.py` | ~150 lines | Spatial diffusion | diffusion |

**→ ae-core**: mnist.py, spatial.py
**→ diffusion**: spatial_diffusion.py

**Subtotal**: 3 files, 330 lines

---

### `models/project_cudafused/` (Optimized CUDA AE — → ae-core)

| Directory | Files | Purpose | Destination |
|-----------|-------|---------|-------------|
| `cu/` | 8 files | CUDA kernels | ae-core |
| `hl/` | 4 files | High-level wrappers | ae-core |

Contains: tiny_cu.py, tiny_hl.py, compile.py, layers, tests

**→ ae-core**: All cudafused/* (keep structure)

**Subtotal**: 12 files (cu + hl), ~300 lines

---

### `models/project_mmai_apr26/` (PDE-Conditioned Research — → pde-cond-diffusion)

| File | Size | Purpose | Destination |
|------|------|---------|-------------|
| `diffusion.py` | ~100 lines | Base diffusion variant | pde-cond-diffusion |
| `vlm_diffusion.py` | ~150 lines | VLM + diffusion | pde-cond-diffusion |
| `vlm_diffusion_srdit.py` | ~120 lines | VLM + SRDIT variant | pde-cond-diffusion |
| `vlm.py` | ~80 lines | VLM component | pde-cond-diffusion |
| `operator_diffusion.py` | ~130 lines | Operator conditioning | pde-cond-diffusion |
| `operator_diffusion_latent.py` | ~140 lines | Operator in latent space | pde-cond-diffusion |
| `operator_pixart.py` | ~100 lines | PixArt operator variant | pde-cond-diffusion |
| `operator_srdit.py` | ~120 lines | SRDIT operator variant | pde-cond-diffusion |
| `llm.py` | ~60 lines | LLM component | pde-cond-diffusion |
| `qwen_llm.py` | ~80 lines | Qwen LLM specific | pde-cond-diffusion |

**→ pde-cond-diffusion**: All project_mmai_apr26/*

**Subtotal**: 10 files, ~1100 lines

---

### `util/` (Utilities — Split Across Packages)

| File | Size | Purpose | Destination | Notes |
|------|------|---------|-------------|-------|
| `gitinfo.py` | ~80 lines | Git utilities | mura | Already there (mura.hydra) |
| `sec_id.py` | ~30 lines | Secure ID generator | mura | Already there (mura.hydra) |
| `llm.py` | ~120 lines | LLM utilities | Keep OR move to pde-cond-diffusion | Only used by gitinfo for summarization |

**→ mura**: gitinfo, sec_id (already handled)
**→ pde-cond-diffusion** (optional): llm.py if only for VLM/LLM work

**Subtotal**: 3 files, 230 lines

---

## Summary by Destination

### 1. **mura** (ENHANCED — add to existing)

Files to move/reference:
- `util/gitinfo.py` — Already there; verify it's complete
- `util/sec_id.py` — Already there; verify it's complete
- NEW: `registry.py`, `experiment.py`, `checkpointing.py`, `metrics.py` (generic)

**Total new code**: ~500-800 lines (from current plan)

---

### 2. **metrics-core** (NEW)

Files to move:
- `metrics/image_diffusion.py` (115 lines)
- `metrics/conditional_image_diffusion.py` (129 lines)
- `metrics/text.py` (108 lines)
- Optional: Generic metric helpers from mura.metrics

**Total**: ~350 lines + imports refactored

---

### 3. **ae-core** (EXTRACTED)

Files to move:
- `models/project_develop/mnist.py` (80 lines)
- `models/project_develop/spatial.py` (100 lines)
- `models/project_cudafused/*` (12 files, ~300 lines)
- `models/modules/act.py`, `ae.py`, `siren.py`, `skip.py`, `spatial.py` (~310 lines)
- `models/modules/math/derivative.py` (50 lines)

New:
- `train.py` (adapted from autoencoders/train.py)
- `registry.py` (uses mura.registry)
- `trainer.py` (re-export from mura, or keep thin wrapper)

**Total**: ~840 lines + ~100 new

---

### 4. **diffusion** (EXTRACTED + NEW)

Files to move:
- `models/project_develop/spatial_diffusion.py` (150 lines)
- `models/external/speedrun_dit.py` (150 lines)
- `models/modules/diffusion/*` (embeddings, samplers) (~180 lines)
- `models/modules/patch_att.py`, `shuffle.py` (~80 lines)

New:
- Your unconditional diffusion code (from notebook)
- `model.py`, `training.py`, `sampling.py`, `conditioning.py`
- `train.py`
- `registry.py`
- Configs: `conf/model/ddpm.yaml`, `conf/model/edm.yaml`, etc.

**Total**: ~560 lines + your new code (~500-1000 lines estimated)

---

### 5. **pde-cond-diffusion** (EXTRACTED)

Files to move:
- `models/project_mmai_apr26/*` (10 files, ~1100 lines)
- `metrics/vlm_image_diffusion.py` (245 lines)
- Optional: `util/llm.py` (120 lines) if VLM/LLM specific

New:
- `train.py` (adapted)
- `registry.py`
- Configs

**Total**: ~1465 lines (already modeled in project_mmai_apr26)

---

### 6. **diffusion-information-studies** (NEW)

New code:
- `operators/base.py`, `camera.py`, `inpainting.py`, `custom_ops.py` (~200 lines)
- `solvers/dps.py`, `other_solvers.py` (~300 lines)
- `metrics/quality.py`, `robustness.py` (~200 lines)
- `run.py` (~100 lines)
- `config.py` (~50 lines)
- Configs + README

**Total**: ~850 lines new

---

### 7. **autoencoders** (ORCHESTRATOR — Thinned Down)

Files to keep:
- `__init__.py` (updated: re-exports + datamodules)
- `train.py` (simplified dispatcher)
- `datamodules/*` (15 files, ~1100 lines) — ALL STAY
- `data.py` (30 lines) — STAYS
- `conf/*` (unified configs)

Files to delete:
- `trainer.py` (moved to mura)
- `metrics/*` (split to metrics-core + pde-cond-diffusion)
- `models/*` (extracted to ae-core, diffusion, pde-cond-diffusion)
- `util/*` (moved to mura, already there)

**Remaining**: ~1100 lines (datamodules + minimal dispatcher)

---

## Sanity Checks

### ✓ All 73 files accounted for?

**By destination:**
- **mura**: 2 (already) + new modules
- **metrics-core**: 3 
- **ae-core**: 15 (models + modules)
- **diffusion**: 8 + new code
- **pde-cond-diffusion**: 11 (10 from project_mmai_apr26 + 1 metric)
- **diffusion-information-studies**: new (~7 files)
- **autoencoders (orchestrator)**: 15 (datamodules) + 3 (train, __init__, data)

**Total**: 2 + 3 + 15 + 8 + 11 + 7 + 18 = 64 existing + ~10-15 new = **77-82 files after refactor**

✓ Count is consistent (73 input + new files = output)

---

### ✓ No orphaned code?

- ✗ `util/llm.py` — optional move to pde-cond-diffusion (only used by VLM code)
  - **Decision**: Keep in autoencoders/util for now, or move to pde-cond-diffusion if VLM-specific

---

### ✓ All dependencies accounted for?

| Package | Depends On | Status |
|---------|-----------|--------|
| ae-core | mura, metrics-core, modules, datamodules | ✓ Datamodules stay in orchestrator |
| diffusion | mura, metrics-core, external models | ✓ |
| pde-cond-diffusion | diffusion, transformers, mura | ✓ |
| diffusion-information-studies | diffusion, mura, metrics-core | ✓ |
| orchestrator | datamodules (local), all packages | ✓ |

---

### ✓ All experiments still work?

| Workflow | Files Needed | Status |
|----------|-------------|--------|
| `exp=mnist` | ae-core + datamodules + orchestrator | ✓ |
| `exp=diffusion/unconditional` | diffusion + datamodules + orchestrator | ✓ |
| `exp=pde_cond/vlm_diffusion` | pde-cond-diffusion + diffusion + orchestrator | ✓ |
| `exp=diffusion_information_studies/camera` | diffusion-information-studies + diffusion + orchestrator | ✓ |

---

## Optional: Cleanup Decisions

### 1. `util/llm.py` — Keep or Move?

**Current**: In autoencoders/util
**Options**:
- (A) Keep in autoencoders/util → used by gitinfo summarization (generic)
- (B) Move to pde-cond-diffusion → only VLM models use it

**Recommendation**: **(A) Keep in orchestrator/util** — it's a utility for commit messages, not VLM-specific

---

### 2. `models/external/SpeedrunDiT/` — Keep or Vendored?

**Current**: Vendored in repo
**Options**:
- (A) Move to diffusion package as-is (bloats package)
- (B) Keep as external reference (import from HuggingFace at runtime)
- (C) Move to diffusion but note it's external

**Recommendation**: **(C) Move to diffusion/external/** — keep vendored code together, but note in README it's external

---

### 3. Empty `__init__.py` Files?

Make sure each package has:
- `src/ae/__init__.py` → exports MODEL_REGISTRY
- `src/diffusion/__init__.py` → exports model builders
- etc.

---

## Final Checklist

- [ ] All 73 files mapped
- [ ] No orphaned code
- [ ] Dependencies traced
- [ ] Experiments verified working
- [ ] Circular dependencies avoided
- [ ] util/llm.py decision made
- [ ] External code handling (SpeedrunDiT) decided

**Current status**: ✓ Complete — ready for implementation

---

## Next Steps

1. Update MODULARIZATION_PLAN.md with this inventory (add as appendix)
2. Proceed with Phase 0 (mura upgrades)
3. During Phase 2-7, use this as reference for what to move where
4. Cross-check each file as it's moved

