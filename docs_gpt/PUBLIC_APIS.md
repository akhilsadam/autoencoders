# Public APIs by Package (Summary)

Quick reference of what each package exports. Use this for integration & import planning.

---

## mura (Foundational)

**What to import from:**
```python
from mura.registry import Registry
from mura.experiment import TASK_REGISTRY, run_task, TaskConfig
from mura.checkpointing import save_checkpoint, load_checkpoint
from mura.metrics import compute_psnr, compute_ssim, aggregate_metrics
from mura.trainer import create_trainer
from mura.hydra import register_resolvers, compute_git_info  # Already there
```

**Used by**: All packages

---

## metrics-core

**What to import from:**
```python
from metrics.image_diffusion import reconstruction, generation, plot
from metrics.conditional_image_diffusion import reconstruction, quick_reconstruction
from metrics.text import metrics, generation, token_accuracy
```

**Used by**: diffusion, pde-cond-diffusion (for training metrics)

---

## ae-core

**What to import from:**
```python
from ae import get_model, get_default_config, MODEL_REGISTRY
from ae.models.develop import MNISTAutoencoder, SpatialAutoencoder
from ae.models.cudafused import FusedLinear, FusedConv2d
from ae.train import train_ae
```

**Entry point**: `ae.train.train_ae(cfg)`

**Registry key format**: `"develop.mnist"`, `"develop.spatial"`, `"cudafused.cu"`, etc.

---

## diffusion

**What to import from:**
```python
from diffusion import MODEL_REGISTRY, get_model, build_diffusion_model
from diffusion.models import DDPMDiffusion, EDMDiffusion
from diffusion.train import train_diffusion
from diffusion.sampling import sample, sample_with_guidance
from diffusion.conditioning import classifier_free_guidance_step
from diffusion.modules.embeddings import TimestepEmbedding
from diffusion.modules.samplers import FlowMatcher
```

**Entry point**: `diffusion.train.train_diffusion(cfg)`

**Registry key format**: `"ddpm"`, `"edm"`, `"speedrun_dit"`, etc.

**Used by**: diffusion-information-studies (loads diffusion model), pde-cond-diffusion (extends)

---

## pde-cond-diffusion

**What to import from:**
```python
from pde_cond_diffusion import MODEL_REGISTRY
from pde_cond_diffusion.models.vlm_diffusion import VLMDiffusion, VLMDiffusionSRDIT
from pde_cond_diffusion.models.operator_diffusion import OperatorDiffusion, OperatorLatentDiffusion
from pde_cond_diffusion.components import VLM, LLM, QwenLLM
from pde_cond_diffusion.train import train_pde_cond
```

**Entry point**: `pde_cond_diffusion.train.train_pde_cond(cfg)`

**Registry key format**: `"vlm_diffusion"`, `"operator_diffusion"`, `"operator_srdit"`, etc.

**Used by**: orchestrator directly (via config)

---

## diffusion-information-studies

**What to import from:**
```python
from diffusion_studies import run_study, OPERATOR_REGISTRY, SOLVER_REGISTRY
from diffusion_studies.operators import CameraOperator, InpaintingOperator, camera_projection, inpainting_mask
from diffusion_studies.solvers import DPSSolver, run_dps_study
from diffusion_studies.metrics import reconstruction_quality, robustness_score
```

**Entry point**: `diffusion_studies.run.run_study(cfg)`

**Operator registry format**: `"camera"`, `"inpainting"`, etc.

**Solver registry format**: `"dps"`, `"other_solver"`, etc.

**Used by**: orchestrator (studies workflow)

---

## autoencoders (Orchestrator)

**What it exports (thin):**
```python
from autoencoders import get_model, get_default_config  # Re-exported from packages
from autoencoders import build_dataloaders  # Local
from autoencoders.datamodules import DATASET_REGISTRY, list_datasets  # Local
from autoencoders.train import main  # Local dispatcher
```

**Entry point**: `python -m autoencoders.train exp=<exp_name> [overrides]`

**What it depends on**: All 5 packages + mura + local datamodules

---

## Registry Pattern (Used Throughout)

All packages follow mura's Registry pattern:

```python
# Register in package __init__.py or train.py
from mura.registry import Registry

MODEL_REGISTRY = Registry()
MODEL_REGISTRY.register("model_key", ModelConfig, build_model_fn)

# Use in train.py
model_class = MODEL_REGISTRY.get("model_key")
model = model_class(cfg)
```

Same for datasets, operators, solvers, tasks.

---

## Import Chains (Data Flow)

### Old Workflow (AE Training)
```
autoencoders.train
  ├─ autoencoders.__init__ (get_model, get_default_config)
  ├─ autoencoders.data (build_dataloaders)
  ├─ autoencoders.trainer (create_trainer)
  └─ autoencoders.datamodules (dataset builders)
```

### New Workflow (AE Training)
```
autoencoders.train
  ├─ mura.experiment (run_task dispatcher)
  ├─ ae.train (train_ae)
  │   ├─ mura.registry (MODEL_REGISTRY)
  │   ├─ mura.trainer (create_trainer)
  │   └─ mura.metrics (metric aggregation)
  └─ autoencoders.datamodules (dataset builders) [LOCAL]
```

### New Workflow (Diffusion Training)
```
autoencoders.train
  ├─ mura.experiment (run_task dispatcher)
  ├─ diffusion.train (train_diffusion)
  │   ├─ mura.registry (MODEL_REGISTRY)
  │   ├─ metrics-core (metrics)
  │   ├─ mura.checkpointing (save/load)
  │   └─ mura.metrics (aggregation)
  └─ autoencoders.datamodules (dataset builders) [LOCAL]
```

### New Workflow (DPS Studies)
```
autoencoders.train
  ├─ mura.experiment (run_task dispatcher)
  ├─ diffusion_studies.run (run_study)
  │   ├─ diffusion.train (load pre-trained diffusion model)
  │   ├─ mura.registry (OPERATOR_REGISTRY, SOLVER_REGISTRY)
  │   ├─ diffusion_studies.operators (camera_projection, etc.)
  │   └─ diffusion_studies.solvers (DPSSolver)
  └─ autoencoders.datamodules [LOCAL, optional for evaluation]
```

### New Workflow (PDE-Conditioned Training)
```
autoencoders.train
  ├─ mura.experiment (run_task dispatcher)
  ├─ pde_cond_diffusion.train (train_pde_cond)
  │   ├─ diffusion.models (base diffusion model to extend)
  │   ├─ mura.registry (MODEL_REGISTRY)
  │   ├─ metrics-core (diffusion metrics)
  │   └─ pde_cond_diffusion.components (VLM, LLM)
  └─ autoencoders.datamodules [LOCAL]
```

---

## Circular Dependency Check

✓ **No circular dependencies**:
- `mura` has no internal dependencies (foundational)
- `metrics-core` depends on nothing except torch
- `ae-core` depends on: mura, metrics-core
- `diffusion` depends on: mura, metrics-core
- `pde-cond-diffusion` depends on: diffusion, mura, metrics-core (NOT on ae-core)
- `diffusion-information-studies` depends on: diffusion, mura, metrics-core
- `autoencoders` depends on: all packages (orchestrator, can depend on anything)

✓ **Dependency DAG is valid** (no cycles)

---

## Breaking Changes: None (By Design)

### Commands that still work:
```bash
python -m autoencoders.train exp=mnist
python -m autoencoders.train exp=ae_spatial
```

### New commands (via config):
```bash
python -m autoencoders.train exp=diffusion/unconditional task=train_diffusion
python -m autoencoders.train exp=diffusion_information_studies/camera task=run_study
python -m autoencoders.train exp=pde_cond/vlm_diffusion task=train_pde_cond
```

### Standalone usage (future papers):
```python
# Don't need this repo, just the packages
pip install git+https://github.com/yourname/diffusion.git
from diffusion import train_diffusion
train_diffusion(cfg)
```

---

## Implementation Checklist

Use this when implementing each phase:

- [ ] **Phase 0**: mura upgrades
  - [ ] `mura.registry.Registry` class
  - [ ] `mura.experiment.run_task()` function
  - [ ] `mura.checkpointing` module
  - [ ] `mura.metrics` module

- [ ] **Phase 2**: metrics-core
  - [ ] `metrics.image_diffusion` (3 functions)
  - [ ] `metrics.conditional_image_diffusion` (2 functions)
  - [ ] `metrics.text` (5 functions)

- [ ] **Phase 3**: ae-core
  - [ ] `ae.models.develop` (2 Lightning modules)
  - [ ] `ae.models.cudafused` (2+ classes, layers)
  - [ ] `ae.modules` (all module classes)
  - [ ] `ae.registry` (uses mura.registry)
  - [ ] `ae.train` (train_ae function)

- [ ] **Phase 4**: diffusion
  - [ ] `diffusion.models` (your new code)
  - [ ] `diffusion.training` (training loop)
  - [ ] `diffusion.sampling` (samplers)
  - [ ] `diffusion.conditioning` (classifier-free guidance)
  - [ ] `diffusion.modules.diffusion` (embeddings, samplers)
  - [ ] `diffusion.registry` (uses mura.registry)
  - [ ] `diffusion.train` (train_diffusion function)

- [ ] **Phase 5**: pde-cond-diffusion
  - [ ] `pde_cond_diffusion.models` (all 10 files)
  - [ ] `pde_cond_diffusion.components` (VLM, LLM)
  - [ ] `pde_cond_diffusion.metrics` (vlm_image_diffusion)
  - [ ] `pde_cond_diffusion.train` (train_pde_cond function)

- [ ] **Phase 6**: diffusion-information-studies
  - [ ] `diffusion_studies.operators` (base + implementations)
  - [ ] `diffusion_studies.solvers` (DPS + others)
  - [ ] `diffusion_studies.metrics` (quality, robustness)
  - [ ] `diffusion_studies.run` (entry point)

- [ ] **Phase 7**: Thin orchestrator
  - [ ] Update `autoencoders.__init__.py` (re-exports)
  - [ ] Simplify `autoencoders.train.py` (dispatcher)
  - [ ] Register all tasks in `mura.experiment.TASK_REGISTRY`
  - [ ] Keep `autoencoders.datamodules/` (all 15 files)

---

**Total API surface**: ~370 definitions (135 classes, 235 functions) across all packages
