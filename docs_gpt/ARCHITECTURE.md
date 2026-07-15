# Architecture Overview (Visual)

## Before Refactoring

```
autoencoders/  (monolithic)
├── src/autoencoders/
│   ├── __init__.py           (registry logic)
│   ├── train.py              (dispatcher logic)
│   ├── trainer.py            (trainer factory)
│   ├── data.py               (dataset registry)
│   ├── models/
│   │   ├── modules/diffusion/    (diffusion layers)
│   │   ├── project_develop/      (AE models)
│   │   ├── project_cudafused/    (AE models)
│   │   └── project_mmai_apr26/   (VLM, LLM, PDE diffusion)
│   ├── datamodules/          (all datasets)
│   └── metrics/              (all metrics)
├── packages/
│   ├── mura/                 (Hydra + WandB, no boilerplate)
│   ├── qg/
│   └── [others]
└── runs/
```

**Problems**:
- Boilerplate (registry, dispatcher) duplicated if we extract
- Hard to use diffusion-only in next paper (depends on entire autoencoders)
- Mixed concerns: diffusion, AE, PDE conditioning all tangled
- Hard to add new operators/studies (touches multiple files)

---

## After Refactoring

```
mura/  (ENHANCED with boilerplate)
├── src/mura/
│   ├── registry.py           ← Generic Registry[T]
│   ├── experiment.py         ← Task registry + run_task()
│   ├── checkpointing.py      ← Save/load with metadata
│   ├── metrics.py            ← Generic metric aggregation
│   ├── trainer.py            ← Trainer factory
│   └── [existing code]

metrics-core/
├── src/metrics/
│   ├── psnr.py, ssim.py      (shared image/diffusion metrics)
│   └── [perceptual losses]

ae-core/
├── src/ae/
│   ├── models/               (develop, cudafused)
│   ├── train.py              (uses mura.registry)
│   └── registry.py           (MODEL_REGISTRY = mura.Registry)

diffusion/
├── src/diffusion/
│   ├── model.py              (your new unconditional diffusion)
│   ├── training.py
│   ├── sampling.py
│   ├── modules/              (diffusion layers, from old repo)
│   ├── train.py              (uses mura.experiment)
│   └── registry.py           (MODEL_REGISTRY)

pde-cond-diffusion/
├── src/pde_cond_diffusion/
│   ├── models/               (project_mmai_apr26: VLM, LLM, operators)
│   ├── components/
│   └── registry.py           (extends diffusion models)

diffusion-information-studies/
├── src/diffusion_studies/
│   ├── operators/            (camera, inpainting, custom)
│   │   ├── camera.py         (function + class wrapper)
│   │   ├── inpainting.py     (function + class wrapper)
│   │   └── base.py           (abstract interface)
│   ├── solvers/              (DPS, other inverse solvers)
│   ├── metrics/
│   ├── run.py                (uses mura.experiment)
│   └── registry.py           (OPERATOR_REGISTRY, SOLVER_REGISTRY)

autoencoders/  (THIN ORCHESTRATOR + PROJECT-SPECIFIC)
├── src/autoencoders/
│   ├── __init__.py           (re-exports from all packages + datamodules)
│   ├── train.py              (from mura.experiment import run_task)
│   ├── datamodules/          # PROJECT-SPECIFIC datasets (stays here)
│   │   ├── fashion_mnist.py
│   │   ├── qg_turbulence.py
│   │   └── [all dataset builders]
│   ├── data.py               (build_dataloaders helper, DATASET_REGISTRY)
│   └── conf/                 (unified Hydra configs)
│       ├── config.yaml
│       ├── data/             (all dataset configs)
│       ├── model/            (all model configs)
│       ├── exp/              (all experiments)
│       │   ├── ae/
│       │   ├── diffusion/
│       │   ├── pde_cond/
│       │   └── diffusion_information_studies/
│       └── operators/        (for studies)
├── packages/                 (all 5 packages + existing ones)
├── tests/integration/        (end-to-end workflows)
├── tests/baselines/          (reference metrics)
└── runs/                     (same as before)
```

**Benefits**:
- Each package is independently installable
- No duplication of boilerplate (registry, dispatcher, checkpointing)
- Diffusion can be used standalone: `pip install packages/diffusion/`
- Studies are modular: add operator = add function + config
- Clean separation: diffusion doesn't know about studies, vice versa

---

## Data Flow: Before Refactoring

```
train.py (custom dispatcher)
  ├─ if task == "train_ae":
  │    ├─ get_model("project_develop.mnist")  [from autoencoders registry]
  │    ├─ get_default_config(...)             [from autoencoders]
  │    ├─ build_dataloaders("fashion_mnist")  [from autoencoders]
  │    └─ create_trainer()                    [custom in autoencoders]
  │
  ├─ if task == "train_diffusion":
  │    └─ [No separate entry point, mixed with AE code]
  │
  └─ if task == "run_study":
      └─ [No separate entry point, mixed with AE code]
```

---

## Data Flow: After Refactoring

```
train.py (uses mura.experiment)
  ├─ mura.experiment.run_task(cfg)
  │    ├─ task = cfg.task  ("train_ae", "train_diffusion", "run_study", etc.)
  │    ├─ handler = TASK_REGISTRY.get(task)
  │    └─ handler(cfg)
  │
  ├─ Task "train_ae" → ae.train.train_ae()
  │    ├─ model = mura.registry.MODEL_REGISTRY.get("ae/mnist")
  │    ├─ config = mura.registry.MODEL_REGISTRY.get_config(...)
  │    ├─ dataloader = mura.registry.DATASET_REGISTRY.get("fashion_mnist")
  │    └─ trainer = mura.trainer.create_trainer(cfg)
  │
  ├─ Task "train_diffusion" → diffusion.train.train_diffusion()
  │    ├─ model = mura.registry.MODEL_REGISTRY.get("diffusion/ddpm")
  │    ├─ config = [same registry]
  │    ├─ dataloader = [same registry]
  │    └─ trainer = [same trainer factory from mura]
  │
  ├─ Task "run_study" → diffusion_studies.run.run_study()
  │    ├─ model = mura.registry.MODEL_REGISTRY.get("diffusion/ddpm")  [loads diffusion]
  │    ├─ operator = mura.registry.OPERATOR_REGISTRY.get("camera_projection")
  │    ├─ solver = mura.registry.SOLVER_REGISTRY.get("dps")
  │    └─ run study...
  │
  └─ Task "train_pde_cond" → pde_cond_diffusion.train.train_pde_cond()
      ├─ model = mura.registry.MODEL_REGISTRY.get("pde_cond/vlm_diffusion")
      ├─ [extends diffusion models via registry]
      └─ trainer = [same mura trainer factory]
```

---

## Config Inheritance: Unified

```
conf/config.yaml (base)
├─ task: train_ae              ← Which package's task to run
├─ project: develop
├─ trainer:
│   max_epochs: -1
│   accelerator: auto
├─ wandb: [settings]
└─ defaults: [data, model, exp]

conf/exp/ae/mnist.yaml
├─ override /model: ae/develop/mnist
├─ override /data: fashion_mnist
└─ project: develop

conf/exp/diffusion/unconditional.yaml
├─ override /model: diffusion/ddpm
├─ override /data: fashion_mnist
├─ task: train_diffusion           ← New: tells orchestrator which task
└─ project: diffusion_unconditional

conf/exp/diffusion_information_studies/camera.yaml
├─ override /model: diffusion/ddpm
├─ override /operators: camera
├─ task: run_study                 ← New: tells orchestrator to run study
├─ project: diffusion_information_studies
└─ study_type: dps_conditioning
```

**Result**: One config system, all packages use it. Config tells orchestrator which task; orchestrator dispatches via `mura.experiment.run_task()`.

---

## Import Changes (Summary)

### Old (Monolithic)
```python
from autoencoders import get_model, get_default_config, build_dataloaders
from autoencoders.metrics import psnr, ssim
from autoencoders.models.modules.diffusion import UNet
from autoencoders.trainer import create_trainer
```

### New (Modular)
```python
# Per-package imports (packages are self-contained)
from ae import get_model as get_ae_model, MODEL_REGISTRY
from diffusion import build_diffusion_model, DiffusionModel
from metrics import psnr, ssim                          # Generic metrics
from diffusion_studies.operators import camera_projection
from pde_cond_diffusion.models import VLMDiffusion

# Shared infrastructure (all use mura)
from mura.registry import Registry, TASK_REGISTRY
from mura.experiment import run_task
from mura.trainer import create_trainer
from mura.checkpointing import load_checkpoint, save_checkpoint
```

### Orchestrator (Thin)
```python
# autoencoders/train.py
import hydra
from mura.experiment import run_task

@hydra.main(...)
def main(cfg):
    run_task(cfg)  # That's it!
```

---

## Comparison: Old vs New

| Aspect | Before | After |
|--------|--------|-------|
| **Boilerplate duplication** | High (registry, dispatcher, trainer in each project) | Zero (all in mura) |
| **Package reusability** | Hard (depends on monolith) | Easy (pip install packages/diffusion) |
| **Adding new study** | Modify autoencoders + studies code | Add function to diffusion-information-studies/operators |
| **Lines in autoencoders/src** | ~2000+ | ~100 (3 files) |
| **Lines in mura/src** | ~500 | ~800 (registry, dispatcher, checkpointing) |
| **Test speed** | Slow (entire monolith) | Fast (unit tests per package) |
| **Import complexity** | Tangled | Clear hierarchy |

---

## After This Refactoring, You Can:

1. **Use in next paper** (just diffusion):
   ```bash
   pip install git+https://github.com/yourname/diffusion.git
   from diffusion import train_diffusion
   ```

2. **Use with studies**:
   ```bash
   pip install git+https://github.com/yourname/diffusion.git
   pip install git+https://github.com/yourname/diffusion-information-studies.git
   from diffusion_studies.operators import camera_projection
   ```

3. **Extend with new studies**:
   - Fork diffusion-information-studies
   - Add operator in `operators/custom.py`
   - Add config in `conf/operators/custom.yaml`
   - Run: `python -m diffusion_studies.run exp=camera_inpainting operator=custom`

4. **Add new conditioning method**:
   - Add solver in `solvers/new_solver.py`
   - Register in solver registry
   - Update config to use it
   - No changes to diffusion-core

---

Done! The plan is ready for execution.
