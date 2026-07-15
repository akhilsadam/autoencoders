# Modularization Plan: Autoencoders → Orchestrator + Packages

**Goal**: Transform this repo into a thin orchestrator that stitches together independently-installable packages. This becomes your experiment logbook and integration point. All new code runs and reproduces results exactly (verified via efficient, containerized integration tests). Move ML boilerplate into `mura` to minimize code duplication across packages.

## Current State Analysis

### What We Have Now
- **Main repo** (`src/autoencoders/`): Mixed concerns
  - Core AE models & trainers (develop, cudafused projects)
  - Diffusion code (scattered in modules/diffusion, project_mmai_apr26)
  - VLM/LLM code (project_mmai_apr26)
  - Datamodules (physics-based + vision)
  - Metrics (tied to specific models)
  - Unified Hydra config system
  - Boilerplate: trainer factory, registry logic, train entrypoint
  
- **Existing packages** (`packages/`):
  - `mura`: Hydra + WandB + git utilities (foundational, but can absorb more)
  - `qg`: QG physics simulation
  - `qg-2d`, `qg-2d-obs-closure`, `delay-neural-operator`: Physics support
  - `vectorspace`: Math utilities

- **Dependencies**:
  - Everything currently depends on `autoencoders` as monolith
  - project_mmai_apr26 imports from `autoencoders.metrics` (image_diffusion, etc.)
  - Models are discovered dynamically via registry pattern in `__init__.py`
  - Boilerplate (trainer factory, registry, Hydra setup) is duplicated if extracted separately

### Boilerplate to Extract to `mura`
- **Registry pattern**: Generic model/dataset/solver registry (used by all packages)
- **Trainer factory**: `create_trainer()` from Lightning config
- **Hydra helpers**: `register_resolvers()`, `compute_git_info()`, etc. (already there, but extend)
- **Entry point factory**: Generic `run_task()` dispatcher that works across packages
- **Logging helpers**: WandB integration, artifact management (expand from existing)
- **Checkpointing**: Model save/load with metadata (git info, config, etc.)

---

## Mura Extensions (ML Boilerplate Consolidation)

**New in `mura`** (in addition to existing Hydra + WandB):

### 1. Registry System
**`mura.registry`**: Generic registry for models, datasets, solvers
```python
class Registry(Generic[T]):
    def register(key: str, config_cls, builder_fn): ...
    def get(key: str) -> T: ...
    def list_all() -> List[str]: ...
    
# Usage:
MODEL_REGISTRY = Registry[LightningModule]()
MODEL_REGISTRY.register("diffusion/ddpm", DDPMConfig, build_ddpm)
```

**Benefits**: 
- One pattern, all packages use it
- No duplicate registry logic per package
- Discoverable: `mura.registry.list_all("models")`

### 2. Trainer Factory (Enhanced)
**`mura.trainer`**: Move `create_trainer()` here, add defaults
```python
def create_trainer(cfg: DictConfig, 
                   logger: Optional[Logger] = None,
                   callbacks: Optional[List[Callback]] = None) -> pl.Trainer:
    # Already mostly there, just move + document
```

**New helpers**:
```python
def create_trainer_with_git_callback(cfg: DictConfig) -> pl.Trainer:
    """Auto-add git callback and WandB logger"""
    
def get_default_trainer_config() -> DictConfig:
    """Default trainer settings from mura"""
```

### 3. Experiment Entry Point
**`mura.experiment`**: Generic task dispatcher
```python
@dataclass
class TaskConfig:
    name: str           # "train_diffusion", "run_study", etc.
    handler: Callable   # Function to call
    
TASK_REGISTRY = Registry[TaskConfig]()

def run_task(cfg: DictConfig) -> Any:
    """Look up cfg.task, find handler in registry, run it"""
```

**Usage in orchestrator** (`autoencoders/train.py`):
```python
from mura.experiment import run_task

@hydra.main(...)
def main(cfg):
    run_task(cfg)  # Dispatcher lives in mura now
```

### 4. Checkpoint & Reproducibility
**`mura.checkpointing`**: Unified save/load
```python
def save_checkpoint(model: pl.LightningModule, 
                    path: str, 
                    cfg: DictConfig, 
                    metadata: dict = None) -> None:
    """Save model + config + git info + custom metadata"""
    
def load_checkpoint(path: str) -> Tuple[pl.LightningModule, DictConfig]:
    """Load model + config, restore git context"""
```

**Stored in checkpoint**:
- Model weights
- Full Hydra config
- Git SHA, commit message, dirty status
- Timestamp, seed, system info
- Custom metadata (study params, etc.)

### 5. Metrics Registry & Aggregation
**`mura.metrics`**: Shared metric pipeline
```python
METRIC_REGISTRY = Registry[Callable]()

def aggregate_metrics(predictions, targets, metric_names: List[str]) -> Dict[str, float]:
    """Look up metrics, compute, log to WandB"""
```

**Location**: Common metrics (PSNR, SSIM) move here, not metrics-core
- metrics-core: diffusion/image-specific
- mura.metrics: generic (PSNR, SSIM, L2, etc.)

---

### Package Hierarchy & Dependencies

```
mura (foundational — Hydra + WandB + git)
  ↓
┌─────────────────────────────────────────────┐
│ Physics support (qg, qg-2d, delay-neural-op)│
└─────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────┐
│  ae-core     diffusion     pde-cond-diffusion│
│  (models)    (models)      (research proj)   │
│             ↓ uses common metrics
│         metrics-core (low-level metrics)
└─────────────────────────────────────────────┘
  ↓
diffusion-information-studies (high-level: inverse problems, information metrics)
  ↓
autoencoders (orchestrator: configs + entrypoints)
```

### Package Definitions

#### 1. **metrics-core** (NEW)
**Purpose**: Shared metric functions that don't depend on model architecture.

**Contents**:
- `image_metrics.py`: PSNR, SSIM, perceptual loss (model-agnostic)
- `operator_metrics.py`: Measurement quality, consistency metrics
- `base.py`: Common interfaces

**Dependencies**: torch, einops, numpy

**Exports**: `psnr()`, `ssim()`, `perceptual_loss()`, etc.

**What leaves this repo**: `src/autoencoders/metrics/` → refactored into metrics-core

---

#### 2. **ae-core** (REFACTORED)
**Purpose**: Autoencoder models and trainer infrastructure.

**Contents**:
- `models/`: develop, cudafused projects (models that are Lightning modules)
- `trainer.py`: Trainer factory (re-export from mura)
- `registry.py`: Model registry logic
- `train.py`: Main training entrypoint

**Dependencies**: mura, torch, lightning, hydra, metrics-core

**Exports**: `get_model()`, `get_default_config()`, `train_ae()`

**Key Change**: Model registry searches `ae` package, not `autoencoders`

**What moves**: `src/autoencoders/models/{develop,cudafused}` → `packages/ae-core/src/ae/models/`

**What stays**: Datamodules stay in orchestrator (project-specific; see Phase 3 note)

---

#### 3. **diffusion** (EXTRACTED + NEW)
**Purpose**: Core unconditional diffusion model training & sampling.

**Contents**:
- `model.py`: UNet / DiT architecture (from modules/diffusion)
- `training.py`: Diffusion training loop
- `sampling.py`: Sampling (DDM, ODE solver variants)
- `conditioning.py`: Simple classifier-free guidance (optional, but separate from advanced stuff)
- `config.py`: Dataclasses for diffusion hyperparams
- `registry.py`: Simple model registry (DDPM, EDM, etc.)

**Dependencies**: mura, torch, einops, hydra, metrics-core, diffusers (optional, for reference)

**Exports**: `DiffusionModel`, `train_diffusion()`, `sample()`, diffusion model builders

**Strict Boundary**: Does NOT know about studies, operators, or advanced conditioning. Just model + loss.

**What moves**:
- `src/autoencoders/models/modules/diffusion/` → `packages/diffusion/src/diffusion/modules/`
- Your new unconditional code → `packages/diffusion/src/diffusion/`
- Diffusion model configs → `packages/diffusion/conf/`

---

#### 4. **diffusion-information-studies** (NEW)
**Purpose**: DPS, conditioning methods, custom operators, measurement operators.

**Contains everything that operates ON a diffusion model, not the model itself.**

**Contents**:
- `operators/`: Observation operators (camera, inpainting, etc.)
  - `base.py`: Abstract operator interface
  - `camera.py`: Camera projection operator
  - `inpainting.py`: Inpainting mask operator
  - Custom measurement operators (user-editable)
  
- `solvers/`: Inverse problem solvers
  - `dps.py`: Diffusion Posterior Sampling
  - `other_solvers.py`: Other conditioning methods
  
- `metrics/`: Study-specific evaluation
  - `quality.py`: Reconstruction quality metrics
  - `robustness.py`: Perturbation robustness
  
- `run.py`: High-level study execution
- `config.py`: Study configuration dataclasses
- `README.md`: Tutorial + examples for high-school level

**Dependencies**: diffusion, metrics-core, torch, numpy

**Exports**: `run_study()`, `DPSSolver`, operator classes

**Design**: Functionally composable, not OOP-heavy. Each operator is a function + simple wrapper.

**What's new**: Your studies live here, easily fork-able for next paper.

---

#### 5. **pde-cond-diffusion** (REFACTOR)
**Purpose**: PDE-conditioned diffusion variants (project_mmai_apr26). Multi-modal (VLM/LLM) + operator-based conditioning for inverse problems.

**Contains**: `src/autoencoders/models/project_mmai_apr26/` refactored

**Contents**:
- `models/`: VLM diffusion, operator diffusion, SRDIT variants
  - `vlm_diffusion.py`
  - `operator_diffusion.py`
  - `operator_diffusion_latent.py`
  - `vlm_diffusion_srdit.py`
- `components/`: VLM, LLM modules
  - `vlm.py`, `llm.py`, `qwen_llm.py`
- `config.py`: PDE-conditioning-specific configs
- `operators.py`: PDE operators (separate from diffusion-information-studies operators)

**Dependencies**: diffusion (as base), metrics-core, torch, transformers (for VLM/LLM)

**Exports**: PDE-conditioned model builders

**Design Goal**: Extends diffusion-core with model-specific conditioning. No knowledge of general studies.

---

### Config Structure (Unified)

**In this repo** (`autoencoders/src/autoencoders/conf/`):

```
conf/
├── config.yaml                    # Base: task, project, logging
├── data/                          # All datasets (from ae-core)
│   ├── fashion_mnist.yaml
│   ├── qg_turbulence.yaml
│   └── ...
├── exp/                           # Experiment configs (ties it all together)
│   ├── autoencoder/
│   │   └── mnist.yaml             # task: train_ae
│   ├── diffusion/
│   │   ├── unconditional.yaml     # task: train_diffusion
│   │   └── variants/
│   └── diffusion_information_studies/
│       ├── camera_inpainting.yaml # task: run_study
│       └── dps_robustness.yaml
├── model/
│   ├── ae/                        # From ae-core
│   │   ├── develop/
│   │   └── cudafused/
│   ├── diffusion/                 # From diffusion package
│   │   ├── ddpm.yaml
│   │   └── edm.yaml
│   └── pde_cond/                  # From pde-cond-diffusion
│       ├── vlm_diffusion.yaml
│       └── operator_diffusion.yaml
└── operators/                     # From diffusion-information-studies
    ├── camera.yaml
    └── inpainting.yaml
```

**Key principle**: Config lives here (orchestrator), code lives in packages.

---

## Migration Plan: Step-by-Step (No External Breaking Changes)

**Principle**: After each phase, the code still trains and produces identical results (verified by tests). No external user-facing changes except config additions (new `task:` field, new `exp=diffusion/...` configs).

### Phase 0: Upgrade `mura` (Infrastructure)
- [ ] Add to `packages/mura/src/mura/`:
  - `registry.py`: Generic `Registry[T]` class
  - `experiment.py`: Task registry + `run_task()` dispatcher
  - `checkpointing.py`: Unified checkpoint save/load
  - `metrics.py`: Shared metric aggregation (generic ones like PSNR, L2)
  - `trainer.py`: Enhance existing trainer factory (bring in defaults)
- [ ] Update `mura/pyproject.toml` (no new external deps)
- [ ] Write tests for new mura modules
- [ ] Update mura README
- [ ] **Checkpoint**: `pip install -e packages/mura && pytest packages/mura/` passes

### Phase 1: Prepare (Create Package Structure)
- [ ] Create directories:
  ```bash
  mkdir -p packages/{metrics-core,ae-core,diffusion,pde-cond-diffusion,diffusion-information-studies}/{src,tests}
  ```
- [ ] Create stub `pyproject.toml` for each (dependencies only, no code yet)
- [ ] Create stub `pytest.ini` per package
- [ ] **Checkpoint**: `ls packages/*/pyproject.toml` shows 5 files

### Phase 2: Extract Low-Hanging Fruit (metrics-core)
- [ ] Move `src/autoencoders/metrics/` → `packages/metrics-core/src/metrics/`
- [ ] Remove model-specific code (save in separate branches per model, or skip for now)
- [ ] Write `packages/metrics-core/pyproject.toml`
- [ ] Write `packages/metrics-core/tests/unit/test_metrics.py`
- [ ] Test: `cd packages/metrics-core && pip install -e . && pytest`
- [ ] Verify: Baseline metrics script produces identical output
  ```python
  # tests/baselines/verify_metrics.py
  baseline_psnr = compute_baseline_metric("test_image.png", "test_ref.png")
  from metrics import psnr
  current_psnr = psnr(test_image, test_ref)
  assert abs(baseline_psnr - current_psnr) < 1e-6
  ```
- [ ] **Checkpoint**: metrics-core installed + tested independently

### Phase 3: Extract AE Infrastructure (ae-core)
- [ ] Move AE-specific models: `src/autoencoders/models/{develop,cudafused}/` → `packages/ae-core/src/ae/models/`
- [ ] Move `src/autoencoders/trainer.py` → `packages/ae-core/src/ae/trainer.py` (or just re-export from mura)
- [ ] Move `src/autoencoders/registry.py` → `packages/ae-core/src/ae/registry.py` (but use `mura.registry.Registry`)
- [ ] Update imports: `from autoencoders` → `from ae` (internally in ae-core)
- [ ] Use `mura.registry.Registry` instead of custom registry
- [ ] Update `ae-core/pyproject.toml` to depend on `metrics-core`, `mura`
- [ ] Create `packages/ae-core/src/ae/train.py` (adapted from autoencoders.train.py)
- [ ] Write comprehensive tests:
  ```python
  # packages/ae-core/tests/unit/test_registry.py
  def test_ae_model_registry():
      from ae.registry import MODEL_REGISTRY
      assert "develop.mnist" in MODEL_REGISTRY.list_all()
  
  # packages/ae-core/tests/repro/test_ae_determinism.py (marked @pytest.mark.slow)
  def test_ae_training_determinism():
      """Same seed → same loss at each step"""
  ```
- [ ] Verify old AE runs still work:
  ```bash
  python packages/ae-core/train.py exp=mnist max_steps=5 seed=42
  # Compare loss against baseline
  ```
- [ ] **Checkpoint**: ae-core works standalone, produces same results
- [ ] **Note**: Datamodules stay in orchestrator (project-specific); ae-core only has models

### Phase 4: Extract Diffusion (diffusion package)
- [ ] Move `src/autoencoders/models/modules/diffusion/` → `packages/diffusion/src/diffusion/modules/`
- [ ] Add your unconditional code:
  - `packages/diffusion/src/diffusion/model.py` (your unconditional architecture)
  - `packages/diffusion/src/diffusion/training.py` (training loop)
  - `packages/diffusion/src/diffusion/sampling.py` (inference)
- [ ] Create minimal diffusion metrics in metrics-core (or keep in diffusion package)
- [ ] Use `mura.registry.Registry` for diffusion model builders
- [ ] Move diffusion configs from `src/autoencoders/conf/model/` → `packages/diffusion/conf/` (if any)
- [ ] Create `packages/diffusion/src/diffusion/train.py`
- [ ] Write tests:
  ```python
  # packages/diffusion/tests/repro/test_diffusion_determinism.py
  def test_diffusion_training_determinism():
      """Same seed → same loss"""
  ```
- [ ] Verify with your notebook:
  ```python
  from diffusion import build_diffusion_model, train_diffusion
  # Run 10 steps, check loss progression
  ```
- [ ] **Checkpoint**: diffusion trains, reproduces notebook results exactly

### Phase 5: Extract PDE-Conditioned Diffusion (pde-cond-diffusion)
- [ ] Move `src/autoencoders/models/project_mmai_apr26/` → `packages/pde-cond-diffusion/src/pde_cond_diffusion/`
- [ ] Update imports:
  - `from autoencoders.metrics` → `from metrics import`
  - `from autoencoders.models.modules.diffusion` → `from diffusion.modules import`
- [ ] Use `mura.registry.Registry` for model builders
- [ ] Update `pde-cond-diffusion/pyproject.toml` (depends on diffusion, metrics-core, transformers)
- [ ] Write tests (unit only for now, these are complex):
  ```python
  # packages/pde-cond-diffusion/tests/unit/test_models_load.py
  def test_vlm_diffusion_imports():
      from pde_cond_diffusion.models.vlm_diffusion import VLMDiffusion
  ```
- [ ] **Checkpoint**: pde-cond-diffusion loads, models can be instantiated

### Phase 6: Create Studies (diffusion-information-studies)
- [ ] Create `packages/diffusion-information-studies/src/diffusion_studies/`:
  - `operators/base.py`: Abstract operator interface
  - `operators/camera.py`, `operators/inpainting.py`: Implementations
  - `solvers/dps.py`: DPS solver
  - `run.py`: Study execution entrypoint
  - `config.py`: Study configs
- [ ] Design operators as **functions first, classes second**:
  ```python
  # packages/diffusion-information-studies/src/diffusion_studies/operators/camera.py
  def camera_projection(x: Tensor, height: int, width: int) -> Tensor:
      """Simple function: no state"""
      return apply_camera_projection(x, height, width)
  
  class CameraOperator:
      """Wrapper if needed for Hydra config"""
      def __init__(self, cfg):
          self.cfg = cfg
      def forward(self, x): return camera_projection(x, self.cfg.height, self.cfg.width)
  ```
- [ ] Use `mura.registry.Registry` for operator builders, solver builders
- [ ] Create `packages/diffusion-information-studies/src/diffusion_studies/run.py`:
  ```python
  @hydra.main(...)
  def main(cfg):
      from mura.experiment import run_task
      run_task(cfg)
  ```
- [ ] Write tests:
  ```python
  # packages/diffusion-information-studies/tests/unit/test_operators.py
  def test_camera_operator():
      x = torch.randn(2, 3, 64, 64)
      y = camera_projection(x, 32, 32)
      assert y.shape == (2, 3, 32, 32)
  
  # packages/diffusion-information-studies/tests/repro/test_dps_determinism.py
  def test_dps_determinism():
      """Same seed → same study results"""
  ```
- [ ] **Checkpoint**: studies run, operators are fast + composable

### Phase 7: Thin Down Orchestrator (autoencoders)
- [ ] Register all task handlers in `mura.experiment.TASK_REGISTRY`:
  ```python
  # In each package's __init__.py or train.py
  from mura.experiment import TASK_REGISTRY
  TASK_REGISTRY.register("train_ae", AETaskConfig)
  TASK_REGISTRY.register("train_diffusion", DiffusionTaskConfig)
  TASK_REGISTRY.register("run_study", StudyTaskConfig)
  ```
- [ ] Simplify `src/autoencoders/train.py`:
  ```python
  import hydra
  from mura.experiment import run_task
  
  @hydra.main(version_base=None, config_path="conf", config_name="config")
  def main(cfg):
      run_task(cfg)
  ```
- [ ] Update `src/autoencoders/__init__.py`:
  ```python
  # Re-export from all packages
  from ae import *
  from diffusion import *
  from pde_cond_diffusion import *
  from diffusion_studies import *
  # Keep datamodules, data registry here (project-specific)
  from .datamodules import DATASET_REGISTRY, build_dataloaders
  ```
- [ ] Delete from `src/autoencoders/`: metrics/, models/ (but keep datamodules/)
- [ ] Keep: `datamodules/`, `data.py`, `__init__.py`, `train.py`, `conf/`
- [ ] Update `pyproject.toml`:
  ```toml
  dependencies = [
      "mura @ file://./packages/mura",
      "metrics-core @ file://./packages/metrics-core",
      "ae-core @ file://./packages/ae-core",
      "diffusion @ file://./packages/diffusion",
      "pde-cond-diffusion @ file://./packages/pde-cond-diffusion",
      "diffusion-information-studies @ file://./packages/diffusion-information-studies",
  ]
  ```
- [ ] Verify all commands still work:
  ```bash
  python -m autoencoders.train exp=mnist max_steps=5
  python -m autoencoders.train exp=diffusion/unconditional max_steps=5
  python -m autoencoders.train exp=diffusion_information_studies/camera max_steps=5
  ```
- [ ] Run full integration test suite:
  ```bash
  pytest tests/integration/ -x
  ```
- [ ] **Checkpoint**: Orchestrator has ~300-400 lines (datamodules + dispatcher), everything works

### Phase 8: Tests & Documentation
- [ ] Create `tests/integration/` with reproducibility tests:
  ```python
  # tests/integration/test_reproducibility.py
  def test_ae_reproduces():
      """Results match pre-refactor baseline"""
  
  def test_diffusion_reproduces():
      """Diffusion results match pre-refactor baseline"""
  
  def test_studies_reproduces():
      """Study results match pre-refactor baseline"""
  ```
- [ ] Create `tests/baselines/`:
  - Save metric checksums from before refactor
  - Use in reproducibility tests
  
- [ ] Update `README.md`:
  - Explain new architecture
  - Quick start for each workflow
  - Link to `PACKAGES.md`
  
- [ ] Create `PACKAGES.md`:
  ```markdown
  # Package Guide
  
  ## metrics-core
  Shared metric functions...
  
  ## ae-core
  Autoencoder models...
  
  [etc.]
  ```

- [ ] Create `TESTING.md`:
  ```markdown
  # Testing Guide
  
  ## Run all tests
  pytest tests/
  
  ## Run unit tests only (fast)
  pytest tests/unit/ -x
  ...
  ```

- [ ] Create Makefile targets:
  ```makefile
  .PHONY: test test-unit test-repro test-integration
  
  test-unit:
      pytest tests/unit/ -x
  
  test-repro:
      pytest tests/repro/ -x -m slow
  
  test-integration:
      pytest tests/integration/ -x -m slow
  ```

- [ ] **Checkpoint**: Documentation complete

---

## What Stays in This Repo

### `src/autoencoders/`
```
autoencoders/
├── __init__.py              # Aggregator: re-export from all packages
├── train.py                 # Entry point: uses mura.experiment.run_task()
├── datamodules/             # Project-specific datasets (stays here)
│   ├── __init__.py
│   ├── fashion_mnist.py
│   ├── qg_turbulence.py
│   ├── [all dataset builders]
│   └── registry.py          # DATASET_REGISTRY (uses mura.registry)
├── data.py                  # build_dataloaders() helper
└── conf/                    # All Hydra configs (unified, minimal)
    ├── config.yaml          # Base: task, project, logging
    ├── data/                # All dataset configs
    ├── model/               # All models
    ├── exp/                 # All experiments
    ├── operators/           # Study operators
    └── test_configs/        # Tiny configs for testing
```

**Why datamodules stay**: They're project-specific (QG turbulence, forced turbulence, etc.) and import from `packages/qg`. Not reusable across projects.

### Root Files
```
MODULARIZATION_PLAN.md    # This document
PACKAGES.md               # Guide to each package, what moved where
TESTING.md                # How to run tests locally + CI
CHANGELOG.md              # What moved + verification steps per phase
README.md                 # Updated: how to run, test, extend
Makefile                  # Targets: test-unit, test-repro, test-integration
pyproject.toml            # Depends on all packages (editable installs)
pytest.ini                # Config for test discovery
```

### `tests/` directory (integration + reproducibility)
```
tests/
├── integration/           # End-to-end workflows
│   ├── test_ae_workflow.py
│   ├── test_diffusion_workflow.py
│   ├── test_studies_workflow.py
│   └── test_reproducibility.py  # Verify results match baselines
├── conftest.py            # Pytest fixtures + helpers
└── baselines/             # Artifact checksums from before refactor
    ├── ae_baseline.json
    ├── diffusion_baseline.json
    └── studies_baseline.json
```

### `runs/` directory (same as always)
**Behavior**: Unchanged. All packages write here. Tagged, logged to WandB, timestamped.

---

## Benefits

### 1. **Reusability**
- Future paper: just `pip install git+...diffusion.git && pip install git+...diffusion-information-studies.git`
- No need to clone this whole repo

### 2. **Modularity**
- Diffusion code doesn't know about studies
- Studies don't know about AE code
- AE trainer doesn't change for diffusion work

### 3. **Clarity**
- Code is DRY: no duplication of metrics, operators, base classes
- Functional style: small functions, easy to test and modify
- Config tells the story: which package, which model, which task

### 4. **Scalability**
- Add new model type? Create new package or extend existing one
- Add new study? Just add to diffusion-information-studies
- Each package has its own `setup.py`, CHANGELOG, versions

### 5. **Experiment Logbook**
- This repo becomes the notebook: configs, notes, commit messages
- Diff against packages shows exactly what changed per paper/project
- When you publish, you publish packages + point to this logbook

---

## Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Circular dependencies | Define clear hierarchy (metrics-core ← diffusion ← studies). No upward deps. |
| Import path changes break old code | Keep old paths as re-exports in autoencoders for 1 release. |
| Config duplication | Inherit configs with Hydra defaults (config.yaml does `defaults: [data, model]`). |
| Different packages out of sync | Commit all together, tag releases together. Use monorepo-style changelogs. |
| Hard to debug cross-package issues | Each package has tests. Orchestrator has integration tests. |

---

## Testing Strategy: Efficient & Containerized

**Goal**: Verify all new code runs + reproduces results exactly, with minimal overhead. Tests are hidden, fast, integrated into the build process.

### Testing Layers

#### 1. **Unit Tests (per package)**
**Location**: `packages/*/tests/unit/`
**Scope**: Functions, registry logic, config loading
**Speed**: < 1s per package
**Run**: `cd packages/X && pytest tests/unit/ -x`

Example:
```python
# packages/diffusion/tests/unit/test_registry.py
def test_diffusion_model_registry():
    from diffusion.registry import MODEL_REGISTRY
    assert "ddpm" in MODEL_REGISTRY.list_all()
    model = MODEL_REGISTRY.get("ddpm")(config)
    assert isinstance(model, pl.LightningModule)
```

#### 2. **Reproducibility Tests (per package)**
**Location**: `packages/*/tests/repro/`
**Scope**: Trains on tiny datasets, compares output hashes/seeds
**Speed**: 10-30s per package (CPU or minimal GPU)
**Run**: `cd packages/X && pytest tests/repro/ -x`

Example:
```python
# packages/diffusion/tests/repro/test_deterministic.py
@pytest.mark.slow
def test_diffusion_determinism():
    """Two runs with same seed produce identical losses"""
    cfg = load_config("exp=diffusion/test_tiny")
    loss1 = train_diffusion(cfg, seed=42, num_steps=10)
    loss2 = train_diffusion(cfg, seed=42, num_steps=10)
    assert loss1 == loss2, "Non-deterministic training!"
```

#### 3. **Integration Tests (orchestrator repo)**
**Location**: `tests/integration/`
**Scope**: End-to-end flows with all packages together
**Speed**: 30-60s total
**Run**: `pytest tests/integration/ -x`

Example:
```python
# tests/integration/test_workflows.py
@pytest.mark.slow
def test_ae_training():
    """Old AE workflow still works"""
    run_cmd = "python -m autoencoders.train exp=mnist num_steps=5"
    result = subprocess.run(run_cmd, shell=True, capture_output=True)
    assert result.returncode == 0
    assert Path("runs/").exists()

@pytest.mark.slow
def test_diffusion_training():
    """New diffusion workflow works"""
    run_cmd = "python -m autoencoders.train exp=diffusion/test_tiny num_steps=5"
    result = subprocess.run(run_cmd, shell=True, capture_output=True)
    assert result.returncode == 0

@pytest.mark.slow
def test_study_workflow():
    """Studies workflow works"""
    run_cmd = "python -m autoencoders.train exp=diffusion_information_studies/test_tiny"
    result = subprocess.run(run_cmd, shell=True, capture_output=True)
    assert result.returncode == 0
```

#### 4. **CI Artifacts (Optional but Recommended)**
**Location**: `.github/workflows/test.yml` (or local)
**What runs**:
- All unit tests (fast, always)
- Reproducibility tests (on push to dev26, or on-demand)
- Integration tests (on push to main or PR)

**Example CI job**:
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: |
          pip install -e packages/mura -e packages/metrics-core -e packages/ae-core
          pip install pytest
          pytest packages/metrics-core/tests/unit/ -x
          pytest packages/ae-core/tests/unit/ -x
      - run: pytest tests/integration/ -x -m slow
```

---

## Verification Plan: Reproduce Results Exactly

**Before/After Comparison**:

1. **Pick baseline experiments**:
   - One AE training run (e.g., `exp=mnist`, 1 epoch)
   - One diffusion run (e.g., `exp=diffusion/test_tiny`, 10 steps)
   - One study run (e.g., `exp=diffusion_information_studies/test_tiny`)

2. **Capture baseline artifacts**:
   ```bash
   python -m autoencoders.train exp=mnist seed=42 max_steps=10
   cp -r runs/fashion_mnist/develop/* baseline/ae_output/
   
   python -m autoencoders.train exp=diffusion/test_tiny seed=42 max_steps=10
   cp -r runs/diffusion_unconditional/ddpm/* baseline/diffusion_output/
   ```

3. **After refactoring, re-run and compare**:
   ```bash
   python -m autoencoders.train exp=mnist seed=42 max_steps=10
   diff -r runs/fashion_mnist/develop/ baseline/ae_output/  # Should have only timestamp diffs
   
   # Check key metrics match:
   python -c "
   import json
   baseline = json.load(open('baseline/ae_output/metrics.json'))
   current = json.load(open('runs/fashion_mnist/develop/metrics.json'))
   for key in baseline:
       assert abs(baseline[key] - current[key]) < 1e-6, f'{key} mismatch'
   print('✓ Results match exactly')
   "
   ```

4. **Test suite captures this**:
   ```python
   # tests/integration/test_reproducibility.py
   def test_ae_reproduces():
       """Results match pre-refactor baseline"""
       baseline_loss = 0.1234  # From runs before refactoring
       current_loss = train_ae(cfg, seed=42, num_steps=10)
       assert abs(current_loss - baseline_loss) < 1e-6
   
   def test_diffusion_reproduces():
       """Diffusion results match pre-refactor baseline"""
       baseline_loss = 2.345
       current_loss = train_diffusion(cfg, seed=42, num_steps=10)
       assert abs(current_loss - baseline_loss) < 1e-6
   ```

---

### Manual Smoke Tests
```bash
python -m autoencoders.train exp=mnist max_steps=5              # Old autoencoder flow
python -m autoencoders.train exp=diffusion/unconditional max_steps=5  # New diffusion flow
python -m autoencoders.train exp=diffusion_information_studies/camera max_steps=5      # New studies flow
```

---

## Test Configuration Files

Each package gets a **tiny test config** (< 1s to run):

```yaml
# packages/diffusion/conf/exp/test_tiny.yaml
defaults:
  - override /model: diffusion/ddpm_tiny
  - override /data: fashion_mnist_tiny

trainer:
  max_steps: 10
  precision: 32

project: test_diffusion
seed: 42
```

Same for ae, studies, pde-cond. This ensures:
- Tests don't need special logic to reduce problem size
- Configs are reusable (user can do `exp=diffusion/test_tiny` for quick debugging)
- Reproducible: always same small dataset, same steps

---

---

## Deliverables After Plan Completes

1. **5 independent packages**, each pip-installable
2. **This repo thinned down** to orchestrator + configs only
3. **Each package has**:
   - Clean `pyproject.toml`
   - README with examples
   - CHANGELOG documenting what moved
   - `src/` with actual code
   - `tests/` with unit tests
   - Optional: `conf/` with default configs

4. **Orchestrator repo has**:
   - Thin `train.py` dispatcher
   - Unified Hydra configs
   - Integration tests
   - `PACKAGES.md` guide
   - `CHANGELOG.md` explaining refactor
   - Still works with all old commands, plus new ones

---

## Timeline Estimate

- **Phase 1 (Prepare)**: 15 min (structural only)
- **Phase 2 (metrics-core)**: 30 min (move, test)
- **Phase 3 (ae-core)**: 45 min (refactor imports)
- **Phase 4 (diffusion)**: 45 min (integrate new code + old modules)
- **Phase 5 (pde-cond-diffusion)**: 30 min (refactor imports)
- **Phase 6 (diffusion-information-studies)**: 30 min (create from new code)
- **Phase 7 (thin orchestrator)**: 30 min (dispatcher + config)
- **Phase 8 (git + docs)**: 30 min (clean history, docs)

**Total**: ~3.5 hours (sequential, but many steps can run in parallel with reviews)

---

## Questions Before We Start

1. Should packages be separate git repos, or all monorepo (one `packages/` dir)?
   - **Recommendation**: Monorepo for now (easier to refactor), can split later
   
2. When to extract to separate repos?
   - **Answer**: After first paper using the package. Then: `git subtree split` the package dir into its own repo
   
3. Should each package have its own CI/tests?
   - **Recommendation**: Local tests in package, orchestrator does integration tests
   
4. Backward compatibility: how long to maintain old import paths?
   - **Recommendation**: 1 release cycle (this repo only). After tag `v1.0.0`, old paths go away.

---

**Next Step**: Approve this plan, then start Phase 1 (structure creation).
