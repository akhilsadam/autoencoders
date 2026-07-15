# Updated Modularization Plan — Key Changes

## Executive Summary

This plan transforms the autoencoders repo into a **thin experiment orchestrator** that stitches together 5 independently-installable packages. Crucially:

1. **All ML boilerplate moves to `mura`** — registry, trainer factory, task dispatcher, checkpointing, metrics aggregation
2. **New package: `pde-cond-diffusion`** (renamed from pde-cond-diffusion) — contains project_mmai_apr26 code
3. **Rigorous reproducibility testing** — every phase verifies exact result matches (seeds, losses, checksums)
4. **Functional operators** — diffusion-information-studies uses functions-first, classes-second design for simplicity
5. **No external breaking changes** — all old commands work, new ones via config additions

---

## What Moved to `mura`

**New in `packages/mura/src/mura/`**:

- **`registry.py`**: Generic `Registry[T]` for models, datasets, solvers (replaces custom per-package logic)
- **`experiment.py`**: Task dispatcher + task registry (moves train.py logic to shared infra)
- **`checkpointing.py`**: Unified checkpoint save/load with metadata (git SHA, config, seed, etc.)
- **`metrics.py`**: Generic metric aggregation (PSNR, SSIM, L2, etc.) — shared across packages
- **`trainer.py`**: Enhance existing trainer factory with defaults

**Why**: Every package would duplicate registry logic, trainer setup, and task dispatch. Centralizing in `mura` makes all packages simpler and keeps boilerplate DRY.

---

## The 5 Packages

| Package | Purpose | Key Move | Tests |
|---------|---------|----------|-------|
| **metrics-core** | Shared diffusion/image metrics (PSNR, SSIM, perceptual loss) | `src/autoencoders/metrics/` | Unit + baseline verification |
| **ae-core** | Autoencoder models & trainer logic | `src/autoencoders/models/{develop,cudafused}` | Unit + repro determinism |
| **diffusion** | Core unconditional diffusion (your new notebook code + modules) | `models/modules/diffusion/` + new code | Unit + repro, notebook verification |
| **pde-cond-diffusion** | PDE-conditioned variants (VLM/LLM + operator conditioning) | `models/project_mmai_apr26/` | Unit only (complex models) |
| **diffusion-information-studies** | DPS, conditioning solvers, custom operators | New package | Unit + functional operator tests |

---

## Testing Strategy (Not Breaking Changes)

**Key principle**: After each phase, code trains and produces **exactly identical results**. Verified by:

1. **Unit tests** (< 1s per package) — fast, always run
2. **Reproducibility tests** (10-30s per package) — tiny configs, seed=42, compare loss checksums
3. **Integration tests** (30-60s total) — end-to-end workflows, all packages together
4. **Manual smoke tests** — run old + new commands, verify no errors

**Example**:
```python
# tests/integration/test_reproducibility.py
def test_diffusion_reproduces():
    """Same seed → same loss as before refactor"""
    baseline_loss = 2.345  # Captured before Phase 4
    current_loss = train_diffusion(cfg, seed=42, num_steps=10)
    assert abs(current_loss - baseline_loss) < 1e-6
```

Each package has tiny test configs (`exp=*/test_tiny.yaml`) that train in seconds.

---

## Migration Phases

| Phase | What | Checkpoint |
|-------|------|------------|
| **0** | Upgrade `mura` with registry, dispatcher, checkpointing | `pip install -e packages/mura && pytest` |
| **1** | Create package directories + stubs | 5 `pyproject.toml` files exist |
| **2** | Extract metrics-core | Standalone metrics, baseline verified |
| **3** | Extract ae-core (uses mura.registry) | AE training works, produces same results |
| **4** | Extract diffusion (your new code) | Diffusion trains, reproduces notebook |
| **5** | Extract pde-cond-diffusion | Models load, unit tests pass |
| **6** | Create diffusion-information-studies (functions-first operators) | Studies run, operators compose cleanly |
| **7** | Thin orchestrator to 3 files | `train.py` uses `mura.experiment.run_task()` |
| **8** | Tests + docs | Full integration suite + `PACKAGES.md` |

**Total: ~4-5 hours** (mostly testing + verification)

---

## Key Design Decisions

### 1. Operators are Functions First
```python
# packages/diffusion-information-studies/src/diffusion_studies/operators/
def camera_projection(x: Tensor, height: int, width: int) -> Tensor:
    """Pure function, no state, easy to test"""
    return apply_camera_projection(x, height, width)

class CameraOperator:
    """Lightweight wrapper for Hydra config if needed"""
    def __init__(self, cfg): self.cfg = cfg
    def forward(self, x): return camera_projection(x, self.cfg.height, self.cfg.width)
```

**Why**: High-school level simplicity. Students can write custom operators by just adding a function.

### 2. Config Lives Here, Code Lives in Packages
- `src/autoencoders/conf/` — unified Hydra configs for all workflows
- `packages/*/src/*/` — actual code (models, trainers, operators)
- Configs pull from all packages via `exp=ae/mnist`, `exp=diffusion/unconditional`, etc.

### 3. Backwards Compatibility = Results Match Exactly
Not about keeping old import paths, but about:
- Same seed → same loss progression
- Same config → same artifacts
- Reproducibility tests verify this before each phase completes

---

## File Structure After Completion

```
autoencoders/  (thin orchestrator)
├── src/autoencoders/
│   ├── __init__.py          # Re-exports from all packages
│   ├── train.py             # @hydra.main + run_task(cfg)
│   └── conf/                # All Hydra configs (unified)
├── packages/
│   ├── mura/                # Enhanced with registry, dispatcher, checkpointing
│   ├── metrics-core/        # Shared metrics
│   ├── ae-core/             # AE models + datamodules
│   ├── diffusion/           # Unconditional diffusion
│   ├── pde-cond-diffusion/  # PDE-conditioned variants
│   └── diffusion-information-studies/   # DPS + operators (functional style)
├── tests/integration/       # End-to-end workflows + reproducibility
├── MODULARIZATION_PLAN.md   # This detailed plan
├── PACKAGES.md              # Guide to each package
├── TESTING.md               # How to run tests
└── README.md                # Quick start + architecture
```

---

## What Doesn't Change (From User Perspective)

- `python -m autoencoders.train exp=mnist` still works
- `runs/` directory structure unchanged
- WandB logging, git tracking, Hydra config system all unchanged
- Same reproducibility guarantees (seeds, determinism)

---

## What's New (Enables Future Work)

1. Each package is independently pip-installable
   - Future paper: `pip install git+...diffusion.git` + `pip install git+...diffusion-information-studies.git`
   - No need to clone this entire repo

2. Modular operators for studies
   - Add custom operator: write a function in `diffusion-information-studies/operators/`
   - High-school student can fork + modify

3. Shared boilerplate in `mura`
   - Next project using diffusion? Just import `mura.registry`, `mura.experiment`
   - No duplication of trainer factory, task dispatch, checkpointing

4. Clean commit history
   - Each phase is a separate commit with clear "what moved where"
   - CHANGELOG documents extraction + verification per phase

---

## Questions?

Approve this plan to proceed with Phase 0 (mura upgrades) → Phase 1 (structure) → etc.
