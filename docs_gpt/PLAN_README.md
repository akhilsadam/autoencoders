# Modularization Plan — Quick Reference

## The Goal

Transform `autoencoders/` from a monolithic repo into a **thin orchestrator + 5 independent packages**. Each package is pip-installable, usable in future papers without this repo. This repo becomes your experiment logbook.

---

## The 5 Packages

```
mura (enhanced)  ← All ML boilerplate (registry, dispatcher, checkpointing, metrics)
  ↓
┌─ metrics-core (shared image/diffusion metrics)
├─ ae-core (autoencoders + datamodules)
├─ diffusion (unconditional diffusion + your new code)
├─ pde-cond-diffusion (project_mmai_apr26: VLM/LLM + operator conditioning)
└─ diffusion-information-studies (DPS + custom operators, functions-first design)
  ↓
autoencoders (orchestrator: thin train.py + unified Hydra configs)
```

---

## Key Changes from Original Plan

| What | Original | Updated | Why |
|------|----------|---------|-----|
| **Boilerplate** | Duplicated per package | Moved to `mura` | DRY; 1 registry, 1 dispatcher, 1 checkpointing |
| **Package 5 name** | pde-cond-diffusion | pde-cond-diffusion | Clearer: includes project_mmai_apr26 |
| **Breaking changes** | Old imports still work | Results match exactly | Reproducibility > backwards compat |
| **Testing** | Basic checks | Determinism + baseline verification | Every phase verified before proceeding |
| **Operators** | OOP-heavy | Functions first, classes second | High-school level simplicity |

---

## The 8 Phases

| Phase | Action | Time | Checkpoint |
|-------|--------|------|-----------|
| **0** | Upgrade `mura` with registry, dispatcher, checkpointing, metrics | 30 min | `pip install -e packages/mura && pytest` |
| **1** | Create 5 package directories + stubs | 15 min | 5 `pyproject.toml` files exist |
| **2** | Extract metrics-core | 30 min | Metrics work standalone, baseline verified |
| **3** | Extract ae-core (uses mura.registry) | 45 min | AE training works, reproduces baseline |
| **4** | Extract diffusion (+ your new code) | 45 min | Diffusion trains, reproduces notebook |
| **5** | Extract pde-cond-diffusion | 30 min | Models load, unit tests pass |
| **6** | Create diffusion-information-studies (functions-first) | 30 min | Studies run, operators compose |
| **7** | Thin orchestrator (3 files: __init__, train, conf/) | 30 min | `train.py` uses `mura.experiment.run_task()` |
| **8** | Tests + documentation | 30 min | Integration suite + `PACKAGES.md` |

**Total: 4-5 hours** (mostly testing)

---

## Testing Per Phase

1. **Unit tests** (< 1s per package) — registry loading, imports
2. **Reproducibility tests** (10-30s per package) — seed=42 determinism
3. **Baseline verification** (1-2 min) — loss matches pre-refactor
4. **Integration tests** (30-60s) — full workflows work
5. **Manual smoke tests** (2 min) — 3 workflows run without error

**Before committing**: `make test-all` must pass

---

## No External Breaking Changes

```bash
# Old commands still work
python -m autoencoders.train exp=mnist max_steps=5

# New commands via config
python -m autoencoders.train exp=diffusion/unconditional max_steps=5
python -m autoencoders.train exp=diffusion_information_studies/camera max_steps=5
python -m autoencoders.train exp=pde_cond/vlm_diffusion max_steps=5
```

**Internally**: Results are verified to match exactly (seeds, losses, checksums).

---

## What's in This Repo After Completion

```
autoencoders/  (orchestrator only)
├── src/autoencoders/
│   ├── __init__.py           # Re-exports from packages
│   ├── train.py              # Uses mura.experiment.run_task()
│   └── conf/                 # Unified Hydra configs
├── packages/                 # 5 independent packages
├── tests/integration/        # Workflows + reproducibility
├── tests/baselines/          # Reference metrics
├── MODULARIZATION_PLAN.md    # Detailed plan (this repo)
├── PLAN_SUMMARY.md           # This quick reference
├── TESTING.md                # How to run tests
├── PACKAGES.md               # Guide to each package
└── README.md                 # Updated with new architecture
```

Each package is independently pip-installable:
```bash
pip install packages/diffusion/
pip install packages/diffusion-information-studies/
# In a future project, just use these packages without this repo
```

---

## Mura Upgrades (New Boilerplate)

**Added to `packages/mura/`**:
- `registry.py` — Generic `Registry[T]` for models, datasets, solvers
- `experiment.py` — Task registry + `run_task()` dispatcher
- `checkpointing.py` — Unified checkpoint save/load + metadata
- `metrics.py` — Generic metric aggregation (PSNR, SSIM, L2)
- Enhance `trainer.py` — Trainer factory with defaults

**Why**: All packages use these. No duplication.

---

## Document Guide

- **MODULARIZATION_PLAN.md** — Full detailed plan (32 KB, all phases + architecture)
- **PLAN_SUMMARY.md** — This summary in table form
- **TESTING.md** — Testing strategy, reproducibility, CI integration
- **Current file** — Quick reference

---

## Next Step

Approve this plan to proceed:

1. Start Phase 0 (mura upgrades)
2. Each phase: write code → run tests → verify reproducibility → commit → move to next

Questions?
