# Modularization Plan — Document Index

This plan transforms the autoencoders repo into a modular system with 5 independent packages + a thin orchestrator. **All documents are in this repo root.**

---

## Start Here

### 1. **PLAN_README.md** ← Start with this
Quick reference. What, why, phases, timeline. ~2 min read.

### 2. **PLAN_SUMMARY.md**
Structured summary. Changes from original plan, design decisions, testing strategy. ~5 min read.

---

## Detailed Documents

### 3. **MODULARIZATION_PLAN.md** (32 KB)
Full detailed plan. Everything: current state, package definitions, mura upgrades, all 8 phases, risks, timeline, Q&A.

**Sections**:
- Current state analysis
- Boilerplate to extract to `mura`
- 5-package architecture definitions
- Unified config structure
- 8-phase migration (detailed checklist)
- Deliverables + timeline

### 4. **ARCHITECTURE.md**
Visual comparison: before/after refactoring.

**Sections**:
- Monolithic architecture (before)
- Modular architecture (after)
- Data flow diagrams
- Config inheritance
- Import changes
- Quick comparison table

### 5. **TESTING.md** (11 KB)
Reproducibility & testing approach.

**Sections**:
- 4 testing layers (unit → repro → baseline → integration)
- Code examples for each layer
- Tiny test configs (reusable)
- Phase verification protocol
- CI/CD integration (Makefile)
- Floating-point tolerance guidance
- Testing checklist

---

## By Role

### I'm a user who wants to understand the plan
1. PLAN_README.md (2 min)
2. ARCHITECTURE.md (5 min)
3. PLAN_SUMMARY.md (5 min)

### I'm implementing Phase X
1. MODULARIZATION_PLAN.md → find Phase X section
2. TESTING.md → understand what tests to run
3. Run verification protocol before committing

### I want to maintain these packages later
1. ARCHITECTURE.md → understand data flow
2. MODULARIZATION_PLAN.md → see final structure
3. Each `packages/X/README.md` → per-package guide

---

## The 5 Packages (After Refactoring)

| Package | Purpose | New in repo? | Key dependency |
|---------|---------|-------------|-----------------|
| **mura** | Boilerplate (registry, dispatcher, checkpointing, metrics) | Enhanced | None (foundational) |
| **metrics-core** | Shared image/diffusion metrics | Yes | mura |
| **ae-core** | Autoencoder models + trainer logic | Extracted | mura, metrics-core |
| **diffusion** | Unconditional diffusion + your code | Extracted + new | mura, metrics-core |
| **pde-cond-diffusion** | PDE-conditioned (project_mmai_apr26) | Renamed | diffusion, mura, metrics-core |
| **diffusion-information-studies** | DPS, operators, study infrastructure | New | diffusion, mura, metrics-core |

---

## The 8 Phases

1. **Phase 0** (30 min): Upgrade `mura` with registry, dispatcher, checkpointing, metrics
2. **Phase 1** (15 min): Create 5 package directories + stubs
3. **Phase 2** (30 min): Extract metrics-core
4. **Phase 3** (45 min): Extract ae-core
5. **Phase 4** (45 min): Extract diffusion (+ your new code)
6. **Phase 5** (30 min): Extract pde-cond-diffusion
7. **Phase 6** (30 min): Create diffusion-information-studies
8. **Phase 7** (30 min): Thin orchestrator to 3 files
9. **Phase 8** (30 min): Tests + documentation

**Total**: 4-5 hours (mostly testing)

---

## Testing Per Phase

Before moving to next phase:
```bash
make test-unit          # Fast, always
make test-repro         # Slow, determinism
make test-integration   # Slow, end-to-end
```

Each phase verifies:
- ✓ All unit tests pass
- ✓ Reproducibility tests pass (seed=42 determinism)
- ✓ Baseline metrics match (pre-refactor comparison)
- ✓ Integration tests pass (workflows work)
- ✓ Manual smoke tests pass (3 workflows run)

---

## File Structure After Completion

```
autoencoders/  (thin orchestrator)
├── src/autoencoders/
│   ├── __init__.py          # Re-exports from packages
│   ├── train.py             # Uses mura.experiment.run_task()
│   └── conf/                # Unified Hydra configs
├── packages/
│   ├── mura/                # Enhanced
│   ├── metrics-core/        # New
│   ├── ae-core/             # Extracted
│   ├── diffusion/           # Extracted + new
│   ├── pde-cond-diffusion/  # Renamed
│   ├── diffusion-information-studies/   # New
│   ├── qg/                  # Existing
│   └── [others]
├── tests/integration/       # End-to-end
├── tests/baselines/         # Reference metrics
├── PLAN_*.md                # These documents
├── ARCHITECTURE.md
├── TESTING.md
├── PACKAGES.md              # Per-package guide (created in Phase 8)
└── README.md                # Updated with new structure
```

---

## Key Design Principles

1. **DRY boilerplate**: Registry, dispatcher, trainer factory → mura (used by all)
2. **Functions-first operators**: Simple to write, test, modify (not OOP-heavy)
3. **One config system**: All packages use Hydra, configs here (orchestrator)
4. **Reproducible extraction**: Each phase verified with determinism tests + baseline comparison
5. **Independent packages**: Each package can be `pip install`'d separately
6. **Thin orchestrator**: 3 files (train.py, __init__.py, conf/) → just glue

---

## Commands After Refactoring

```bash
# Old (still works)
python -m autoencoders.train exp=mnist max_steps=10

# New (via config)
python -m autoencoders.train exp=diffusion/unconditional max_steps=10
python -m autoencoders.train exp=diffusion_information_studies/camera max_steps=10
python -m autoencoders.train exp=pde_cond/vlm_diffusion max_steps=10

# Or standalone (without this repo)
pip install git+https://github.com/yourname/diffusion.git
python -m diffusion.train exp=unconditional
```

---

## What's New vs. Old

### For Users: No Breaking Changes
- Same commands work (old behavior preserved)
- Same `runs/` directory structure
- Same WandB logging
- Same reproducibility guarantees

### Internally: Modular
- Each package is independent
- Boilerplate in mura (DRY)
- Functions-first operators (simplicity)
- Verified reproducibility (tests prove results match)

---

## Next Steps

1. **Read PLAN_README.md** (2 min)
2. **Read ARCHITECTURE.md** (5 min)
3. **Approve plan** ← You are here
4. **Start Phase 0** (mura upgrades)
5. Each phase: implement → test → verify → commit

---

## Questions?

See MODULARIZATION_PLAN.md section "Questions Before We Start" (bottom of document).

---

**Status**: Plan ready for execution. All details documented. Proceed with Phase 0.
