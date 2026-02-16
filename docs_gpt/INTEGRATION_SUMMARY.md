# Integration Summary: Autoencoders + QG + Mura

## Executive Summary

**Goal:** Integrate QG physics simulations as a dataset source for autoencoder training, with automatic caching and version tracking, while preserving all existing functionality.

**Approach:** Minimal, non-breaking integration following existing patterns.

---

## Key Design Decisions

### 1. **QG as a Datamodule** (Not a separate workflow)
- QG integrates exactly like `fashion_mnist` or `aesthetic4k`
- Uses same registry pattern
- Same Hydra config structure
- **Benefit:** Zero learning curve, consistent interface

### 2. **Hash-Based Caching** (Not manual versioning)
- Config parameters → deterministic hash → cache key
- Different params = automatic new cache
- **Benefit:** No manual version management needed

### 3. **Selective Mura Integration** (Not full adoption)
- Import only: callbacks, utility functions
- Don't adopt: workflow orchestration, version manager
- **Benefit:** Get benefits without breaking Hydra patterns

### 4. **Submodule Dependencies** (Not PyPI)
- Add `qg` and `mura` as git submodules in `lib/`
- Install via `make install`
- **Benefit:** Control versions, easy local development

---

## What Gets Added

### New Files (6 total)
```
src/autoencoders/
├── datamodules/
│   └── qg_dataset.py              # ~150 lines - QG integration
├── conf/data/
│   └── qg_turbulence.yaml         # ~15 lines - default config
└── util/
    ├── mura_callbacks.py          # ~50 lines - git tracking
    └── gpu_utils.py               # ~15 lines - cluster helper
lib/
├── qg/                            # git submodule
└── mura/                          # git submodule
```

### Modified Files (3 total)
```
src/autoencoders/datamodules/__init__.py  # +2 lines - registry entry
Makefile                                  # +20 lines - install targets
pyproject.toml                            # +3 lines - dependencies
```

**Total New Code:** ~250 lines  
**Total Modified Code:** ~25 lines

---

## What Stays the Same

### Unchanged ✅
- All existing datamodules (`fashion_mnist`, `aesthetic4k`)
- All existing models (`mnist`, `tiny_cu`, `tiny_hl`)
- Training logic (`train.py`)
- WandB integration
- Hydra configuration system
- Git tracking (`gitinfo.py`, `sec_id.py`)
- Slurm deployment (`hpc_deploy.py`)
- All existing Hydra configs

### Backward Compatibility ✅
```bash
# These work exactly as before (no changes needed)
python -m src.autoencoders.train data=fashion_mnist
python -m src.autoencoders.train data=aesthetic4k
make train
make test-cu
make test-hl
```

---

## Usage Examples

### Basic QG Training
```bash
# First run: generates QG data + trains
python -m src.autoencoders.train data=qg_turbulence model=tiny_cu

# Second run: loads from cache (fast) + trains
python -m src.autoencoders.train data=qg_turbulence model=tiny_cu
```

### Custom QG Parameters
```bash
# Higher resolution (creates new cache automatically)
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.grid_size=256 \
    data.params.num_samples=1000

# Different physics (creates another cache)
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.time_steps=500
```

### Cache Management
```bash
# Check cache
ls -lh data/qg_cache/

# Clear cache (forces regeneration)
rm -rf data/qg_cache/

# Force regenerate specific config
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.force_regenerate=true
```

---

## Installation

### One Command
```bash
make install
```

### What It Does
1. Installs `uv` package manager
2. Creates virtual environment
3. Installs QG package (`lib/qg`)
4. Installs Mura package (`lib/mura`)
5. Installs autoencoders + dependencies
6. Configures Python paths

### Post-Install (one-time)
```bash
wandb login
huggingface-cli login  # Only for some datasets
```

---

## Caching Details

### How It Works
1. **Config → Hash:** QG parameters hashed to 12-char ID
2. **Cache Check:** Look for `data/qg_cache/qg_v1_<hash>/data.pt`
3. **Generate or Load:**
   - Not found → Run QG solver, save to cache
   - Found → Load from disk (instant)
4. **Train:** Use cached data like any other dataset

### Cache Structure
```
data/qg_cache/
├── qg_v1_a1b2c3d4e5f6/    # grid_size=128, num_samples=1000
│   ├── data.pt             # Cached tensor
│   └── metadata.json       # Config snapshot
├── qg_v1_f6e5d4c3b2a1/    # grid_size=256, num_samples=1000
│   ├── data.pt
│   └── metadata.json
└── ...
```

### Version Tracking
- **Automatic:** Config change → new hash → new cache
- **No manual tracking needed**
- **Clear naming:** Hash in path for debugging

---

## Pros/Cons

### ✅ Pros

1. **No Breaking Changes**
   - Existing code untouched
   - All configs work as-is
   - 100% backward compatible

2. **Fast Iteration**
   - QG generation once per config
   - Subsequent runs instant
   - Easy parameter sweeps

3. **Clean Architecture**
   - Follows existing patterns
   - Minimal new code
   - Easy to understand

4. **Simple Installation**
   - One command
   - All deps managed
   - Works on cluster

5. **Flexible**
   - Easy to add QG variants
   - Can mix with other datasets
   - CLI overrides for everything

### ❌ Cons

1. **First-Run Slowness**
   - QG generation takes time (minutes)
   - **Mitigation:** Small defaults, progress bars

2. **Disk Usage**
   - Each config variant = new cache
   - **Mitigation:** Clear structure, easy cleanup

3. **QG Complexity**
   - Many physics parameters
   - **Mitigation:** Good defaults, docs

4. **Dependency Weight**
   - Adds QG + Mura packages
   - **Mitigation:** Clean separation, optional

---

## Risk Assessment

### Low Risk ✅
- Phase 1 (Dependencies): Additive only
- Phase 2 (QG Dataset): Follows existing pattern
- Phase 3 (Mura Utils): Completely optional
- Phase 4 (Install): Makefile updates only

### Medium Risk ⚠️
- Git submodule management (need clear docs)
- Cache directory permissions (cluster FS)
- QG solver stability (depends on QG package)

### Mitigation Strategies
- Extensive testing before merge
- Clear error messages
- Rollback plan (revert commits)
- Documentation with examples

---

## Implementation Timeline

### Week 1: Core (5 days)
- **Day 1-2:** Add submodules, update Makefile, test existing
- **Day 3-5:** Implement `qg_dataset.py`, test caching

### Week 2: Polish (5 days)
- **Day 6-7:** Add Mura callbacks, test on cluster
- **Day 8-9:** Finalize installation, documentation
- **Day 10:** Integration testing, CI/CD

**Total:** 10 days for complete, tested implementation

---

## Success Metrics

- ✅ `make install` works on clean system
- ✅ All existing tests pass
- ✅ QG dataset generates successfully
- ✅ Caching works (2nd run is fast)
- ✅ Can train autoencoders on QG data
- ✅ WandB logs QG experiments correctly
- ✅ Documentation is clear and complete

---

## Next Steps

1. **Review:** Discuss this plan with team
2. **Prototype:** Implement Phase 1 in branch
3. **Test:** Validate on sample model
4. **Iterate:** Refine based on feedback
5. **Deploy:** Merge and document

---

## Questions?

See detailed plan in `INTEGRATION_PLAN.md` for:
- Complete code examples
- Testing strategy
- Dependency options
- Future enhancements
- FAQ

---

## TL;DR

**What:** Add QG physics simulations as cached dataset source  
**How:** New datamodule following existing pattern  
**Risk:** Low - no breaking changes  
**Effort:** ~250 lines of new code  
**Benefit:** Automated dataset generation with version tracking  
**Timeline:** 2 weeks for polished implementation
