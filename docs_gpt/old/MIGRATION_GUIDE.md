# Migration Guide: v0.1.0 → v0.2.0

## Overview

Version 0.2.0 introduces:
- **QG turbulence dataset** integration with automatic caching
- **Mura v0.3.0** Hydra utilities (refactored for Hydra compatibility)
- **Reduced boilerplate** in training scripts
- **No breaking changes** to existing configs or models

---

## What Changed

### ✅ Fully Backward Compatible
- All existing datasets work unchanged (`fashion_mnist`, `aesthetic4k`)
- All existing models work unchanged (`mnist`, `tiny_cu`, `tiny_hl`)
- All existing configs work unchanged
- All existing commands work unchanged

### 🆕 New Features

#### 1. QG Turbulence Dataset
```bash
# New dataset available
python -m src.autoencoders.train data=qg_turbulence model=tiny_cu

# Automatic caching (second run is instant)
python -m src.autoencoders.train data=qg_turbulence  # First: generates
python -m src.autoencoders.train data=qg_turbulence  # Second: loads cache
```

#### 2. Mura Hydra Utilities
The custom `util.sec_id` and `util.gitinfo` modules are replaced by Mura's unified resolvers:

**Before (v0.1.0):**
```python
from .util import sec_id
from .util import gitinfo
```

**After (v0.2.0):**
```python
from mura.hydra import register_resolvers
register_resolvers()
```

**Impact:** Your configs still work! The resolvers `${sec_id:}` and `${gitinfo:sha}` still function identically.

#### 3. Enhanced Git Tracking
Git diffs are now automatically saved to artifacts when the repo is dirty.

#### 4. Simplified Installation
```bash
# Now installs mura, qg, and autoencoders
make install
```

---

## Installation

### Fresh Install
```bash
cd /path/to/ml/autoencoders
make install
wandb login
huggingface-cli login  # Only for some datasets
```

### Upgrading Existing Installation
```bash
cd /path/to/ml/autoencoders
git pull
make install  # Reinstalls with new dependencies
```

---

## New Commands

### Quick Reference
```bash
# Train with QG dataset
make train-qg

# Quick test with small QG params (offline WandB)
make test-qg

# Clean QG cache
make clean-qg-cache

# Regular training (unchanged)
make train
```

### QG Configuration Examples
```bash
# High resolution
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.grid_size=256

# More samples
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.num_samples=1000

# Force regenerate (ignore cache)
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.force_regenerate=true
```

---

## Cache Management

### Cache Location
```
data/qg_cache/
└── qg_turbulence_v1_<hash>/
    ├── data.pt
    └── metadata.json
```

### How Caching Works
1. Config parameters → SHA256 hash
2. Check if `data/qg_cache/qg_turbulence_v1_<hash>/` exists
3. If yes: load from cache (instant)
4. If no: generate via QG solver, save to cache

### Cache Keys
These parameters affect the cache hash:
- `grid_size`, `num_samples`, `time_steps`, `save_rate`
- `dt`, `nu`, `mu`, `beta`, `seed`

Changing any of these creates a new cache automatically.

### Managing Cache
```bash
# View cache
ls -lh data/qg_cache/

# Clear specific cache
rm -rf data/qg_cache/qg_turbulence_v1_abc123def456/

# Clear all cache
make clean-qg-cache
```

---

## For Developers

### Changes to `train.py`
The boilerplate in `train.py` was reduced by using Mura utilities:

**Changed:**
- Imports: Now uses `mura.hydra` utilities
- Logger creation: Uses `create_wandb_logger()`
- Trainer creation: Uses `create_trainer_with_defaults()`
- Callbacks: Added `EnhancedGitCallback()`

**Unchanged:**
- Model instantiation
- Data loading logic
- Training loop
- Artifact management

### New Dependencies
- `mura>=0.3.0` - Hydra utilities
- `qg` - QG turbulence solver

Both installed automatically via `make install`.

---

## Troubleshooting

### "QG package not found"
```bash
# Reinstall
make install-qg
```

### "Mura module not found"
```bash
# Reinstall
make install-mura
```

### Cache issues
```bash
# Clear cache and regenerate
make clean-qg-cache
python -m src.autoencoders.train data=qg_turbulence
```

### Old imports not working
If you have custom scripts using old imports:
```python
# Old (still works in configs)
from autoencoders.util import sec_id
from autoencoders.util import gitinfo

# New (recommended for code)
from mura.hydra import register_resolvers, compute_git_info
```

---

## Testing Your Setup

### 1. Test existing functionality
```bash
make test-cu
make test-hl
make train  # Should work exactly as before
```

### 2. Test QG integration
```bash
make test-qg  # Quick test with small params
```

### 3. Test full QG training
```bash
make train-qg  # Full training run
```

---

## What to Expect

### First QG Run
- **Time:** Few minutes (depends on params)
- **Output:** Progress bars, cache save messages
- **Result:** Data cached for future use

### Subsequent QG Runs (same params)
- **Time:** <1 second (loads from cache)
- **Output:** "Loading cached data from..."
- **Result:** Training starts immediately

---

## Need Help?

1. Check `QUICK_REFERENCE.md` for common commands
2. Check `INTEGRATION_PLAN.md` for detailed documentation
3. Check `APPROACH_COMPARISON.md` for design rationale

---

## Rollback (if needed)

If you need to revert to v0.1.0:
```bash
git checkout v0.1.0  # Or your previous commit
make install-standalone  # Installs without mura/qg
```

---

## Summary

✅ **All existing code works unchanged**  
✅ **New QG dataset available**  
✅ **Automatic caching with version tracking**  
✅ **Reduced boilerplate via Mura utilities**  
✅ **Simple installation: `make install`**
