# Autoencoders v0.2.0 - Quick Start

## What's New in v0.2.0

✨ **QG Turbulence Dataset** - Physics-based dataset generation with automatic caching  
🔧 **Mura Utilities** - Reduced boilerplate, enhanced git tracking  
📦 **Unified Install** - One command installs everything  
✅ **100% Backward Compatible** - All existing code works unchanged

---

## Installation

```bash
cd autoencoders
make install
wandb login
huggingface-cli login  # Only for some datasets
```

**What it installs:**
- Mura v0.3.0 (Hydra utilities)
- QG (physics simulation)
- Autoencoders with all dependencies

---

## Quick Start

### Existing Datasets (unchanged)
```bash
# FashionMNIST
python -m src.autoencoders.train data=fashion_mnist model=mnist

# Aesthetic4K
python -m src.autoencoders.train data=aesthetic4k model=tiny_cu
```

### New: QG Turbulence Dataset
```bash
# Basic (uses defaults: 128x128, 100 samples)
python -m src.autoencoders.train data=qg_turbulence model=tiny_cu

# Or use make target
make train-qg

# High resolution
python -m src.autoencoders.train data=qg_turbulence data.params.grid_size=256

# More samples
python -m src.autoencoders.train data=qg_turbulence data.params.num_samples=1000

# Quick test (small params, offline WandB)
make test-qg
```

---

## QG Dataset Features

### Automatic Caching
- **First run:** Generates physics simulation (takes minutes)
- **Second run:** Loads from cache (instant!)
- **Different params:** Automatically creates new cache

### Cache Management
```bash
# View cache
ls -lh data/qg_cache/

# Clear cache
make clean-qg-cache

# Force regenerate
python -m src.autoencoders.train data=qg_turbulence data.params.force_regenerate=true
```

### Configurable Parameters
```yaml
# In config or via CLI
data.params:
  grid_size: 128        # Resolution: 64, 128, 256, 512
  num_samples: 100      # Number of simulations
  time_steps: 200       # Simulation length
  save_rate: 10         # Temporal sampling
  nu: 1.025e-5         # Viscosity (physics)
  mu: 0.0              # Linear drag (physics)
  beta: 0.0            # Beta plane (physics)
```

---

## Makefile Targets

```bash
make install          # Install everything (mura + qg + autoencoders)
make train            # Train with default config
make train-qg         # Train with QG dataset
make test-qg          # Quick QG test (small params)
make test-cu          # Test CUDA models
make test-hl          # Test high-level models
make clean-qg-cache   # Clear QG cache
```

---

## Migration from v0.1.0

**Good news:** Zero breaking changes! 

All your existing code, configs, and commands work exactly as before.

**What changed:**
- Added QG dataset (optional)
- Simplified imports via Mura utilities (transparent)
- Enhanced git tracking (automatic)

See `MIGRATION_GUIDE.md` for details.

---

## Documentation

- **`QUICK_REFERENCE.md`** - Command cheat sheet
- **`MIGRATION_GUIDE.md`** - v0.1.0 → v0.2.0 migration
- **`INTEGRATION_PLAN.md`** - Detailed implementation
- **`APPROACH_COMPARISON.md`** - Design decisions

---

## Examples

### Sweep QG Parameters
```bash
python -m src.autoencoders.train -m \
    data=qg_turbulence \
    data.params.grid_size=64,128,256
```

### Custom QG Config
Create `src/autoencoders/conf/data/qg_highres.yaml`:
```yaml
name: qg_turbulence
params:
  batch_size: 16
  grid_size: 512
  num_samples: 500
  time_steps: 1000
```

Use it:
```bash
python -m src.autoencoders.train data=qg_highres model=tiny_cu
```

---

## Cluster Usage

Same as before! Mura utilities automatically handle:
- Git tracking with diff saving
- WandB logging with full config
- Version tracking

```bash
# On cluster (Slurm)
sbatch scripts/train_qg.sh

# Or use deployment system
make deploy cluster=slurm data=qg_turbulence
```

---

## Architecture

```
autoencoders/
├── src/autoencoders/
│   ├── datamodules/
│   │   ├── fashion_mnist.py    # Existing
│   │   ├── aesthetic4k.py      # Existing
│   │   └── qg_turbulence.py    # NEW - QG integration
│   ├── models/                  # Unchanged
│   ├── train.py                 # Updated with Mura utilities
│   └── conf/
│       └── data/
│           └── qg_turbulence.yaml  # NEW
└── data/
    └── qg_cache/                # Auto-generated cache
        └── qg_turbulence_v1_*/
```

**Key:** QG is just another datamodule. Same pattern as existing datasets.

---

## Troubleshooting

### "QG package not found"
```bash
cd ../qg
uv pip install -e . --python ../autoencoders/.venv/bin/python
```

### "Mura module not found"
```bash
cd ../mura
uv pip install -e . --python ../autoencoders/.venv/bin/python
```

### Cache issues
```bash
make clean-qg-cache
python -m src.autoencoders.train data=qg_turbulence
```

---

## FAQ

**Q: Will this break my existing code?**  
A: No! 100% backward compatible.

**Q: Do I need to use QG?**  
A: No! It's completely optional. Existing datasets work as before.

**Q: How long does QG generation take?**  
A: First run: minutes (depends on params). Subsequent runs: instant (cached).

**Q: Can I customize QG physics?**  
A: Yes! Override any parameter via config or CLI.

**Q: How much disk space does cache use?**  
A: Depends on params. Typical: ~100MB per config variant.

---

## Credits

- **Autoencoders:** Akhil Sadam
- **Mura:** Akhil Sadam
- **QG:** Akhil Sadam

Integration follows Approach A (Minimal) with selective Approach B (Hybrid) features.

---

**Questions?** Check the documentation or open an issue!
