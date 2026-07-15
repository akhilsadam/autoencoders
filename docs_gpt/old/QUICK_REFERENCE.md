# Quick Reference: Autoencoders + QG Integration

## Installation

```bash
# Clone and install everything
git clone <autoencoders-repo>
cd autoencoders
make install

# One-time setup
wandb login
huggingface-cli login  # Only for some datasets
```

## Training Commands

### Existing Datasets (unchanged)
```bash
# FashionMNIST
python -m src.autoencoders.train data=fashion_mnist model=mnist

# Aesthetic4K
python -m src.autoencoders.train data=aesthetic4k model=tiny_cu
```

### QG Datasets (new)
```bash
# Basic - uses defaults (128x128, 1000 samples)
python -m src.autoencoders.train data=qg_turbulence model=tiny_cu

# High resolution
python -m src.autoencoders.train data=qg_turbulence data.params.grid_size=256

# More samples
python -m src.autoencoders.train data=qg_turbulence data.params.num_samples=5000

# Longer simulation
python -m src.autoencoders.train data=qg_turbulence data.params.time_steps=500

# Combine overrides
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.grid_size=256 \
    data.params.num_samples=2000 \
    model=tiny_cu \
    trainer.max_steps=50000 \
    wandb.project=qg-experiments
```

## Cache Management

```bash
# View cache
ls -lh data/qg_cache/

# Each directory is one config variant
# qg_v1_<hash> where hash = f(grid_size, num_samples, time_steps, seed)

# Clear all cache (forces regeneration)
rm -rf data/qg_cache/

# Clear specific cache
rm -rf data/qg_cache/qg_v1_a1b2c3d4e5f6/

# Force regenerate (keep old cache)
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.force_regenerate=true
```

## Cluster Usage

```bash
# On cluster (with Slurm)
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=qg-train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

module load cuda/13.0
source .venv/bin/activate

python -m src.autoencoders.train \
    data=qg_turbulence \
    model=tiny_cu \
    trainer.devices=1
EOF

# Or use existing deployment system
make deploy cluster=slurm data=qg_turbulence
```

## Development

```bash
# Run tests
make test-cu
make test-hl
pytest tests/test_qg_dataset.py

# Quick debug run (offline, small steps)
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.num_samples=10 \
    trainer.max_steps=5 \
    wandb.mode=offline

# Check config without running
python -m src.autoencoders.train \
    data=qg_turbulence \
    --cfg job
```

## Common QG Parameters

```yaml
# Grid resolution
data.params.grid_size: 64, 128, 256, 512

# Number of independent samples
data.params.num_samples: 10, 100, 1000, 10000

# Simulation time steps
data.params.time_steps: 100, 200, 500, 1000

# Save frequency (affects temporal resolution)
data.params.save_rate: 5, 10, 20, 50

# Random seed (for reproducibility)
data.params.seed: 42
```

## Troubleshooting

### QG generation fails
```bash
# Check QG package installed
python -c "import qg; print('QG OK')"

# Test QG directly
cd packages/qg
python -m pytest src/qg/test.py
```

### Cache loading fails
```bash
# Check cache exists
ls data/qg_cache/

# Check permissions
chmod -R u+rw data/qg_cache/

# Force regenerate
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.force_regenerate=true
```

### Out of memory
```bash
# Reduce batch size
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.batch_size=8

# Reduce grid size
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.grid_size=64

# Reduce samples
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.num_samples=100
```

### Slow generation
```bash
# Use smaller defaults
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.grid_size=64 \
    data.params.num_samples=50 \
    data.params.time_steps=100

# Or wait for cache on first run (progress bar shows status)
```

## File Locations

```
autoencoders/
├── data/
│   └── qg_cache/              # QG datasets cached here
├── runs/                      # Hydra output dirs
│   └── qg_turbulence/tiny_cu/
│       └── <sec_id>_<git_msg>/
│           ├── artifacts/     # Checkpoints, reconstructions
│           ├── config.yaml    # Full resolved config
│           ├── git_diff.patch # If repo was dirty
│           └── ...
├── packages/
│   ├── qg/                    # QG package (submodule)
│   └── mura/                  # Mura package (submodule)
└── src/autoencoders/
    ├── datamodules/
    │   └── qg_dataset.py      # QG integration
    └── conf/data/
        └── qg_turbulence.yaml # QG config
```

## Configuration Files

### Default QG Config
`src/autoencoders/conf/data/qg_turbulence.yaml`
```yaml
name: qg_turbulence
params:
  batch_size: 32
  num_workers: 4
  grid_size: 128
  num_samples: 1000
  time_steps: 200
  save_rate: 10
  cache_root: ${paths.data_root}/qg_cache
  force_regenerate: false
  seed: 42
```

### Custom QG Variant
Create `src/autoencoders/conf/data/qg_highres.yaml`
```yaml
name: qg_turbulence
params:
  batch_size: 16
  grid_size: 512
  num_samples: 500
  time_steps: 1000
  save_rate: 50
```

Use with:
```bash
python -m src.autoencoders.train data=qg_highres model=tiny_cu
```

## WandB Integration

```bash
# All QG runs are logged to WandB automatically
# Run name includes dataset: "qg_turbulence-tiny_cu-..."
# Tags include: ["qg_turbulence", "tiny_cu"]

# Custom project
python -m src.autoencoders.train \
    data=qg_turbulence \
    wandb.project=my-qg-experiments

# Offline mode (no internet)
python -m src.autoencoders.train \
    data=qg_turbulence \
    wandb.mode=offline

# Disable WandB
python -m src.autoencoders.train \
    data=qg_turbulence \
    wandb.mode=disabled
```

## Pro Tips

1. **Start small:** Test with small params first
   ```bash
   data.params.grid_size=64 data.params.num_samples=10
   ```

2. **Use cache:** Same params = instant loading
   
3. **Sweep params:** Hydra makes it easy
   ```bash
   python -m src.autoencoders.train -m \
       data=qg_turbulence \
       data.params.grid_size=64,128,256
   ```

4. **Check cache before long runs:**
   ```bash
   ls data/qg_cache/  # See what's already generated
   ```

5. **Name your runs:**
   ```bash
   run.name=my-qg-experiment
   ```

## Further Reading

- `INTEGRATION_PLAN.md` - Detailed implementation plan
- `INTEGRATION_SUMMARY.md` - High-level overview
- `README.md` - General autoencoders documentation
 - `packages/qg/README.md` - QG package documentation
