# Integration Plan: Autoencoders + QG + Mura

**Goal:** Integrate QG dataset generation with autoencoder training while preserving all existing functionality and adding selective Mura utilities.

**Priorities:**
1. ✅ Don't break existing autoencoders + WandB integration
2. ✅ QG generates datasets automatically (cached + versioned)
3. ✅ Fast, easy, trackable training jobs
4. ✅ Simple installation (`make install`)

---

## Architecture Overview

```
autoencoders/
├── src/autoencoders/
│   ├── datamodules/
│   │   ├── fashion_mnist.py    (existing)
│   │   ├── aesthetic4k.py      (existing)
│   │   └── qg_dataset.py       (NEW - QG integration)
│   ├── models/                  (existing - unchanged)
│   ├── util/
│   │   ├── gitinfo.py          (existing - keep)
│   │   ├── sec_id.py           (existing - keep)
│   │   ├── mura_callbacks.py   (NEW - from mura)
│   │   └── cache.py            (NEW - dataset caching)
│   ├── train.py                (existing - minor updates)
│   └── conf/
│       ├── config.yaml         (existing - minor updates)
│       └── data/
│           └── qg_turbulence.yaml  (NEW)
├── packages/
│   ├── qg/                      (submodule or copy)
│   └── mura/                    (submodule or copy)
└── pyproject.toml              (updated dependencies)
```

---

## Implementation Strategy

### Phase 1: Dependency Integration (No Breaking Changes)
**Goal:** Add QG and Mura as dependencies without changing existing code

**Actions:**
1. Add `qg` and `mura` as git submodules or direct dependencies
2. Update `pyproject.toml` with new dependencies
3. Update `Makefile` to install all packages
4. Test existing training still works

**Files Changed:**
- `pyproject.toml` - add dependencies
- `Makefile` - update install target
- `src/install/requirements.txt` - add qg, mura

**Testing:**
```bash
make install
make train  # Should work exactly as before
```

**Risk:** Low - additive only

---

### Phase 2: QG Dataset Module (Core Integration)
**Goal:** Create a datamodule that generates/caches QG datasets

**Design:**

```python
# src/autoencoders/datamodules/qg_dataset.py
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import hashlib
import json

@dataclass
class QGDatasetConfig:
    """Config for QG-generated datasets."""
    batch_size: int
    num_workers: int = 4
    
    # QG simulation params
    grid_size: int = 128
    num_samples: int = 1000
    time_steps: int = 200
    save_rate: int = 10
    
    # Caching
    cache_root: str = "${paths.data_root}/qg_cache"
    force_regenerate: bool = False
    
    seed: int = 42

def _config_hash(cfg: QGDatasetConfig) -> str:
    """Generate deterministic hash from config."""
    # Hash only params that affect data generation
    key_params = {
        'grid_size': cfg.grid_size,
        'num_samples': cfg.num_samples,
        'time_steps': cfg.time_steps,
        'save_rate': cfg.save_rate,
        'seed': cfg.seed,
    }
    cfg_str = json.dumps(key_params, sort_keys=True)
    return hashlib.sha256(cfg_str.encode()).hexdigest()[:12]

def _generate_qg_data(cfg: QGDatasetConfig, cache_path: Path):
    """Generate QG data using qg package."""
    from qg.solver.qg import QG
    from qg._input.validate_configuration import config as qg_config
    
    # Build QG config from our params
    qg_cfg = qg_config()
    qg_cfg.grid.Nx = cfg.grid_size
    qg_cfg.grid.Ny = cfg.grid_size
    qg_cfg.time.T = cfg.time_steps
    qg_cfg.time.save_rate = cfg.save_rate
    qg_cfg.seed = cfg.seed
    
    # Generate data
    solver = QG(qg_cfg)
    data = solver._run()  # Returns torch tensor: [B, T, C, H, W]
    
    # Save to cache
    cache_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        'data': data,
        'config': cfg,
        'version': '1.0',
    }, cache_path / 'data.pt')
    
    return data

def build_dataloaders(cfg: QGDatasetConfig):
    """Build train/val loaders from cached or generated QG data."""
    from omegaconf import OmegaConf
    
    # Resolve cache path (handle Hydra interpolations)
    cache_root = Path(OmegaConf.to_container(cfg.cache_root, resolve=True))
    
    # Create cache key from config hash
    config_hash = _config_hash(cfg)
    cache_path = cache_root / f"qg_v1_{config_hash}"
    cache_file = cache_path / 'data.pt'
    
    # Load or generate
    if cache_file.exists() and not cfg.force_regenerate:
        print(f"Loading cached QG data from {cache_path}")
        checkpoint = torch.load(cache_file)
        data = checkpoint['data']
    else:
        print(f"Generating QG data (will cache to {cache_path})")
        data = _generate_qg_data(cfg, cache_path)
    
    # Split into train/val (80/20)
    B, T, C, H, W = data.shape
    split_idx = int(0.8 * B)
    
    train_data = data[:split_idx].reshape(-1, C, H, W)  # Flatten B, T
    val_data = data[split_idx:].reshape(-1, C, H, W)
    
    # Create datasets (self-supervised: input = output)
    train_ds = TensorDataset(train_data, train_data)
    val_ds = TensorDataset(val_data, val_data)
    
    # Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
```

**Registry Update:**
```python
# src/autoencoders/datamodules/__init__.py
from .qg_dataset import QGDatasetConfig, build_dataloaders as build_qg

DATASET_REGISTRY: Dict[str, DatasetEntry] = {
    "fashion_mnist": DatasetEntry(FashionMNISTConfig, build_fashion_mnist),
    "aesthetic4k": DatasetEntry(Aesthetic4KConfig, build_aesthetic4k),
    "qg_turbulence": DatasetEntry(QGDatasetConfig, build_qg),  # NEW
}
```

**Hydra Config:**
```yaml
# src/autoencoders/conf/data/qg_turbulence.yaml
name: qg_turbulence

params:
  batch_size: 32
  num_workers: 4
  
  # QG simulation
  grid_size: 128
  num_samples: 100
  time_steps: 200
  save_rate: 10
  
  # Caching
  cache_root: ${paths.data_root}/qg_cache
  force_regenerate: false
  
  seed: 42
```

**Usage:**
```bash
# Generate QG dataset and train (automatic caching)
python -m src.autoencoders.train data=qg_turbulence model=tiny_cu

# Force regenerate dataset
python -m src.autoencoders.train data=qg_turbulence data.params.force_regenerate=true

# Different QG params = different cache (automatic versioning via hash)
python -m src.autoencoders.train data=qg_turbulence data.params.grid_size=256
```

**Testing:**
```bash
# Should cache on first run
python -m src.autoencoders.train data=qg_turbulence trainer.max_steps=100

# Should load from cache on second run (instant)
python -m src.autoencoders.train data=qg_turbulence trainer.max_steps=100
```

**Risk:** Low - follows existing pattern exactly

---

### Phase 3: Mura Utilities (Optional Enhancements)
**Goal:** Add useful Mura features without breaking anything

**Selective Integration:**

#### 3a. Git Callback (Already exists, enhance)
Current `gitinfo.py` works, but add Mura's `GitTrackerCallback`:

```python
# src/autoencoders/util/mura_callbacks.py
"""Lightning callbacks from Mura (selectively imported)."""
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import git
import hashlib

class EnhancedGitCallback(Callback):
    """Enhanced git tracking with diff logging."""
    
    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        try:
            repo = git.Repo(search_parent_directories=True)
            commit_hash = repo.head.commit.hexsha
            branch = repo.active_branch.name
            diff = repo.git.diff()
            diff_hash = hashlib.sha256(diff.encode()).hexdigest()
            
            # Log to WandB
            if trainer.logger:
                trainer.logger.experiment.config.update({
                    "git/commit": commit_hash,
                    "git/branch": branch,
                    "git/diff_sha256": diff_hash,
                    "git/dirty": bool(diff),
                })
                
                # Save diff to artifacts dir if dirty
                if diff and hasattr(trainer, 'log_dir'):
                    diff_path = Path(trainer.log_dir) / "git_diff.patch"
                    diff_path.write_text(diff)
                    
        except Exception as e:
            print(f"Git tracking failed: {str(e)}")
```

**Optional** - Use in `train.py`:
```python
# Add to imports
from .util.mura_callbacks import EnhancedGitCallback

# Add to callbacks
callbacks=[checkpoint_cb, lr_cb, EnhancedGitCallback()]
```

#### 3b. Version Manager (Optional)
Mura's `VersionManager` is useful but conflicts with Hydra's run directories.

**Decision:** Skip for now, Hydra's `${sec_id:}` + git SHA is sufficient.

#### 3c. Smart GPU Selection (Useful for cluster)
Copy from Mura:

```python
# src/autoencoders/util/gpu_utils.py (NEW)
"""GPU utilities for cluster environments."""
import torch

def get_free_gpus(memory_threshold_mb: float = 8.0):
    """Find GPUs with low utilization (cluster workaround)."""
    free_gpus = []
    for device_id in range(torch.cuda.device_count()):
        if torch.cuda.utilization(device_id) == 0:
            allocated = torch.cuda.memory_allocated(device_id) / 1e6
            if allocated <= memory_threshold_mb:
                free_gpus.append(device_id)
    return free_gpus
```

**Optional** - Use in trainer creation (only if needed on cluster).

**Risk:** Low - completely optional

---

### Phase 4: Installation System
**Goal:** One command installs everything

**Makefile Updates:**
```makefile
# Updated autoencoders/Makefile

# Add submodules or dependencies
.PHONY: install install-deps install-qg install-mura

install: install-deps install-qg install-mura install-autoencoders
	@echo "Installation complete!"
	@echo "Don't forget to run:"
	@echo "  wandb login"
	@echo "  huggingface-cli login"

install-deps:
	@command -v $(UV) >/dev/null 2>&1 || $(SYS_PYTHON) -m pip install --user uv
	@[ -d "$(VENV)" ] || $(UV) venv $(VENV)

install-qg:
    @echo "Installing QG package..."
    cd packages/qg && $(UV) pip install -e . --python $(PYTHON)

install-mura:
    @echo "Installing Mura package..."
    cd packages/mura && $(UV) pip install -e . --python $(PYTHON)

install-autoencoders:
	@echo "Installing autoencoders package..."
	$(UV) pip install -r $(INSTALL)/requirements.txt --python $(PYTHON)
	$(MAKE) py3-conf

# Test QG integration
test-qg-data:
	source "$(VENV)/bin/activate" && \
	$(PYTHON) -c "from autoencoders.datamodules import qg_dataset; print('QG dataset module loaded successfully')"

# Quick train with QG data
train-qg: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train data=qg_turbulence trainer.max_steps=100
```

**Pyproject.toml Updates:**
```toml
[project]
name = "autoencoders"
version = "0.2.0"  # Bump minor version
description = "Autoencoders with QG dataset generation"
# ... existing fields ...

dependencies = [
    # Existing
    "torch>=2.4.1",
    "lightning>=2.0.0",
    "wandb>=0.17.0",
    "hydra-core",
    "omegaconf",
    # ... other existing deps ...
    
    # NEW: Local packages (relative paths)
    "qg @ file:///${PROJECT_ROOT}/packages/qg",
    "mura @ file:///${PROJECT_ROOT}/packages/mura",
]

[project.optional-dependencies]
qg = [
    "qg @ file:///${PROJECT_ROOT}/packages/qg",
]
```

**Alternative: Git Submodules**
```bash
# In autoencoders repo
git submodule add ../qg packages/qg
git submodule add ../mura packages/mura
git submodule update --init --recursive
```

**Testing:**
```bash
# Fresh install
rm -rf .venv
make install

# Should work
make train-qg
```

**Risk:** Medium - depends on dependency resolution

---

## Pros/Cons Analysis

### ✅ Pros

1. **Zero Breaking Changes**
   - Existing `fashion_mnist`, `aesthetic4k` work exactly as before
   - All existing configs unchanged
   - WandB integration untouched
   
2. **Automatic Dataset Caching**
   - QG runs once per unique config
   - Hash-based versioning (change config = new cache)
   - Fast subsequent runs (load from disk)
   
3. **Clean Integration**
   - QG is just another datamodule
   - Follows existing pattern exactly
   - No special cases in training code
   
4. **Flexible**
   - Easy to add more QG variants
   - Can mix QG + other datasets
   - Override any param via CLI
   
5. **Minimal Code**
   - ~150 lines for full QG integration
   - Only 2 new files (`qg_dataset.py`, config)
   - No changes to existing files (except registry)
   
6. **Simple Installation**
   - One `make install` command
   - All dependencies managed
   - Clear error messages

### ❌ Cons

1. **Dataset Generation Time**
   - First run with new QG config will be slow
   - Mitigated by: caching, clear progress bars, small defaults
   
2. **Cache Management**
   - Disk space grows with QG variants
   - Mitigated by: clear cache structure, easy to delete
   
3. **QG Config Complexity**
   - Many physics params to understand
   - Mitigated by: good defaults, documentation
   
4. **Dependency Weight**
   - Adds QG + Mura packages
   - Mitigated by: optional dependencies, clear separation
   
5. **Mura Integration Incomplete**
   - Only using callbacks, not full workflow
   - Mitigated by: we only want utilities anyway

---

## Implementation Timeline

### Week 1: Core Integration
- **Day 1-2:** Phase 1 (Dependencies)
  - Add submodules
  - Update Makefile
  - Test existing functionality
  
- **Day 3-5:** Phase 2 (QG Dataset)
  - Implement `qg_dataset.py`
  - Create config
  - Test caching logic
  - Integration test with training

### Week 2: Polish
- **Day 6-7:** Phase 3 (Mura Utilities)
  - Add callbacks
  - Test on cluster
  
- **Day 8-9:** Phase 4 (Installation)
  - Finalize Makefile
  - Write documentation
  - Test fresh install
  
- **Day 10:** Documentation & Testing
  - README updates
  - Example commands
  - CI/CD (if applicable)

---

## Testing Strategy

### Unit Tests
```python
# tests/test_qg_dataset.py
def test_qg_config_hash():
    """Test config hashing is deterministic."""
    cfg1 = QGDatasetConfig(batch_size=32, grid_size=128)
    cfg2 = QGDatasetConfig(batch_size=32, grid_size=128)
    assert _config_hash(cfg1) == _config_hash(cfg2)
    
    cfg3 = QGDatasetConfig(batch_size=32, grid_size=256)
    assert _config_hash(cfg1) != _config_hash(cfg3)

def test_qg_caching(tmp_path):
    """Test dataset caching works."""
    cfg = QGDatasetConfig(
        batch_size=8,
        num_samples=10,
        grid_size=64,
        cache_root=str(tmp_path)
    )
    
    # First call generates
    train_loader1, _ = build_dataloaders(cfg)
    
    # Second call loads from cache
    train_loader2, _ = build_dataloaders(cfg)
    
    # Data should be identical
    batch1 = next(iter(train_loader1))
    batch2 = next(iter(train_loader2))
    assert torch.allclose(batch1[0], batch2[0])
```

### Integration Tests
```bash
# Test existing functionality preserved
make test-cu
make test-hl

# Test QG data generation
python -m pytest tests/test_qg_dataset.py

# Test full training pipeline
HYDRA_FULL_ERROR=1 python -m src.autoencoders.train \
    data=qg_turbulence \
    model=tiny_cu \
    trainer.max_steps=10 \
    wandb.mode=offline
```

### Manual Testing
```bash
# Fresh install
rm -rf .venv
make install

# Train with existing datasets (should work as before)
make train

# Train with QG (should generate and cache)
make train-qg

# Train with QG again (should load from cache, be fast)
make train-qg
```

---

## Documentation Updates

### README.md
```markdown
## Installation

```bash
make install
wandb login
huggingface-cli login  # Only needed for some datasets
```

## Quick Start

```bash
# Train with FashionMNIST (existing)
python -m src.autoencoders.train data=fashion_mnist model=mnist

# Train with QG-generated turbulence data
python -m src.autoencoders.train data=qg_turbulence model=tiny_cu

# Customize QG parameters
python -m src.autoencoders.train \
    data=qg_turbulence \
    data.params.grid_size=256 \
    data.params.num_samples=500
```

## Dataset Caching

QG datasets are automatically cached based on configuration parameters:
- First run: generates and saves to `data/qg_cache/qg_v1_<hash>/`
- Subsequent runs: loads from cache (instant)
- Different params = different cache (automatic versioning)

To force regeneration:
```bash
python -m src.autoencoders.train data=qg_turbulence data.params.force_regenerate=true
```

To clean cache:
```bash
rm -rf data/qg_cache/
```
```

---

## Migration Path

For existing users:

1. **No action needed** - existing configs work unchanged
2. **Optional**: Try QG dataset with `data=qg_turbulence`
3. **Optional**: Add Mura callbacks for enhanced git tracking

**Backward Compatibility:** 100% guaranteed

---

## Success Criteria

- ✅ Existing training commands work unchanged
- ✅ `make install` sets up everything
- ✅ QG dataset generates on first run
- ✅ QG dataset loads from cache on subsequent runs
- ✅ Different QG configs create separate caches
- ✅ WandB logging works for QG experiments
- ✅ Can train autoencoders on QG data successfully
- ✅ Cache management is intuitive
- ✅ Documentation is clear

---

## Future Enhancements (Not in Scope)

1. **Multi-Resolution QG**
   - Generate multiple grid sizes in one run
   - Pyramid caching

2. **Distributed QG Generation**
   - Run QG solver on multiple GPUs
   - Parallel batch generation

3. **QG-Specific Augmentations**
   - Time-reversal augmentation
   - Spectral augmentation

4. **Dataset Mixing**
   - Train on QG + real data
   - Multi-task learning

5. **Full Mura Workflow**
   - Version manager integration
   - Cluster deployment scripts

---

## Questions for Review

1. **Dependency Strategy:** Git submodules vs. `file://` installs?
2. **Cache Location:** `data/qg_cache/` vs. configurable root?
3. **Mura Integration:** Just callbacks or more features?
4. **QG Defaults:** What grid sizes, samples, etc.?
5. **Testing:** CI/CD setup or manual for now?

---

## Conclusion

This plan provides:
- ✅ **Zero risk** to existing functionality
- ✅ **Clean integration** following existing patterns
- ✅ **Automatic caching** with version tracking
- ✅ **Simple installation** with one command
- ✅ **Minimal code changes** (~200 LOC total)
- ✅ **Clear testing strategy**
- ✅ **Backward compatibility**

**Recommended:** Proceed with implementation, starting with Phase 1.
