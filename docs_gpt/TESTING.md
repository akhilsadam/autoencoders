# Reproducibility & Testing Approach

## Core Principle

**After each refactoring phase, the new code produces byte-identical results (modulo timestamps/random UUIDs).** This is verified before proceeding to the next phase.

---

## Testing Layers (Nested, Progressive)

### Layer 1: Unit Tests (Fast, Always)
**Per package**: `packages/X/tests/unit/`
**Time**: < 1s per package
**What**: Registry loading, config parsing, imports, basic function tests

```python
# packages/diffusion/tests/unit/test_registry.py
def test_diffusion_models_register():
    from mura.registry import Registry
    from diffusion.registry import MODEL_REGISTRY
    
    models = MODEL_REGISTRY.list_all()
    assert "ddpm" in models
    assert "edm" in models

# packages/diffusion-information-studies/tests/unit/test_operators.py
def test_camera_operator():
    from diffusion_studies.operators import camera_projection
    import torch
    
    x = torch.randn(2, 3, 64, 64)
    y = camera_projection(x, height=32, width=32)
    assert y.shape == (2, 3, 32, 32)
```

**Run**: `cd packages/X && pytest tests/unit/ -x`

---

### Layer 2: Reproducibility Tests (Determinism Checks)
**Per package**: `packages/X/tests/repro/`
**Time**: 10-30s per package (marked `@pytest.mark.slow`)
**What**: Train on tiny dataset with seed=42, verify deterministic

```python
# packages/ae-core/tests/repro/test_ae_determinism.py
import pytest
from omegaconf import OmegaConf

@pytest.mark.slow
def test_ae_training_deterministic():
    """Two AE trainings with seed=42 produce identical losses"""
    from ae.train import train_ae
    
    cfg1 = OmegaConf.load("../conf/exp/test_tiny.yaml")
    cfg1.seed = 42
    cfg1.trainer.max_steps = 10
    
    losses1 = train_ae(cfg1)
    losses2 = train_ae(cfg1)  # Same seed
    
    for l1, l2 in zip(losses1, losses2):
        assert abs(l1 - l2) < 1e-6, f"Loss mismatch: {l1} vs {l2}"

# packages/diffusion/tests/repro/test_diffusion_determinism.py
@pytest.mark.slow
def test_diffusion_training_deterministic():
    """Diffusion with seed=42 is deterministic"""
    from diffusion.train import train_diffusion
    
    cfg = OmegaConf.load("../conf/exp/test_tiny.yaml")
    cfg.seed = 42
    cfg.trainer.max_steps = 10
    
    loss1 = train_diffusion(cfg)
    loss2 = train_diffusion(cfg)
    
    assert abs(loss1 - loss2) < 1e-6
```

**Run**: `cd packages/X && pytest tests/repro/ -x -m slow`

---

### Layer 3: Before/After Baselines (Phase Verification)
**In orchestrator**: `tests/baselines/`
**Time**: 1-2 min per phase (run once before refactoring)
**What**: Capture reference metrics, verify they match after refactoring

```python
# tests/baselines/capture_baseline.py
"""Run before Phase 3 starts"""
import subprocess
import json

def capture_ae_baseline():
    """Run AE training, save loss progression"""
    cmd = "python -m autoencoders.train exp=test_ae_tiny max_steps=10 seed=42"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    
    # Load the loss from wandb/artifacts
    with open("runs/*/metrics.json") as f:
        metrics = json.load(f)
    
    baseline = {
        "final_loss": metrics["final_loss"],
        "loss_progression": metrics["loss_per_step"]
    }
    
    with open("baselines/ae_baseline.json", "w") as f:
        json.dump(baseline, f)

# tests/baselines/verify_baseline.py
"""Run after Phase 3 refactoring"""
def test_ae_baseline_matches():
    """After refactoring, AE produces same loss"""
    cmd = "python -m autoencoders.train exp=test_ae_tiny max_steps=10 seed=42"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    
    with open("runs/*/metrics.json") as f:
        current = json.load(f)
    
    with open("baselines/ae_baseline.json") as f:
        baseline = json.load(f)
    
    assert abs(current["final_loss"] - baseline["final_loss"]) < 1e-5
    for i, (c, b) in enumerate(zip(current["loss_per_step"], baseline["loss_per_step"])):
        assert abs(c - b) < 1e-5, f"Step {i}: {c} vs {b}"
```

**Run**: 
```bash
# Before Phase 3
pytest tests/baselines/capture_baseline.py::test_capture_ae_baseline

# After Phase 3
pytest tests/baselines/verify_baseline.py::test_ae_baseline_matches
```

---

### Layer 4: Integration Tests (End-to-End)
**In orchestrator**: `tests/integration/`
**Time**: 30-60s total (all packages together)
**What**: Full workflows, all packages loaded, config + task dispatch working

```python
# tests/integration/test_ae_workflow.py
import subprocess
from pathlib import Path

@pytest.mark.slow
def test_ae_training_workflow():
    """Old AE workflow still works end-to-end"""
    run_dir = Path("runs/test_ae_workflow")
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    
    cmd = "python -m autoencoders.train exp=test_ae_tiny max_steps=5 hydra.run.dir=runs/test_ae_workflow"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"
    assert run_dir.exists(), "Run directory not created"
    assert (run_dir / "metrics.json").exists(), "Metrics not logged"

@pytest.mark.slow
def test_diffusion_workflow():
    """New diffusion workflow works"""
    cmd = "python -m autoencoders.train exp=diffusion/test_tiny task=train_diffusion max_steps=5"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    assert result.returncode == 0, f"Diffusion training failed:\n{result.stderr}"

@pytest.mark.slow
def test_diffusion_information_studies_workflow():
    """Studies workflow works"""
    cmd = "python -m autoencoders.train exp=diffusion_information_studies/test_tiny task=run_study"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    assert result.returncode == 0, f"Study failed:\n{result.stderr}"

@pytest.mark.slow
def test_pde_cond_workflow():
    """PDE-conditioning workflow works"""
    cmd = "python -m autoencoders.train exp=pde_cond/test_tiny task=train_pde_cond max_steps=5"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    assert result.returncode == 0, f"PDE conditioning failed:\n{result.stderr}"
```

**Run**: `pytest tests/integration/ -x -m slow`

---

## Tiny Test Configs (Reusable)

Each package + orchestrator defines minimal configs for testing:

```yaml
# packages/ae-core/conf/exp/test_tiny.yaml
defaults:
  - override /model: ae/develop/test_tiny
  - override /data: fashion_mnist_tiny

trainer:
  max_steps: 10
  precision: 32
  log_every_n_steps: 1

project: test_ae
seed: 42
```

```yaml
# packages/diffusion/conf/exp/test_tiny.yaml
defaults:
  - override /model: diffusion/test_tiny_ddpm
  - override /data: fashion_mnist_tiny

trainer:
  max_steps: 10
  precision: 32

project: test_diffusion
seed: 42
```

**Benefits**:
- Tests don't need custom logic to reduce problem size
- Users can debug with `exp=*/test_tiny` quickly
- Configs are reproducible (same small dataset, same steps)

---

## Phase Verification Protocol

**For each phase, before committing:**

1. **Run all unit tests** (< 2 min total)
   ```bash
   pytest packages/*/tests/unit/ -x
   ```

2. **Run reproducibility tests** (< 2 min per new package)
   ```bash
   pytest packages/X/tests/repro/ -x -m slow
   ```

3. **Run baseline verification** (if applicable)
   ```bash
   # Phase 2: First time capturing
   pytest tests/baselines/capture_baseline.py::test_capture_metrics
   
   # Phases 3+: Verify against baseline
   pytest tests/baselines/verify_baseline.py -x
   ```

4. **Manual smoke tests**
   ```bash
   python -m autoencoders.train exp=test_ae_tiny max_steps=5
   python -m autoencoders.train exp=diffusion/test_tiny max_steps=5
   python -m autoencoders.train exp=diffusion_information_studies/test_tiny max_steps=5
   ```

5. **If all pass**: Commit phase
   ```bash
   git add packages/X tests/baselines/
   git commit -m "phase X: extract/refactor [component], verified reproducible"
   ```

---

## CI/CD Integration (Optional, Local First)

```makefile
.PHONY: test test-unit test-repro test-integration test-all

test-unit:
	cd packages/metrics-core && pytest tests/unit/ -x
	cd packages/ae-core && pytest tests/unit/ -x
	cd packages/diffusion && pytest tests/unit/ -x
	cd packages/pde-cond-diffusion && pytest tests/unit/ -x
	cd packages/diffusion-information-studies && pytest tests/unit/ -x

test-repro:
	cd packages/metrics-core && pytest tests/repro/ -x -m slow || true
	cd packages/ae-core && pytest tests/repro/ -x -m slow
	cd packages/diffusion && pytest tests/repro/ -x -m slow
	cd packages/diffusion-information-studies && pytest tests/repro/ -x -m slow || true

test-integration:
	pytest tests/integration/ -x -m slow

test-all: test-unit test-repro test-integration
	@echo "✓ All tests passed"
```

```bash
# Before committing a phase:
make test-all
```

---

## Handling Floating-Point Precision

Since we're checking for exact matches, use appropriate tolerances:

```python
# For losses and metrics (usually float32)
assert abs(baseline - current) < 1e-6

# For model weights (if comparing checkpoints)
torch.testing.assert_close(baseline_weights, current_weights, rtol=1e-5, atol=1e-6)

# For multi-step sequences
for i, (b, c) in enumerate(zip(baseline_seq, current_seq)):
    assert abs(b - c) < 1e-6, f"Step {i}: {b} vs {c}"
```

---

## What Gets Committed to Baselines

**In `tests/baselines/`**:
```
baselines/
├── ae_baseline.json              # Loss progression (Phase 3)
├── diffusion_baseline.json       # Loss progression (Phase 4)
├── pde_cond_baseline.json        # Loss progression (Phase 5)
├── dps_study_baseline.json       # Study metric results (Phase 6)
└── README.md                     # How baselines were captured
```

Each baseline is a JSON with:
```json
{
  "phase": 3,
  "date": "2026-07-13",
  "commit": "abc1234",
  "seed": 42,
  "steps": 10,
  "final_loss": 2.345,
  "loss_per_step": [10.5, 8.2, 6.1, ...],
  "metrics": {"accuracy": 0.95, ...}
}
```

---

## Summary: Testing Checklist Per Phase

- [ ] Unit tests pass (all new packages)
- [ ] Reproducibility tests pass (10 runs, same seed → same loss)
- [ ] Baseline captured/verified (if first time or refactor)
- [ ] Integration tests pass (full workflows)
- [ ] Manual smoke tests pass (3 workflows)
- [ ] No import errors
- [ ] Commit with clear message

---

**Goal**: After each phase, you can run `make test-all` and be 100% confident the refactoring didn't break anything.
