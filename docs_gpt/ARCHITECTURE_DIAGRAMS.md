# Architecture Diagrams

Visual representations of the integration architecture.

---

## Current Architecture (Before Integration)

```
┌─────────────────────────────────────────────────────────────┐
│                      autoencoders                            │
│                                                              │
│  ┌────────────┐         ┌──────────────────────────────┐   │
│  │  train.py  │────────▶│     Hydra Config System      │   │
│  │  (entry)   │         │  - model: mnist/tiny_cu/...  │   │
│  └────────────┘         │  - data: fashion_mnist/...   │   │
│        │                └──────────────────────────────┘   │
│        │                                                     │
│        ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           DataModule Registry                         │  │
│  │  - fashion_mnist  → FashionMNIST Dataset             │  │
│  │  - aesthetic4k    → HuggingFace Dataset              │  │
│  └──────────────────────────────────────────────────────┘  │
│        │                                                     │
│        ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         PyTorch Lightning Trainer                     │  │
│  │  - Model (mnist/tiny_cu/tiny_hl)                     │  │
│  │  - Callbacks (checkpoint, learning rate)             │  │
│  │  - Logger (WandB)                                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Proposed Architecture (After Integration - Approach A)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         autoencoders                                  │
│                                                                       │
│  ┌────────────┐         ┌─────────────────────────────────────┐    │
│  │  train.py  │────────▶│      Hydra Config System            │    │
│  │  (entry)   │         │  - model: mnist/tiny_cu/...         │    │
│  └────────────┘         │  - data: fashion_mnist/aesthetic4k/ │    │
│        │                │          qg_turbulence  ◄─── NEW    │    │
│        │                └─────────────────────────────────────┘    │
│        │                                                             │
│        ▼                                                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              DataModule Registry                                │ │
│  │  - fashion_mnist  → FashionMNIST Dataset                       │ │
│  │  - aesthetic4k    → HuggingFace Dataset                        │ │
│  │  - qg_turbulence  → QG Generated Dataset  ◄─── NEW             │ │
│  └────────────────────────────────────────────────────────────────┘ │
│         │                                                            │
│         │ if qg_turbulence:                                         │
│         ▼                                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │           QG Dataset Module (NEW)                               │ │
│  │                                                                  │ │
│  │  1. Compute config hash                                         │ │
│  │  2. Check cache: data/qg_cache/qg_v1_<hash>/                   │ │
│  │     ├─ Exists?   → Load from disk (instant)                    │ │
│  │     └─ Missing?  → Generate via QG solver ────┐                │ │
│  │                                                │                │ │
│  └────────────────────────────────────────────────│────────────────┘ │
│                                                   │                  │
│                                                   ▼                  │
│         ┌───────────────────────────────────────────────────┐       │
│         │  lib/qg (submodule or dependency)                 │       │
│         │  - QG Solver (physics simulation)                 │       │
│         │  - Generates: [B, T, C, H, W] tensor              │       │
│         │  - Saves to cache                                 │       │
│         └───────────────────────────────────────────────────┘       │
│                                                   │                  │
│                                                   ▼                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │         PyTorch Lightning Trainer                               │ │
│  │  - Model (mnist/tiny_cu/tiny_hl)                               │ │
│  │  - Callbacks (checkpoint, lr, git ◄─ Enhanced if Approach B)  │ │
│  │  - Logger (WandB)                                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Optional: lib/mura (submodule or dependency)                       │
│            - Callbacks, utilities (not full workflow)               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: QG Dataset Generation

```
┌──────────────────────────────────────────────────────────────────┐
│ User Command:                                                     │
│ python -m src.autoencoders.train data=qg_turbulence model=tiny_cu│
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Hydra loads config:                                              │
│   data.params.grid_size = 128                                    │
│   data.params.num_samples = 1000                                 │
│   data.params.time_steps = 200                                   │
│   data.params.cache_root = "data/qg_cache"                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ build_dataloaders(cfg) called                                    │
│   1. Compute hash: h = sha256(grid_size, num_samples, ...)      │
│      Result: h = "a1b2c3d4e5f6"                                  │
│   2. Check cache path: data/qg_cache/qg_v1_a1b2c3d4e5f6/        │
└──────────────────────────────────────────────────────────────────┘
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
         ┌────────────────┐    ┌────────────────┐
         │ Cache EXISTS?  │    │ Cache MISSING? │
         │ (2nd+ run)     │    │ (1st run)      │
         └────────────────┘    └────────────────┘
                   │                     │
                   ▼                     ▼
     ┌──────────────────────┐  ┌─────────────────────────┐
     │ Load from cache:     │  │ Generate via QG solver: │
     │ data.pt              │  │                         │
     │ [B, T, C, H, W]      │  │ 1. Create QG config     │
     │                      │  │ 2. Initialize solver    │
     │ Time: <1 second      │  │ 3. Run simulation       │
     │                      │  │ 4. Save to cache        │
     │                      │  │                         │
     │                      │  │ Time: minutes           │
     └──────────────────────┘  └─────────────────────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
           ┌────────────────────────────────────────┐
           │ Create PyTorch DataLoaders:            │
           │ - Train: 80% of data                   │
           │ - Val:   20% of data                   │
           │ - Batch size from config               │
           └────────────────────────────────────────┘
                              │
                              ▼
           ┌────────────────────────────────────────┐
           │ Return to train.py                     │
           │ → Pass to Lightning Trainer            │
           │ → Training begins                      │
           └────────────────────────────────────────┘
```

---

## Cache Structure

```
data/
└── qg_cache/
    ├── qg_v1_a1b2c3d4e5f6/          # grid_size=128, num_samples=1000
    │   ├── data.pt                   # Cached tensor [B, T, C, H, W]
    │   └── metadata.json             # Config snapshot
    │
    ├── qg_v1_f6e5d4c3b2a1/          # grid_size=256, num_samples=1000
    │   ├── data.pt
    │   └── metadata.json
    │
    └── qg_v1_1234567890ab/          # grid_size=128, num_samples=5000
        ├── data.pt
        └── metadata.json

Notes:
- Each unique config gets its own cache directory
- Hash is deterministic (same config = same hash)
- Version prefix (qg_v1) allows format evolution
```

---

## Hydra Output Structure

```
runs/
└── qg_turbulence/                   # Dataset name
    └── tiny_cu/                     # Model name
        └── A1B2C3_add-qg-dataset/   # sec_id + git message
            ├── .hydra/              # Hydra metadata
            │   ├── config.yaml      # Full resolved config
            │   ├── overrides.yaml
            │   └── hydra.yaml
            │
            ├── artifacts/           # Training artifacts
            │   ├── checkpoints/
            │   │   ├── last.ckpt
            │   │   └── epoch=01-val_loss=0.1234.ckpt
            │   └── reconstructions/
            │       ├── inputs.png
            │       └── reconstructions.png
            │
            ├── config.yaml          # Saved by train.py
            ├── git_diff.patch       # If repo was dirty
            └── train.log            # Training logs

Notes:
- Directory structure is hierarchical: dataset/model/run_id
- sec_id is time-based sortable ID
- git message helps identify changes
```

---

## Installation Dependency Graph

```
┌────────────────────┐
│  make install      │
└────────────────────┘
          │
          ├─────────────────────────────────────────────────┐
          │                                                  │
          ▼                                                  ▼
┌──────────────────┐                            ┌──────────────────┐
│ Install UV       │                            │ Create venv      │
│ (package mgr)    │                            │ .venv/           │
└──────────────────┘                            └──────────────────┘
          │                                                  │
          └──────────────────┬───────────────────────────────┘
                             ▼
          ┌──────────────────────────────────────┐
          │ Install Dependencies                 │
          │ (in parallel where possible)         │
          └──────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐
│ Install QG   │   │ Install Mura │   │ Install Autoencoders│
│ (lib/qg)     │   │ (lib/mura)   │   │ (requirements.txt)  │
└──────────────┘   └──────────────┘   └─────────────────────┘
        │                    │                    │
        └────────────────────┴────────────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │ Configure Paths  │
                  │ (python3-config) │
                  └──────────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │ Done!            │
                  │ Ready to train   │
                  └──────────────────┘
```

---

## Comparison: Approach A vs B vs C

```
┌────────────────────────────────────────────────────────────────┐
│                        Approach A (Minimal)                     │
│  autoencoders (Hydra)                                          │
│    ├─ QG (as datamodule)        ◄── Integration point         │
│    └─ Mura (utilities only)     ◄── Optional callbacks        │
│                                                                 │
│  Complexity: Low    │  Risk: Low    │  Time: 1 week            │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                      Approach B (Hybrid)                        │
│  autoencoders (Hydra)                                          │
│    ├─ QG (as datamodule)        ◄── Integration point         │
│    └─ Mura (deep integration)   ◄── Refactored Mura           │
│         ├─ Callbacks                                           │
│         ├─ Utilities                                           │
│         └─ Factory functions                                   │
│                                                                 │
│  Complexity: Medium │  Risk: Medium │  Time: 3 weeks           │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                   Approach C (Full Migration)                   │
│  autoencoders (Mura-based) ◄── Replaces Hydra                 │
│    ├─ QG (via Mura workflow)                                  │
│    └─ Mura (full adoption)                                     │
│         ├─ lightning_run()                                     │
│         ├─ TOML configs                                        │
│         └─ Version manager                                     │
│                                                                 │
│  Complexity: High   │  Risk: High   │  Time: 6 weeks           │
│  ❌ Loses Hydra composition                                     │
│  ❌ Loses CLI overrides                                         │
└────────────────────────────────────────────────────────────────┘

Recommendation: Approach A ✅
```

---

## Integration Timeline

```
Week 1: Core Integration (Phase 1)
┌─────┬─────┬─────┬─────┬─────┐
│ Mon │ Tue │ Wed │ Thu │ Fri │
├─────┼─────┼─────┼─────┼─────┤
│ Add │ Add │ QG  │ QG  │ Test│
│ Deps│ Deps│ Data│ Data│ All │
│     │     │ Mod │ Mod │     │
│     │     │     │     │     │
│Setup│Test │Write│Cache│Intg │
└─────┴─────┴─────┴─────┴─────┘

Week 2: Polish (Phase 2, Optional)
┌─────┬─────┬─────┬─────┬─────┐
│ Mon │ Tue │ Wed │ Thu │ Fri │
├─────┼─────┼─────┼─────┼─────┤
│ Mura│ GPU │ Test│ Docs│Docs │
│ Call│ Util│Clust│     │     │
│back │     │ er  │     │     │
│     │     │     │     │     │
│Enh  │Add  │Test │Write│Final│
└─────┴─────┴─────┴─────┴─────┘

Week 3+: Deploy & Monitor
┌──────────────────────────┐
│ Deploy to cluster        │
│ Monitor usage            │
│ Gather feedback          │
│ Iterate improvements     │
└──────────────────────────┘
```

---

## Code Size Comparison

```
Approach A (Minimal):
┌──────────────────────────┐
│ New Code:                │
│ ████████████ 250 lines   │
│                          │
│ Modified Code:           │
│ █ 25 lines               │
└──────────────────────────┘

Approach B (Hybrid):
┌──────────────────────────┐
│ New Code:                │
│ ████████████████████████ │
│ ████████ 500 lines       │
│                          │
│ Modified Code:           │
│ ████ 100 lines           │
└──────────────────────────┘

Approach C (Full Migration):
┌──────────────────────────┐
│ Modified Code:           │
│ ████████████████████████ │
│ ████████████████████████ │
│ ████████████████████████ │
│ ████████████████████████ │
│ ████ 2000 lines          │
└──────────────────────────┘

Clear winner: Approach A ✅
```

---

## Test Coverage

```
┌────────────────────────────────────────────────┐
│ Test Pyramid (Approach A)                      │
│                                                 │
│                    ┌─────┐                     │
│                    │ E2E │  Integration Test   │
│                    │  1  │  (full training)    │
│                    └─────┘                     │
│                  ┌─────────┐                   │
│                  │  Integ  │  Component Tests  │
│                  │    5    │  (cache, load)    │
│                  └─────────┘                   │
│              ┌───────────────┐                 │
│              │  Unit Tests   │  Function Tests │
│              │      10       │  (hash, config) │
│              └───────────────┘                 │
│                                                 │
│ Total: 16 tests (manageable)                   │
└────────────────────────────────────────────────┘
```

---

## Summary Visual

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Integration Summary (Approach A)            ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                              ┃
┃  ✅ What Works:                              ┃
┃     • QG generates datasets                 ┃
┃     • Automatic caching & versioning        ┃
┃     • Zero breaking changes                 ┃
┃     • Simple installation                   ┃
┃     • Same CLI as existing                  ┃
┃                                              ┃
┃  📊 Metrics:                                 ┃
┃     • New code: ~250 lines                  ┃
┃     • Time: 1-2 weeks                       ┃
┃     • Risk: Low                             ┃
┃                                              ┃
┃  🎯 Next: Implement Phase 1                 ┃
┃                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```
