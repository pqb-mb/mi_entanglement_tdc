# MI-Entanglement Pipeline Architecture

This document provides a detailed view of the quantum-inspired pipeline for molecular property prediction.

> **Note**: This pipeline uses classical simulation of quantum circuits via PennyLane. The algorithm is designed for future deployment on quantum hardware.

## Pipeline Overview

```mermaid
flowchart TD
    subgraph Entry["run_experiment.py"]
        A[CLI Args Parser] --> B{--generate-features?}
        B -->|Yes| C[Generate MapLight Features Only]
        B -->|No| D[run_benchmark]
    end

    subgraph CUDA["CUDA Initialization"]
        A0[Load libcudart.so with RTLD_GLOBAL] --> A1[Initialize CUDA Driver API]
        A1 --> A2[Initialize CUDA Runtime API]
        A2 --> A3[Import PyTDC safely]
    end

    A0 -.-> A

    subgraph Step1["Step 1: Feature Generation"]
        D --> E[generate_maplight_for_dataframe]
        E --> E1[Check Cache]
        E1 -->|Cache Hit| E2[Load .npy]
        E1 -->|Cache Miss| E3[Compute 2563-dim features]
        E3 --> E4[ECFP counts 1024]
        E3 --> E5[Avalon counts 1024]
        E3 --> E6[ERG 315]
        E3 --> E7[RDKit props 200]
        E4 & E5 & E6 & E7 --> E8[Save to Cache]
    end

    subgraph Step2["Step 2: MI Discovery"]
        E2 & E8 --> F[run_maplight_mi_discovery]
        F --> F1[Prefilter top K features by univariate MI]
        F1 --> F2[Compute pairwise MI matrix]
        F2 --> F3[Select top pairs by MI threshold]
        F2 --> F4[Select top triads by MI threshold]
        F3 & F4 --> F5[Dynamic circuit params]
        F5 --> F6[Enforce max_qubits constraint]
        F6 --> F7[Cache MI results]
    end

    subgraph Step3["Step 3: Quantum-Inspired Encoding"]
        F7 --> G[quantum_encode]
        G --> G1[Scale features MinMax 0-1]
        G1 --> G2[2Q Encoder]
        G1 --> G3[3Q Encoder]
        G2 --> G4[HamiltonianProjectedEncoder<br/>MI-weighted coupling]
        G3 --> G5[HamiltonianProjectedEncoder<br/>MI-weighted coupling]
        G4 & G5 --> G6[Encoded features]
    end

    subgraph Step4["Step 4: Feature Assembly"]
        G6 --> H[Concatenate Features]
        H --> H1[MapLight raw 2563-dim]
        H --> H2[2Q projected features]
        H --> H3[3Q projected features]
        H1 & H2 & H3 --> H4[Final feature matrix]
    end

    subgraph Step5["Step 5: CatBoost Training"]
        H4 --> I[For each seed in BENCHMARK_SEEDS]
        I --> I1[CatBoostClassifier/Regressor]
        I1 --> I2[Train on train_val]
        I2 --> I3[Predict on test]
        I3 --> I4[Store predictions]
    end

    subgraph Eval["Evaluation"]
        I4 --> J[group.evaluate_many]
        J --> J1[TDC Benchmark Results]
        J1 --> K[Save results.json]
    end

    subgraph Config["Configuration Sources"]
        C1[DEFAULT_CONFIG] --> C2[configs/benchmark.json]
        C2 --> C3[CLI overrides]
        C3 --> C4[Effective Config]
    end

    C4 -.-> D
```

## Module Responsibilities

| Module | File | Description |
|--------|------|-------------|
| Entry Point | `src/run_experiment.py` | CLI parsing, CUDA initialization, dispatch |
| Feature Generation | `src/features.py` | MapLight 2563-dim molecular descriptors |
| MI Discovery | `src/mi_discovery.py` | Univariate MI prefilter + pairwise MI for pairs/triads |
| Quantum-Inspired Encoding | `src/quantum_encoding.py` | Hamiltonian time evolution simulation |
| Encoder Core | `src/projected_q_encoder.py` | PennyLane-based HamiltonianProjectedEncoder (classical simulation) |
| Benchmark Runner | `src/benchmark.py` | Feature assembly, CatBoost training, TDC evaluation |
| Configuration | `src/config.py` | Parameters, caching, metrics |
| MI Utilities | `src/mi_entanglement_utils.py` | Pairwise MI computation, pair/triad selection |

## Data Flow

```
SMILES strings (train_val, test)
    │
    ▼
┌─────────────────────────────────────┐
│  MapLight Features (2563-dim)       │
│  ├── ECFP counts (1024)             │
│  ├── Avalon counts (1024)           │
│  ├── ErG descriptors (315)          │
│  └── RDKit properties (200)         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  MI Pre-filtering                   │
│  └── Top 100 features by I(X;Y)     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Pairwise MI Matrix (100×100)       │
│  └── I(Xᵢ; Xⱼ) for feature pairs    │
└─────────────────────────────────────┘
    │
    ├──────────────────┐
    ▼                  ▼
┌──────────────┐  ┌──────────────┐
│  Top Pairs   │  │  Top Triads  │
│  (i,j)       │  │  (i,j,k)     │
│  max 30      │  │  max 15      │
└──────────────┘  └──────────────┘
    │                  │
    ▼                  ▼
┌──────────────┐  ┌──────────────┐
│  2Q Circuit  │  │  3Q Circuit  │
│  ≤28 qubits  │  │  ≤28 qubits  │
└──────────────┘  └──────────────┘
    │                  │
    ▼                  ▼
┌──────────────┐  ┌──────────────┐
│  ⟨σᶻᵢ⟩       │  │  ⟨σᶻᵢ⟩       │
│  features    │  │  features    │
└──────────────┘  └──────────────┘
    │                  │
    └────────┬─────────┘
             ▼
┌─────────────────────────────────────┐
│  Final Features                     │
│  ├── MapLight raw (2563)            │
│  ├── 2Q projections (simulated)     │
│  └── 3Q projections (simulated)     │
└─────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  CatBoost (5 seeds)                 │
│  └── Gradient boosted classifier    │
└─────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  TDC Evaluation                     │
│  └── group.evaluate_many()          │
└─────────────────────────────────────┘
```

## Caching Strategy

The pipeline caches intermediate results for efficiency:

| Cache | Location | Key | Contents |
|-------|----------|-----|----------|
| MapLight features | `cache/maplight_<hash>.npy` | SHA256 of SMILES list | NumPy array (n, 2563) |
| MI discovery | `cache/mi_phase1_<version>_<hash>_<params>.json` | Data + config hash | Pairs, triads, index maps |

Cache invalidation: Bump `CACHE_VERSION` in `config.py` when feature generation or MI logic changes.
