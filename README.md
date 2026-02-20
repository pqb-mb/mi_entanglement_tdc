# MI-Entanglement: Quantum-Inspired Feature Augmentation for TDC ADMET Benchmarks

Mutual Information-guided quantum-inspired feature engineering for the TDC ADMET benchmark suite.

> **Note**: This implementation uses classical simulation of quantum circuits via [PennyLane](https://pennylane.ai/). The algorithm is designed for future deployment on quantum hardware, but currently runs on classical CPUs/GPUs.

## Method

This approach uses data-driven quantum-inspired entanglement patterns to augment classical molecular features:

1. **MapLight Feature Generation** — 2563 molecular descriptors from SMILES (ECFP + Avalon fingerprints + ErG descriptors + RDKit physicochemical properties)
2. **MI Pre-filtering** — Top-K features selected by univariate mutual information with the target variable
3. **MI Discovery** — Pairwise mutual information identifies correlated feature pairs and triads for encoding
4. **Hamiltonian Encoding** — MI-selected pairs/triads define the entanglement topology (simulated classically):
   ```
   H(x) = Σ xᵢσᶻᵢ + Σ cᵢⱼ σᶻᵢσᶻⱼ + Σ cᵢⱼₖ σᶻᵢσᶻⱼσᶻₖ
   ```
   Time evolution `exp(-iHt)` encodes molecules into simulated quantum states. Pauli Z expectations extract projected features.
5. **CatBoost Classification** — Raw MapLight + quantum-inspired projected features train a gradient boosted classifier.

Inspired by [Simen et al. (arXiv:2510.13807)](https://arxiv.org/abs/2510.13807).

## Results

**CYP3A4_Substrate_CarbonMangels** (AUROC, 5 seeds):

| Method | Score | Δ vs Baseline |
|--------|-------|---------------|
| Classical baseline (MapLight + CatBoost) | 0.656 ± 0.006 | — |
| **MI-Entanglement (ours)** | **0.673 ± 0.004** | **+0.017** |
| Previous SOTA (CFA) | 0.667 ± 0.019 | +0.011 |
| MapLight (published) | 0.650 ± 0.006 | -0.006 |

Quantum-inspired feature augmentation improves over the classical baseline by +0.017 AUROC, achieving new state-of-the-art.

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. Install dependencies

```bash
make sync       # CPU only
make sync-gpu   # GPU support (requires NVIDIA driver)
```

### 3. Run tests

```bash
make test
```

### 4. Run benchmark

```bash
# List available benchmarks
make list-benchmarks

# Run a benchmark (CPU)
make run BENCHMARK=Caco2_Wang

# Run with GPU
make run BENCHMARK=Caco2_Wang LOCAL_DEVICE=lightning.gpu
```

## Repository Structure

```
mi_entanglement_tdc/
├── pyproject.toml
├── Makefile
├── README.md
├── configs/                          # Best hyperparams per benchmark
├── data/                             # TDC downloads (auto-populated)
├── cache/                            # MI computation cache
├── results/                          # Output JSON files
├── docs/
│   └── pipeline.md                   # Full pipeline architecture diagram
├── src/
│   ├── run_experiment.py             # CLI entry point
│   ├── config.py                     # Constants, metrics, cache I/O
│   ├── features.py                   # MapLight feature generation
│   ├── mi_discovery.py               # MI pre-filtering, pair/triad selection
│   ├── quantum_encoding.py           # Hamiltonian encoding orchestration
│   ├── benchmark.py                  # Full benchmark pipeline (5 seeds)
│   ├── projected_q_encoder.py        # HamiltonianProjectedEncoder (PennyLane)
│   ├── encoder_worker.py             # Multiprocessing worker
│   ├── process_bio88.py              # Feature computation
│   └── mi_entanglement_utils.py      # MI utilities
└── tests/
```

## CLI

```
python src/run_experiment.py [options]

Options:
  --device DEVICE           PennyLane device (default: lightning.gpu)
  --cache-dir DIR           MI cache directory (default: cache/)
  --output-json FILE        Output JSON file (default: results.json)
  --data-path DIR           TDC data directory (default: data/)
  --n-workers N             Parallel workers for encoding (default: 1)
  --benchmark NAME          TDC ADMET benchmark name
  --generate-features       Generate MapLight features only (no GPU needed)
  --configs-dir DIR         Saved configs directory (default: configs/)
  --max-qubits N            Max qubits per circuit (default: 28)
```

## Supported Benchmarks

Classification (AUROC): `Caco2_Wang`, `HIA_Hou`, `Pgp_Broccatelli`, `Bioavailability_Ma`, `BBB_Martins`, `CYP2C9_Veith`, `CYP2D6_Veith`, `CYP3A4_Veith`, `CYP2C9_Substrate_CarbonMangels`, `CYP2D6_Substrate_CarbonMangels`, `CYP3A4_Substrate_CarbonMangels`, `hERG`, `AMES`, `DILI`

Regression (MAE): `Lipophilicity_AstraZeneca`, `Solubility_AqSolDB`, `PPBR_AZ`, `LD50_Zhu`

## Quantum-Inspired Pipeline

The pipeline uses mutual information to guide simulated quantum entanglement topology, creating quantum-inspired projected features that capture higher-order correlations between molecular descriptors.

```mermaid
flowchart LR
    A[SMILES] --> B[MapLight<br/>2563 features]
    B --> C[MI Prefilter<br/>Top 100]
    C --> D[Pairwise MI<br/>Matrix]
    D --> E[Select Pairs<br/>& Triads]
    E --> F[Quantum<br/>Encoding]
    F --> G[Feature<br/>Assembly]
    G --> H[CatBoost]
    H --> I[Predictions]
```

### Step 1: MapLight Feature Generation

Converts SMILES strings into 2563-dimensional molecular descriptors:

| Component | Dimensions | Description |
|-----------|------------|-------------|
| ECFP Counts | 1024 | Extended connectivity fingerprints (radius 2) |
| Avalon Counts | 1024 | Avalon fingerprint bit counts |
| ErG Descriptors | 315 | Extended reduced graph descriptors |
| RDKit Properties | 200 | Physicochemical properties (LogP, TPSA, etc.) |

Features are cached as `.npy` files keyed by SMILES hash for fast reuse.

### Step 2: MI Discovery

Identifies which feature pairs and triads exhibit high mutual information, indicating statistical dependencies worth capturing via simulated entanglement.

1. **Univariate Pre-filter**: Rank all 2563 features by MI with target variable `I(Xᵢ; Y)`, keep top K (default: 100)
2. **Pairwise MI Matrix**: Compute `I(Xᵢ; Xⱼ)` for all pairs in the filtered set
3. **Pair Selection**: Extract top pairs exceeding MI threshold (default: 0.1, max 30 pairs)
4. **Triad Selection**: Extract top triads exceeding MI threshold (default: 0.15, max 15 triads)
5. **Qubit Constraint**: Trim pairs/triads to fit within max qubits (default: 28)

MI results are cached to avoid recomputation across runs.

### Step 3: Quantum-Inspired Encoding

Simulates quantum circuits where entanglement topology mirrors the MI-discovered correlations:

```
Hamiltonian: H(x) = Σᵢ xᵢσᶻᵢ + Σᵢⱼ cᵢⱼσᶻᵢσᶻⱼ + Σᵢⱼₖ cᵢⱼₖσᶻᵢσᶻⱼσᶻₖ
```

- **Feature encoding**: Input features scaled to [0,1] become rotation angles
- **MI-weighted coupling**: Coupling strengths `cᵢⱼ` proportional to mutual information
- **Time evolution**: Apply `exp(-iHt)` via Trotterization
- **Measurement**: Extract Pauli-Z expectations `⟨σᶻᵢ⟩` as projected features

Two separate circuits handle pairs (2Q) and triads (3Q) for efficiency.

### Step 4: Feature Assembly

Concatenate all features for the final model:

```
Final = [MapLight raw (2563)] + [2Q quantum] + [3Q quantum]
```

The quantum-inspired features capture nonlinear correlations that complement the raw descriptors.

### Step 5: CatBoost Training & Evaluation

- Train CatBoost classifier/regressor on assembled features
- Run 5 seeds with scaffold splits per TDC protocol
- Evaluate via `group.evaluate_many()` for official benchmark metrics

### Pipeline Diagram

See [docs/pipeline.md](docs/pipeline.md) for the full architecture diagram with all components and data flow.

## Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Feature set | MapLight (2563) |
| MI pre-filter K | 100 |
| Coupling scale | direct |
| Evolution time | 0.5 |
| Trotter steps | 1 |
| MI pair threshold | 0.1 |
| MI triad threshold | 0.15 |
| Max pairs | 30 |
| Max triads | 15 |
| Max qubits | 28 |
| CatBoost iterations | 1000 |
| CatBoost depth | 6 |
| CatBoost learning rate | 0.05 |

## Hardware Requirements

The quantum circuit simulation benefits significantly from GPU acceleration via PennyLane Lightning GPU.

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | CUDA-capable with 6GB+ VRAM | NVIDIA RTX 3060 (12GB), NVIDIA GB10 |
| RAM | 16GB | 32GB |
| CPU | 4 cores | 8+ cores (for parallel feature generation) |

**Notes**:
- CPU-only execution works but is significantly slower for circuit simulation
- The `--max-qubits` flag can reduce VRAM usage (default: 28)
- Feature generation (`--generate-features`) does not require a GPU
