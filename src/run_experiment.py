#!/usr/bin/env python3
"""
MI-Entanglement Quantum Feature Augmentation — TDC ADMET Benchmark

Pipeline:
  1. Generate MapLight features from SMILES (2563-dim)
  2. MI Discovery — prefilter + single-pass pairwise MI for pairs/triads
  3. Quantum Encoding — Hamiltonian time evolution with MI-weighted coupling
  4. Train CatBoost on raw + quantum features, predict on test
  5. Evaluate via TDC's group.evaluate_many()

Usage:
    python run_experiment.py
    python run_experiment.py --device lightning.gpu --cache-dir cache/
    python run_experiment.py --generate-features --benchmark Caco2_Wang
"""
from __future__ import annotations

import argparse
import logging
import sys

# CUDA boot sequence — order matters!
#
# Problem: PyTDC imports PyTorch (CPU-only) which loads stub CUDA *runtime*
# symbols (cudaSetDevice, cudaGetDeviceCount, etc.). When PennyLane Lightning
# GPU's C++ extension later calls these via DevTag::refresh(), it hits the
# stubs → "device busy or unavailable".
#
# Fix: Pre-load the REAL CUDA runtime (libcudart.so.12 from the nvidia/cuda
# base image) with RTLD_GLOBAL *before* PyTDC can import PyTorch. This pins
# the real symbols in the global symbol table so stubs can't replace them.
# Then initialize both driver and runtime APIs.
import ctypes as _ctypes

# Step 1: Load real CUDA runtime with RTLD_GLOBAL (before PyTorch stubs)
_cudart = None
for _lib_name in ("libcudart.so.12", "libcudart.so.11"):
    try:
        _cudart = _ctypes.CDLL(_lib_name, mode=_ctypes.RTLD_GLOBAL)
        break
    except OSError:
        continue

# Step 2: Initialize CUDA driver API (skip if not available, e.g., CPU-only)
_libcuda = None
try:
    _libcuda = _ctypes.cdll.LoadLibrary("libcuda.so.1")
    _libcuda.cuInit(0)
except OSError:
    pass  # CPU-only environment

# Step 3: Initialize CUDA runtime API (locks in real symbols)
if _cudart is not None and _libcuda is not None:
    _cudart.cudaFree(_ctypes.c_void_p(0))

# Step 4: NOW safe to import PyTDC — PyTorch stubs can't replace real runtime
from tdc.benchmark_group import admet_group

# Step 5: Now safe to import modules that transitively load PennyLane.
from config import BENCHMARK_NAME, ADMET_METRICS

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure root logger for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def _benchmark_name_type(value: str) -> str:
    """Argparse type validator for --benchmark flag."""
    if value.lower() not in ADMET_METRICS:
        raise argparse.ArgumentTypeError(
            f"Unknown benchmark: '{value}'. "
            f"Valid benchmarks: {sorted(ADMET_METRICS.keys())}"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MI-Entanglement Quantum Feature Augmentation — TDC ADMET Benchmark"
    )
    parser.add_argument(
        '--device', default='default.qubit',
        help='PennyLane device (default: default.qubit, use lightning.gpu for GPU)'
    )
    parser.add_argument(
        '--cache-dir', default='cache/',
        help='Directory for MI cache files (default: cache/)'
    )
    parser.add_argument(
        '--output-json', default='results.json',
        help='Output JSON file for results (default: results.json)'
    )
    parser.add_argument(
        '--data-path', default='data/',
        help='Path for TDC data download (default: data/)'
    )
    parser.add_argument(
        '--n-workers', type=int, default=1,
        help='Number of parallel workers for quantum encoding (default: 1)'
    )
    parser.add_argument(
        '--benchmark', default=BENCHMARK_NAME, type=_benchmark_name_type,
        help=f'TDC ADMET benchmark name (default: {BENCHMARK_NAME})'
    )
    parser.add_argument(
        '--generate-features', action='store_true',
        help='Generate and cache MapLight features only (no GPU needed)'
    )
    parser.add_argument(
        '--encoding-results', default=None,
        help='Path to encoding config JSON (optional, overrides configs/<benchmark>.json)'
    )
    parser.add_argument(
        '--model-results', default=None,
        help='Path to model config JSON (optional, overrides configs/<benchmark>.json)'
    )
    parser.add_argument(
        '--max-qubits', type=int, default=None,
        help='Max qubits per circuit (overrides DEFAULT_CONFIG, default: 28)'
    )
    parser.add_argument(
        '--configs-dir', default='configs/',
        help='Directory for saved best configs (default: configs/)'
    )
    parser.add_argument(
        '--baseline', action='store_true',
        help='Run classical baseline (no quantum features, sets max_pairs=0 and max_triads=0)'
    )
    args = parser.parse_args()

    setup_logging()

    group = admet_group(path=args.data_path)

    if args.generate_features:
        from features import generate_maplight_for_dataframe
        benchmark = group.get(args.benchmark)
        train_val, test = benchmark['train_val'], benchmark['test']
        logger.info("Generating MapLight features for %s (%d train, %d test, %d workers)",
                    args.benchmark, len(train_val), len(test), args.n_workers)
        generate_maplight_for_dataframe(train_val, n_workers=args.n_workers, cache_dir=args.cache_dir)
        generate_maplight_for_dataframe(test, n_workers=args.n_workers, cache_dir=args.cache_dir)
    else:
        from benchmark import run_benchmark
        run_benchmark(
            device_type=args.device,
            cache_dir=args.cache_dir,
            output_json=args.output_json,
            n_workers=args.n_workers,
            benchmark_name=args.benchmark,
            encoding_results=args.encoding_results,
            model_results=args.model_results,
            configs_dir=args.configs_dir,
            group=group,
            max_qubits=args.max_qubits,
            baseline=args.baseline,
        )

    logger.info("Done.")


if __name__ == '__main__':
    main()
