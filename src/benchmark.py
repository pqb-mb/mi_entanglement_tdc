"""
TDC ADMET benchmark runner for MI-entanglement pipeline.

Runs the full pipeline: MapLight features -> MI discovery -> quantum encoding -> CatBoost -> evaluation.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional

from config import (
    DEFAULT_CONFIG, BENCHMARK_SEEDS, get_metric_fn, get_metric_config,
    load_best_config, NumpyEncoder,
)
from features import generate_maplight_for_dataframe
from mi_discovery import run_maplight_mi_discovery
from quantum_encoding import quantum_encode

logger = logging.getLogger(__name__)


def run_benchmark(device_type: str = 'default.qubit', cache_dir: str = 'cache/',
                  output_json: Optional[str] = None, n_workers: int = 1,
                  benchmark_name: Optional[str] = None,
                  encoding_results: Optional[str] = None,
                  model_results: Optional[str] = None,
                  configs_dir: str = '../configs/',
                  group: Any = None,
                  max_qubits: Optional[int] = None,
                  baseline: bool = False) -> Dict[str, Any]:
    """Run TDC ADMET benchmark.

    Uses MapLight features (2563-dim) as primary feature set.
    MI discovery prefilters top-K MapLight features by univariate MI,
    then selects top pairs/triads via single-pass pairwise MI.
    CatBoost gets: all MapLight raw + quantum features.

    Config priority (highest wins):
        1. CLI --encoding-results / --model-results
        2. configs/<benchmark>.json (saved by grid search)
        3. DEFAULT_CONFIG defaults

    Follows TDC benchmark protocol:
        - 5 seeds with scaffold splits
        - group.evaluate_many() for final metrics
    """
    mcfg = get_metric_config(benchmark_name)
    score_fn, metric_label = mcfg['fn'], mcfg['label']
    regression = mcfg['regression']

    # Build effective config: start from DEFAULT_CONFIG, override with saved/CLI winners
    config = dict(DEFAULT_CONFIG)

    # Layer 1: Load saved config from configs/<benchmark>.json if it exists
    saved = load_best_config(benchmark_name, configs_dir)
    if saved:
        config.update(saved)
        logger.info("Loaded saved config from configs/%s.json", benchmark_name.lower())

    # CLI --max-qubits always wins over saved config
    if max_qubits is not None:
        config['max_qubits'] = max_qubits

    # Baseline mode: disable quantum features
    if baseline:
        config['max_pairs'] = 0
        config['max_triads'] = 0
        logger.info("Baseline mode: disabled quantum features (max_pairs=0, max_triads=0)")

    # Layer 2: CLI overrides (backwards compat)
    if encoding_results:
        with open(encoding_results) as f:
            phase1 = json.load(f)
        enc = phase1['best_encoding']
        config['coupling_scale'] = enc['coupling_scale']
        config['evolution_time'] = enc['evolution_time']
        config['n_cd_layers'] = enc.get('n_cd_layers', 0)
        config['cd_strength'] = enc.get('cd_strength', 0.0)
        config['include_correlations'] = enc['include_correlations']
        logger.info("Loaded encoding winners from %s", encoding_results)

    if model_results:
        with open(model_results) as f:
            phase2 = json.load(f)
        cb = phase2['best_catboost']
        config['catboost_iterations'] = cb['iterations']
        config['catboost_depth'] = cb['depth']
        config['catboost_learning_rate'] = cb['learning_rate']
        config['catboost_random_strength'] = cb['random_strength']
        if 'l2_leaf_reg' in cb:
            config['catboost_l2_leaf_reg'] = cb['l2_leaf_reg']
        # Also load encoding config from phase 2 if no separate encoding_results
        if not encoding_results and 'encoding_config' in phase2:
            enc = phase2['encoding_config']
            config['coupling_scale'] = enc['coupling_scale']
            config['evolution_time'] = enc['evolution_time']
            config['n_cd_layers'] = enc.get('n_cd_layers', 0)
            config['cd_strength'] = enc.get('cd_strength', 0.0)
            config['include_correlations'] = enc['include_correlations']
        logger.info("Loaded CatBoost winners from %s", model_results)

    logger.info("=" * 70)
    logger.info("MapLight MI-Entanglement Quantum Feature Augmentation")
    logger.info("Benchmark: %s  (metric: %s)", benchmark_name, metric_label)
    logger.info("Device: %s", device_type)
    logger.info("Config: %s", json.dumps(config, indent=2))
    logger.info("=" * 70)

    # Get data (train_val/test are identical across seeds)
    benchmark = group.get(benchmark_name)
    name = benchmark['name']
    train_val, test = benchmark['train_val'], benchmark['test']
    label_dtype = float if regression else int
    labels_train_val = train_val['Y'].values.astype(label_dtype)
    labels_test = test['Y'].values.astype(label_dtype)

    logger.info("Train_val: %d, Test: %d", len(train_val), len(test))

    # =========================================================================
    # Step 1: MapLight Feature Generation (shared across seeds)
    # =========================================================================
    logger.info("--- Step 1: MapLight Feature Generation (%d workers) ---", n_workers)

    ml_train_val = generate_maplight_for_dataframe(train_val, n_workers=n_workers, cache_dir=cache_dir)
    ml_test = generate_maplight_for_dataframe(test, n_workers=n_workers, cache_dir=cache_dir)

    logger.info("train_val: %s, test: %s", ml_train_val.shape, ml_test.shape)

    # =========================================================================
    # Step 2: MI Discovery on MapLight features (shared across seeds)
    # =========================================================================
    logger.info("--- Step 2: MI Discovery (prefilter + single-pass MI) ---")

    mi_result = run_maplight_mi_discovery(
        ml_train_val, labels_train_val, config, cache_dir=cache_dir,
        regression=regression
    )

    n_qubits_2q = mi_result['n_qubits_2q']
    n_qubits_3q = mi_result['n_qubits_3q']
    top_indices = mi_result['top_feature_indices']
    logger.info("Prefiltered: %d features from %d", len(top_indices), ml_train_val.shape[1])
    logger.info("2Q circuit: %d qubits, %d pairs", n_qubits_2q, len(mi_result['remapped_pairs']))
    logger.info("3Q circuit: %d qubits, %d triads", n_qubits_3q, len(mi_result['remapped_triads']))

    # =========================================================================
    # Step 3: Quantum Encoding on prefiltered MapLight features (shared across seeds)
    # =========================================================================
    encoding_workers = 1 if 'gpu' in device_type.lower() else n_workers
    logger.info("--- Step 3: Quantum Encoding (%d workers) ---", encoding_workers)

    # Quantum encode uses the prefiltered feature subset
    ml_train_val_subset = ml_train_val[:, top_indices]
    ml_test_subset = ml_test[:, top_indices]

    X_train_2q, X_test_2q, X_train_3q, X_test_3q = quantum_encode(
        ml_train_val_subset, ml_test_subset, mi_result,
        config, device_type, n_workers=encoding_workers
    )

    X_train_val_quantum = np.hstack([X_train_2q, X_train_3q])
    X_test_quantum = np.hstack([X_test_2q, X_test_3q])
    logger.info("Quantum features: train_val=%s, test=%s",
                X_train_val_quantum.shape, X_test_quantum.shape)

    # =========================================================================
    # Step 4: Feature Assembly (all MapLight raw + quantum)
    # =========================================================================
    logger.info("--- Step 4: Feature Assembly ---")

    X_train_val_final = np.hstack([ml_train_val, X_train_val_quantum])
    X_test_final = np.hstack([ml_test, X_test_quantum])

    logger.info("MapLight raw: %d", ml_train_val.shape[1])
    logger.info("Quantum features: %d", X_train_val_quantum.shape[1])
    logger.info("Final features: train_val=%s, test=%s",
                X_train_val_final.shape, X_test_final.shape)

    # =========================================================================
    # Step 5: CatBoost Training (per seed, on all train_val)
    # =========================================================================
    predictions_list = []
    seed_details = []

    for seed_idx, seed in enumerate(BENCHMARK_SEEDS):
        seed_start = time.time()
        logger.info("--- Step 5: CatBoost Training -- Seed %d (%d/%d) ---",
                     seed, seed_idx + 1, len(BENCHMARK_SEEDS))

        logger.info("Training on all train_val: %d samples", len(X_train_val_final))

        if regression:
            from catboost import CatBoostRegressor as CatBoostModel
        else:
            from catboost import CatBoostClassifier as CatBoostModel
        clf = CatBoostModel(
            iterations=config['catboost_iterations'],
            depth=config['catboost_depth'],
            learning_rate=config['catboost_learning_rate'],
            random_strength=config.get('catboost_random_strength', 1),
            l2_leaf_reg=config.get('catboost_l2_leaf_reg', 3),
            thread_count=10,
            random_seed=seed,
            verbose=False
        )

        clf.fit(X_train_val_final, labels_train_val)
        if regression:
            y_pred = clf.predict(X_test_final)
        else:
            y_pred = clf.predict_proba(X_test_final)[:, 1]

        predictions = {name: y_pred}
        predictions_list.append(predictions)

        seed_elapsed = time.time() - seed_start
        seed_score = score_fn(labels_test, y_pred)
        logger.info("Seed %d: %s=%.4f (%.1fs)", seed, metric_label, seed_score, seed_elapsed)

        seed_details.append({
            'seed': seed,
            'score': round(seed_score, 4),
            'metric': metric_label,
            'n_qubits_2q': n_qubits_2q,
            'n_qubits_3q': n_qubits_3q,
            'n_pairs': len(mi_result['remapped_pairs']),
            'n_triads': len(mi_result['remapped_triads']),
            'n_prefilter_k': len(top_indices),
            'n_maplight_features': ml_train_val.shape[1],
            'n_quantum_features': X_train_val_quantum.shape[1],
            'n_total_features': X_train_val_final.shape[1],
        })

    # =========================================================================
    # Final Evaluation
    # =========================================================================
    logger.info("=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)

    results = group.evaluate_many(predictions_list)
    logger.info("TDC Benchmark Results: %s", results)

    # Save results
    output = {
        'benchmark': benchmark_name,
        'approach': 'maplight_mi_quantum',
        'results': results,
        'config': config,
        'device_type': device_type,
        'timestamp': datetime.now().isoformat(),
        'seeds': BENCHMARK_SEEDS,
        'seed_details': seed_details,
    }

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        logger.info("Results saved to: %s", output_json)

    return results
