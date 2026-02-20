"""
Configuration, constants, and cache I/O for the MI-Entanglement pipeline.
"""
from __future__ import annotations

import json
import hashlib
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

logger = logging.getLogger(__name__)

# Cache version — bump when feature generation or MI logic changes to
# invalidate stale caches. See CODE_REVIEW.md issue #15.
CACHE_VERSION = "v2"

# ============================================================================
# Default Parameters
# ============================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    # MI Pre-filter (select top K MapLight features by univariate MI with target)
    'mi_prefilter_k': 100,

    # MI Discovery (single-pass pairwise MI on prefiltered features)
    'pair_threshold': 0.1,
    'triad_threshold': 0.15,
    'max_pairs': 30,
    'max_triads': 15,
    'max_qubits': 28,

    # Quantum Encoding (grid search Phase 1 winner)
    'coupling_scale': 'direct',
    'evolution_time': 0.5,
    'n_trotter_steps': 1,
    'include_correlations': False,
    'n_cd_layers': 0,
    'cd_strength': 0.0,

    # CatBoost (defaults for CB)
    'catboost_iterations': 1000,
    'catboost_depth': 6,
    'catboost_learning_rate': 0.05,
    'catboost_random_strength': 1,
    'catboost_l2_leaf_reg': 3,
}

BENCHMARK_NAME = 'CYP3A4_Substrate_CarbonMangels'
BENCHMARK_SEEDS = [1, 2, 3, 4, 5]
GRID_SEEDS = [1, 2, 3]  # Fewer seeds for grid search speed; full 5 seeds for final run
N_GRID_FOLDS = 5  # Stratified k-fold CV for grid search

# TDC ADMET metric mapping (from tdc/metadata.py)
ADMET_METRICS: Dict[str, str] = {
    'caco2_wang': 'mae',
    'hia_hou': 'roc-auc',
    'pgp_broccatelli': 'roc-auc',
    'bioavailability_ma': 'roc-auc',
    'lipophilicity_astrazeneca': 'mae',
    'solubility_aqsoldb': 'mae',
    'bbb_martins': 'roc-auc',
    'ppbr_az': 'mae',
    'vdss_lombardo': 'spearman',
    'cyp2c9_veith': 'pr-auc',
    'cyp2d6_veith': 'pr-auc',
    'cyp3a4_veith': 'pr-auc',
    'cyp2c9_substrate_carbonmangels': 'pr-auc',
    'cyp2d6_substrate_carbonmangels': 'pr-auc',
    'cyp3a4_substrate_carbonmangels': 'roc-auc',
    'half_life_obach': 'spearman',
    'clearance_hepatocyte_az': 'spearman',
    'clearance_microsome_az': 'spearman',
    'ld50_zhu': 'mae',
    'herg': 'roc-auc',
    'ames': 'roc-auc',
    'dili': 'roc-auc',
}


METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
    'roc-auc': {'fn': roc_auc_score, 'label': 'AUROC', 'higher_is_better': True, 'regression': False},
    'pr-auc':  {'fn': average_precision_score, 'label': 'AUPRC', 'higher_is_better': True, 'regression': False},
    'mae':     {'fn': mean_absolute_error, 'label': 'MAE', 'higher_is_better': False, 'regression': True},
}


def get_metric_fn(benchmark_name: str) -> Tuple[Callable, str]:
    """Return (score_fn, metric_label) for the given benchmark."""
    metric = ADMET_METRICS.get(benchmark_name.lower(), 'roc-auc')
    cfg = METRIC_CONFIG.get(metric)
    if cfg is None:
        raise ValueError(f"Unsupported metric '{metric}' for benchmark '{benchmark_name}'")
    return cfg['fn'], cfg['label']


def get_metric_config(benchmark_name: str) -> Dict[str, Any]:
    """Return full metric config dict for the given benchmark.

    Keys: fn, label, higher_is_better, regression.
    """
    metric = ADMET_METRICS.get(benchmark_name.lower(), 'roc-auc')
    cfg = METRIC_CONFIG.get(metric)
    if cfg is None:
        raise ValueError(f"Unsupported metric '{metric}' for benchmark '{benchmark_name}'")
    return cfg


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# Config Persistence
# ============================================================================

def save_best_config(benchmark_name: str, config: Dict[str, Any],
                     score: float, configs_dir: str) -> None:
    """Save best hyperparameters to configs/<benchmark_lower>.json."""
    Path(configs_dir).mkdir(parents=True, exist_ok=True)
    fname = benchmark_name.lower() + '.json'
    path = Path(configs_dir) / fname
    output = {
        'benchmark': benchmark_name,
        'timestamp': datetime.now().isoformat(),
        'grid_search_score': round(score, 4),
        'config': {k: v for k, v in config.items()},
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    logger.info("Best config saved to: %s", path)


def load_best_config(benchmark_name: str, configs_dir: str) -> Optional[Dict[str, Any]]:
    """Load saved hyperparameters from configs/<benchmark_lower>.json, or None."""
    fname = benchmark_name.lower() + '.json'
    path = Path(configs_dir) / fname
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('config')


# ============================================================================
# MI Caching
# ============================================================================

def generate_mi_cache_key(features: np.ndarray, labels: np.ndarray,
                          config: Dict[str, Any]) -> str:
    """Generate deterministic cache key for MI Phase 1 results."""
    data_fingerprint = hashlib.sha256()
    data_fingerprint.update(CACHE_VERSION.encode())
    data_fingerprint.update(features.tobytes())
    data_fingerprint.update(labels.tobytes())
    data_hash = data_fingerprint.hexdigest()[:16]

    param_str = (f"pt{config['pair_threshold']}"
                 f"_tt{config['triad_threshold']}_mp{config['max_pairs']}"
                 f"_mt{config['max_triads']}_mq{config.get('max_qubits', 28)}")

    return f"mi_phase1_{CACHE_VERSION}_{data_hash}_{param_str}"


def save_mi_cache(cache_path: str, **kwargs: Any) -> None:
    """Save MI Phase 1 results to cache file."""
    kwargs['timestamp'] = datetime.now().isoformat()

    cache_dir = Path(cache_path).parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_path, 'w') as f:
        json.dump(kwargs, f, indent=2, cls=NumpyEncoder)

    logger.info("Saved MI cache to: %s", cache_path)


def load_mi_cache(cache_path: str) -> Dict[str, Any]:
    """Load MI Phase 1 results from cache file."""
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)

    # Convert string keys back to ints for index maps
    cache_data['index_map_2q'] = {int(k): v for k, v in cache_data['index_map_2q'].items()}
    cache_data['index_map_3q'] = {int(k): v for k, v in cache_data['index_map_3q'].items()}

    # Convert MI scores back to tuples
    if cache_data.get('mi_pairs_scored'):
        cache_data['mi_pairs_scored'] = [
            (tuple(pair), score) for pair, score in cache_data['mi_pairs_scored']
        ]
    if cache_data.get('mi_triads_scored'):
        cache_data['mi_triads_scored'] = [
            (tuple(triad), score) for triad, score in cache_data['mi_triads_scored']
        ]

    logger.info("Loaded MI cache from: %s", cache_path)
    logger.info("Cached at: %s", cache_data.get('timestamp', 'unknown'))

    return cache_data
