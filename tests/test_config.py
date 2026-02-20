"""Tests for config.py — constants, metric functions, and cache I/O."""
from __future__ import annotations

import json

import numpy as np
import pytest

from config import (
    DEFAULT_CONFIG,
    ADMET_METRICS,
    METRIC_CONFIG,
    get_metric_fn,
    get_metric_config,
    save_best_config,
    load_best_config,
    generate_mi_cache_key,
    NumpyEncoder,
    CACHE_VERSION,
)


# ---------------------------------------------------------------------------
# get_metric_fn
# ---------------------------------------------------------------------------

class TestGetMetricFn:
    def test_roc_auc_benchmark(self):
        fn, label = get_metric_fn('CYP3A4_Substrate_CarbonMangels')
        assert label == 'AUROC'
        assert callable(fn)

    def test_pr_auc_benchmark(self):
        fn, label = get_metric_fn('CYP2C9_Veith')
        assert label == 'AUPRC'
        assert callable(fn)

    def test_mae_benchmark(self):
        fn, label = get_metric_fn('Lipophilicity_AstraZeneca')
        assert label == 'MAE'
        assert callable(fn)
        assert fn([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
        assert fn([1.0, 2.0], [2.0, 3.0]) == 1.0

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported metric"):
            get_metric_fn('Half_Life_Obach')  # 'spearman' — not yet supported

    def test_case_insensitive(self):
        fn1, _ = get_metric_fn('CYP3A4_Substrate_CarbonMangels')
        fn2, _ = get_metric_fn('cyp3a4_substrate_carbonmangels')
        assert fn1 == fn2


class TestGetMetricConfig:
    def test_classification_config(self):
        cfg = get_metric_config('CYP3A4_Substrate_CarbonMangels')
        assert cfg['higher_is_better'] is True
        assert cfg['regression'] is False

    def test_mae_config(self):
        cfg = get_metric_config('Lipophilicity_AstraZeneca')
        assert cfg['higher_is_better'] is False
        assert cfg['regression'] is True
        assert cfg['label'] == 'MAE'

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported metric"):
            get_metric_config('Half_Life_Obach')


# ---------------------------------------------------------------------------
# save_best_config / load_best_config
# ---------------------------------------------------------------------------

class TestConfigPersistence:
    def test_round_trip(self, tmp_configs_dir):
        config = {'learning_rate': 0.05, 'depth': 6}
        save_best_config('TestBenchmark', config, 0.85, tmp_configs_dir)
        loaded = load_best_config('TestBenchmark', tmp_configs_dir)
        assert loaded is not None
        assert loaded['learning_rate'] == 0.05
        assert loaded['depth'] == 6

    def test_creates_dirs(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        save_best_config('Test', {'x': 1}, 0.5, nested)
        loaded = load_best_config('Test', nested)
        assert loaded is not None

    def test_none_for_missing(self, tmp_configs_dir):
        loaded = load_best_config('NonExistent', tmp_configs_dir)
        assert loaded is None


# ---------------------------------------------------------------------------
# generate_mi_cache_key
# ---------------------------------------------------------------------------

class TestMICacheKey:
    def test_deterministic(self):
        feats = np.ones((10, 5))
        labels = np.zeros(10)
        config = {'pair_threshold': 0.1, 'triad_threshold': 0.15,
                  'max_pairs': 30, 'max_triads': 15}
        key1 = generate_mi_cache_key(feats, labels, config)
        key2 = generate_mi_cache_key(feats, labels, config)
        assert key1 == key2

    def test_changes_with_data(self):
        config = {'pair_threshold': 0.1, 'triad_threshold': 0.15,
                  'max_pairs': 30, 'max_triads': 15}
        feats1 = np.ones((10, 5))
        feats2 = np.zeros((10, 5))
        labels = np.zeros(10)
        key1 = generate_mi_cache_key(feats1, labels, config)
        key2 = generate_mi_cache_key(feats2, labels, config)
        assert key1 != key2

    def test_changes_with_params(self):
        feats = np.ones((10, 5))
        labels = np.zeros(10)
        config1 = {'pair_threshold': 0.1, 'triad_threshold': 0.15,
                   'max_pairs': 30, 'max_triads': 15}
        config2 = {'pair_threshold': 0.2, 'triad_threshold': 0.15,
                   'max_pairs': 30, 'max_triads': 15}
        key1 = generate_mi_cache_key(feats, labels, config1)
        key2 = generate_mi_cache_key(feats, labels, config2)
        assert key1 != key2

    def test_includes_cache_version(self):
        feats = np.ones((10, 5))
        labels = np.zeros(10)
        config = {'pair_threshold': 0.1, 'triad_threshold': 0.15,
                  'max_pairs': 30, 'max_triads': 15}
        key = generate_mi_cache_key(feats, labels, config)
        assert CACHE_VERSION in key


# ---------------------------------------------------------------------------
# NumpyEncoder
# ---------------------------------------------------------------------------

class TestNumpyEncoder:
    def test_int64(self):
        data = {'x': np.int64(42)}
        result = json.dumps(data, cls=NumpyEncoder)
        assert '"x": 42' in result

    def test_float32(self):
        data = {'x': np.float32(3.14)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert abs(parsed['x'] - 3.14) < 0.01

    def test_ndarray(self):
        data = {'x': np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed['x'] == [1, 2, 3]


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG sanity
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_has_required_keys(self):
        required = [
            'mi_prefilter_k', 'pair_threshold', 'triad_threshold',
            'max_pairs', 'max_triads', 'max_qubits',
            'coupling_scale', 'evolution_time', 'n_trotter_steps',
            'include_correlations', 'catboost_iterations',
            'catboost_depth', 'catboost_learning_rate',
        ]
        for key in required:
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"
