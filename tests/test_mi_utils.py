"""Tests for mi_entanglement_utils.py MI computation functions."""
from __future__ import annotations

import numpy as np
import pytest

from mi_entanglement_utils import (
    prefilter_by_univariate_mi,
    compute_pairwise_mi,
    select_filtered_pairs,
    select_filtered_triplets,
    get_dynamic_circuit_params,
    get_deduped_feature_indices,
    define_feature_tiles,
    compute_tile_encoding_params,
)


# ---------------------------------------------------------------------------
# prefilter_by_univariate_mi
# ---------------------------------------------------------------------------

class TestPrefilterByUnivMI:
    def test_correct_count(self, sample_feature_matrix, sample_labels):
        top_k = 5
        indices, scores = prefilter_by_univariate_mi(
            sample_feature_matrix, sample_labels, top_k=top_k, verbose=False
        )
        assert len(indices) == top_k

    def test_indices_in_range(self, sample_feature_matrix, sample_labels):
        indices, scores = prefilter_by_univariate_mi(
            sample_feature_matrix, sample_labels, top_k=5, verbose=False
        )
        for idx in indices:
            assert 0 <= idx < sample_feature_matrix.shape[1]

    def test_scores_non_negative(self, sample_feature_matrix, sample_labels):
        _, scores = prefilter_by_univariate_mi(
            sample_feature_matrix, sample_labels, top_k=5, verbose=False
        )
        assert np.all(scores >= 0)

    def test_regression_continuous_labels(self, sample_feature_matrix, sample_continuous_labels):
        indices, scores = prefilter_by_univariate_mi(
            sample_feature_matrix, sample_continuous_labels, top_k=5,
            verbose=False, regression=True
        )
        assert len(indices) == 5
        assert np.all(scores >= 0)

    def test_regression_false_with_continuous_raises(self, sample_feature_matrix, sample_continuous_labels):
        with pytest.raises(ValueError, match="Unknown label type"):
            prefilter_by_univariate_mi(
                sample_feature_matrix, sample_continuous_labels, top_k=5,
                verbose=False, regression=False
            )


# ---------------------------------------------------------------------------
# compute_pairwise_mi
# ---------------------------------------------------------------------------

class TestPairwiseMI:
    def test_symmetric(self, sample_feature_matrix):
        mi = compute_pairwise_mi(sample_feature_matrix[:, :5], show_progress=False)
        np.testing.assert_array_almost_equal(mi, mi.T)

    def test_zero_diagonal(self, sample_feature_matrix):
        mi = compute_pairwise_mi(sample_feature_matrix[:, :5], show_progress=False)
        np.testing.assert_array_almost_equal(np.diag(mi), 0)

    def test_non_negative(self, sample_feature_matrix):
        mi = compute_pairwise_mi(sample_feature_matrix[:, :5], show_progress=False)
        assert np.all(mi >= -1e-10)  # small tolerance for numerical noise

    def test_correct_shape(self, sample_feature_matrix):
        n_feat = 5
        mi = compute_pairwise_mi(sample_feature_matrix[:, :n_feat], show_progress=False)
        assert mi.shape == (n_feat, n_feat)


# ---------------------------------------------------------------------------
# select_filtered_pairs
# ---------------------------------------------------------------------------

class TestSelectFilteredPairs:
    def test_respects_max_pairs(self, small_mi_matrix):
        pairs = select_filtered_pairs(small_mi_matrix, max_pairs=2, min_threshold=0.0)
        assert len(pairs) <= 2

    def test_respects_threshold(self, small_mi_matrix):
        pairs = select_filtered_pairs(small_mi_matrix, max_pairs=100, min_threshold=0.5)
        for pair, score in pairs:
            assert score >= 0.5

    def test_sorted_descending(self, small_mi_matrix):
        pairs = select_filtered_pairs(small_mi_matrix, max_pairs=100, min_threshold=0.0)
        scores = [s for _, s in pairs]
        assert scores == sorted(scores, reverse=True)

    def test_correct_format(self, small_mi_matrix):
        pairs = select_filtered_pairs(small_mi_matrix, max_pairs=100, min_threshold=0.0)
        for pair, score in pairs:
            assert len(pair) == 2
            assert isinstance(score, (int, float, np.floating))


# ---------------------------------------------------------------------------
# select_filtered_triplets
# ---------------------------------------------------------------------------

class TestSelectFilteredTriplets:
    def test_respects_max_triplets(self, small_mi_matrix):
        triplets = select_filtered_triplets(small_mi_matrix, max_triplets=2, min_threshold=0.0)
        assert len(triplets) <= 2

    def test_respects_threshold(self, small_mi_matrix):
        triplets = select_filtered_triplets(small_mi_matrix, max_triplets=100, min_threshold=0.5)
        for triplet, score in triplets:
            assert score >= 0.5

    def test_sorted_descending(self, small_mi_matrix):
        triplets = select_filtered_triplets(small_mi_matrix, max_triplets=100, min_threshold=0.0)
        scores = [s for _, s in triplets]
        assert scores == sorted(scores, reverse=True)

    def test_correct_format(self, small_mi_matrix):
        triplets = select_filtered_triplets(small_mi_matrix, max_triplets=100, min_threshold=0.0)
        for triplet, score in triplets:
            assert len(triplet) == 3


# ---------------------------------------------------------------------------
# get_dynamic_circuit_params
# ---------------------------------------------------------------------------

class TestDynamicCircuitParams:
    def test_known_remapping(self):
        pairs = [(2, 5), (5, 8)]
        active, idx_map, remapped_pairs, remapped_triads = get_dynamic_circuit_params(pairs, [])
        assert set(active) == {2, 5, 8}
        assert len(idx_map) == 3
        # Remapped pairs should use 0-indexed consecutive qubits
        for p in remapped_pairs:
            for q in p:
                assert q < len(active)

    def test_empty_inputs(self):
        active, idx_map, pairs, triads = get_dynamic_circuit_params([], [])
        assert active == []
        assert idx_map == {}
        assert pairs == []
        assert triads == []

    def test_triads(self):
        triads = [(0, 3, 7)]
        active, idx_map, _, remapped_triads = get_dynamic_circuit_params([], triads)
        assert set(active) == {0, 3, 7}
        assert len(remapped_triads) == 1
        assert len(remapped_triads[0]) == 3


# ---------------------------------------------------------------------------
# get_deduped_feature_indices
# ---------------------------------------------------------------------------

class TestDedupedFeatureIndices:
    def test_returns_86(self):
        indices = get_deduped_feature_indices()
        assert len(indices) == 86

    def test_excludes_duplicates(self):
        indices = get_deduped_feature_indices()
        assert 1 not in indices
        assert 78 not in indices

    def test_sorted(self):
        indices = get_deduped_feature_indices()
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# define_feature_tiles
# ---------------------------------------------------------------------------

class TestDefineFeatureTiles:
    def test_returns_5_tiles(self):
        tiles = define_feature_tiles()
        assert len(tiles) == 5

    def test_all_indices_covered(self):
        tiles = define_feature_tiles()
        all_indices = set()
        for tile in tiles:
            all_indices.update(tile)
        # Should cover all 86 deduped indices
        deduped = set(get_deduped_feature_indices())
        assert all_indices.issubset(deduped)


# ---------------------------------------------------------------------------
# compute_tile_encoding_params
# ---------------------------------------------------------------------------

class TestTileEncodingParams:
    def test_output_keys(self, small_mi_matrix):
        deduped = list(range(5))
        tile_indices = [0, 1, 2]
        params = compute_tile_encoding_params(
            small_mi_matrix, deduped, tile_indices,
            max_pairs=5, max_triads=2,
        )
        assert 'n_qubits' in params
        assert 'pairs' in params
        assert 'triads' in params
        assert 'feature_indices' in params

    def test_n_qubits_matches(self, small_mi_matrix):
        deduped = list(range(5))
        tile_indices = [0, 1, 2]
        params = compute_tile_encoding_params(
            small_mi_matrix, deduped, tile_indices,
            max_pairs=5, max_triads=2,
        )
        assert params['n_qubits'] == len(tile_indices)
