"""Tests for projected_q_encoder.py (using default.qubit, no GPU required)."""
from __future__ import annotations

import numpy as np
import pytest

from projected_q_encoder import HamiltonianProjectedEncoder


# ---------------------------------------------------------------------------
# _process_mi_scores
# ---------------------------------------------------------------------------

class TestProcessMIScores:
    def _make_encoder(self, coupling_scale='direct'):
        return HamiltonianProjectedEncoder(
            num_qubits=4,
            n_features=4,
            device_type='default.qubit',
            entanglement='custom',
            custom_entanglement=[(0, 1), (2, 3)],
            custom_triads=[],
            mi_pairs_with_scores=[((0, 1), 0.5), ((2, 3), 0.3)],
            mi_triads_with_scores=[],
            evolution_time=0.5,
            n_trotter_steps=1,
            coupling_scale=coupling_scale,
            include_correlations=False,
        )

    def test_direct_scaling(self):
        enc = self._make_encoder('direct')
        result = enc._process_mi_scores(
            scored_items=[((0, 1), 0.5), ((2, 3), 0.3)]
        )
        scores = [s for _, s in result]
        assert abs(scores[0] - 0.5) < 1e-6
        assert abs(scores[1] - 0.3) < 1e-6

    def test_normalized_scaling(self):
        enc = self._make_encoder('normalized')
        result = enc._process_mi_scores(
            scored_items=[((0, 1), 0.5), ((2, 3), 0.3)]
        )
        scores = [s for _, s in result]
        # Max should be 1.0
        assert abs(max(scores) - 1.0) < 1e-6

    def test_sqrt_scaling(self):
        enc = self._make_encoder('sqrt')
        result = enc._process_mi_scores(
            scored_items=[((0, 1), 0.25)]
        )
        scores = [s for _, s in result]
        assert abs(scores[0] - 0.5) < 1e-6  # sqrt(0.25) = 0.5

    def test_weak_scaling(self):
        enc = self._make_encoder('weak')
        result = enc._process_mi_scores(
            scored_items=[((0, 1), 1.0)]
        )
        scores = [s for _, s in result]
        assert abs(scores[0] - 0.1) < 1e-6  # 1.0 * 0.1


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestEncoderInit:
    def test_output_dim_no_correlations(self):
        enc = HamiltonianProjectedEncoder(
            num_qubits=4,
            n_features=4,
            device_type='default.qubit',
            entanglement='custom',
            custom_entanglement=[(0, 1)],
            custom_triads=[],
            evolution_time=0.5,
            n_trotter_steps=1,
            coupling_scale='direct',
            include_correlations=False,
        )
        assert enc.output_dim == 4  # n_qubits expectations only

    def test_output_dim_with_correlations(self):
        enc = HamiltonianProjectedEncoder(
            num_qubits=4,
            n_features=4,
            device_type='default.qubit',
            entanglement='custom',
            custom_entanglement=[(0, 1), (2, 3)],
            custom_triads=[(0, 1, 2)],
            evolution_time=0.5,
            n_trotter_steps=1,
            coupling_scale='direct',
            include_correlations=True,
        )
        # n_qubits + n_pairs + n_triads = 4 + 2 + 1 = 7
        assert enc.output_dim == 7


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

class TestEncode:
    @pytest.fixture
    def encoder(self):
        return HamiltonianProjectedEncoder(
            num_qubits=3,
            n_features=3,
            device_type='default.qubit',
            entanglement='custom',
            custom_entanglement=[(0, 1)],
            custom_triads=[],
            evolution_time=0.5,
            n_trotter_steps=1,
            coupling_scale='direct',
            include_correlations=False,
        )

    def test_correct_output_shape(self, encoder):
        features = np.array([0.5, 0.3, 0.8])
        result = encoder.encode(features)
        assert result.shape == (encoder.output_dim,)

    def test_values_in_range(self, encoder):
        features = np.array([0.5, 0.3, 0.8])
        result = encoder.encode(features)
        # Pauli Z expectations are in [-1, 1]
        assert np.all(result >= -1.0 - 1e-6)
        assert np.all(result <= 1.0 + 1e-6)

    def test_deterministic(self, encoder):
        features = np.array([0.5, 0.3, 0.8])
        r1 = encoder.encode(features)
        r2 = encoder.encode(features)
        np.testing.assert_array_almost_equal(r1, r2)


# ---------------------------------------------------------------------------
# _init_device raises RuntimeError (not SystemExit)
# ---------------------------------------------------------------------------

class TestInitDevice:
    def test_raises_runtime_error_on_bad_device(self):
        """_init_device should raise RuntimeError, not sys.exit."""
        with pytest.raises(RuntimeError):
            HamiltonianProjectedEncoder(
                num_qubits=2,
                n_features=2,
                device_type='nonexistent.device.type',
                entanglement='custom',
                custom_entanglement=[(0, 1)],
                custom_triads=[],
                evolution_time=0.5,
                n_trotter_steps=1,
                coupling_scale='direct',
                include_correlations=False,
            )
