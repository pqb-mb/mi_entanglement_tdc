"""
Quantum encoding functions for MI-entanglement pipeline.

Provides MapLight-style 2Q/3Q encoding and tiled Kipu-style encoding.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, List, Tuple

from projected_q_encoder import HamiltonianProjectedEncoder

logger = logging.getLogger(__name__)


def remap_mi_scores(scored_items: List[Tuple],
                    stable_items: List,
                    remapped_items: List) -> List[Tuple]:
    """Remap MI scores from original feature indices to circuit indices."""
    stable_tuples = [tuple(x) if isinstance(x, list) else x for x in stable_items]
    remapped = []

    for orig_item, score in scored_items:
        orig_tuple = tuple(orig_item) if isinstance(orig_item, list) else orig_item
        if orig_tuple in stable_tuples:
            idx = stable_tuples.index(orig_tuple)
            remapped.append((remapped_items[idx], score))

    return remapped


def quantum_encode(features_train: np.ndarray, features_test: np.ndarray | None,
                   mi_result: Dict[str, Any], config: Dict[str, Any],
                   device_type: str, n_workers: int = 1
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantum encode features using Hamiltonian time evolution.

    Creates separate 2Q and 3Q encoders with MI-weighted coupling.
    If features_test is None, test outputs are empty arrays.

    Returns:
        (X_train_2q, X_test_2q, X_train_3q, X_test_3q) encoded features
    """
    if n_workers > 1 and 'gpu' in device_type.lower():
        logger.info("GPU detected (%s), forcing n_workers=1 for quantum encoding", device_type)
        n_workers = 1

    n_qubits_2q = mi_result['n_qubits_2q']
    n_qubits_3q = mi_result['n_qubits_3q']
    remapped_pairs = mi_result['remapped_pairs']
    remapped_triads = mi_result['remapped_triads']
    active_qubits_2q = mi_result['active_qubits_2q']
    active_qubits_3q = mi_result['active_qubits_3q']
    mi_pairs_scored = mi_result.get('mi_pairs_scored', [])
    mi_triads_scored = mi_result.get('mi_triads_scored', [])

    # Scale features for quantum encoding (MinMax to [0, 1])
    scaler_q = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_q.fit_transform(features_train)
    X_test_scaled = scaler_q.transform(features_test) if features_test is not None else None

    n_test = len(features_test) if features_test is not None else 0

    # --- 2Q Encoding ---
    X_train_2q = np.zeros((len(features_train), 0))
    X_test_2q = np.zeros((n_test, 0))

    if n_qubits_2q > 0 and remapped_pairs:
        logger.info("2Q Encoding: %d qubits, %d pairs", n_qubits_2q, len(remapped_pairs))

        mi_pairs_remapped = remap_mi_scores(
            mi_pairs_scored, mi_result['stable_pairs'], remapped_pairs
        )

        encoder_2q = HamiltonianProjectedEncoder(
            num_qubits=n_qubits_2q,
            n_features=n_qubits_2q,
            device_type=device_type,
            entanglement='custom',
            custom_entanglement=list(remapped_pairs),
            custom_triads=[],
            mi_pairs_with_scores=mi_pairs_remapped,
            mi_triads_with_scores=[],
            evolution_time=config['evolution_time'],
            n_trotter_steps=config['n_trotter_steps'],
            coupling_scale=config['coupling_scale'],
            include_correlations=config['include_correlations'],
            n_cd_layers=config.get('n_cd_layers', 0),
            cd_strength=config.get('cd_strength', 0.3),
        )

        X_train_2q_input = X_train_scaled[:, active_qubits_2q]
        X_train_2q = encoder_2q.encode_batch(X_train_2q_input, show_progress=True, n_workers=n_workers)

        if X_test_scaled is not None:
            X_test_2q_input = X_test_scaled[:, active_qubits_2q]
            X_test_2q = encoder_2q.encode_batch(X_test_2q_input, show_progress=True, n_workers=n_workers)

    # --- 3Q Encoding ---
    X_train_3q = np.zeros((len(features_train), 0))
    X_test_3q = np.zeros((n_test, 0))

    if n_qubits_3q > 0 and remapped_triads:
        logger.info("3Q Encoding: %d qubits, %d triads", n_qubits_3q, len(remapped_triads))

        mi_triads_remapped = remap_mi_scores(
            mi_triads_scored, mi_result['stable_triads'], remapped_triads
        )

        encoder_3q = HamiltonianProjectedEncoder(
            num_qubits=n_qubits_3q,
            n_features=n_qubits_3q,
            device_type=device_type,
            entanglement='custom',
            custom_entanglement=[],
            custom_triads=list(remapped_triads),
            mi_pairs_with_scores=[],
            mi_triads_with_scores=mi_triads_remapped,
            evolution_time=config['evolution_time'],
            n_trotter_steps=config['n_trotter_steps'],
            coupling_scale=config['coupling_scale'],
            include_correlations=config['include_correlations'],
            n_cd_layers=config.get('n_cd_layers', 0),
            cd_strength=config.get('cd_strength', 0.3),
        )

        X_train_3q_input = X_train_scaled[:, active_qubits_3q]
        X_train_3q = encoder_3q.encode_batch(X_train_3q_input, show_progress=True, n_workers=n_workers)

        if X_test_scaled is not None:
            X_test_3q_input = X_test_scaled[:, active_qubits_3q]
            X_test_3q = encoder_3q.encode_batch(X_test_3q_input, show_progress=True, n_workers=n_workers)

    return X_train_2q, X_test_2q, X_train_3q, X_test_3q


def tiled_quantum_encode(features_88_train: np.ndarray, features_88_test: np.ndarray,
                         tile_params_list: List[Dict[str, Any]],
                         config: Dict[str, Any], device_type: str,
                         n_workers: int = 1
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Encode features using tiled Kipu-style circuits.

    Each tile creates a separate HamiltonianProjectedEncoder with MI-weighted
    couplings. All tile outputs are concatenated.

    Returns:
        (X_train_quantum, X_test_quantum) concatenated quantum features
    """
    if n_workers > 1 and 'gpu' in device_type.lower():
        logger.info("GPU detected (%s), forcing n_workers=1", device_type)
        n_workers = 1

    # Scale all features once (MinMax to [0, 1])
    scaler_q = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_q.fit_transform(features_88_train)
    X_test_scaled = scaler_q.transform(features_88_test)

    all_train_q = []
    all_test_q = []

    for i, tile_params in enumerate(tile_params_list):
        feature_indices = tile_params['feature_indices']
        n_qubits = tile_params['n_qubits']

        # Extract tile features from full 88-feature matrix
        X_train_tile = X_train_scaled[:, feature_indices]
        X_test_tile = X_test_scaled[:, feature_indices]

        tile_label = tile_params.get('tile_num', i + 1)
        logger.info("Tile %d (%d/%d): %dq, %d pairs, %d triads",
                     tile_label, i + 1, len(tile_params_list), n_qubits,
                     len(tile_params['pairs']), len(tile_params['triads']))

        encoder = HamiltonianProjectedEncoder(
            num_qubits=n_qubits,
            n_features=n_qubits,
            device_type=device_type,
            entanglement='custom',
            custom_entanglement=tile_params['pairs'],
            custom_triads=tile_params['triads'],
            mi_pairs_with_scores=tile_params['mi_pairs_scored'],
            mi_triads_with_scores=tile_params['mi_triads_scored'],
            evolution_time=config['evolution_time'],
            n_trotter_steps=config['n_trotter_steps'],
            coupling_scale=config['coupling_scale'],
            include_correlations=config['include_correlations'],
        )

        X_train_q = encoder.encode_batch(X_train_tile, show_progress=True, n_workers=n_workers)
        X_test_q = encoder.encode_batch(X_test_tile, show_progress=True, n_workers=n_workers)

        all_train_q.append(X_train_q)
        all_test_q.append(X_test_q)
        logger.info("Tile %d output: train=%s, test=%s", tile_label, X_train_q.shape, X_test_q.shape)

    return np.hstack(all_train_q), np.hstack(all_test_q)
