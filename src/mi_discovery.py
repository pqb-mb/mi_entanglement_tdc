"""
MI Discovery orchestration for quantum entanglement pair/triad selection.

Provides MapLight-based MI discovery (prefilter + single-pass pairwise MI)
and tiled Kipu-style MI discovery.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Optional, Tuple

from config import generate_mi_cache_key, save_mi_cache, load_mi_cache
from mi_entanglement_utils import (
    compute_pairwise_mi,
    prefilter_by_univariate_mi,
    select_filtered_pairs,
    select_filtered_triplets,
    get_dynamic_circuit_params,
    get_deduped_feature_indices,
    define_feature_tiles,
    define_mi_tiles,
    compute_tile_encoding_params,
)

logger = logging.getLogger(__name__)


def run_maplight_mi_discovery(features_ml: np.ndarray, labels: np.ndarray,
                              config: Dict[str, Any],
                              cache_dir: Optional[str] = None,
                              regression: bool = False) -> Dict[str, Any]:
    """MI discovery on MapLight features (2563-dim).

    Since pairwise MI on 2563 features is too expensive (esp. triplet search),
    we first prefilter to top K features by univariate MI with target, then
    compute a single-pass pairwise MI matrix on those K features to select
    top pairs/triads for quantum encoding.

    No CV folds -- MI is computed once on the full train_val set. No data
    leakage since we never touch the test set.

    Args:
        features_ml: MapLight features (n_samples, 2563)
        labels: Target labels (n_samples,)
        config: Dict with mi_prefilter_k, pair_threshold, etc.
        cache_dir: Optional cache directory
        regression: Use mutual_info_regression for continuous targets

    Returns:
        dict with MI discovery results + top_feature_indices (into MapLight space)
    """
    top_k = config.get('mi_prefilter_k', 100)

    # Step 1: Prefilter by univariate MI
    logger.info("Pre-filtering to top %d features by univariate MI...", top_k)
    top_indices, univ_mi_scores = prefilter_by_univariate_mi(
        features_ml, labels, top_k=top_k, regression=regression
    )

    # Extract subset
    features_subset = features_ml[:, top_indices]

    # Check cache (based on subset hash)
    if cache_dir:
        cache_key = generate_mi_cache_key(features_subset, labels, config)
        cache_path = Path(cache_dir) / f"{cache_key}.json"

        if cache_path.exists():
            logger.info("Found cached MI results")
            cache_data = load_mi_cache(str(cache_path))
            if cache_data.get('mi_pairs_scored') is not None:
                cache_data['top_feature_indices'] = top_indices
                cache_data['univ_mi_scores'] = univ_mi_scores.tolist()
                return cache_data

    # Step 2: Scale for MI computation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_subset)

    logger.info("Computing single-pass pairwise MI on %d features, %d samples",
                top_k, len(labels))

    # Step 3: Single-pass pairwise MI matrix
    mi_matrix = compute_pairwise_mi(X_scaled)

    # Step 4: Select top pairs and triads by MI
    logger.info("Selecting top pairs (threshold=%s, max=%s)...",
                config['pair_threshold'], config['max_pairs'])
    selected_pairs = select_filtered_pairs(
        mi_matrix,
        max_pairs=config['max_pairs'],
        min_threshold=config['pair_threshold'],
    )
    stable_pairs = [pair for pair, score in selected_pairs]
    mi_pairs_scored = selected_pairs
    logger.info("Selected %d pairs", len(stable_pairs))

    logger.info("Selecting top triads (threshold=%s, max=%s)...",
                config['triad_threshold'], config['max_triads'])
    selected_triads = select_filtered_triplets(
        mi_matrix,
        max_triplets=config['max_triads'],
        min_threshold=config['triad_threshold'],
    )
    stable_triads = [triad for triad, score in selected_triads]
    mi_triads_scored = selected_triads
    logger.info("Selected %d triads", len(stable_triads))

    # Step 5: Dynamic circuit mappings (enforce max_qubits)
    active_qubits_2q, index_map_2q, remapped_pairs, _ = get_dynamic_circuit_params(
        stable_pairs, []
    )
    n_qubits_2q = len(active_qubits_2q)

    active_qubits_3q, index_map_3q, _, remapped_triads = get_dynamic_circuit_params(
        [], stable_triads
    )
    n_qubits_3q = len(active_qubits_3q)

    # Enforce max_qubits constraint
    max_q = config.get('max_qubits', 28)
    if n_qubits_2q > max_q:
        logger.warning("2Q circuit needs %d qubits > max %d, trimming pairs...",
                       n_qubits_2q, max_q)
        while n_qubits_2q > max_q and stable_pairs:
            stable_pairs = stable_pairs[:-1]
            mi_pairs_scored = mi_pairs_scored[:-1]
            active_qubits_2q, index_map_2q, remapped_pairs, _ = get_dynamic_circuit_params(stable_pairs, [])
            n_qubits_2q = len(active_qubits_2q)
        logger.info("Trimmed to %d pairs, %d qubits", len(stable_pairs), n_qubits_2q)

    if n_qubits_3q > max_q:
        logger.warning("3Q circuit needs %d qubits > max %d, trimming triads...",
                       n_qubits_3q, max_q)
        while n_qubits_3q > max_q and stable_triads:
            stable_triads = stable_triads[:-1]
            mi_triads_scored = mi_triads_scored[:-1]
            active_qubits_3q, index_map_3q, _, remapped_triads = get_dynamic_circuit_params([], stable_triads)
            n_qubits_3q = len(active_qubits_3q)
        logger.info("Trimmed to %d triads, %d qubits", len(stable_triads), n_qubits_3q)

    result = {
        'stable_pairs': stable_pairs,
        'stable_triads': stable_triads,
        'active_qubits_2q': active_qubits_2q,
        'active_qubits_3q': active_qubits_3q,
        'index_map_2q': index_map_2q,
        'index_map_3q': index_map_3q,
        'remapped_pairs': remapped_pairs,
        'remapped_triads': remapped_triads,
        'n_qubits_2q': n_qubits_2q,
        'n_qubits_3q': n_qubits_3q,
        'mi_pairs_scored': mi_pairs_scored,
        'mi_triads_scored': mi_triads_scored,
        'top_feature_indices': top_indices,
        'univ_mi_scores': univ_mi_scores.tolist(),
    }

    # Save to cache
    if cache_dir:
        save_mi_cache(str(cache_path), **{k: v for k, v in result.items()
                                           if k not in ('univ_mi_scores',)})

    return result


def run_tiled_mi_discovery(features_88: np.ndarray, config: Dict[str, Any],
                           cache_dir: Optional[str] = None
                           ) -> Tuple[List[Dict[str, Any]], np.ndarray, List[int]]:
    """Kipu-style tiled MI discovery: compute full MI matrix, define semantic tiles.

    Args:
        features_88: Full BIO88 features (n_samples, 88)
        config: Dict with 'tile_max_pairs' and 'tile_max_triads'
        cache_dir: Optional cache directory for MI matrix

    Returns:
        (tile_params_list, mi_matrix, deduped_indices)
    """
    deduped_indices = get_deduped_feature_indices()
    features_deduped = features_88[:, deduped_indices]

    n_deduped = len(deduped_indices)
    logger.info("Deduped: %d -> %d features (dropped indices %s)",
                features_88.shape[1], n_deduped,
                sorted(set(range(88)) - set(deduped_indices)))

    # Check cache
    mi_matrix = None
    if cache_dir:
        cache_key = hashlib.sha256(features_deduped.tobytes()).hexdigest()[:16]
        cache_path = Path(cache_dir) / f"tiled_mi_{cache_key}.npy"
        if cache_path.exists():
            mi_matrix = np.load(str(cache_path))
            logger.info("Loaded cached MI matrix from: %s", cache_path)

    if mi_matrix is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_deduped)

        logger.info("Computing %dx%d MI matrix...", n_deduped, n_deduped)
        mi_matrix = compute_pairwise_mi(X_scaled)

        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), mi_matrix)
            logger.info("Saved MI matrix to: %s", cache_path)

    # Define tiles
    tile_mode = config.get('tile_mode', 'semantic')
    if tile_mode == 'mi':
        max_tile = config.get('max_tile_size', 28)
        logger.info("Tile mode: MI-driven (rank by total MI, slice into groups of %d)", max_tile)
        raw_tiles = define_mi_tiles(mi_matrix, deduped_indices, max_tile_size=max_tile)
    else:
        logger.info("Tile mode: semantic (fixed grouping by feature type)")
        raw_tiles = define_feature_tiles()

    # Apply active_tiles filter if present
    active_tiles = config.get('active_tiles', None)
    if active_tiles:
        tiles = [(idx, raw_tiles[idx - 1]) for idx in active_tiles]
        logger.info("Active tiles: %s (skipping %s)", active_tiles,
                    [i+1 for i in range(len(raw_tiles)) if i+1 not in active_tiles])
    else:
        tiles = [(i + 1, t) for i, t in enumerate(raw_tiles)]

    tile_params_list = []
    total_qubits = 0
    total_pairs = 0
    total_triads = 0

    for tile_num, tile_indices in tiles:
        params = compute_tile_encoding_params(
            mi_matrix, deduped_indices, tile_indices,
            max_pairs=config['tile_max_pairs'],
            max_triads=config['tile_max_triads'],
        )
        params['tile_num'] = tile_num
        tile_params_list.append(params)
        total_qubits += params['n_qubits']
        total_pairs += len(params['pairs'])
        total_triads += len(params['triads'])
        logger.info("Tile %d: %dq, %d pairs, %d triads",
                    tile_num, params['n_qubits'], len(params['pairs']), len(params['triads']))

    logger.info("Total: %d qubits across %d tiles, %d pairs, %d triads",
                total_qubits, len(tiles), total_pairs, total_triads)

    return tile_params_list, mi_matrix, deduped_indices
