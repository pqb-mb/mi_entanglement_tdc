"""
Mutual Information-based Entanglement Selection

Computes pairwise mutual information between features to select
data-driven entanglement pairs and triads for quantum encoding.

Inspired by Simen et al. (arXiv:2510.13807) - "Data-informed Qubit Assignment
with a Variable Arity Quantum Convolution"
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)


def prefilter_by_univariate_mi(X: np.ndarray, y: np.ndarray, top_k: int = 100,
                                random_state: int = 42, verbose: bool = True,
                                regression: bool = False) -> Tuple[List[int], np.ndarray]:
    """
    Pre-filter features by univariate MI with target label.

    Computes MI between each feature and the target, returns
    indices of the top K features. Used to reduce dimensionality before
    expensive pairwise MI computation.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        top_k: Number of top features to select
        random_state: Random seed
        verbose: Print progress
        regression: Use mutual_info_regression instead of mutual_info_classif

    Returns:
        (top_indices, mi_scores) — sorted indices and full MI score array
    """
    if verbose:
        logger.info(f"  Computing univariate MI for {X.shape[1]} features...")

    mi_fn = mutual_info_regression if regression else mutual_info_classif
    mi_scores = mi_fn(X, y, random_state=random_state)

    # Rank by MI score (descending)
    ranked = np.argsort(mi_scores)[::-1]
    top_indices = [int(i) for i in sorted(ranked[:top_k])]

    if verbose:
        logger.info(f"  Top {top_k} features selected (MI range: "
                    f"{mi_scores[ranked[0]]:.4f} — {mi_scores[ranked[min(top_k-1, len(ranked)-1)]]:.4f})")
        n_zero = np.sum(mi_scores == 0)
        if n_zero > 0:
            logger.info(f"  {n_zero}/{X.shape[1]} features have zero MI with target")

    return top_indices, mi_scores


def compute_pairwise_mi(X: np.ndarray, n_neighbors: int = 3, random_state: int = 42,
                       show_progress: bool = True) -> np.ndarray:
    """
    Compute mutual information matrix between all feature pairs.

    Uses k-nearest neighbors estimation via sklearn.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        n_neighbors: Number of neighbors for MI estimation (default 3)
        random_state: Random seed for reproducibility
        show_progress: Show tqdm progress bar (default True)

    Returns:
        Symmetric MI matrix of shape (n_features, n_features)
    """
    import time
    n_features = X.shape[1]
    mi_matrix = np.zeros((n_features, n_features))

    start_time = time.time()

    iterator = tqdm(range(n_features), desc="Computing MI", disable=not show_progress)
    for i in iterator:
        mi_values = mutual_info_regression(
            X, X[:, i],
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        mi_matrix[i, :] = mi_values

    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"    MI computation took {elapsed:.2f}s")

    # Symmetrize (average of both directions)
    mi_matrix = (mi_matrix + mi_matrix.T) / 2

    # Zero diagonal (MI with self is not meaningful for entanglement)
    np.fill_diagonal(mi_matrix, 0)

    return mi_matrix


def select_filtered_pairs(
    mi_matrix: np.ndarray,
    max_pairs: int = 30,
    min_threshold: float = 0.1
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Select top pairs by MI with both top-N and minimum threshold constraints.

    Returns:
        List of ((i, j), mi_score) tuples sorted by MI descending
    """
    n = mi_matrix.shape[0]
    pairs_with_mi = []

    for i in range(n):
        for j in range(i + 1, n):
            mi_score = mi_matrix[i, j]
            if mi_score >= min_threshold:
                pairs_with_mi.append(((i, j), mi_score))

    pairs_with_mi.sort(key=lambda x: x[1], reverse=True)
    return pairs_with_mi[:max_pairs]


def select_filtered_triplets(
    mi_matrix: np.ndarray,
    max_triplets: int = 15,
    min_threshold: float = 0.15
) -> List[Tuple[Tuple[int, int, int], float]]:
    """
    Select top triplets by average pairwise MI.

    The triplet score is the average of the three pairwise MI values:
        score(i,j,k) = (MI[i,j] + MI[j,k] + MI[i,k]) / 3

    Returns:
        List of ((i, j, k), avg_mi) tuples sorted by avg_mi descending
    """
    n = mi_matrix.shape[0]
    triplets_with_mi = []

    for i, j, k in combinations(range(n), 3):
        avg_mi = (mi_matrix[i, j] + mi_matrix[j, k] + mi_matrix[i, k]) / 3
        if avg_mi >= min_threshold:
            triplets_with_mi.append(((i, j, k), avg_mi))

    triplets_with_mi.sort(key=lambda x: x[1], reverse=True)
    return triplets_with_mi[:max_triplets]


def get_mi_scores_for_pairs(
    X: np.ndarray,
    pairs: List[Tuple[int, int]],
    triads: List[Tuple[int, int, int]],
    n_neighbors: int = 3,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[List[Tuple[Tuple[int, int], float]],
           List[Tuple[Tuple[int, int, int], float]],
           np.ndarray]:
    """
    Compute MI scores for given stable pairs and triads.

    Use this after compute_stable_mi_pairs() to get MI scores for the
    stable entanglement structure identified via cross-validation.

    Returns:
        Tuple of (pairs_with_scores, triads_with_scores, mi_matrix)
    """
    if verbose:
        logger.info(f"  Computing MI matrix for scoring pairs/triads...")

    mi_matrix = compute_pairwise_mi(X, n_neighbors=n_neighbors,
                                    random_state=random_state,
                                    show_progress=verbose)

    pairs_with_scores = []
    for (i, j) in pairs:
        mi_score = mi_matrix[i, j]
        pairs_with_scores.append(((i, j), mi_score))

    triads_with_scores = []
    for (i, j, k) in triads:
        avg_mi = (mi_matrix[i, j] + mi_matrix[j, k] + mi_matrix[i, k]) / 3
        triads_with_scores.append(((i, j, k), avg_mi))

    if verbose and pairs_with_scores:
        top_pair_mi = max(mi for _, mi in pairs_with_scores)
        min_pair_mi = min(mi for _, mi in pairs_with_scores)
        logger.info(f"  Pair MI scores: range [{min_pair_mi:.4f}, {top_pair_mi:.4f}]")

    if verbose and triads_with_scores:
        top_triad_mi = max(mi for _, mi in triads_with_scores)
        min_triad_mi = min(mi for _, mi in triads_with_scores)
        logger.info(f"  Triad MI scores: range [{min_triad_mi:.4f}, {top_triad_mi:.4f}]")

    return pairs_with_scores, triads_with_scores, mi_matrix


def get_dynamic_circuit_params(
    mi_pairs: List[Tuple[int, int]],
    mi_triads: List[Tuple[int, int, int]],
) -> Tuple[List[int], Dict[int, int], List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Compute dynamic circuit parameters based on MI-selected entanglement.

    Finds the unique qubits used in pairs/triads and creates a mapping
    to remap indices for a smaller circuit.

    Returns:
        Tuple of (active_qubits, index_map, remapped_pairs, remapped_triads)
    """
    used_qubits = set()
    for i, j in mi_pairs:
        used_qubits.add(i)
        used_qubits.add(j)
    for i, j, k in mi_triads:
        used_qubits.add(i)
        used_qubits.add(j)
        used_qubits.add(k)

    active_qubits = sorted(used_qubits)
    index_map = {orig: new for new, orig in enumerate(active_qubits)}

    remapped_pairs = [(index_map[i], index_map[j]) for i, j in mi_pairs]
    remapped_triads = [(index_map[i], index_map[j], index_map[k]) for i, j, k in mi_triads]

    return active_qubits, index_map, remapped_pairs, remapped_triads


def _process_single_fold(
    X: np.ndarray,
    train_idx: np.ndarray,
    fold_idx: int,
    n_neighbors: int,
    random_state: int,
    max_pairs: int,
    pair_threshold: float,
    max_triads: int,
    triad_threshold: float,
    verbose: bool,
    print_progress: bool = False
) -> Tuple[int, List[Tuple[Tuple[int, int], float]], List[Tuple[Tuple[int, int, int], float]]]:
    """Process a single CV fold for MI computation (used for parallel processing)."""
    import time

    X_train = X[train_idx]
    n_features = X_train.shape[1]

    if print_progress:
        mi_matrix = np.zeros((n_features, n_features))
        start_time = time.time()
        milestone_step = max(1, n_features // 10)

        for i in range(n_features):
            mi_values = mutual_info_regression(
                X_train, X_train[:, i],
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            mi_matrix[i, :] = mi_values

            if (i + 1) % milestone_step == 0 or i == n_features - 1:
                pct = int(100 * (i + 1) / n_features)
                elapsed = time.time() - start_time
                logger.warning(f"    [Fold {fold_idx + 1}] {pct}% complete ({i + 1}/{n_features} features, {elapsed:.1f}s)")

        mi_matrix = (mi_matrix + mi_matrix.T) / 2
        np.fill_diagonal(mi_matrix, 0)
    else:
        mi_matrix = compute_pairwise_mi(X_train, n_neighbors=n_neighbors, random_state=random_state,
                                        show_progress=verbose)

    filtered_pairs = select_filtered_pairs(mi_matrix, max_pairs=max_pairs, min_threshold=pair_threshold)
    filtered_triads = select_filtered_triplets(mi_matrix, max_triplets=max_triads, min_threshold=triad_threshold)

    return fold_idx, filtered_pairs, filtered_triads


def compute_stable_mi_pairs(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    pair_threshold: float = 0.1,
    triad_threshold: float = 0.15,
    max_pairs: int = 30,
    max_triads: int = 15,
    stability_threshold: int = 4,
    max_qubits: int = 28,
    n_neighbors: int = 3,
    random_state: int = 42,
    verbose: bool = True,
    n_jobs: Optional[int] = None
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]], Dict]:
    """
    Phase 1: Compute MI across CV folds and return stable pairs/triads.

    Runs k-fold CV purely for MI discovery, tracking which pairs/triads
    pass thresholds consistently across folds.

    Args:
        X: Feature matrix of shape (n_samples, n_features), already scaled
        y: Label array for stratified splitting
        n_folds: Number of CV folds for stability assessment
        pair_threshold: Minimum MI for pair inclusion per fold
        triad_threshold: Minimum avg MI for triad inclusion per fold
        max_pairs: Maximum pairs to select per fold
        max_triads: Maximum triads to select per fold
        stability_threshold: Minimum folds a pair/triad must appear in
        max_qubits: Maximum qubits allowed (default 28)
        n_neighbors: k for MI estimation
        random_state: Random seed for CV splitting
        verbose: Print progress info
        n_jobs: Number of parallel jobs (default None = n_folds)

    Returns:
        Tuple of (stable_pairs, stable_triads, stability_stats)
    """
    if n_jobs is None:
        n_jobs = n_folds

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    pair_counts: Counter = Counter()
    pair_mi_scores: Dict[Tuple[int, int], list] = {}
    triad_counts: Counter = Counter()
    triad_mi_scores: Dict[Tuple[int, int, int], list] = {}

    fold_stats = []

    if verbose:
        logger.info(f"  Phase 1: MI Discovery across {n_folds} folds")
        logger.info(f"  Thresholds: pair >= {pair_threshold}, triad >= {triad_threshold}")
        logger.info(f"  Stability: must appear in >= {stability_threshold}/{n_folds} folds")
        if n_jobs != 1:
            logger.info(f"  Using {n_jobs if n_jobs > 0 else 'all'} parallel jobs")

    fold_indices = list(enumerate(skf.split(X, y)))

    if n_jobs == 1:
        fold_iterator = tqdm(fold_indices, desc="CV Folds", disable=not verbose)
        fold_results = []
        for fold_idx, (train_idx, _) in fold_iterator:
            if verbose:
                fold_iterator.set_description(f"Fold {fold_idx + 1}/{n_folds}")

            result = _process_single_fold(
                X, train_idx, fold_idx, n_neighbors, random_state,
                max_pairs, pair_threshold, max_triads, triad_threshold,
                verbose=False, print_progress=False
            )
            fold_results.append(result)

            if verbose:
                _, filtered_pairs, filtered_triads = result
                fold_iterator.set_postfix({'pairs': len(filtered_pairs), 'triads': len(filtered_triads)})
    else:
        if verbose:
            logger.info(f"  Processing {n_folds} folds in parallel...")

        fold_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_process_single_fold)(
                X, train_idx, fold_idx, n_neighbors, random_state,
                max_pairs, pair_threshold, max_triads, triad_threshold,
                verbose=False, print_progress=verbose
            )
            for fold_idx, (train_idx, _) in fold_indices
        )

    # Aggregate results from all folds
    for fold_idx, filtered_pairs, filtered_triads in fold_results:
        fold_pairs = [p for p, _ in filtered_pairs]
        fold_pair_scores = {p: score for p, score in filtered_pairs}
        fold_triads = [t for t, _ in filtered_triads]
        fold_triad_scores = {t: score for t, score in filtered_triads}

        for pair in fold_pairs:
            normalized_pair = tuple(sorted(pair))
            pair_counts[normalized_pair] += 1
            if normalized_pair not in pair_mi_scores:
                pair_mi_scores[normalized_pair] = []
            pair_mi_scores[normalized_pair].append(fold_pair_scores[pair])

        for triad in fold_triads:
            normalized_triad = tuple(sorted(triad))
            triad_counts[normalized_triad] += 1
            if normalized_triad not in triad_mi_scores:
                triad_mi_scores[normalized_triad] = []
            triad_mi_scores[normalized_triad].append(fold_triad_scores[triad])

        fold_stats.append({
            'fold': fold_idx + 1,
            'n_pairs': len(fold_pairs),
            'n_triads': len(fold_triads),
        })

    # Filter by stability threshold
    stable_pairs = []
    for pair, count in pair_counts.items():
        if count >= stability_threshold:
            avg_mi = np.mean(pair_mi_scores[pair])
            stable_pairs.append((pair, avg_mi))
    stable_pairs.sort(key=lambda x: x[1], reverse=True)

    stable_triads = []
    for triad, count in triad_counts.items():
        if count >= stability_threshold:
            avg_mi = np.mean(triad_mi_scores[triad])
            stable_triads.append((triad, avg_mi))
    stable_triads.sort(key=lambda x: x[1], reverse=True)

    # Apply max_qubits constraint INDEPENDENTLY for 2Q and 3Q
    final_pairs = []
    used_qubits_2q = set()
    for pair, mi_score in stable_pairs[:max_pairs]:
        potential_qubits = used_qubits_2q | {pair[0], pair[1]}
        if len(potential_qubits) <= max_qubits:
            final_pairs.append(pair)
            used_qubits_2q = potential_qubits
        else:
            break
    n_qubits_2q = len(used_qubits_2q)

    final_triads = []
    used_qubits_3q = set()
    for triad, mi_score in stable_triads[:max_triads]:
        potential_qubits = used_qubits_3q | {triad[0], triad[1], triad[2]}
        if len(potential_qubits) <= max_qubits:
            final_triads.append(triad)
            used_qubits_3q = potential_qubits
        else:
            break
    n_qubits_3q = len(used_qubits_3q)

    stability_stats = {
        'n_folds': n_folds,
        'stability_threshold': stability_threshold,
        'pair_threshold': pair_threshold,
        'triad_threshold': triad_threshold,
        'max_qubits': max_qubits,
        'n_qubits_2q': n_qubits_2q,
        'n_qubits_3q': n_qubits_3q,
        'n_features': X.shape[1],
        'total_unique_pairs': len(pair_counts),
        'total_unique_triads': len(triad_counts),
        'stable_pairs_before_limit': len(stable_pairs),
        'stable_triads_before_limit': len(stable_triads),
        'stable_pairs': len(final_pairs),
        'stable_triads': len(final_triads),
        'fold_stats': fold_stats,
        'pair_fold_distribution': dict(Counter(pair_counts.values())),
        'triad_fold_distribution': dict(Counter(triad_counts.values())),
    }

    if verbose:
        logger.info(f"\n  Stability Results:")
        logger.info(f"    Input features: {X.shape[1]}")
        logger.info(f"    Unique pairs across folds: {len(pair_counts)}")
        logger.info(f"    Stable pairs (>= {stability_threshold} folds): {len(stable_pairs)}")
        logger.info(f"    Unique triads across folds: {len(triad_counts)}")
        logger.info(f"    Stable triads (>= {stability_threshold} folds): {len(stable_triads)}")
        logger.info(f"\n  Qubit Constraint (max {max_qubits} per circuit):")
        logger.info(f"    2Q circuit: {len(final_pairs)} pairs, {n_qubits_2q} qubits")
        logger.info(f"    3Q circuit: {len(final_triads)} triads, {n_qubits_3q} qubits")
        if len(final_pairs) < len(stable_pairs):
            logger.info(f"    WARNING: 2Q qubit limit reached, {len(stable_pairs) - len(final_pairs)} pairs excluded")
        if len(final_triads) < len(stable_triads):
            logger.info(f"    WARNING: 3Q qubit limit reached, {len(stable_triads) - len(final_triads)} triads excluded")

        logger.info(f"\n  Distribution:")
        logger.info(f"    Pair fold counts: {stability_stats['pair_fold_distribution']}")
        logger.info(f"    Triad fold counts: {stability_stats['triad_fold_distribution']}")

    return final_pairs, final_triads, stability_stats


# ============================================================================
# Tiled Kipu-Style Encoding (MI as coupling weight, not selection gate)
# ============================================================================

# BIO88-specific near-duplicate feature indices to drop before tiled encoding.
# Index 1 = ExactMolWt (nearly identical to MolWt at index 0)
# Index 78 = Struct_aromatic_ratio (nearly identical to AromaticAtomFraction at index 24)
# Only valid for the 88-feature layout defined in process_bio88.py.
DEDUP_DROP_INDICES = {1, 78}


def get_deduped_feature_indices() -> List[int]:
    """Return sorted list of 86 original BIO88 feature indices (excluding duplicates)."""
    return [i for i in range(88) if i not in DEDUP_DROP_INDICES]


def define_feature_tiles() -> List[List[int]]:
    """Legacy: Return 5 semantic tile definitions (arbitrary grouping by feature type)."""
    return [
        [0] + list(range(2, 18)),      # Tile 1: 17 features
        list(range(18, 36)),            # Tile 2: 18 features
        list(range(36, 54)),            # Tile 3: 18 features
        list(range(54, 78)),            # Tile 4: 24 features
        list(range(79, 88)),            # Tile 5: 9 features
    ]


def define_mi_tiles(mi_matrix: np.ndarray, deduped_indices: List[int], max_tile_size: int = 28) -> List[List[int]]:
    """
    Data-driven tiling: rank features by total MI, slice into groups of max_tile_size.

    Features with the highest total MI (sum of MI with all other features) go
    into tile 1, next highest into tile 2, etc. This ensures the most strongly
    interacting features share a tile and get ZZ/ZZZ couplings.

    Args:
        mi_matrix: Full MI matrix (n×n) on deduped features
        deduped_indices: List mapping deduped position → original feature index
        max_tile_size: Max features per tile (default 28, our qubit limit)

    Returns:
        List of tiles, each tile is a list of original BIO88 feature indices.
    """
    n = mi_matrix.shape[0]

    # Total MI for each feature = sum of its row (excluding diagonal)
    total_mi = mi_matrix.sum(axis=1) - np.diag(mi_matrix)

    # Rank by total MI (highest first)
    ranked_positions = np.argsort(-total_mi)

    # Slice into tiles of max_tile_size
    tiles = []
    for start in range(0, n, max_tile_size):
        tile_positions = ranked_positions[start:start + max_tile_size]
        # Map back to original BIO88 feature indices
        tile_orig = [deduped_indices[pos] for pos in tile_positions]
        tiles.append(sorted(tile_orig))  # Sort for determinism

    # Print summary
    for i, tile in enumerate(tiles):
        tile_positions = [deduped_indices.index(idx) for idx in tile]
        tile_total_mi = sum(total_mi[p] for p in tile_positions)
        logger.info(f"    Tile {i+1}: {len(tile)} features, total MI={tile_total_mi:.1f}")

    return tiles


def compute_tile_encoding_params(mi_matrix: np.ndarray, deduped_indices: List[int],
                                  tile_orig_indices: List[int],
                                  max_pairs: int = 35, max_triads: int = 8) -> Dict[str, Any]:
    """
    Compute encoding parameters for a single tile (Kipu-style).

    Every feature in the tile gets a 1-body Z term. MI scores weight the
    2-body (ZZ) and 3-body (ZZZ) coupling coefficients — top pairs/triads
    selected by MI strength within the tile.

    Args:
        mi_matrix: Full MI matrix (86×86) computed on deduped features
        deduped_indices: List mapping deduped position → original feature index
        tile_orig_indices: List of original BIO88 feature indices for this tile
        max_pairs: Max 2-body interactions per tile
        max_triads: Max 3-body interactions per tile

    Returns:
        dict with n_qubits, feature_indices, pairs, triads, MI scores
    """
    # Map original indices to deduped positions
    orig_to_pos = {orig: pos for pos, orig in enumerate(deduped_indices)}
    tile_dedup_positions = [orig_to_pos[i] for i in tile_orig_indices]
    n_tile = len(tile_orig_indices)

    # Extract sub-matrix for this tile
    sub_mi = mi_matrix[np.ix_(tile_dedup_positions, tile_dedup_positions)]

    # Select top pairs by MI (using tile-local indices 0..n_tile-1)
    pairs_with_mi = []
    for i in range(n_tile):
        for j in range(i + 1, n_tile):
            mi_score = sub_mi[i, j]
            if mi_score > 0:
                pairs_with_mi.append(((i, j), float(mi_score)))
    pairs_with_mi.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pairs_with_mi[:max_pairs]

    # Select top triads by average pairwise MI
    triads_with_mi = []
    for i, j, k in combinations(range(n_tile), 3):
        avg_mi = (sub_mi[i, j] + sub_mi[j, k] + sub_mi[i, k]) / 3
        if avg_mi > 0:
            triads_with_mi.append(((i, j, k), float(avg_mi)))
    triads_with_mi.sort(key=lambda x: x[1], reverse=True)
    top_triads = triads_with_mi[:max_triads]

    return {
        'n_qubits': n_tile,
        'feature_indices': tile_orig_indices,
        'pairs': [p for p, _ in top_pairs],
        'triads': [t for t, _ in top_triads],
        'mi_pairs_scored': top_pairs,
        'mi_triads_scored': top_triads,
    }
