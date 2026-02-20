"""
Feature generation wrappers for BIO88 and MapLight features.
"""
from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional

from rdkit import Chem

from process_bio88 import (
    compute_all_features, ALL_FEATURE_NAMES, _process_single,
    N_MAPLIGHT_FEATURES, _process_maplight_single, compute_maplight_features,
)

logger = logging.getLogger(__name__)


def generate_features_for_dataframe(df: pd.DataFrame,
                                    smiles_col: str = 'Drug',
                                    n_workers: int = 1) -> np.ndarray:
    """Generate BIO88 features for a DataFrame of SMILES strings.

    Returns:
        np.ndarray of shape (n_samples, 88) -- rows with failed SMILES get NaN-filled
    """
    smiles_list = df[smiles_col].tolist()
    n_samples = len(smiles_list)
    n_features = len(ALL_FEATURE_NAMES)

    features = np.full((n_samples, n_features), np.nan)

    if n_workers > 1:
        ctx = mp.get_context('spawn')
        work_items = list(enumerate(smiles_list))
        with ctx.Pool(n_workers) as pool:
            for idx, result in tqdm(
                pool.imap_unordered(_process_single, work_items),
                total=n_samples, desc="Generating BIO88 features"
            ):
                if result is not None:
                    features[idx] = [result[name] for name in ALL_FEATURE_NAMES]
    else:
        for i, smi in enumerate(tqdm(smiles_list, desc="Generating BIO88 features")):
            result = compute_all_features(smi)
            if result is not None:
                features[i] = [result[name] for name in ALL_FEATURE_NAMES]

    n_failures = np.sum(np.isnan(features[:, 0]))
    if n_failures > 0:
        logger.warning("%d/%d SMILES failed feature generation", n_failures, n_samples)
        col_medians = np.nanmedian(features, axis=0)
        for j in range(n_features):
            mask = np.isnan(features[:, j])
            features[mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

    return features


def generate_maplight_for_dataframe(df: pd.DataFrame,
                                    smiles_col: str = 'Drug',
                                    n_workers: int = 1,
                                    cache_dir: Optional[str] = None) -> np.ndarray:
    """Generate 2563-dim MapLight features for a DataFrame of SMILES strings.

    ECFP counts (1024) + Avalon counts (1024) + ERG (315) + RDKit props (200).

    Args:
        cache_dir: If set, cache features as .npy keyed by SMILES hash.

    Returns:
        np.ndarray of shape (n_samples, 2563)
    """
    smiles_list = df[smiles_col].tolist()
    n_samples = len(smiles_list)

    # Check cache
    cache_path = None
    if cache_dir:
        h = hashlib.sha256('\n'.join(smiles_list).encode()).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f'maplight_{h}.npy')
        if os.path.exists(cache_path):
            features = np.load(cache_path)
            logger.info("Loaded cached MapLight features from %s (%d samples)",
                        cache_path, features.shape[0])
            return features

    features = np.zeros((n_samples, N_MAPLIGHT_FEATURES), dtype=np.float32)
    n_failures = 0

    if n_workers > 1:
        ctx = mp.get_context('spawn')
        work_items = list(enumerate(smiles_list))
        with ctx.Pool(n_workers) as pool:
            for idx, result in tqdm(
                pool.imap_unordered(_process_maplight_single, work_items),
                total=n_samples, desc="Generating MapLight features"
            ):
                if result is not None:
                    features[idx] = result
                else:
                    n_failures += 1
    else:
        for i, smi in enumerate(tqdm(smiles_list, desc="Generating MapLight features")):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                features[i] = compute_maplight_features(mol)
            else:
                n_failures += 1

    if n_failures > 0:
        logger.warning("%d/%d SMILES failed MapLight generation", n_failures, n_samples)

    # Save to cache
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, features)
        logger.info("Saved MapLight features to cache: %s", cache_path)

    return features
