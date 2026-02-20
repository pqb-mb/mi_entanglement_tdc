"""Shared fixtures for mi_entanglement tests."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# SMILES fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_smiles():
    """List of 4 known-good SMILES strings."""
    return ['CCO', 'CC(=O)Oc1ccccc1C(=O)O', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O', 'c1ccccc1']


@pytest.fixture
def ethanol_mol():
    from rdkit import Chem
    return Chem.MolFromSmiles('CCO')


@pytest.fixture
def aspirin_mol():
    from rdkit import Chem
    return Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')


@pytest.fixture
def caffeine_mol():
    from rdkit import Chem
    return Chem.MolFromSmiles('Cn1c(=O)c2c(ncn2C)n(C)c1=O')


# ---------------------------------------------------------------------------
# Numerical fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_feature_matrix():
    """10x20 random feature matrix (seeded for reproducibility)."""
    rng = np.random.RandomState(42)
    return rng.randn(10, 20)


@pytest.fixture
def sample_labels():
    """Binary labels for 10 samples."""
    return np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])


@pytest.fixture
def sample_continuous_labels():
    """Continuous labels for 10 samples (regression targets)."""
    rng = np.random.RandomState(42)
    return rng.randn(10)


@pytest.fixture
def small_mi_matrix():
    """5x5 symmetric MI matrix with known values."""
    mi = np.array([
        [0.0, 0.5, 0.3, 0.1, 0.2],
        [0.5, 0.0, 0.4, 0.6, 0.1],
        [0.3, 0.4, 0.0, 0.2, 0.7],
        [0.1, 0.6, 0.2, 0.0, 0.3],
        [0.2, 0.1, 0.7, 0.3, 0.0],
    ])
    return mi


# ---------------------------------------------------------------------------
# Temp directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_configs_dir(tmp_path):
    """Temporary directory for config persistence tests."""
    d = tmp_path / "configs"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# GPU auto-skip
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked @pytest.mark.gpu when no CUDA is available."""
    try:
        import ctypes
        libcuda = ctypes.cdll.LoadLibrary("libcuda.so.1")
        count = ctypes.c_int()
        rc_init = libcuda.cuInit(0)
        rc_count = libcuda.cuDeviceGetCount(ctypes.byref(count))
        has_gpu = (rc_init == 0 and rc_count == 0 and count.value > 0)
    except (OSError, AttributeError):
        has_gpu = False

    skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
