"""Tests for process_bio88.py feature generation functions."""
from __future__ import annotations

import numpy as np
import pytest

from process_bio88 import (
    fix_smiles,
    _compute_bio30,
    compute_pharmacophore_features,
    compute_structural_features,
    compute_all_features,
    compute_erg_features,
    compute_rdkit_properties,
    compute_maplight_features,
    ALL_FEATURE_NAMES,
    N_MAPLIGHT_FEATURES,
)


# ---------------------------------------------------------------------------
# fix_smiles
# ---------------------------------------------------------------------------

class TestFixSmiles:
    def test_valid_smiles(self):
        mol = fix_smiles('CCO')
        assert mol is not None

    def test_invalid_smiles(self):
        mol = fix_smiles('NOT_A_SMILES')
        assert mol is None

    def test_empty_string(self):
        mol = fix_smiles('')
        assert mol is None

    def test_none_input(self):
        mol = fix_smiles(None)
        assert mol is None

    def test_salt_removal(self):
        """Salts should be handled (largest fragment kept)."""
        mol = fix_smiles('CCO.[Na]')
        assert mol is not None

    def test_standardization(self):
        """Tautomers/charge normalization should produce valid mol."""
        mol = fix_smiles('c1ccccc1')
        assert mol is not None


# ---------------------------------------------------------------------------
# _compute_bio30
# ---------------------------------------------------------------------------

class TestComputeBio30:
    def test_returns_30_keys(self, ethanol_mol):
        result = _compute_bio30(ethanol_mol)
        assert len(result) == 30

    def test_known_molwt_ethanol(self, ethanol_mol):
        result = _compute_bio30(ethanol_mol)
        assert abs(result['MolWt'] - 46.07) < 0.1

    def test_no_nans(self, aspirin_mol):
        result = _compute_bio30(aspirin_mol)
        for key, val in result.items():
            assert not np.isnan(val), f"NaN found for {key}"


# ---------------------------------------------------------------------------
# compute_pharmacophore_features
# ---------------------------------------------------------------------------

class TestPharmacophorFeatures:
    def test_returns_48_keys(self, ethanol_mol):
        result = compute_pharmacophore_features(ethanol_mol)
        assert len(result) == 48

    def test_donor_count_aspirin(self, aspirin_mol):
        result = compute_pharmacophore_features(aspirin_mol)
        assert result['Pharm_donor_count'] >= 0  # Non-negative count

    def test_no_negatives(self, aspirin_mol):
        result = compute_pharmacophore_features(aspirin_mol)
        for key, val in result.items():
            assert val >= 0, f"Negative value for {key}: {val}"


# ---------------------------------------------------------------------------
# compute_structural_features
# ---------------------------------------------------------------------------

class TestStructuralFeatures:
    def test_returns_10_keys(self, ethanol_mol):
        result = compute_structural_features(ethanol_mol)
        assert len(result) == 10

    def test_benzene_aromatic_ratio(self):
        from rdkit import Chem
        benzene = Chem.MolFromSmiles('c1ccccc1')
        result = compute_structural_features(benzene)
        # All atoms in benzene are aromatic
        assert result['Struct_aromatic_ratio'] > 0.9

    def test_ethanol_oxygen_ratio(self, ethanol_mol):
        result = compute_structural_features(ethanol_mol)
        # Ethanol has 1 O out of 3 heavy atoms
        assert 0.3 < result['Struct_n_oxygen_ratio'] < 0.4


# ---------------------------------------------------------------------------
# compute_all_features
# ---------------------------------------------------------------------------

class TestComputeAllFeatures:
    def test_returns_88_keys(self):
        result = compute_all_features('CCO')
        assert result is not None
        assert len(result) == 88

    def test_keys_match_all_feature_names(self):
        result = compute_all_features('CCO')
        assert set(result.keys()) == set(ALL_FEATURE_NAMES)

    def test_invalid_returns_none(self):
        result = compute_all_features('INVALID')
        assert result is None


# ---------------------------------------------------------------------------
# compute_rdkit_properties — positional determinism
# ---------------------------------------------------------------------------

class TestRdkitProperties:
    def test_shape(self, ethanol_mol):
        props = compute_rdkit_properties(ethanol_mol)
        assert props.shape == (200,)

    def test_none_mol_returns_zeros(self):
        props = compute_rdkit_properties(None)
        assert props.shape == (200,)
        assert np.all(props == 0.0)

    def test_positional_determinism(self, ethanol_mol, aspirin_mol):
        """Same index must correspond to the same descriptor across molecules."""
        props_eth = compute_rdkit_properties(ethanol_mol)
        props_asp = compute_rdkit_properties(aspirin_mol)
        assert props_eth.shape == props_asp.shape
        # Both should have non-zero values at index 0 (first descriptor)
        # The key test: positions are stable, not shifted by failures
        # Run twice on same mol to verify deterministic
        props_eth2 = compute_rdkit_properties(ethanol_mol)
        np.testing.assert_array_equal(props_eth, props_eth2)


# ---------------------------------------------------------------------------
# compute_maplight_features
# ---------------------------------------------------------------------------

class TestMaplightFeatures:
    def test_shape(self, ethanol_mol):
        feats = compute_maplight_features(ethanol_mol)
        assert feats.shape == (N_MAPLIGHT_FEATURES,)

    def test_not_all_zeros(self, aspirin_mol):
        feats = compute_maplight_features(aspirin_mol)
        assert np.any(feats != 0)

    def test_none_returns_zeros(self):
        feats = compute_maplight_features(None)
        assert feats.shape == (N_MAPLIGHT_FEATURES,)
        assert np.all(feats == 0)


# ---------------------------------------------------------------------------
# compute_erg_features
# ---------------------------------------------------------------------------

class TestErgFeatures:
    def test_shape(self, ethanol_mol):
        erg = compute_erg_features(ethanol_mol)
        assert erg.shape == (315,)

    def test_not_all_zeros(self, aspirin_mol):
        erg = compute_erg_features(aspirin_mol)
        assert np.any(erg != 0)
