"""
BIO88 Feature Generation from SMILES

Computes 88 molecular features from SMILES strings:
  - BIO30: 30 RDKit physicochemical descriptors
  - Pharmacophore: 48 pharmacophore distance/count features
  - Structural: 10 molecular graph topology features

Usage:
    from process_bio88 import compute_all_features
    features = compute_all_features("CCO")  # returns dict with 88 features

CLI:
    python process_bio88.py -i input.csv -o output.csv
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import os
import argparse
import time
import re
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdReducedGraphs
from rdkit.Avalon import pyAvalonTools

RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Timeout for individual molecule processing (seconds)
MOLECULE_TIMEOUT = 300  # 5 minutes

BIO30_FEATURE_NAMES = [
    "MolWt",
    "ExactMolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "HeavyAtomCount",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "FractionCSP3",
    "LabuteASA",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "NumAromaticHeterocycles",
    "NumAliphaticHeterocycles",
    "LipinskiRO5Violations",
    "VeberViolations",
    "HBondDonorFraction",
    "HBondAcceptorFraction",
    "AromaticAtomFraction",
    "HeteroAtomFraction",
    "TPSAperHeavyAtom",
    "LabuteASAperHeavyAtom",
    "FlexibilityIndex",
    "PolarityLipophilicityBalance",
]

# Pharmacophore SMARTS patterns
PHARMACOPHORE_SMARTS = {
    'donor': '[#7,#8][H]',                # H-bond donors (N-H, O-H)
    'acceptor': '[#7,#8;!$([#7,#8][H])]', # H-bond acceptors (N, O without H)
    'hydrophobic': '[C;!$(C=O);!$(C#N)]', # Hydrophobic carbons
    'aromatic': 'a',                       # Aromatic atoms
    'positive': '[+;!$([*]~[-])]',         # Positively charged
    'negative': '[-;!$([*]~[+])]',         # Negatively charged
}

PHARM_TYPES = ['donor', 'acceptor', 'hydrophobic', 'aromatic', 'positive', 'negative']

# Structural Graph Features (10 features)
STRUCTURAL_FEATURE_NAMES = [
    "Struct_aromatic_ratio",
    "Struct_conjugated_ratio",
    "Struct_avg_electronegativity",
    "Struct_edge_density",
    "Struct_clustering_coef",
    "Struct_avg_degree",
    "Struct_n_paths_2",
    "Struct_n_nitrogen_ratio",
    "Struct_n_oxygen_ratio",
    "Struct_n_halogen_ratio",
]

# Electronegativity values (Pauling scale)
ELECTRONEGATIVITY = {
    1: 2.20,   # H
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    35: 2.96,  # Br
    53: 2.66,  # I
}

HALOGEN_ATOMIC_NUMS = {9, 17, 35, 53}  # F, Cl, Br, I

# All 88 feature names
ALL_FEATURE_NAMES = BIO30_FEATURE_NAMES.copy()

# Generate pharmacophore feature names
PHARMACOPHORE_FEATURE_NAMES = []
# Counts (6)
for _ptype in PHARM_TYPES:
    PHARMACOPHORE_FEATURE_NAMES.append(f"Pharm_{_ptype}_count")
# Distance stats for each pair (21 pairs x 2 stats = 42)
for _i, _t1 in enumerate(PHARM_TYPES):
    for _t2 in PHARM_TYPES[_i:]:
        PHARMACOPHORE_FEATURE_NAMES.append(f"Pharm_{_t1}_{_t2}_min_dist")
        PHARMACOPHORE_FEATURE_NAMES.append(f"Pharm_{_t1}_{_t2}_mean_dist")

ALL_FEATURE_NAMES += PHARMACOPHORE_FEATURE_NAMES
ALL_FEATURE_NAMES += STRUCTURAL_FEATURE_NAMES

# Compile SMARTS patterns at module load
COMPILED_SMARTS = {}
for _name, _smarts in PHARMACOPHORE_SMARTS.items():
    _pattern = Chem.MolFromSmarts(_smarts)
    if _pattern is not None:
        COMPILED_SMARTS[_name] = _pattern


def fix_smiles(smiles: str) -> Optional[Any]:
    """Attempt to fix/standardize a SMILES string using RDKit's MolStandardize."""
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            cleaned = smiles
            cleaned = re.sub(r'\[([A-Z][a-z]?)([+-])H(\d*)\]', r'[\1H\3\2]', cleaned)
            cleaned = re.sub(r'\[([A-Z][a-z]?)H(\d*)\+\]', r'[\1H\2+]', cleaned)
            cleaned = re.sub(r'\[([A-Z][a-z]?)\+H(\d*)\]', r'[\1H\2+]', cleaned)
            cleaned = re.sub(r'\[([A-Z][a-z]?)-H(\d*)\]', r'[\1H\2-]', cleaned)
            mol = Chem.MolFromSmiles(cleaned)

        if mol is None:
            return None

        largest_frag = rdMolStandardize.LargestFragmentChooser()
        mol = largest_frag.choose(mol)

        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        Chem.SanitizeMol(mol)
        return mol

    except (ValueError, RuntimeError) as e:
        import logging
        logging.getLogger(__name__).debug("fix_smiles failed for '%s': %s",
                                          smiles[:80] if smiles else '', e)
        return None


def _compute_bio30(mol: Any) -> Dict[str, float]:
    """Compute 30 RDKit physicochemical descriptors from a Mol object."""
    mw = Descriptors.MolWt(mol)
    exact_mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    mr = Descriptors.MolMR(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_aliph_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
    n_sat_rings = rdMolDescriptors.CalcNumSaturatedRings(mol)
    frac_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    asa = rdMolDescriptors.CalcLabuteASA(mol)
    n_hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    n_nhoh = Descriptors.NHOHCount(mol)
    n_no = Descriptors.NOCount(mol)
    n_arom_hetero = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    n_aliph_hetero = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)

    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    heavy_safe = max(heavy, 1)
    denom_h = hbd + hba + 1e-6

    lipinski_viol = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    veber_viol = int(n_rot > 10) + int(tpsa > 140)

    return {
        "MolWt": mw,
        "ExactMolWt": exact_mw,
        "MolLogP": logp,
        "MolMR": mr,
        "TPSA": tpsa,
        "NumHAcceptors": hba,
        "NumHDonors": hbd,
        "NumRotatableBonds": n_rot,
        "HeavyAtomCount": heavy,
        "RingCount": n_rings,
        "NumAromaticRings": n_arom_rings,
        "NumAliphaticRings": n_aliph_rings,
        "NumSaturatedRings": n_sat_rings,
        "FractionCSP3": frac_sp3,
        "LabuteASA": asa,
        "NumHeteroatoms": n_hetero,
        "NHOHCount": n_nhoh,
        "NOCount": n_no,
        "NumAromaticHeterocycles": n_arom_hetero,
        "NumAliphaticHeterocycles": n_aliph_hetero,
        "LipinskiRO5Violations": lipinski_viol,
        "VeberViolations": veber_viol,
        "HBondDonorFraction": hbd / denom_h,
        "HBondAcceptorFraction": hba / denom_h,
        "AromaticAtomFraction": aromatic_atoms / heavy_safe,
        "HeteroAtomFraction": n_hetero / heavy_safe,
        "TPSAperHeavyAtom": tpsa / heavy_safe,
        "LabuteASAperHeavyAtom": asa / heavy_safe,
        "FlexibilityIndex": n_rot / heavy_safe,
        "PolarityLipophilicityBalance": tpsa / (logp + 1.0) if logp is not None else 0.0,
    }


def compute_pharmacophore_features(mol: Any) -> Dict[str, float]:
    """Compute 48 pharmacophore features (6 counts + 42 distance stats)."""
    features = {name: 0.0 for name in PHARMACOPHORE_FEATURE_NAMES}

    if mol is None:
        return features

    pharm_atoms = {}
    for name, pattern in COMPILED_SMARTS.items():
        matches = mol.GetSubstructMatches(pattern)
        atoms = [m[0] for m in matches]
        pharm_atoms[name] = atoms
        features[f"Pharm_{name}_count"] = len(atoms)

    try:
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol_3d, params)

        if result != 0:
            return features

        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
        conf = mol_3d.GetConformer()

        pharm_atoms_3d = {}
        for name, pattern in COMPILED_SMARTS.items():
            matches = mol_3d.GetSubstructMatches(pattern)
            pharm_atoms_3d[name] = [m[0] for m in matches]

        for i, t1 in enumerate(PHARM_TYPES):
            for t2 in PHARM_TYPES[i:]:
                atoms1 = pharm_atoms_3d.get(t1, [])
                atoms2 = pharm_atoms_3d.get(t2, [])

                if len(atoms1) == 0 or len(atoms2) == 0:
                    continue

                distances = []
                for a1 in atoms1:
                    pos1 = conf.GetAtomPosition(a1)
                    for a2 in atoms2:
                        if t1 == t2 and a1 >= a2:
                            continue
                        pos2 = conf.GetAtomPosition(a2)
                        distances.append(pos1.Distance(pos2))

                if distances:
                    features[f"Pharm_{t1}_{t2}_min_dist"] = min(distances)
                    features[f"Pharm_{t1}_{t2}_mean_dist"] = sum(distances) / len(distances)

    except (ValueError, RuntimeError) as e:
        import logging
        logging.getLogger(__name__).debug("3D pharmacophore embedding failed: %s", e)

    return features


def compute_structural_features(mol: Any) -> Dict[str, float]:
    """Compute 10 structural graph features."""
    features = {name: 0.0 for name in STRUCTURAL_FEATURE_NAMES}

    if mol is None:
        return features

    try:
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if n_atoms == 0:
            return features

        # Composition (3)
        n_nitrogen = sum(1 for a in atoms if a.GetAtomicNum() == 7)
        n_oxygen = sum(1 for a in atoms if a.GetAtomicNum() == 8)
        n_halogen = sum(1 for a in atoms if a.GetAtomicNum() in HALOGEN_ATOMIC_NUMS)

        features["Struct_n_nitrogen_ratio"] = n_nitrogen / n_atoms
        features["Struct_n_oxygen_ratio"] = n_oxygen / n_atoms
        features["Struct_n_halogen_ratio"] = n_halogen / n_atoms

        # Electronic (3)
        n_aromatic = sum(1 for a in atoms if a.GetIsAromatic())
        features["Struct_aromatic_ratio"] = n_aromatic / n_atoms

        if n_bonds > 0:
            n_conjugated = sum(1 for b in bonds if b.GetIsConjugated())
            features["Struct_conjugated_ratio"] = n_conjugated / n_bonds

        electroneg_sum = 0.0
        for a in atoms:
            electroneg_sum += ELECTRONEGATIVITY.get(a.GetAtomicNum(), 2.5)
        features["Struct_avg_electronegativity"] = electroneg_sum / n_atoms

        # Topological (4)
        max_edges = n_atoms * (n_atoms - 1) / 2 if n_atoms > 1 else 1
        features["Struct_edge_density"] = n_bonds / max_edges

        adj = {i: set() for i in range(n_atoms)}
        for bond in bonds:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[i].add(j)
            adj[j].add(i)

        degrees = [len(neighbors) for neighbors in adj.values()]
        avg_degree = sum(degrees) / n_atoms if n_atoms > 0 else 0
        features["Struct_avg_degree"] = avg_degree / 4.0

        clustering_sum = 0.0
        for i in range(n_atoms):
            neighbors = list(adj[i])
            k = len(neighbors)
            if k < 2:
                continue
            edges_between = 0
            for ni in range(len(neighbors)):
                for nj in range(ni + 1, len(neighbors)):
                    if neighbors[nj] in adj[neighbors[ni]]:
                        edges_between += 1
            max_neighbor_edges = k * (k - 1) / 2
            clustering_sum += edges_between / max_neighbor_edges
        features["Struct_clustering_coef"] = clustering_sum / n_atoms if n_atoms > 0 else 0

        n_paths_2 = 0
        for i in range(n_atoms):
            k = len(adj[i])
            n_paths_2 += k * (k - 1) // 2
        features["Struct_n_paths_2"] = n_paths_2 / 10.0

    except (ValueError, RuntimeError, ZeroDivisionError) as e:
        import logging
        logging.getLogger(__name__).warning("Structural feature computation failed: %s", e)

    return features


def compute_all_features(smiles: str) -> Optional[Dict[str, float]]:
    """
    Compute all 88 features from a SMILES string.

    Returns dict with 88 features (BIO30 + pharmacophore + structural),
    or None if SMILES parsing fails.
    """
    try:
        mol = fix_smiles(smiles)
        if mol is None:
            return None

        bio30 = _compute_bio30(mol)
        pharm = compute_pharmacophore_features(mol)
        struct = compute_structural_features(mol)

        bio30.update(pharm)
        bio30.update(struct)
        return bio30

    except (ValueError, RuntimeError) as e:
        import logging
        logging.getLogger(__name__).warning("compute_all_features failed: %s", e)
        return None
    except Exception:
        import logging
        logging.getLogger(__name__).error("Unexpected error in compute_all_features",
                                          exc_info=True)
        return None


def _process_single(item: Tuple[int, str]) -> Tuple[int, Optional[Dict[str, float]]]:
    """Process single indexed SMILES for parallel execution."""
    idx, smi = item
    return idx, compute_all_features(smi)


# ============================================================================
# MapLight Features — 2563-dim (ECFP 1024 + Avalon 1024 + ERG 315 + RDKit 200)
# From: arXiv:2310.00174 (MapLight)
# ============================================================================

N_ECFP_FEATURES = 1024
N_AVALON_FEATURES = 1024
N_ERG_FEATURES = 315
N_RDKIT_PROP_FEATURES = 200
N_MAPLIGHT_FEATURES = N_ECFP_FEATURES + N_AVALON_FEATURES + N_ERG_FEATURES + N_RDKIT_PROP_FEATURES  # 2563


def compute_ecfp_counts(mol: Any, radius: int = 2, nBits: int = 1024) -> np.ndarray:
    """Compute ECFP (Morgan) count fingerprint — count-based, not binary."""
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
    arr = np.zeros(nBits, dtype=np.float32)
    for idx, count in fp.GetNonzeroElements().items():
        arr[idx] = count
    return arr


def compute_avalon_counts(mol: Any, nBits: int = 1024) -> np.ndarray:
    """Compute Avalon count fingerprint — count-based, not binary."""
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=nBits)
    arr = np.zeros(nBits, dtype=np.float32)
    for i in range(nBits):
        arr[i] = fp[i]
    return arr


def compute_erg_features(mol: Any) -> np.ndarray:
    """
    Compute 315-dim Extended Reduced Graph (ErG) fingerprint.

    Uses RDKit's rdReducedGraphs.GetErGFingerprint which reduces the molecule
    to a pharmacophore graph and computes fuzzy property distributions over
    reduced graph distances (Stiefl et al., J. Chem. Inf. Model. 2006).
    """
    if mol is None:
        return np.zeros(N_ERG_FEATURES, dtype=np.float32)

    try:
        erg = rdReducedGraphs.GetErGFingerprint(mol)
        return np.array(erg, dtype=np.float32)
    except (ValueError, RuntimeError) as e:
        import logging
        logging.getLogger(__name__).debug("ErG fingerprint failed: %s", e)
        return np.zeros(N_ERG_FEATURES, dtype=np.float32)


def compute_composite_erg_features(mol: Any) -> np.ndarray:
    """
    Composite 315-dim structural fingerprint (NOT real ErG).

    Atom pair FP (128) + topological torsion FP (128) + feature Morgan FP (59).
    Binary bits, not continuous. Slightly outperformed real ErG on CYP3A4:
      composite: 0.697 ± 0.011 AUROC
      real ErG:  0.690 ± 0.013 AUROC
    """
    if mol is None:
        return np.zeros(N_ERG_FEATURES, dtype=np.float32)
    try:
        ap_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=128)
        ap = np.array(ap_fp, dtype=np.float32)
        tt_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=128)
        tt = np.array(tt_fp, dtype=np.float32)
        fm_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=59, useFeatures=True)
        fm = np.array(fm_fp, dtype=np.float32)
        return np.concatenate([ap, tt, fm])
    except Exception:
        return np.zeros(N_ERG_FEATURES, dtype=np.float32)


def get_maplight_descriptors():
    """Return MapLight's curated list of 200 RDKit descriptor names.

    From arXiv:2310.00174 (MapLight-TDC).
    """
    return [
        'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1',
        'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
        'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
        'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
        'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',
        'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount',
        'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
        'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
        'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge',
        'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
        'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
        'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors',
        'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds',
        'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
        'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
        'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
        'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1',
        'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
        'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
        'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
        'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
        'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
        'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
        'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
        'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
        'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH',
        'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
        'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo',
        'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
        'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
        'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
        'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
        'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
        'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
        'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid',
        'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
        'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone',
        'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
        'fr_unbrch_alkane', 'fr_urea', 'qed',
    ]


# Build calculator once at module load
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
_MAPLIGHT_CALCULATOR = MolecularDescriptorCalculator(get_maplight_descriptors())


def compute_rdkit_properties(mol: Any) -> np.ndarray:
    """Compute 200 RDKit molecular descriptors (continuous values).

    Uses MapLight's curated descriptor list (arXiv:2310.00174) via
    MolecularDescriptorCalculator for consistent feature alignment.
    """
    if mol is None:
        return np.zeros(N_RDKIT_PROP_FEATURES, dtype=np.float32)

    try:
        vals = _MAPLIGHT_CALCULATOR.CalcDescriptors(mol)
        props = np.array(vals, dtype=np.float32)
        props = np.nan_to_num(props, nan=0.0, posinf=0.0, neginf=0.0)
        return props
    except (ValueError, RuntimeError, ZeroDivisionError, OverflowError):
        return np.zeros(N_RDKIT_PROP_FEATURES, dtype=np.float32)


def compute_maplight_features(mol: Any) -> np.ndarray:
    """
    Compute full MapLight feature vector (2563-dim).
    ECFP counts (1024) + Avalon counts (1024) + ERG (315) + RDKit props (200).
    """
    ecfp = compute_ecfp_counts(mol)
    avalon = compute_avalon_counts(mol)
    erg = compute_erg_features(mol)
    props = compute_rdkit_properties(mol)
    return np.concatenate([ecfp, avalon, erg, props])


def _process_maplight_single(item: Tuple[int, str]) -> Tuple[int, Optional[np.ndarray]]:
    """Process single indexed SMILES for MapLight features (parallel execution)."""
    idx, smi = item
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return idx, None
    return idx, compute_maplight_features(mol)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute BIO88 features (30 bio + 48 pharmacophore + 10 structural) from SMILES."
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    parser.add_argument("--smiles-column", "-s", default=None, help="Name of the SMILES column")
    parser.add_argument("--processes", "-p", type=int, default=None, help="Number of parallel processes")
    parser.add_argument("--chunk-size", "-c", type=int, default=50, help="Chunk size for parallel processing")
    args = parser.parse_args()

    print(f"Reading input file: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")

    # Auto-detect SMILES column
    smiles_col = args.smiles_column
    if smiles_col is None:
        for col_name in ["smiles", "SMILES", "Smiles", "canonical_smiles", "fixed_smiles", "Drug"]:
            if col_name in df.columns:
                smiles_col = col_name
                break
        if smiles_col is None:
            raise ValueError(f"Could not find SMILES column. Available columns: {list(df.columns)}")
    print(f"Using SMILES column: {smiles_col}")

    n_processes = args.processes or max(1, cpu_count() - 1)
    print(f"Using {n_processes} processes — computing all 88 features")

    smiles_list = df[smiles_col].tolist()
    all_results = [None] * len(smiles_list)
    work_items = list(enumerate(smiles_list))

    with Pool(n_processes) as pool:
        pending = {}
        completed = 0
        next_submit = 0

        pbar = tqdm(total=len(smiles_list), desc="Processing molecules")

        while next_submit < len(smiles_list) and len(pending) < n_processes:
            idx, smi = work_items[next_submit]
            ar = pool.apply_async(_process_single, ((idx, smi),))
            pending[idx] = (ar, time.time())
            next_submit += 1

        while pending or next_submit < len(smiles_list):
            done_idxs = []
            for idx, (ar, start_time) in list(pending.items()):
                if ar.ready():
                    try:
                        result_idx, result = ar.get(timeout=0)
                        all_results[result_idx] = result
                    except Exception:
                        all_results[idx] = None
                    done_idxs.append(idx)
                    completed += 1
                    pbar.update(1)
                elif time.time() - start_time > MOLECULE_TIMEOUT:
                    print(f"\nTimeout: Skipping molecule {idx} after {MOLECULE_TIMEOUT}s")
                    all_results[idx] = None
                    done_idxs.append(idx)
                    completed += 1
                    pbar.update(1)

            for idx in done_idxs:
                del pending[idx]

            while next_submit < len(smiles_list) and len(pending) < n_processes:
                idx, smi = work_items[next_submit]
                ar = pool.apply_async(_process_single, ((idx, smi),))
                pending[idx] = (ar, time.time())
                next_submit += 1

            if pending:
                time.sleep(0.1)

        pbar.close()

    n_failures = sum(1 for r in all_results if r is None)
    if n_failures > 0:
        print(f"Warning: {n_failures} SMILES could not be parsed or timed out")

    valid_indices = [i for i, r in enumerate(all_results) if r is not None]
    valid_results = [all_results[i] for i in valid_indices]

    bio_df = pd.DataFrame(valid_results)

    output_df = pd.DataFrame()
    output_df[smiles_col] = [df[smiles_col].iloc[i] for i in valid_indices]
    if "Y" in df.columns:
        output_df["Y"] = [df["Y"].iloc[i] for i in valid_indices]
    for col in ALL_FEATURE_NAMES:
        output_df[col] = bio_df[col] if col in bio_df.columns else None

    rows_before = len(output_df)
    output_df = output_df.dropna()
    rows_after = len(output_df)
    if rows_before != rows_after:
        print(f"Dropped {rows_before - rows_after} rows with missing feature values")

    output_df.to_csv(args.output, index=False)
    print(f"Saved output to: {args.output}")
    print(f"Output shape: {output_df.shape}")


if __name__ == "__main__":
    main()
