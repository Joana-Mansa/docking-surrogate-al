"""Atom and bond featurisation.

Design note
-----------
Featurisation lives in its own module because it is the part of a GNN pipeline
most often changed and least often documented, and because a change here
invalidates every trained checkpoint. ``FEATURISER_VERSION`` is bumped on any
change to the feature set and is written into checkpoints, so loading a
checkpoint built with a different featuriser fails loudly instead of silently
mis-indexing feature columns.

What the features are, and why
------------------------------
The surrogate's job is to imitate a docking score. Docking scores are
dominated by shape complementarity, hydrogen bonding and hydrophobic burial,
so the features chosen are the 2D graph quantities that most directly proxy
those:

* **Element** (one-hot over the allowed set). Proxies atom radius, polarity
  and the oracle's own atom typing, which is element-based.
* **Degree** and **total number of Hs**. Steric bulk and H-bond donor count.
* **Formal charge** and **radical electrons**. Electrostatics; kept even
  though standardisation neutralises where possible, because zwitterions and
  quaternary nitrogens legitimately survive it.
* **Hybridisation** and **aromaticity**. Geometry and ring planarity.
* **In-ring**, and ring size membership for sizes 3 to 7. Conformational
  rigidity, which is what drives the entropic term in docking scoring.
* **Scaled atomic mass**, as one continuous feature, to break ties between
  halogens that otherwise differ only in their one-hot slot.

Bond features: bond type, conjugation, ring membership, stereochemistry.
These matter because the oracle docks a 3D conformer, and rotatable-bond count
(derivable from bond type plus ring membership) is an explicit term in the
smina scoring function.

Deliberately excluded: 3D coordinates and any descriptor computed from the
docked pose. Including those would leak the oracle's output into its own
input, and the surrogate's whole purpose is to predict the score *before*
paying for the pose.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

# Bump on ANY change to the feature set below.
FEATURISER_VERSION = "1.0.0"

ATOM_ELEMENTS: tuple[str, ...] = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B")
ATOM_DEGREES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
ATOM_NUM_HS: tuple[int, ...] = (0, 1, 2, 3, 4)
ATOM_FORMAL_CHARGES: tuple[int, ...] = (-2, -1, 0, 1, 2)
ATOM_HYBRIDISATIONS: tuple[Chem.HybridizationType, ...] = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
)
RING_SIZES: tuple[int, ...] = (3, 4, 5, 6, 7)

BOND_TYPES: tuple[Chem.BondType, ...] = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
BOND_STEREO: tuple[Chem.BondStereo, ...] = (
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
)


def _one_hot(value: Any, choices: tuple[Any, ...]) -> list[float]:
    """One-hot encode with a trailing "other" slot.

    The extra slot means an unseen element or hybridisation state degrades to
    a generic category instead of raising, which matters when the molecule set
    is scaled from 5k to 50k and rarer chemistry appears.
    """
    vector = [0.0] * (len(choices) + 1)
    try:
        vector[choices.index(value)] = 1.0
    except ValueError:
        vector[-1] = 1.0
    return vector


def atom_features(atom: Chem.Atom) -> list[float]:
    """Return the feature vector for one atom."""
    ring_info = atom.GetOwningMol().GetRingInfo()
    features: list[float] = []
    features += _one_hot(atom.GetSymbol(), ATOM_ELEMENTS)
    features += _one_hot(atom.GetTotalDegree(), ATOM_DEGREES)
    features += _one_hot(atom.GetTotalNumHs(includeNeighbors=True), ATOM_NUM_HS)
    features += _one_hot(atom.GetFormalCharge(), ATOM_FORMAL_CHARGES)
    features += _one_hot(atom.GetHybridization(), ATOM_HYBRIDISATIONS)
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.IsInRing()))
    features.append(float(atom.GetNumRadicalElectrons() > 0))
    features += [float(ring_info.IsAtomInRingOfSize(atom.GetIdx(), s)) for s in RING_SIZES]
    # Scaled to roughly unit range so it does not dominate the one-hots.
    features.append(atom.GetMass() / 100.0)
    return features


def bond_features(bond: Chem.Bond) -> list[float]:
    """Return the feature vector for one bond."""
    features: list[float] = []
    features += _one_hot(bond.GetBondType(), BOND_TYPES)
    features += _one_hot(bond.GetStereo(), BOND_STEREO)
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))
    return features


ATOM_FEATURE_DIM: int = (
    (len(ATOM_ELEMENTS) + 1)
    + (len(ATOM_DEGREES) + 1)
    + (len(ATOM_NUM_HS) + 1)
    + (len(ATOM_FORMAL_CHARGES) + 1)
    + (len(ATOM_HYBRIDISATIONS) + 1)
    + 3
    + len(RING_SIZES)
    + 1
)
BOND_FEATURE_DIM: int = (len(BOND_TYPES) + 1) + (len(BOND_STEREO) + 1) + 2


def mol_to_graph(
    smiles: str, y: float | None = None, index: int | None = None
) -> Data | None:
    """Convert a SMILES string to a PyTorch Geometric graph.

    Args:
        smiles: Canonical SMILES (already standardised).
        y: Target docking score, or None at prediction time.
        index: Position in the pool, carried through so predictions can be
            mapped back to molecules after shuffling. Stored as ``pool_id``,
            deliberately NOT ``pool_index``: PyTorch Geometric increments any
            attribute whose name contains "index" by the node offset when it
            builds a batch, treating it as an edge index. That silently
            corrupts a per-graph identifier, which is exactly the kind of
            misalignment that produces plausible but wrong results.

    Returns:
        A :class:`torch_geometric.data.Data` object, or None if the molecule
        cannot be parsed or has no bonds (a lone atom has no message passing
        to do and would produce an empty edge index).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumBonds() == 0:
        return None

    x = torch.tensor(
        np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32),
        dtype=torch.float32,
    )

    # Undirected graph stored as both directions, which is what message
    # passing expects.
    sources: list[int] = []
    targets: list[int] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_features(bond)
        sources += [i, j]
        targets += [j, i]
        edge_attrs += [features, features]

    data = Data(
        x=x,
        edge_index=torch.tensor([sources, targets], dtype=torch.long),
        edge_attr=torch.tensor(np.array(edge_attrs, dtype=np.float32), dtype=torch.float32),
    )
    data.smiles = smiles
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    if index is not None:
        data.pool_id = torch.tensor([int(index)], dtype=torch.long)
    return data


def feature_summary() -> dict[str, Any]:
    """Return a manifest-ready description of the featuriser."""
    return {
        "featuriser_version": FEATURISER_VERSION,
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "elements": list(ATOM_ELEMENTS),
    }
