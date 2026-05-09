"""
Path B — Custom Geometric Embeddings.

For each residue with a backbone amide (N, CA, C atoms):
  1. Define neighbors: through-space (5 Å from N) ∪ sequence (i±3).
  2. Build a local coordinate frame anchored on N, Cα, and C(carbonyl).
  3. Rotate all neighbor coordinates into the local frame.
  4. Encode element as one-hot, append in_space / in_seq flags.
  5. Pad / truncate neighbor list to max_neighbors; flatten to 1-D vector.

Feature dimension per residue = max_neighbors × 14
  (3 coord_mean + 3 coord_std + 1 in_space + 1 in_seq + 6 element_onehot)
Default max_neighbors=128 → feature_dim = 1792.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

ELEMENTS = ["C", "N", "O", "S", "H"]  # last bin = "other"
FEATURES_PER_NEIGHBOR = 3 + 3 + 1 + 1 + (len(ELEMENTS) + 1)  # = 14


def one_hot_element(element: str) -> list[float]:
    vec = [0.0] * (len(ELEMENTS) + 1)
    e = element.strip().upper()[:1]
    if e in ELEMENTS:
        vec[ELEMENTS.index(e)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


def build_local_frame(N_coord: np.ndarray, Ca_coord: np.ndarray, C_coord: np.ndarray) -> np.ndarray:
    """
    Returns 3×3 rotation matrix R such that:
      R @ (Ca - N) is along +X (Y=0, Z=0)
      R @ (C  - N) has Z=0, Y > 0
    """
    ca = Ca_coord - N_coord
    c  = C_coord  - N_coord

    # R1a: rotate about Z to align ca_XY onto +X
    ca_xy = np.array([ca[0], ca[1], 0.0])
    norm_xy = np.linalg.norm(ca_xy)
    if norm_xy < 1e-8:
        Rz = np.eye(3)
    else:
        cos_z =  ca_xy[0] / norm_xy
        sin_z = -ca_xy[1] / norm_xy
        Rz = np.array([[cos_z, -sin_z, 0.0],
                        [sin_z,  cos_z, 0.0],
                        [0.0,    0.0,   1.0]])

    ca1 = Rz @ ca

    # R1b: rotate about Y to eliminate Z component of ca1
    norm_xz = np.sqrt(ca1[0]**2 + ca1[2]**2)
    if norm_xz < 1e-8:
        Ry = np.eye(3)
    else:
        cos_y =  ca1[0] / norm_xz
        sin_y =  ca1[2] / norm_xz
        Ry = np.array([[ cos_y, 0.0, sin_y],
                        [ 0.0,  1.0, 0.0  ],
                        [-sin_y, 0.0, cos_y]])

    R1 = Ry @ Rz

    # R2: rotate about X so C falls onto +Y half-plane (Z=0)
    c1 = R1 @ c
    norm_yz = np.sqrt(c1[1]**2 + c1[2]**2)
    if norm_yz < 1e-8:
        Rx = np.eye(3)
    else:
        cos_x =  c1[1] / norm_yz
        sin_x = -c1[2] / norm_yz
        Rx = np.array([[1.0, 0.0,    0.0   ],
                        [0.0, cos_x, -sin_x],
                        [0.0, sin_x,  cos_x]])

    return Rx @ R1  # (3, 3)


def _get_coord(atom_list: list[dict], name: str) -> Optional[np.ndarray]:
    for a in atom_list:
        if a["name"] == name:
            return a["coord"]
    return None


def compute_residue_features(
    residue_key: tuple,
    residue_atoms: dict[tuple, list[dict]],
    seq_order: list[tuple],  # ordered list of all residue keys in this chain
    space_distance_threshold: float = 5.0,
    seq_neighbor_range: int = 3,
    max_neighbors: int = 128,
    neighbor_flexibility_cutoff: float = 1.5,
) -> Optional[np.ndarray]:
    """
    Compute feature vector for one residue.

    Returns np.ndarray of shape (max_neighbors * FEATURES_PER_NEIGHBOR,)
    or None if backbone atoms N / CA / C are missing.
    """
    if residue_key not in residue_atoms:
        return None

    my_atoms = residue_atoms[residue_key]
    N_coord  = _get_coord(my_atoms, "N")
    Ca_coord = _get_coord(my_atoms, "CA")
    C_coord  = _get_coord(my_atoms, "C")

    if N_coord is None or Ca_coord is None or C_coord is None:
        return None

    R = build_local_frame(N_coord, Ca_coord, C_coord)

    # ── Build neighbor atom list ───────────────────────────────────────
    chain_id, seq_id, _ = residue_key
    my_idx = next((i for i, k in enumerate(seq_order) if k == residue_key), None)

    neighbors: list[dict] = []
    seen: set[tuple] = set()  # (res_key, atom_name)

    for other_key, other_atoms in residue_atoms.items():
        if other_key == residue_key:
            continue
        other_chain, other_seq, _ = other_key
        if other_chain != chain_id:
            continue

        # Compute flags
        in_seq = 0
        if my_idx is not None:
            other_idx = next((i for i, k in enumerate(seq_order) if k == other_key), None)
            if other_idx is not None and abs(other_idx - my_idx) <= seq_neighbor_range:
                in_seq = 1

        for atom in other_atoms:
            atom_key = (other_key, atom["name"])
            if atom_key in seen:
                continue

            coord = atom["coord"]
            dist = np.linalg.norm(coord - N_coord)
            in_space = 1 if dist <= space_distance_threshold else 0

            if in_space == 0 and in_seq == 0:
                continue

            # Skip highly flexible neighbors
            if atom.get("rmsd", 0.0) > neighbor_flexibility_cutoff:
                continue

            seen.add(atom_key)
            neighbors.append({
                "coord":      coord,
                "coord_std":  atom.get("coord_std", np.zeros(3)),
                "element":    atom["element"],
                "in_space":   float(in_space),
                "in_seq":     float(in_seq),
            })

    # ── Sort by distance for deterministic ordering ───────────────────
    neighbors.sort(key=lambda nb: np.linalg.norm(nb["coord"] - N_coord))

    # ── Encode features ───────────────────────────────────────────────
    feature_rows: list[list[float]] = []
    for nb in neighbors[:max_neighbors]:
        local_coord = R @ (nb["coord"] - N_coord)
        local_std   = R @ nb["coord_std"]
        row = [
            *local_coord.tolist(),
            *local_std.tolist(),
            nb["in_space"],
            nb["in_seq"],
            *one_hot_element(nb["element"]),
        ]
        feature_rows.append(row)

    # Pad with zeros to max_neighbors
    while len(feature_rows) < max_neighbors:
        feature_rows.append([0.0] * FEATURES_PER_NEIGHBOR)

    return np.array(feature_rows[:max_neighbors], dtype=np.float32).flatten()


def embed_protein(
    residue_atoms: dict[tuple, list[dict]],
    target_keys: list[tuple],  # residues that have HSQC labels
    flexible_ok: Optional[set[tuple]] = None,
    config: Optional[dict] = None,
) -> dict[tuple, np.ndarray]:
    """
    Compute custom geometric features for all residues in target_keys.

    Args:
        residue_atoms   : full atom dictionary for the protein
        target_keys     : residue keys with available HSQC shifts
        flexible_ok     : set of residues passing flexibility filter (None = no filter)
        config          : embedding config dict (keys: neighbor_distance_threshold_angstrom, etc.)

    Returns:
        { residue_key: feature_vector }  — only for successfully embedded residues
    """
    cfg = config or {}
    dist_thresh = cfg.get("neighbor_distance_threshold_angstrom", 5.0)
    seq_range   = cfg.get("sequence_neighbor_range", 3)
    max_nb      = cfg.get("max_neighbors", 128)
    nb_cutoff   = cfg.get("neighbor_flexibility_cutoff_angstrom", 1.5)

    # Build per-chain sequence order from residue_atoms keys
    chain_orders: dict[str, list[tuple]] = defaultdict(list)
    for key in sorted(residue_atoms.keys(), key=lambda k: k[1]):
        chain_orders[key[0]].append(key)
    seq_order = [k for keys in chain_orders.values() for k in keys]

    features = {}
    for key in target_keys:
        if flexible_ok is not None and key not in flexible_ok:
            continue
        vec = compute_residue_features(
            key, residue_atoms, seq_order,
            space_distance_threshold=dist_thresh,
            seq_neighbor_range=seq_range,
            max_neighbors=max_nb,
            neighbor_flexibility_cutoff=nb_cutoff,
        )
        if vec is not None:
            features[key] = vec

    return features
