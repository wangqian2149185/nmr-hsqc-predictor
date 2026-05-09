"""
Build PyTorch Geometric Data objects and protein-level train/val/test splits.

Graph structure (per protein):
  Nodes  = residues with HSQC labels (after flexibility filtering)
  Edges  = spatial (N atoms within 8 Å) + sequential (|i−j| ≤ edge_seq_range)
  x      = custom geometric feature vectors   shape (N_nodes, feature_dim)
  y      = (δ¹H, δ¹⁵N)                        shape (N_nodes, 2)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset

from src.data.embed_custom import embed_protein
from src.data.parse_bmrb import parse_hsqc_shifts, validate_referencing
from src.data.parse_pdb import load_residue_atoms

log = logging.getLogger(__name__)


# ── Graph construction helpers ────────────────────────────────────────────────

def _residue_N_coord(residue_atoms: dict, key: tuple) -> Optional[np.ndarray]:
    for atom in residue_atoms.get(key, []):
        if atom["name"] == "N":
            return atom["coord"]
    return None


def build_edges(
    node_keys: list[tuple],
    residue_atoms: dict[tuple, list[dict]],
    spatial_threshold: float = 8.0,
    seq_range: int = 3,
) -> torch.Tensor:
    """
    Returns edge_index of shape (2, E) — bidirectional.
    Connects nodes i,j if:
      - distance between their N atoms ≤ spatial_threshold, OR
      - |seq_id_i − seq_id_j| ≤ seq_range AND same chain
    """
    n = len(node_keys)
    coords = [_residue_N_coord(residue_atoms, k) for k in node_keys]

    src, dst = [], []
    for i in range(n):
        ci, si, _ = node_keys[i]
        for j in range(i + 1, n):
            cj, sj, _ = node_keys[j]

            connected = False
            if ci == cj and abs(si - sj) <= seq_range:
                connected = True
            if not connected and coords[i] is not None and coords[j] is not None:
                if np.linalg.norm(coords[i] - coords[j]) <= spatial_threshold:
                    connected = True

            if connected:
                src += [i, j]
                dst += [j, i]

    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


# ── Per-protein processing ────────────────────────────────────────────────────

def process_protein(
    bmrb_file: str,
    pdb_file: str,
    embed_config: dict,
    gcn_config: dict,
) -> Optional[Data]:
    """
    Parse BMRB + PDB, compute features, build graph.
    Returns torch_geometric.data.Data or None if insufficient data.
    """
    shifts = parse_hsqc_shifts(bmrb_file)
    if not shifts:
        return None

    ref_check = validate_referencing(shifts)
    if ref_check["out_of_range"] > ref_check["in_range"]:
        log.debug("Skipping %s: majority of shifts out of expected range", bmrb_file)
        return None

    backbone_cutoff = embed_config.get("flexibility_cutoff_angstrom", 1.0)
    residue_atoms, flexible_ok = load_residue_atoms(pdb_file, backbone_cutoff=backbone_cutoff)

    # Intersect BMRB keys with PDB keys
    target_keys = [k for k in shifts if k in residue_atoms]
    if not target_keys:
        return None

    features = embed_protein(
        residue_atoms,
        target_keys,
        flexible_ok=flexible_ok,
        config=embed_config,
    )
    if not features:
        return None

    # Align keys that survived embedding
    valid_keys = [k for k in target_keys if k in features]
    if len(valid_keys) < 3:
        return None

    X = np.stack([features[k] for k in valid_keys], axis=0)  # (N, F)
    y = np.array([shifts[k] for k in valid_keys], dtype=np.float32)  # (N, 2)

    edge_index = build_edges(
        valid_keys,
        residue_atoms,
        spatial_threshold=gcn_config.get("spatial_edge_threshold_angstrom", 8.0),
        seq_range=gcn_config.get("sequence_edge_range", 3),
    )

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float32),
        num_nodes=len(valid_keys),
    )
    return data


# ── Dataset class ─────────────────────────────────────────────────────────────

class HSQCDataset(Dataset):
    def __init__(self, data_list: list[Data]):
        super().__init__()
        self._data_list = data_list

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]


# ── Protein-level splits ──────────────────────────────────────────────────────

def split_proteins(
    protein_ids: list[str],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
) -> tuple[list[str], list[str], list[str]]:
    """70% train / 20% val / 10% test — split at protein level."""
    test_ratio = 1.0 - train_ratio - val_ratio
    train, tmp = train_test_split(protein_ids, test_size=(1 - train_ratio), random_state=seed)
    val_frac_of_tmp = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(tmp, test_size=(1 - val_frac_of_tmp), random_state=seed)
    return train, val, test


def save_splits(train_ids, val_ids, test_ids, split_dir: str):
    Path(split_dir).mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        with open(os.path.join(split_dir, f"{name}.txt"), "w") as fh:
            fh.write("\n".join(ids))


def load_splits(split_dir: str) -> tuple[list[str], list[str], list[str]]:
    results = []
    for name in ("train", "val", "test"):
        path = os.path.join(split_dir, f"{name}.txt")
        with open(path) as fh:
            results.append([line.strip() for line in fh if line.strip()])
    return tuple(results)


# ── Full pipeline runner ──────────────────────────────────────────────────────

def build_all_datasets(
    manifest_path: str,
    embed_config: dict,
    gcn_config: dict,
    cache_dir: str = "data/processed",
    split_dir: str = "data/splits",
    seed: int = 42,
    force_rebuild: bool = False,
) -> tuple[HSQCDataset, HSQCDataset, HSQCDataset]:
    """
    Build (or reload from cache) train/val/test HSQCDatasets.
    """
    cache_path = os.path.join(cache_dir, "graphs.pkl")

    if os.path.exists(cache_path) and not force_rebuild:
        log.info("Loading cached graphs from %s", cache_path)
        with open(cache_path, "rb") as fh:
            graph_map = pickle.load(fh)
    else:
        with open(manifest_path) as fh:
            manifest = json.load(fh)

        valid_entries = [r for r in manifest if r.get("ok")]
        log.info("Processing %d entries …", len(valid_entries))

        graph_map: dict[str, Data] = {}
        for entry in valid_entries:
            pid = entry["bmrb_id"]
            data = process_protein(
                entry["bmrb_path"],
                entry["pdb_path"],
                embed_config,
                gcn_config,
            )
            if data is not None:
                graph_map[pid] = data
            else:
                log.debug("Skipped %s", pid)

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(graph_map, fh)
        log.info("Cached %d graphs to %s", len(graph_map), cache_path)

    protein_ids = sorted(graph_map.keys())
    train_ids, val_ids, test_ids = split_proteins(protein_ids, seed=seed)
    save_splits(train_ids, val_ids, test_ids, split_dir)

    def _make(ids: list[str]) -> HSQCDataset:
        return HSQCDataset([graph_map[pid] for pid in ids if pid in graph_map])

    return _make(train_ids), _make(val_ids), _make(test_ids)
