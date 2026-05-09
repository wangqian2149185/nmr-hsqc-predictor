"""
Parse PDB files using BioPython.

Handles both single-model (X-ray) and multi-model (NMR ensemble) structures.
For NMR ensembles:
  - Computes per-atom mean coordinate and per-axis std across all models.
  - Computes per-residue backbone RMSD for flexibility filtering.
For X-ray structures: std is zero, RMSD is zero (single model).
"""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser

log = logging.getLogger(__name__)
_parser = PDBParser(QUIET=True)

BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def load_structure(pdb_file: str, pdb_id: str = "X"):
    return _parser.get_structure(pdb_id, pdb_file)


# ── Single-model helper (X-ray) ───────────────────────────────────────────────

def get_residue_atoms(structure) -> dict[tuple, list[dict]]:
    """
    Returns:
        { (chain_id, seq_id, res_name): [ {name, coord, element, coord_std} ] }

    coord_std is np.zeros(3) for single-model structures.
    """
    result = {}
    for model in structure:
        for chain in model:
            for res in chain:
                hetflag, seq_id, _ = res.id
                if hetflag.strip():
                    continue
                atoms = [
                    {
                        "name": atom.name,
                        "coord": atom.coord.copy(),
                        "coord_std": np.zeros(3, dtype=np.float32),
                        "element": (atom.element or atom.name[0]).upper(),
                        "rmsd": 0.0,
                    }
                    for atom in res.get_atoms()
                ]
                key = (chain.id, seq_id, res.resname.strip())
                result[key] = atoms
        break  # first model only for X-ray
    return result


# ── Multi-model helper (NMR ensemble) ─────────────────────────────────────────

def compute_ensemble_stats(pdb_file: str, pdb_id: str = "X") -> dict[tuple, dict]:
    """
    For NMR ensemble PDB files.

    Returns per-atom stats:
        { (chain_id, seq_id, res_name, atom_name):
            { mean: np.ndarray(3,), std: np.ndarray(3,), rmsd: float, n: int } }
    """
    structure = load_structure(pdb_file, pdb_id)
    models = list(structure.get_models())

    coords_by_atom: dict[tuple, list] = defaultdict(list)
    for model in models:
        for chain in model:
            for res in chain:
                hetflag, seq_id, _ = res.id
                if hetflag.strip():
                    continue
                for atom in res.get_atoms():
                    key = (chain.id, seq_id, res.resname.strip(), atom.name)
                    coords_by_atom[key].append(atom.coord.copy())

    stats = {}
    for key, coord_list in coords_by_atom.items():
        arr = np.array(coord_list, dtype=np.float32)  # (n_models, 3)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        rmsd = float(np.sqrt(((arr - mean) ** 2).sum(axis=1).mean()))
        stats[key] = {"mean": mean, "std": std, "rmsd": rmsd, "n": len(coord_list)}
    return stats


def get_residue_atoms_ensemble(
    ensemble_stats: dict[tuple, dict],
) -> dict[tuple, list[dict]]:
    """
    Convert ensemble atom stats → per-residue atom list format
    (same schema as get_residue_atoms, but with real coord_std).
    """
    # Group by (chain_id, seq_id, res_name)
    residues: dict[tuple, list[dict]] = defaultdict(list)
    for (chain_id, seq_id, res_name, atom_name), s in ensemble_stats.items():
        key = (chain_id, seq_id, res_name)
        residues[key].append({
            "name": atom_name,
            "coord": s["mean"],
            "coord_std": s["std"],
            "element": atom_name[0].upper() if atom_name else "C",
            "rmsd": s["rmsd"],
        })
    return dict(residues)


def filter_by_flexibility(
    ensemble_stats: dict[tuple, dict],
    backbone_cutoff: float = 1.0,
) -> set[tuple]:
    """
    Returns set of (chain_id, seq_id, res_name) whose backbone RMSD is below threshold.
    """
    bb_rmsds: dict[tuple, list[float]] = defaultdict(list)
    for (chain_id, seq_id, res_name, atom_name), s in ensemble_stats.items():
        if atom_name in BACKBONE_ATOMS:
            bb_rmsds[(chain_id, seq_id, res_name)].append(s["rmsd"])

    return {
        res_key
        for res_key, rmsds in bb_rmsds.items()
        if rmsds and np.mean(rmsds) < backbone_cutoff
    }


def load_residue_atoms(
    pdb_file: str,
    pdb_id: str = "X",
    backbone_cutoff: float = 1.0,
) -> tuple[dict[tuple, list[dict]], Optional[set[tuple]]]:
    """
    Unified loader — automatically detects NMR ensemble (>1 model) vs. X-ray.

    Returns:
        residue_atoms : { (chain_id, seq_id, res_name): [ atom_dict ] }
        flexible_ok   : set of residue keys passing RMSD filter (None for X-ray)
    """
    structure = load_structure(pdb_file, pdb_id)
    n_models = len(list(structure.get_models()))

    if n_models > 1:
        stats = compute_ensemble_stats(pdb_file, pdb_id)
        residue_atoms = get_residue_atoms_ensemble(stats)
        flexible_ok = filter_by_flexibility(stats, backbone_cutoff)
        return residue_atoms, flexible_ok
    else:
        residue_atoms = get_residue_atoms(structure)
        return residue_atoms, None  # no flexibility filter for X-ray
