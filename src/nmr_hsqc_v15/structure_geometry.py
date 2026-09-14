"""Label-numbered mmCIF geometry with explicit continuity checks."""

from pathlib import Path
import gzip
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .coordinate_primitives import select_residue_atoms, polymer_connection

AA3 = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY",
))


def normalize_missing(value):
    value = str(value).strip()
    return "" if value in {"", ".", "?"} else value


def read_structure_residues(path):
    """Return all polymer residues in the first encountered model."""
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt") as handle:
        cif = MMCIF2Dict(handle)

    fields = [
        "label_asym_id", "label_seq_id", "label_entity_id",
        "auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code",
        "label_comp_id", "label_atom_id", "label_alt_id",
        "occupancy", "Cartn_x", "Cartn_y", "Cartn_z",
        "pdbx_PDB_model_num",
    ]
    columns = {}
    for name in fields:
        value = cif.get("_atom_site." + name)
        if value is None:
            raise ValueError(f"Missing mmCIF field: {name}")
        columns[name] = value if isinstance(value, list) else [value]

    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("Invalid or empty atom-site table.")

    models = list(dict.fromkeys(columns["pdbx_PDB_model_num"]))
    first_model = models[0]
    groups = {}

    for i in range(len(columns["label_seq_id"])):
        if columns["pdbx_PDB_model_num"][i] != first_model:
            continue
        seq_value = columns["label_seq_id"][i]
        if seq_value in {"", ".", "?"}:
            continue  # Nonpolymer atoms have no label sequence position.
        try:
            position = int(seq_value)
        except ValueError as error:
            raise ValueError("Invalid polymer label_seq_id") from error

        chain = columns["label_asym_id"][i]
        key = (chain, position)
        identity = (
            columns["label_entity_id"][i],
            columns["auth_asym_id"][i],
            columns["auth_seq_id"][i],
            normalize_missing(columns["pdbx_PDB_ins_code"][i]),
            columns["label_comp_id"][i],
        )
        if key not in groups:
            groups[key] = {"identity": identity, "records": []}
        elif groups[key]["identity"] != identity:
            raise ValueError(f"Conflicting residue identity: {key}")

        def number(name):
            try:
                return float(columns[name][i])
            except (ValueError, TypeError):
                return np.nan

        groups[key]["records"].append({
            "atom_name": columns["label_atom_id"][i],
            "altloc": columns["label_alt_id"][i],
            "occupancy": number("occupancy"),
            "xyz": [float(np.float32(number(name))) for name in ("Cartn_x", "Cartn_y", "Cartn_z")],
        })

    chains = {}
    for (chain, position), group in sorted(groups.items()):
        entity, author_chain, author_seq, icode, component = group["identity"]
        atoms, audit = select_residue_atoms(group["records"])
        chains.setdefault(chain, []).append({
            "label_chain": chain,
            "label_seq_id": position,
            "label_entity_id": entity,
            "auth_chain": author_chain,
            "auth_seq_id": author_seq,
            "insertion_code": icode,
            "comp_id": component,
            "aa_code": AA3.get(component, "X"),
            "atoms": atoms,
            "altloc_audit": audit,
        })

    return chains, {
        "model_count": len(models),
        "selected_model_number": first_model,
    }


def dihedral(p0, p1, p2, p3):
    points = np.asarray([p0, p1, p2, p3], dtype=float)
    if points.shape != (4, 3) or not np.isfinite(points).all():
        return np.nan
    b0 = -(points[1] - points[0])
    b1 = points[2] - points[1]
    b2 = points[3] - points[2]
    norm = np.linalg.norm(b1)
    if norm < 1e-8:
        return np.nan
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) < 1e-8 or np.linalg.norm(w) < 1e-8:
        return np.nan
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def phi_psi(residues, index):
    current = residues[index]
    atoms = current["atoms"]
    if current["aa_code"] == "X" or not {"N", "CA", "C"} <= atoms.keys():
        return np.nan, np.nan

    phi = psi = np.nan
    if index > 0:
        previous = residues[index - 1]
        if previous["aa_code"] != "X" and polymer_connection(previous, current):
            phi = dihedral(
                previous["atoms"]["C"], atoms["N"], atoms["CA"], atoms["C"]
            )
    if index + 1 < len(residues):
        following = residues[index + 1]
        if following["aa_code"] != "X" and polymer_connection(current, following):
            psi = dihedral(
                atoms["N"], atoms["CA"], atoms["C"], following["atoms"]["N"]
            )
    return phi, psi


def angle_components(angle):
    if not np.isfinite(angle):
        return {"sin": 0.0, "cos": 0.0, "mask": 0.0}
    return {
        "sin": float(np.sin(angle)),
        "cos": float(np.cos(angle)),
        "mask": 1.0,
    }
