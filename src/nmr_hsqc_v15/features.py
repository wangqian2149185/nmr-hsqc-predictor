"""Shared training/deployment feature builder for v15."""

import numpy as np
import pandas as pd

from .structure_geometry import (
    read_structure_residues, phi_psi, angle_components,
)

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
BASE_COLUMNS = [
    *(prefix + aa for prefix in ("aa_", "prev_", "next_")
      for aa in AA_ORDER),
    "position_fraction", "log_chain_length",
    "sin_phi", "cos_phi", "phi_mask",
    "sin_psi", "cos_psi", "psi_mask",
]
STRUCTURE_COLUMNS = [
    "ca_contacts_6a", "ca_contacts_8a", "ca_contacts_10a",
    "nearest_backbone_o_distance", "backbone_o_within_3p5a",
    "hbond_proxy_mask",
]
IDENTITY_COLUMNS = [
    "chain_id", "pdb_seq_position", "pdb_resseq", "pdb_icode",
    "comp_id", "aa_code", "label_chain_id", "label_seq_id",
    "pdb_entity_id",
]


def one_hot(prefix, aa):
    return {prefix + code: float(code == aa) for code in AA_ORDER}


def build_features_from_residues(chains, label_chain, include_proline=False):
    """Input comes from read_structure_residues; no target data required."""
    if label_chain not in chains:
        raise KeyError(f"Missing label chain: {label_chain}")
    target_all = chains[label_chain]
    standard = [r for r in target_all if r["aa_code"] in AA_ORDER]
    if len(standard) < 2:
        raise ValueError("Target chain has fewer than two standard residues.")

    positions = [r["label_seq_id"] for r in target_all]
    if len(positions) != len(set(positions)) or positions != sorted(positions):
        raise ValueError("Target label sequence keys are invalid.")

    full_index = {r["label_seq_id"]: i for i, r in enumerate(target_all)}
    environment = [
        r for residues in chains.values() for r in residues
        if r["aa_code"] in AA_ORDER
    ]
    ca_environment = [
        r for r in environment if "CA" in r["atoms"]
    ]
    oxygen_environment = [
        r for r in environment if "O" in r["atoms"]
    ]

    rows = []
    for index, residue in enumerate(standard):
        aa = residue["aa_code"]
        if aa == "P" and not include_proline:
            continue
        previous = standard[index - 1]["aa_code"] if index else None
        following = (
            standard[index + 1]["aa_code"]
            if index + 1 < len(standard) else None
        )
        phi, psi = phi_psi(target_all, full_index[residue["label_seq_id"]])
        phi_values, psi_values = angle_components(phi), angle_components(psi)

        atoms = residue["atoms"]
        contacts = {cutoff: np.nan for cutoff in (6, 8, 10)}
        if "CA" in atoms:
            distances = [
                float(np.linalg.norm(atoms["CA"] - item["atoms"]["CA"]))
                for item in ca_environment
                if not (
                    item["label_chain"] == label_chain
                    and item["label_seq_id"] == residue["label_seq_id"]
                )
            ]
            for cutoff in contacts:
                contacts[cutoff] = float(sum(d <= cutoff for d in distances))

        oxygen_distances = []
        if "N" in atoms:
            oxygen_distances = [
                float(np.linalg.norm(atoms["N"] - item["atoms"]["O"]))
                for item in oxygen_environment
                if not (
                    item["label_chain"] == label_chain
                    and abs(item["label_seq_id"] - residue["label_seq_id"]) <= 1
                )
            ]
        try:
            author_number = int(residue["auth_seq_id"])
        except (ValueError, TypeError) as error:
            raise ValueError(
                "Author residue number is noninteger; requires inspection."
            ) from error

        rows.append({
            "chain_id": residue["auth_chain"],
            "pdb_seq_position": index + 1,
            "pdb_resseq": author_number,
            "pdb_icode": residue["insertion_code"],
            "comp_id": residue["comp_id"],
            "aa_code": aa,
            "label_chain_id": label_chain,
            "label_seq_id": residue["label_seq_id"],
            "pdb_entity_id": residue["label_entity_id"],
            **one_hot("aa_", aa),
            **one_hot("prev_", previous),
            **one_hot("next_", following),
            "position_fraction": (index + 1) / len(standard),
            "log_chain_length": float(np.log(len(standard)) / 6.0),
            "sin_phi": phi_values["sin"],
            "cos_phi": phi_values["cos"],
            "phi_mask": phi_values["mask"],
            "sin_psi": psi_values["sin"],
            "cos_psi": psi_values["cos"],
            "psi_mask": psi_values["mask"],
            "ca_contacts_6a": contacts[6],
            "ca_contacts_8a": contacts[8],
            "ca_contacts_10a": contacts[10],
            "nearest_backbone_o_distance": (
                min(oxygen_distances) if oxygen_distances else np.nan
            ),
            "backbone_o_within_3p5a": (
                float(sum(d <= 3.5 for d in oxygen_distances))
                if oxygen_distances else np.nan
            ),
            "hbond_proxy_mask": float(bool(oxygen_distances)),
        })

    frame = pd.DataFrame(
        rows, columns=IDENTITY_COLUMNS + BASE_COLUMNS + STRUCTURE_COLUMNS
    )
    if not np.isfinite(frame[BASE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite base features.")
    if np.isinf(frame[STRUCTURE_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError("Infinite structure features.")
    return frame


def build_pdb_feature_frame(structure_path, label_chain_id,
                            include_proline=False):
    """Accept plain or gzipped mmCIF; chain argument is a LABEL chain."""
    chains, metadata = read_structure_residues(structure_path)
    frame = build_features_from_residues(
        chains, str(label_chain_id), include_proline
    )
    frame.attrs["structure_metadata"] = metadata
    return frame
