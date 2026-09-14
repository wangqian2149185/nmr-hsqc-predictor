"""Coordinate primitives implementing the frozen v15 feature policy."""

from fractions import Fraction
import numpy as np

BACKBONE = frozenset({"N", "CA", "C", "O"})
SHARED = frozenset({"", ".", "?"})


def valid_atom(record):
    """Records have atom_name, altloc, occupancy and xyz fields."""
    try:
        occupancy = float(record["occupancy"])
        xyz = np.asarray(record["xyz"], dtype=float)
        return bool(
            np.isfinite(occupancy) and occupancy > 0
            and xyz.shape == (3,) and np.isfinite(xyz).all()
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def select_residue_atoms(records):
    """Select one residue altloc; return atoms and auditable metadata."""
    normalized = []
    seen = set()
    for record in records:
        item = dict(record)
        label = str(item.get("altloc", "")).strip()
        label = "" if label in SHARED else label
        name = str(item["atom_name"]).strip()
        key = (name, label)
        if key in seen:
            raise ValueError(
                f"Duplicate atom/altloc records require inspection: {key}"
            )
        seen.add(key)
        item["altloc"], item["atom_name"] = label, name
        normalized.append(item)

    labels = sorted({r["altloc"] for r in normalized if r["altloc"]})
    scores = {}
    for label in labels:
        values = [
            Fraction(str(r["occupancy"]))
            for r in normalized
            if r["altloc"] == label
            and r["atom_name"] in BACKBONE and valid_atom(r)
        ]
        if values:
            scores[label] = sum(values, Fraction(0)) / len(values)

    if scores:
        maximum = max(scores.values())
        winners = sorted(k for k, value in scores.items() if value == maximum)
        selected_label = winners[0]
    else:
        winners = labels
        selected_label = labels[0] if labels else ""

    atoms = {}
    for name in sorted({r["atom_name"] for r in normalized}):
        selected = [
            r for r in normalized
            if r["atom_name"] == name
            and r["altloc"] == selected_label and valid_atom(r)
        ]
        shared = [
            r for r in normalized
            if r["atom_name"] == name
            and r["altloc"] == "" and valid_atom(r)
        ]
        candidates = selected or shared
        if candidates:
            atoms[name] = np.asarray(candidates[0]["xyz"], dtype=float)

    return atoms, {
        "selected_altloc": selected_label,
        "mean_backbone_occupancy_scores": {
            k: str(value) for k, value in scores.items()
        },
        "maximum_score_tie": bool(scores and len(winners) > 1),
        "fallback_without_backbone_score": not bool(scores),
        "missing_backbone_atoms": sorted(BACKBONE - atoms.keys()),
    }


def polymer_connection(previous, current):
    """Residues contain label_chain, label_seq_id and selected atoms."""
    if previous["label_chain"] != current["label_chain"]:
        return False
    if int(current["label_seq_id"]) != int(previous["label_seq_id"]) + 1:
        return False
    c = previous["atoms"].get("C")
    n = current["atoms"].get("N")
    if c is None or n is None:
        return False
    c, n = np.asarray(c, dtype=float), np.asarray(n, dtype=float)
    if c.shape != (3,) or n.shape != (3,):
        return False
    if not np.isfinite(c).all() or not np.isfinite(n).all():
        return False
    return bool(np.linalg.norm(c - n) <= 2.0)
