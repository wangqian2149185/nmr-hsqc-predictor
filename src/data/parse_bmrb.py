"""
Parse NMR-STAR 3 files to extract backbone amide ¹H / ¹⁵N chemical shifts.

Returns:
    { (chain_id, seq_id, res_name): (delta_1H, delta_15N) }

Keys use entity_assembly_id as chain_id (string) and integer seq_id.
Prolines are excluded (no backbone amide proton).
"""

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Backbone amide H atom names seen in BMRB entries
AMIDE_H_NAMES = {"H", "HN", "H1"}
PROLINE = "PRO"


def parse_hsqc_shifts(star_file: str) -> dict[tuple, tuple[float, float]]:
    """
    Returns { (chain_id, seq_id, res_name): (delta_1H, delta_15N) }.

    chain_id : str  (entity_assembly_id, e.g. "1" or "A")
    seq_id   : int
    res_name : str  (3-letter code, upper-case)
    """
    try:
        import pynmrstar
    except ImportError as exc:
        raise ImportError("pip install pynmrstar") from exc

    result: dict[tuple, dict] = {}  # (chain, seq, res) → {"H": float, "N": float}

    try:
        entry = pynmrstar.Entry.from_file(star_file)
    except Exception as exc:
        log.warning("Failed to parse %s: %s", star_file, exc)
        return {}

    for saveframe in entry:
        if saveframe.category != "assigned_chemical_shifts":
            continue

        for loop in saveframe:
            tags_lower = [t.lower() for t in loop.tags]
            required = {"atom_id", "val", "seq_id", "comp_id"}

            # Build normalised tag index
            tag_map: dict[str, int] = {}
            for i, t in enumerate(tags_lower):
                # Strip prefix (e.g. '_atom_chem_shift.atom_id' → 'atom_id')
                short = t.split(".")[-1]
                tag_map[short] = i

            if not required.issubset(tag_map.keys()):
                continue

            for row in loop.data:
                try:
                    atom_id  = str(row[tag_map["atom_id"]]).upper()
                    val_raw  = row[tag_map["val"]]
                    seq_id   = int(row[tag_map["seq_id"]])
                    res_name = str(row[tag_map["comp_id"]]).upper()
                except (ValueError, KeyError):
                    continue

                if res_name == PROLINE:
                    continue

                if val_raw in (".", "?", "", None):
                    continue
                try:
                    val = float(val_raw)
                except ValueError:
                    continue

                # Chain ID
                chain_id = "1"
                for chain_tag in ("entity_assembly_id", "assembly_atom_id"):
                    if chain_tag in tag_map:
                        chain_id = str(row[tag_map[chain_tag]])
                        break

                # Isotope number (prefer explicit tag; else infer from atom name)
                isotope: Optional[int] = None
                if "atom_isotope_number" in tag_map:
                    iso_raw = row[tag_map["atom_isotope_number"]]
                    if iso_raw not in (".", "?", "", None):
                        try:
                            isotope = int(iso_raw)
                        except ValueError:
                            pass

                key = (chain_id, seq_id, res_name)
                result.setdefault(key, {})

                is_amide_h = atom_id in AMIDE_H_NAMES and (isotope == 1 or isotope is None)
                is_amide_n = atom_id == "N" and (isotope == 15 or isotope is None)

                # Isotope-less entries: infer from atom symbol
                if isotope is None:
                    is_amide_h = atom_id in AMIDE_H_NAMES
                    is_amide_n = atom_id == "N"

                if is_amide_h and "H" not in result[key]:
                    result[key]["H"] = val
                if is_amide_n and "N" not in result[key]:
                    result[key]["N"] = val

    return {
        key: (d["H"], d["N"])
        for key, d in result.items()
        if "H" in d and "N" in d
    }


def validate_referencing(shifts: dict[tuple, tuple[float, float]]) -> dict:
    """
    Soft validation of chemical shift referencing ranges.
    ¹H amide: typically 6–11 ppm (DSS reference).
    ¹⁵N amide: typically 100–135 ppm (liquid ammonia reference).
    Returns counts of in-range and out-of-range residues.
    """
    in_range, out_range = 0, 0
    for (_, _, _), (dH, dN) in shifts.items():
        if 5.0 <= dH <= 12.0 and 90.0 <= dN <= 145.0:
            in_range += 1
        else:
            out_range += 1
    return {"in_range": in_range, "out_of_range": out_range}
