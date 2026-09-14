import json
from pathlib import Path

import numpy as np

from nmr_hsqc_v15.features import (
    BASE_COLUMNS, STRUCTURE_COLUMNS, build_features_from_residues,
)


def residue(position, aa="A", chain="L"):
    components = {"A": "ALA", "P": "PRO", "X": "UNK"}
    return {
        "label_chain": chain, "label_seq_id": position,
        "label_entity_id": "1", "auth_chain": "A",
        "auth_seq_id": str(position + 100), "insertion_code": "",
        "comp_id": components[aa], "aa_code": aa,
        "atoms": {
            "N": np.array([position * 2., 0., 0.]),
            "CA": np.array([position * 2., 1., 0.]),
            "C": np.array([position * 2. + 1., 1., 1.]),
            "O": np.array([position * 2. + 1., 2., 1.]),
        },
    }


def test_columns_match_frozen_manifest():
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((
        repo / "releases/v13_deployable_step33/config/feature_manifest.json"
    ).read_text())
    assert BASE_COLUMNS == manifest["base_feature_columns"]
    assert STRUCTURE_COLUMNS == manifest["h_structure_feature_columns"]
    assert len(BASE_COLUMNS) == 68


def test_proline_filter_keeps_sequence_context_and_position():
    frame = build_features_from_residues(
        {"L": [residue(1), residue(2, "P"), residue(3)]}, "L"
    )
    assert frame["pdb_seq_position"].tolist() == [1, 3]
    assert frame.iloc[0]["next_P"] == 1
    assert frame.iloc[1]["prev_P"] == 1
    assert frame.iloc[0]["pdb_resseq"] == 101
    assert frame.iloc[0]["label_seq_id"] == 1


def test_unknown_position_does_not_bridge_angles():
    frame = build_features_from_residues(
        {"L": [residue(1), residue(2, "X"), residue(3)]}, "L"
    )
    assert frame.iloc[0]["psi_mask"] == 0
    assert frame.iloc[1]["phi_mask"] == 0


def test_missing_ca_retains_raw_nan():
    first = residue(1)
    del first["atoms"]["CA"]
    frame = build_features_from_residues(
        {"L": [first, residue(2)]}, "L"
    )
    assert np.isnan(frame.iloc[0]["ca_contacts_6a"])
    assert frame.iloc[0]["phi_mask"] == 0
    assert frame.iloc[0]["psi_mask"] == 0


def test_other_chain_environment_is_included():
    target = [residue(1), residue(2)]
    other = residue(1, chain="M")
    frame = build_features_from_residues(
        {"L": target, "M": [other]}, "L"
    )
    assert frame.iloc[0]["ca_contacts_6a"] == 2
    assert frame.iloc[0]["hbond_proxy_mask"] == 1


def test_oxygen_exclusion_uses_label_positions_not_filtered_offsets():
    frame = build_features_from_residues(
        {"L": [residue(1), residue(3)]}, "L"
    )
    # The second residue is list-adjacent but not polymer-sequence-adjacent.
    assert frame.iloc[0]["hbond_proxy_mask"] == 1
    assert np.isfinite(frame.iloc[0]["nearest_backbone_o_distance"])
    assert frame.iloc[0]["psi_mask"] == 0


def test_no_usable_oxygen_environment_is_missing():
    residues = [residue(1), residue(2)]
    frame = build_features_from_residues({"L": residues}, "L")
    assert frame["hbond_proxy_mask"].tolist() == [0, 0]
    assert frame["nearest_backbone_o_distance"].isna().all()
