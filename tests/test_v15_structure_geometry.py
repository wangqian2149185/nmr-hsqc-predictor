import gzip
import numpy as np

from nmr_hsqc_v15.structure_geometry import (
    read_structure_residues, phi_psi, angle_components, dihedral,
)


def residue(position, aa="A", shift=0):
    return {
        "label_chain": "A", "label_seq_id": position, "aa_code": aa,
        "atoms": {
            "N": np.array([shift, 0., 0.]),
            "CA": np.array([shift, 1., 0.]),
            "C": np.array([shift + 1., 1., 1.]),
        },
    }


def test_missing_polymer_position_blocks_angles():
    residues = [residue(1), residue(3, shift=2)]
    assert np.isnan(phi_psi(residues, 0)[1])
    assert np.isnan(phi_psi(residues, 1)[0])


def test_unknown_middle_residue_is_not_bridged():
    residues = [
        residue(1), residue(2, aa="X", shift=2), residue(3, shift=4)
    ]
    assert np.isnan(phi_psi(residues, 0)[1])
    assert np.isnan(phi_psi(residues, 2)[0])


def test_continuous_pair_has_finite_angles():
    residues = [residue(1), residue(2, shift=2)]
    assert np.isfinite(phi_psi(residues, 0)[1])
    assert np.isfinite(phi_psi(residues, 1)[0])


def test_large_cn_distance_blocks_angles():
    residues = [residue(1), residue(2, shift=20)]
    assert np.isnan(phi_psi(residues, 0)[1])
    assert np.isnan(phi_psi(residues, 1)[0])


def test_missing_atom_and_degenerate_angle_encoding():
    residues = [residue(1), residue(2, shift=2)]
    del residues[1]["atoms"]["CA"]
    assert all(np.isnan(x) for x in phi_psi(residues, 1))
    assert np.isnan(dihedral(*[np.zeros(3)] * 4))
    assert angle_components(np.nan) == {"sin": 0., "cos": 0., "mask": 0.}


def test_cif_gzip_mapping_and_first_model(tmp_path):
    fields = [
        "label_asym_id", "label_seq_id", "label_entity_id",
        "auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code",
        "label_comp_id", "label_atom_id", "label_alt_id",
        "occupancy", "Cartn_x", "Cartn_y", "Cartn_z",
        "pdbx_PDB_model_num",
    ]
    text = "data_test\nloop_\n"
    text += "".join("_atom_site." + field + "\n" for field in fields)
    text += "L 10 1 A 2 B ALA N . 1 0 0 0 1\n"
    text += "L 10 1 A 2 B ALA CA . 1 0 1 0 1\n"
    text += "L 10 1 A 2 B ALA C . 1 1 1 1 1\n"
    text += "L 10 1 A 2 B ALA O . 1 1 2 1 1\n"
    text += "L 10 1 A 2 B ALA N . 1 99 0 0 2\n#\n"
    path = tmp_path / "sample.cif.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(text)

    chains, metadata = read_structure_residues(path)
    item = chains["L"][0]
    assert item["label_seq_id"] == 10
    assert item["auth_seq_id"] == "2"
    assert item["auth_chain"] == "A"
    assert item["insertion_code"] == "B"
    assert metadata == {"model_count": 2, "selected_model_number": "1"}
    assert item["atoms"]["N"][0] == 0
