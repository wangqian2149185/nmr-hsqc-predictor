import numpy as np
from nmr_hsqc_v15.structure_geometry import read_structure_residues


def test_parser_matches_historical_coordinate_precision(tmp_path):
    fields = [
        "label_asym_id", "label_seq_id", "label_entity_id",
        "auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code",
        "label_comp_id", "label_atom_id", "label_alt_id",
        "occupancy", "Cartn_x", "Cartn_y", "Cartn_z",
        "pdbx_PDB_model_num",
    ]
    text = "data_precision\nloop_\n"
    text += "".join("_atom_site." + f + "\n" for f in fields)
    text += "L 1 1 A 1 ? ALA N . 1 26.981 53.977 40.085 1\n#\n"
    path = tmp_path / "precision.cif"
    path.write_text(text)
    chains, _ = read_structure_residues(path)
    actual = chains["L"][0]["atoms"]["N"]
    expected = np.array([26.981, 53.977, 40.085], dtype=np.float32).astype(np.float64)
    assert actual.dtype == np.float64
    assert np.array_equal(actual, expected)
