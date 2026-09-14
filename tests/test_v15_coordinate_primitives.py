import numpy as np
import pytest

from nmr_hsqc_v15.coordinate_primitives import (
    valid_atom, select_residue_atoms, polymer_connection,
)


def atom(name, label="", occupancy=1.0, xyz=(0, 0, 0)):
    return {
        "atom_name": name, "altloc": label,
        "occupancy": occupancy, "xyz": xyz,
    }


def test_atom_validity():
    assert valid_atom(atom("N"))
    assert not valid_atom(atom("N", occupancy=0))
    assert not valid_atom(atom("N", occupancy=None))
    assert not valid_atom(atom("N", xyz=(np.nan, 0, 0)))


def test_residue_mean_occupancy_selection():
    records = [
        atom("N", "A", .9), atom("CA", "A", .1),
        atom("N", "B", .6), atom("CA", "B", .6),
    ]
    _, audit = select_residue_atoms(records)
    assert audit["selected_altloc"] == "B"


def test_tie_is_independent_of_input_order():
    records = [atom("N", "B", .5), atom("N", "A", .5)]
    _, first = select_residue_atoms(records)
    _, second = select_residue_atoms(list(reversed(records)))
    assert first == second
    assert first["selected_altloc"] == "A"
    assert first["maximum_score_tie"]


def test_shared_fallback_without_other_conformer_borrowing():
    records = [
        atom("N", "A", .9), atom("CA", "A", .9),
        atom("O", "B", .2), atom("C"),
    ]
    atoms, audit = select_residue_atoms(records)
    assert audit["selected_altloc"] == "A"
    assert "C" in atoms and "O" not in atoms


def test_invalid_selected_atom_can_fall_back_to_shared():
    atoms, _ = select_residue_atoms([
        atom("N", "A", .8), atom("CA", "A", 0),
        atom("CA"),
    ])
    assert "CA" in atoms


def test_duplicate_normalized_shared_records_are_flagged():
    with pytest.raises(ValueError):
        select_residue_atoms([atom("N", "."), atom("N", "?")])


def test_no_backbone_score_fallback():
    atoms, audit = select_residue_atoms([
        atom("CB", "B", .9), atom("CB", "A", .1), atom("N"),
    ])
    assert audit["selected_altloc"] == "A"
    assert audit["fallback_without_backbone_score"]
    assert "N" in atoms


def residue(position, atoms, chain="A"):
    return {
        "label_chain": chain, "label_seq_id": position, "atoms": atoms,
    }


def test_connection_threshold_and_label_sequence():
    previous = residue(10, {"C": np.array([0., 0., 0.])})
    assert polymer_connection(
        previous, residue(11, {"N": np.array([2., 0., 0.])})
    )
    assert not polymer_connection(
        previous, residue(11, {"N": np.array([2.01, 0., 0.])})
    )
    assert not polymer_connection(
        previous, residue(12, {"N": np.array([1.3, 0., 0.])})
    )
    assert not polymer_connection(
        previous, residue(11, {"N": np.array([1.3, 0., 0.])}, "B")
    )


def test_missing_and_nonfinite_connection_atoms():
    previous = residue(10, {"C": np.zeros(3)})
    assert not polymer_connection(previous, residue(11, {}))
    assert not polymer_connection(
        previous, residue(11, {"N": np.array([np.nan, 0., 0.])})
    )
