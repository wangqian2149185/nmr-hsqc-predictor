import numpy as np
import torch

from nmr_hsqc_v13.alignment import ResidueRecord, align_bmrb_to_pdb
from nmr_hsqc_v13.losses import masked_reference_aware_loss
from nmr_hsqc_v13.referencing import estimate_entry_offsets
from nmr_hsqc_v13.splits import grouped_three_way_split


def test_chain_alignment_does_not_collapse_duplicate_residue_numbers():
    bmrb = [ResidueRecord("entity-1", i, aa) for i, aa in enumerate(["ALA", "GLY", "SER"], 1)]
    pdb = [ResidueRecord("A", i + 10, aa) for i, aa in enumerate(["ALA", "GLY", "SER"], 1)]
    result = align_bmrb_to_pdb(bmrb, pdb)
    assert len(result) == 1
    assert [m.pdb.seq_id for m in result[0].matches] == [11, 12, 13]


def test_group_split_keeps_clusters_together():
    rows = [{"id": i, "cluster": str(i // 2)} for i in range(20)]
    parts = grouped_three_way_split(rows, lambda row: row["cluster"])
    locations = {}
    for partition, rows_in_partition in enumerate(parts):
        for row in rows_in_partition:
            locations.setdefault(row["cluster"], set()).add(partition)
    assert all(len(value) == 1 for value in locations.values())


def test_offsets_are_head_specific_and_shrunken():
    observed = np.array([[0.2, 2.0], [0.2, 2.0], [0.2, 2.0]])
    offsets = estimate_entry_offsets(observed, np.zeros_like(observed), np.ones_like(observed, bool))
    assert 0 < offsets[0] < 0.2
    assert 0 < offsets[1] < 2.0
    assert offsets[0] != offsets[1]


def test_masked_loss_accepts_missing_h_or_n():
    pred = torch.tensor([[0.0, 1.0], [2.0, 0.0]], requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, False], [False, True]])
    loss = masked_reference_aware_loss(
        pred, target, mask, torch.tensor([0, 0]), head_scale=torch.tensor([0.3, 2.0])
    )
    loss.backward()
    assert torch.isfinite(loss)
