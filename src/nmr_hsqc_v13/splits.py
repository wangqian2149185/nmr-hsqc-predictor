"""Group-safe train/validation/test splitting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import random
from typing import TypeVar

T = TypeVar("T")


def grouped_three_way_split(
    records: Sequence[T],
    group_key: Callable[[T], str],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[T], list[T], list[T]]:
    """Split whole protein/sequence clusters, never individual BMRB entries.

    ``group_key`` should be a precomputed sequence-cluster identifier. All entries
    sharing a PDB, UniProt target, exact sequence, or homology cluster must receive
    the same identifier before calling this function.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be below 1")

    groups: dict[str, list[T]] = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = len(records)
    targets = (total * train_fraction, total * validation_fraction)
    partitions: list[list[T]] = [[], [], []]
    for key in sorted(keys, key=lambda k: len(groups[k]), reverse=True):
        train_deficit = targets[0] - len(partitions[0])
        val_deficit = targets[1] - len(partitions[1])
        destination = 0 if train_deficit >= max(val_deficit, 0) else 1
        if train_deficit <= 0 and val_deficit <= 0:
            destination = 2
        partitions[destination].extend(groups[key])
    return partitions[0], partitions[1], partitions[2]
