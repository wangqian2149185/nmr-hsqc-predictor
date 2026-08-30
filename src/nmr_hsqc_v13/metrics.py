"""Metrics that report both residue-micro and entry-macro performance."""

from __future__ import annotations

import numpy as np


def masked_mae_by_head(target, prediction, mask) -> np.ndarray:
    target, prediction, mask = map(np.asarray, (target, prediction, mask))
    return np.array([
        np.mean(np.abs(prediction[mask[:, h], h] - target[mask[:, h], h]))
        if np.any(mask[:, h]) else np.nan
        for h in range(target.shape[1])
    ])


def macro_entry_mae(target, prediction, mask, entry_ids) -> np.ndarray:
    target, prediction, mask, entry_ids = map(np.asarray, (target, prediction, mask, entry_ids))
    per_entry = []
    for entry in np.unique(entry_ids):
        selected = entry_ids == entry
        per_entry.append(masked_mae_by_head(target[selected], prediction[selected], mask[selected]))
    return np.nanmean(np.stack(per_entry), axis=0)
