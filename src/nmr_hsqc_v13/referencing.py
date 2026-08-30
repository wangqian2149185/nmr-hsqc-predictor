"""Robust entry-level chemical-shift offset estimation."""

from __future__ import annotations

import numpy as np


def estimate_entry_offsets(
    observed: np.ndarray,
    baseline_prediction: np.ndarray,
    mask: np.ndarray,
    *,
    prior_scale: tuple[float, float] = (0.20, 2.0),
    noise_scale: tuple[float, float] = (0.35, 2.5),
) -> np.ndarray:
    """Estimate shrunken H/N entry offsets without forcing equal ppm offsets.

    The robust median residual is shrunk toward zero using an approximate normal
    prior. This is intended for training-label remediation and diagnostics; a new
    structure with no experimental anchor uses the zero-offset prior mean.
    """
    observed = np.asarray(observed, dtype=float)
    baseline_prediction = np.asarray(baseline_prediction, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if observed.shape != baseline_prediction.shape or observed.shape != mask.shape:
        raise ValueError("observed, baseline_prediction, and mask must have equal shape")
    if observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("expected arrays shaped [residue, 2] for H and N")

    offsets = np.zeros(2, dtype=float)
    for head in range(2):
        residual = observed[mask[:, head], head] - baseline_prediction[mask[:, head], head]
        if residual.size == 0:
            continue
        raw = float(np.median(residual))
        prior_var = prior_scale[head] ** 2
        mean_var = noise_scale[head] ** 2 / residual.size
        offsets[head] = raw * prior_var / (prior_var + mean_var)
    return offsets
