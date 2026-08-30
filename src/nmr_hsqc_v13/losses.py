"""Masked multi-nucleus losses robust to partial labels and entry translations."""

from __future__ import annotations

import torch


def _stable_log_cosh(value: torch.Tensor) -> torch.Tensor:
    return value + torch.nn.functional.softplus(-2.0 * value) - torch.log(
        value.new_tensor(2.0)
    )


def masked_reference_aware_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    entry_index: torch.Tensor,
    *,
    head_scale: torch.Tensor,
    centered_weight: float = 0.25,
) -> torch.Tensor:
    """Absolute robust loss plus an entry-translation-invariant centered term.

    H and N labels are independent masks. ``head_scale`` expresses meaningful
    ppm scales and prevents N from dominating model selection or training.
    """
    mask = target_mask.bool()
    scale = head_scale.to(prediction).clamp_min(1e-6)
    error = (prediction - target) / scale
    absolute_terms = _stable_log_cosh(error)[mask]
    if absolute_terms.numel() == 0:
        return prediction.sum() * 0.0
    absolute = absolute_terms.mean()

    centered_terms = []
    for entry in torch.unique(entry_index):
        in_entry = entry_index == entry
        for head in range(prediction.shape[1]):
            selected = in_entry & mask[:, head]
            if selected.sum() < 2:
                continue
            residual = error[selected, head]
            residual = residual - residual.mean()
            centered_terms.append(_stable_log_cosh(residual).mean())
    if not centered_terms:
        return absolute
    return absolute + centered_weight * torch.stack(centered_terms).mean()
