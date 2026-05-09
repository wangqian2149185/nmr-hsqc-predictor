"""
Full evaluation: MAE, Pearson r, and scatter plots on the test set.

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pt \
                           --config configs/gcn.yaml \
                           [--output-dir output]
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

log = logging.getLogger(__name__)


def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred), each shape (N_residues, 2)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            preds.append(pred.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae_H = float(np.mean(np.abs(y_pred[:, 0] - y_true[:, 0])))
    mae_N = float(np.mean(np.abs(y_pred[:, 1] - y_true[:, 1])))
    r_H, _ = pearsonr(y_true[:, 0], y_pred[:, 0])
    r_N, _ = pearsonr(y_true[:, 1], y_pred[:, 1])
    return {"mae_H": mae_H, "mae_N": mae_N, "pearson_H": float(r_H), "pearson_N": float(r_N)}


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, label, unit, target_mae in [
        (axes[0], 0, "¹H", "ppm", 0.3),
        (axes[1], 1, "¹⁵N", "ppm", 2.0),
    ]:
        xt, xp = y_true[:, col], y_pred[:, col]
        ax.scatter(xt, xp, alpha=0.3, s=8, rasterized=True)
        lim = [min(xt.min(), xp.min()) - 0.5, max(xt.max(), xp.max()) + 0.5]
        ax.plot(lim, lim, "k--", lw=1, label="ideal")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"Experimental δ{label} ({unit})")
        ax.set_ylabel(f"Predicted δ{label} ({unit})")
        key_mae = f"mae_{label[1]}"  # mae_H or mae_N
        key_r   = f"pearson_{label[1]}"
        ax.set_title(
            f"δ{label}  MAE={metrics[key_mae]:.3f} ppm  r={metrics[key_r]:.3f}"
        )
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, "scatter_hsqc.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Scatter plot saved to %s", out_path)


# ── Targets ───────────────────────────────────────────────────────────────────

TARGETS = {
    "mae_H": ("<", 0.3, "ppm"),
    "mae_N": ("<", 2.0, "ppm"),
    "pearson_H": (">", 0.90, ""),
    "pearson_N": (">", 0.90, ""),
}


def print_report(metrics: dict):
    print("\n=== Evaluation Report ===")
    all_pass = True
    for key, (op, thr, unit) in TARGETS.items():
        val = metrics[key]
        passed = (val < thr) if op == "<" else (val > thr)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {key:<12}: {val:.4f} {unit}  [{op} {thr}{unit}]  → {status}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME TARGETS NOT MET'}")
    print("========================\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config", default="configs/gcn.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    with open("configs/custom_embed.yaml") as fh:
        embed_cfg = yaml.safe_load(fh)

    from src.data.dataset import build_all_datasets
    _, _, test_ds = build_all_datasets(
        manifest_path="data/raw/manifest.json",
        embed_config=embed_cfg,
        gcn_config=cfg,
    )

    node_feat_dim = cfg["node_feat_dim"]
    hidden_dim    = cfg.get("hidden_dim", 256)

    if args.baseline:
        from src.models.baseline_mlp import NMRShiftMLP
        model = NMRShiftMLP(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim)
    else:
        from src.models.gcn import NMRShiftGCN
        model = NMRShiftGCN(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=cfg.get("num_layers", 4),
        )

    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device)

    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 32), shuffle=False)
    y_true, y_pred = collect_predictions(model, test_loader, args.device)
    metrics = compute_metrics(y_true, y_pred)
    print_report(metrics)
    plot_scatter(y_true, y_pred, metrics, args.output_dir)


if __name__ == "__main__":
    main()
