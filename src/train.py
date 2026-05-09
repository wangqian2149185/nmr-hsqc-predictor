"""
Training script for NMR HSQC chemical shift prediction.

Usage:
    python src/train.py --config configs/gcn.yaml [--baseline]

Checkpoints saved to: checkpoints/best_model.pt
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

log = logging.getLogger(__name__)


def evaluate(model, loader, device: str) -> tuple[float, float]:
    """Returns (mae_H, mae_N) on the given loader."""
    model.eval()
    total_abs_H = total_abs_N = total_n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            diff = (pred - batch.y).abs()
            total_abs_H += diff[:, 0].sum().item()
            total_abs_N += diff[:, 1].sum().item()
            total_n += diff.shape[0]
    if total_n == 0:
        return float("inf"), float("inf")
    return total_abs_H / total_n, total_abs_N / total_n


def train(
    model,
    train_loader,
    val_loader,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    checkpoint_dir: str = "checkpoints",
    device: str = "cuda",
):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.HuberLoss()

    best_val_mae = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_mae_H, val_mae_N = evaluate(model, val_loader, device)
        val_mae = val_mae_H + val_mae_N  # combined for early stopping

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d | loss=%.4f | val MAE ¹H=%.3f ¹⁵N=%.3f ppm",
                epoch, avg_loss, val_mae_H, val_mae_N,
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), checkpoint_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                log.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    log.info("Best val MAE: %.4f | checkpoint saved to %s", best_val_mae, checkpoint_path)
    return checkpoint_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gcn.yaml")
    parser.add_argument("--baseline", action="store_true", help="Train MLP baseline instead of GCN")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    with open("configs/custom_embed.yaml") as fh:
        embed_cfg = yaml.safe_load(fh)

    # Build datasets
    from src.data.dataset import build_all_datasets
    train_ds, val_ds, test_ds = build_all_datasets(
        manifest_path="data/raw/manifest.json",
        embed_config=embed_cfg,
        gcn_config=cfg,
        seed=args.seed,
    )

    log.info("Dataset: train=%d, val=%d, test=%d proteins", len(train_ds), len(val_ds), len(test_ds))

    batch_size = cfg.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    node_feat_dim = cfg["node_feat_dim"]
    hidden_dim    = cfg.get("hidden_dim", 256)

    if args.baseline:
        from src.models.baseline_mlp import NMRShiftMLP
        model = NMRShiftMLP(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim)
        log.info("Training MLP baseline")
    else:
        from src.models.gcn import NMRShiftGCN
        model = NMRShiftGCN(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=cfg.get("num_layers", 4),
        )
        log.info("Training GCN")

    checkpoint_path = train(
        model,
        train_loader,
        val_loader,
        epochs=cfg.get("epochs", 200),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
        patience=cfg.get("early_stop_patience", 20),
        device=args.device,
    )

    # Final test evaluation
    log.info("Loading best model for test evaluation …")
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    test_mae_H, test_mae_N = evaluate(model, test_loader, args.device)
    log.info("Test MAE ¹H=%.3f ppm | ¹⁵N=%.3f ppm", test_mae_H, test_mae_N)


if __name__ == "__main__":
    main()
