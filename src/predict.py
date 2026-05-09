"""
Inference CLI — predict ¹H/¹⁵N chemical shifts for a single PDB file.

Usage:
    python src/predict.py --pdb my_protein.pdb \
                          --checkpoint checkpoints/best_model.pt \
                          --config configs/gcn.yaml \
                          [--output predictions.csv]
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import yaml

log = logging.getLogger(__name__)


def predict_pdb(
    pdb_file: str,
    model,
    embed_config: dict,
    gcn_config: dict,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run the full feature-extraction → GCN inference pipeline for one PDB.
    Returns a DataFrame with columns: chain, seq_id, res_name, pred_1H, pred_15N.
    """
    from src.data.parse_pdb import load_residue_atoms
    from src.data.embed_custom import embed_protein
    from src.data.dataset import build_edges
    from collections import defaultdict

    backbone_cutoff = embed_config.get("flexibility_cutoff_angstrom", 1.0)
    residue_atoms, flexible_ok = load_residue_atoms(pdb_file, backbone_cutoff=backbone_cutoff)

    if not residue_atoms:
        raise ValueError(f"No residues loaded from {pdb_file}")

    # At inference time all residues are targets
    target_keys = sorted(residue_atoms.keys(), key=lambda k: (k[0], k[1]))

    features = embed_protein(
        residue_atoms,
        target_keys,
        flexible_ok=flexible_ok,
        config=embed_config,
    )

    valid_keys = [k for k in target_keys if k in features]
    if not valid_keys:
        raise ValueError("No residues survived feature extraction — check backbone atoms.")

    X = np.stack([features[k] for k in valid_keys], axis=0)
    x_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    edge_index = build_edges(
        valid_keys,
        residue_atoms,
        spatial_threshold=gcn_config.get("spatial_edge_threshold_angstrom", 8.0),
        seq_range=gcn_config.get("sequence_edge_range", 3),
    ).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x_tensor, edge_index).cpu().numpy()  # (N, 2)

    rows = []
    for i, (chain_id, seq_id, res_name) in enumerate(valid_keys):
        rows.append({
            "chain":     chain_id,
            "seq_id":    seq_id,
            "res_name":  res_name,
            "pred_1H":   round(float(pred[i, 0]), 4),
            "pred_15N":  round(float(pred[i, 1]), 4),
        })

    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Predict ¹H-¹⁵N HSQC shifts from a PDB file.")
    parser.add_argument("--pdb",        required=True,  help="Input PDB file")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config",     default="configs/gcn.yaml")
    parser.add_argument("--output",     default=None,   help="Output CSV path (default: stdout)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline",   action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    with open("configs/custom_embed.yaml") as fh:
        embed_cfg = yaml.safe_load(fh)

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

    df = predict_pdb(args.pdb, model, embed_cfg, cfg, device=args.device)

    if args.output:
        df.to_csv(args.output, index=False)
        log.info("Predictions saved to %s", args.output)
    else:
        print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
