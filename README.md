# NMR ¹H-¹⁵N HSQC Chemical Shift Predictor

Predict backbone amide ¹H and ¹⁵N chemical shifts from protein PDB structures using a Graph Convolutional Network.

```
PDB file → Custom Geometric Embeddings → GCN → (δ¹H, δ¹⁵N) per residue
```

## Quick start

```bash
# 1. Create environment
bash setup_env.sh

# 2. Download ~500 BMRB+PDB pairs
python src/data/download.py --max-entries 500

# 3. Train GCN (builds features + graph on first run, cached afterwards)
python src/train.py --config configs/gcn.yaml

# 4. Evaluate on test set + generate scatter plots
python src/evaluate.py --checkpoint checkpoints/best_model.pt

# 5. Predict on a new structure
python src/predict.py --pdb my_protein.pdb --output predictions.csv
```

## Project structure

```
nmr_hsqc_predictor/
├── data/
│   ├── raw/{bmrb,pdb}/     # Downloaded files
│   ├── processed/          # Cached feature graphs
│   └── splits/             # train/val/test protein ID lists
├── src/
│   ├── data/
│   │   ├── download.py     # BMRB + PDB batch downloader
│   │   ├── parse_bmrb.py   # NMR-STAR → amide shift dict
│   │   ├── parse_pdb.py    # PDB → atom coords (handles NMR ensembles)
│   │   ├── embed_custom.py # Local-frame geometric feature vectors
│   │   └── dataset.py      # PyG Data objects + train/val/test split
│   ├── models/
│   │   ├── gcn.py          # NMRShiftGCN (primary)
│   │   └── baseline_mlp.py # MLP baseline
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── configs/
│   ├── custom_embed.yaml
│   └── gcn.yaml
└── checkpoints/            # Saved model weights
```

## Embedding — Path B (Custom Geometric)

For each residue `i`:

1. **Local frame**: translate amide N to origin; rotate so N→Cα aligns with +X and N→C(carbonyl) falls in the +Y half-plane.
2. **Neighbors**: union of through-space atoms within 5 Å of N, and sequence neighbors i±3.
3. **Per-neighbor features** (14 values): local (x,y,z) mean, local (x,y,z) std across NMR ensemble models, `in_space` flag, `in_seq` flag, 6-dim one-hot element.
4. **Padding**: sorted by distance, padded/truncated to 128 neighbors → **1792-dim** feature vector.

## Performance targets

| Metric     | Target  |
|------------|---------|
| MAE ¹H     | < 0.3 ppm |
| MAE ¹⁵N    | < 2.0 ppm |
| Pearson r ¹H  | > 0.90 |
| Pearson r ¹⁵N | > 0.90 |

## Key design decisions

- **Protein-level split**: train/val/test split at the protein level (not residue level) to prevent data leakage.
- **NMR ensembles**: all models used; mean coordinate + per-axis std are both features.
- **Flexibility filter**: residues with backbone RMSD > 1.0 Å across NMR models are excluded.
- **Prolines excluded**: no backbone amide proton.
- **Huber loss**: robust to outlier chemical shifts.
