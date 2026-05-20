# NMR-HSQC-GNN: Graph Neural Network Prediction of Protein Amide Chemical Shifts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.x-orange.svg)](https://pyg.org/)
[![Colab](https://img.shields.io/badge/Runs%20on-Google%20Colab-yellow.svg)](https://colab.research.google.com/)

Predict backbone amide **¹H** and **¹⁵N** NMR chemical shifts from protein 3D structure using a GATv2-based graph neural network. Trained on 4,337 proteins from the BMRB/PDB archives.

---

## Table of Contents

1. [Scientific Background](#scientific-background)
2. [Project Goals](#project-goals)
3. [Repository Structure](#repository-structure)
4. [Data Sources](#data-sources)
5. [Data Pipeline](#data-pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Model Architecture](#model-architecture)
8. [Training Configuration](#training-configuration)
9. [Results Summary](#results-summary)
10. [Limitations and Future Work](#limitations-and-future-work)
11. [Environment Setup](#environment-setup)
12. [Quick Start](#quick-start)
13. [Citation](#citation)

---

## Scientific Background

In protein NMR spectroscopy, the **¹H–¹⁵N HSQC spectrum** provides one peak per backbone amide group, with peak positions (chemical shifts) sensitive to the local structural and electronic environment of each residue. Accurate prediction of these shifts from protein 3D coordinates enables:

- Rapid validation of NMR assignments
- Structure-based chemical shift perturbation (CSP) analysis
- Folded/disordered state discrimination
- Cross-validation of computational protein models

### What makes HN / ¹⁵N shifts hard to predict

**HN (¹H amide proton):** Dominated by hydrogen-bond geometry, aromatic ring currents, and electrostatic field along the N–H bond. Total dispersion ~4–5 ppm; structure-sensitive range ±2 ppm.

**¹⁵N (backbone nitrogen):** Simultaneously sensitive to φ/ψ dihedral angles, the preceding residue type (ψᵢ₋₁ effect, ±5 ppm), hydrogen-bond length, χ₁ rotamer of the residue itself, and long-range electrostatics. Total dispersion ~25–30 ppm; state-of-the-art prediction RMSD ~2.0 ppm.

---

## Project Goals

| Metric | Target | Best achieved (v9) |
|--------|--------|--------------------|
| GNN MAE ¹H | < 0.30 ppm | 0.311 ppm |
| GNN MAE ¹⁵N | < 2.00 ppm | **1.775 ppm ✓** |
| Pearson r ¹H | > 0.90 | 0.732 |
| Pearson r ¹⁵N | > 0.90 | 0.777 |

---

## Repository Structure

```
nmr-hsqc-gnn/
├── nmr_hsqc_colab_v5.ipynb   # Proof of concept — atom stacking features
├── nmr_hsqc_colab_v6.ipynb   # Physics feature engineering + data scale-up
├── nmr_hsqc_colab_v7.ipynb   # Regularisation experiments (DropEdge, log-cosh)
├── nmr_hsqc_colab_v8.ipynb   # Learned aa embeddings + sinusoidal edges
├── nmr_hsqc_colab_v9.ipynb   # Stable best model (A2 revert, current)
├── VERSION_SUMMARY.md         # Per-version technical changelog + metrics
└── README.md                  # This file
```

All notebooks are self-contained and designed to run end-to-end on **Google Colab (T4 GPU)**. Data is downloaded and cached to Google Drive automatically.

---

## Data Sources

### BMRB (Biological Magnetic Resonance Data Bank)
- **URL:** https://bmrb.io
- **Format:** NMR-STAR v3 (`.str` files)
- **Content used:** Backbone ¹H and ¹⁵N chemical shifts (`Atom_chem_shift` loop), optional relaxation parameters (T₁, T₂, T₁ρ, NOE, S²)
- **Total entries with linked PDB:** 8,712

### RCSB PDB (Protein Data Bank)
- **URL:** https://rcsb.org
- **Format:** PDB format (`.pdb` files)
- **Content used:** 3D atomic coordinates, NMR ensemble models (for RMSD calculation), B-factors (X-ray structures)
- **Preferred:** NMR structures (multi-model PDB) for ensemble RMSD; X-ray structures for precise single-model coordinates

### BMRB→PDB Mapping
Retrieved via BMRB REST API: `https://api.bmrb.io/v2/mappings/bmrb/pdb`

---

## Data Pipeline

```
BMRB .str files (8,712)          PDB .pdb files
        │                                │
        ▼                                ▼
parse_hsqc_shifts()         complete_hydrogens()  ← pdbfixer, pH 7.0
 └─ extract ¹H + ¹⁵N SCS    load_residue_atoms()
    per residue              └─ NMR: ensemble RMSD
                                X-ray: normalised B-factor
        │                                │
        └──────────┬────────────────────┘
                   ▼
        _find_bmrb_pdb_offset()
        ├─ search offset ∈ [-10, +10]
        ├─ maximise (seq_id + offset, res_name) overlap
        └─ apply best offset to BMRB seq_ids

                   ▼
        embed_residue()  →  73-dim continuous + 2 index dims = 75-dim node feature
        build_edge_features()  →  37-dim edge feature
        compute_scs()  →  obs − _RC_SHIFTS[res_name]

                   ▼
        SCS outlier filter:
        ├─ |Δδ_HN|  > 3.0 ppm  → discard residue
        └─ |Δδ_¹⁵N| > 15.0 ppm → discard residue

                   ▼
        Data(x, edge_index, edge_attr, y, weight)
        cached to graphs.pkl with CACHE_VERSION tag

                   ▼
        70/20/10 protein-level split  (4337 / 1240 / 620)
        global SCS normalisation (μ, σ per head, from train split)
```

**Skip rate:** ~28% of entries discarded due to:
- No BMRB↔PDB sequence match even after offset search (~10%)
- No ¹⁵N data in BMRB entry (~15%)
- Fewer than 5 matched residues after filtering (~3%)

---

## Feature Engineering

### Node Features (75 dims per residue)

| Block | Features | Dims | Notes |
|-------|----------|------|-------|
| **Geometry continuous** | aa_oh(20) + sin/cos φψω(6) + sin/cos χ₁₋₄(8) + χ_mask(4) + Cβ_dir_local(3) | 41 | φ,ψ,ω,χ encoded as sin/cos pairs to preserve periodicity |
| **Neighbour aa index** | aa_prev_idx + aa_next_idx | 2 | Integer indices for Embedding lookup (20=missing/terminus) |
| **Ring current** | Haigh-Mallion sum over all aromatic rings (Phe/Tyr/Trp/His) | 1 | Δδ_RC = Σ B_j(1−3cos²θ)/r³ |
| **SASA** | total_norm + backbone_norm (FreeSASA, Tien 2013 reference) | 2 | Normalised to [0,1] per residue type |
| **H-bond geometry** | exists + r_N···O + r_H···O + cos∠NHO + cos∠HOC + count + strength_sum | 7 | KD-tree O-atom search, r_H···O < 2.5 Å threshold |
| **Ensemble RMSD** | rmsd_backbone + rmsd_sidechain | 2 | NMR: true RMSD; X-ray: normalised B-factor |
| **Electrostatics** | φ_DH + E_z (Debye-Hückel, pH 7, κ=0.125 Å⁻¹) | 2 | Full-protein sum, no spatial cutoff |
| **n→π* interaction** | d_O···C + cos∠(O···C=O) | 2 | Bartlett 2010 geometry; active when d < 3.2 Å |
| **AM1 charges** | q_N + q_H + q_C=O (GFN1-xTB heuristic) | 3 | Simplified; full xTB optional |
| **Dynamics** | S² + Rex_proxy + τₑ + T1ρ_norm + train_weight + B_bb + B_sc + disorder + rotamer_entropy | 9 | From BMRB relaxation data where available |
| **Dynamics mask** | bmrb_avail + ensemble_avail + bfactor_avail + seq_avail | 4 | Binary flags for data-source availability |

### Edge Features (37 dims per directed edge)

| Feature | Dims | Description |
|---------|------|-------------|
| Sinusoidal seq encoding | 8 | enc[2k]=sin(Δ/10^(2k/8)), enc[2k+1]=cos(Δ/10^(2k/8)); encodes signed sequence distance Δ=j−i |
| RBF distance | 16 | Gaussian radial basis functions, μ ∈ [2, 20] Å, σ=1.2 Å |
| Direction in local frame | 3 | Unit vector Cαᵢ→Cαⱼ expressed in residue i's N-Cα-C frame |
| Relative orientation | 6 | First 2 columns of R_frame_j @ R_frame_i^T (SO(3) rotation) |
| Bond type | 4 | One-hot: (peptide_bond, seq_neighbour, spatial_contact, h_bond) |

**Graph construction:**
- Sequence edges: all residues within |i−j| ≤ 4 in the same chain
- Spatial edges: Cα–Cα distance ≤ 10 Å
- Edges are directed and undirected (both i→j and j→i stored)

### y-Target

**Secondary chemical shifts (SCS):**

```
Δδ_HN  = δ_HN_obs  − δ_HN_RC(res_name)
Δδ_¹⁵N = δ_¹⁵N_obs − δ_¹⁵N_RC(res_name)
```

Random-coil reference values `_RC_SHIFTS` sourced from Kjaergaard & Poulsen (2011). SCS values are then globally standardised using training-set mean and standard deviation before model input.

---

## Model Architecture

### NMRShiftGNN

```
Input x: [N, 75]
  ├─ x[:, :-2]  (73 continuous) → node_proj: Linear(73, 256) → h: [N, 256]
  ├─ x[:, :20].argmax()         → aa_embed(21, 8)           → emb_i: [N, 8]
  ├─ x[:, -2].long()            → aa_embed(21, 8)           → emb_prev: [N, 8]
  └─ x[:, -1].long()            → aa_embed(21, 8)           → emb_next: [N, 8]
                                   cat([emb_i, emb_prev, emb_next]) → [N, 24]
                                   aa_embed_proj: Linear(24, 256)   → [N, 256]
                                   h = h + aa_embed_proj(emb)

edge_attr: [E, 37] → edge_proj: Linear(37, 256) → ea: [E, 256]

GATv2Conv × 4 layers:
  each: GATv2Conv(256, 64, heads=4, edge_dim=256) → LayerNorm(256) → residual

Output:
  head_HN:  Linear(256, 64) → ReLU → Dropout(0.2) → Linear(64, 1)  → Δδ_HN_norm
  head_15N: Linear(256, 64) → ReLU → Dropout(0.2) → Linear(64, 1)  → Δδ_¹⁵N_norm
  aa_bias:  Embedding(20, 2)[aa_idx]   → residue-type output correction

Total parameters: ~861,000
```

### NMRShiftMLP (baseline)

```
Input: cat(x[:, :-2], aa_emb(i/prev/next)) → [N, 73+24=97]
net: Linear(97, 256) → LN → ReLU → Dropout → Linear(256, 256) → LN → ReLU → Dropout → Linear(256, 2)
Total parameters: ~92,000
```

### Loss Function

**Weighted log-cosh loss (per-head):**

```
L = mean_over_residues( w_i × [log cosh(ŷ_HN,i − y_HN,i) + log cosh(ŷ_N,i − y_N,i)] )
```

Where `w_i` is the per-residue dynamics weight (geometric mean of S², RMSD-derived weight, and disorder propensity). Log-cosh is preferred over Huber loss for its smooth gradient (bounded by `tanh(x)`) at large residuals, improving robustness to occasional mis-aligned residues.

---

## Training Configuration

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Optimiser | AdamW | |
| Initial LR | 1×10⁻³ | |
| LR schedule | Linear warmup (10 ep) + Cosine decay | final LR = 1×10⁻⁵ |
| Weight decay | 5×10⁻⁴ | |
| Dropout | 0.2 | applied in GATv2 attention + output heads |
| DropEdge | 0.0 | disabled — harmful for sparse protein graphs |
| Batch size | 32 proteins | |
| Max epochs | 500 | |
| Early stopping | patience=60 on EMA(α=0.1) of val MAE_H + MAE_N | |
| Seed | 42 | |

---

## Results Summary

### v9 Final Results (Test Set, 620 proteins)

| Model | MAE ¹H | MAE ¹⁵N | r ¹H | r ¹⁵N |
|-------|--------|---------|------|-------|
| GNN (v9) | 0.311 ppm | **1.775 ppm** | 0.732 | 0.777 |
| MLP baseline | 0.416 ppm | 2.333 ppm | 0.476 | 0.602 |
| GNN advantage | −25% | −24% | +0.256 | +0.175 |
| SPARTA+ (literature) | ~0.25 ppm | ~1.80 ppm | ~0.90 | ~0.88 |

The GNN outperforms the sequence-only MLP baseline by ~25% in MAE and ~+0.20 in r for both nuclei, confirming that graph-encoded 3D structural information provides substantial predictive value beyond sequence features alone.

### Progression Across Versions

```
Version  ¹H MAE   ¹⁵N MAE   r_H   r_N   Key change
───────  ───────  ────────  ────  ────  ──────────────────────────────────
v5       0.470    3.065     0.43  0.31  Initial build, 56 proteins
v6       0.318    1.801     0.72  0.77  Physics features + 4337 proteins ★
v7       0.361    1.988     0.63  0.73  DropEdge regression (p=0.1)
v7+      0.312    1.738*    —     —     DropEdge disabled, val metrics
v8       0.207    6.587     0.71  0.77  A2 centering broke ¹⁵N
v9       0.311    1.775     0.73  0.78  A2 reverted; best balanced ★★
```
*val metric at ep 250

---

## Limitations and Future Work

### Current limitations

1. **Ligand / metal blindness:** HETATM records (small molecules, metals) are entirely ignored. Paramagnetic metals (Fe, Cu, Co, Mn) in training entries corrupt the loss. Estimated ~30% of BMRB entries are affected.

2. **Static structure assumption:** A single PDB model is used. Solution-state chemical shifts reflect a time-averaged ensemble (ps–ns dynamics). Fast internal motions are partially captured via the dynamics feature block but not structurally.

3. **Reference standard noise:** ~15% of BMRB entries contain unremediated referencing offsets (< 1 ppm for ¹H, < 3 ppm for ¹⁵N). The current SCS outlier filter (|Δδ| > 3/15 ppm) removes extreme cases only.

4. **Simple random-coil reference:** `_RC_SHIFTS` are single-value per residue type. pH, temperature, and ±2 sequence-context corrections (as in POTENCI) are not applied.

5. **No PTM support:** Phosphorylated, glycosylated, or otherwise modified residues are treated as their unmodified parent type.

### Planned improvements

- [ ] LACS-based ¹H/¹⁵N re-referencing in preprocessing
- [ ] POTENCI sequence-context random-coil corrections
- [ ] Paramagnetic metal detection and entry filtering
- [ ] Antidiamagnetic metal coordination feature (Zn, Ca, Mg)
- [ ] BMRB relaxation data (S², T₁, T₂) integration at scale
- [ ] Optional MD trajectory input for dynamic averaging

---

## Environment Setup

### Requirements
```
python >= 3.10
torch >= 2.0
torch-geometric >= 2.3
torch-scatter, torch-sparse
biopython >= 1.81
pynmrstar >= 3.3
freesasa >= 2.1
pdbfixer >= 1.9
openmm >= 8.0
scipy >= 1.10
scikit-learn >= 1.3
pandas >= 2.0
matplotlib >= 3.7
seaborn >= 0.12
tqdm
requests
```

### Google Colab (recommended)
All notebooks install dependencies automatically in the first cell. No manual setup required beyond mounting Google Drive.

### Local installation
```bash
conda create -n nmr-gnn python=3.10
conda activate nmr-gnn

# Install PyTorch (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyG
pip install torch-geometric
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install remaining dependencies
pip install biopython pynmrstar freesasa pdbfixer openmm \
            scipy scikit-learn pandas matplotlib seaborn tqdm requests
```

---

## Quick Start

### Google Colab
1. Upload `nmr_hsqc_colab_v9.ipynb` to Google Colab
2. Connect to a T4 GPU runtime (`Runtime > Change runtime type > T4 GPU`)
3. Mount Google Drive when prompted
4. Edit the **USER CONFIG** cell to set `DRIVE_ROOT` and optionally `FEATURE_LEVEL`
5. Run all cells (`Ctrl+F9`)

Data download and graph construction take ~3 hours for the full 8,646-entry dataset. Subsequent runs use the cached `graphs.pkl`.

### Predict chemical shifts for a new PDB file
```python
# After training, run the predict_pdb cell:
df = predict_pdb("/path/to/my_protein.pdb", gnn_model, is_gnn=True)
print(df)
# Output columns: chain, seq_id, res_name,
#                 pred_1H_scs, pred_15N_scs,   ← secondary chemical shifts (ppm)
#                 pred_1H_abs, pred_15N_abs     ← absolute chemical shifts  (ppm)
```

---

## Citation

If you use this code or the trained models in your research, please cite:

```bibtex
@misc{nmr-hsqc-gnn,
  title  = {NMR-HSQC-GNN: Graph Neural Network Prediction of Protein Amide Chemical Shifts},
  year   = {2025},
  note   = {GitHub repository},
  url    = {https://github.com/[your-username]/nmr-hsqc-gnn}
}
```

### Key references for methods used
- **BMRB:** Ulrich et al. (2008) *Nucleic Acids Research* 36, D402–D408
- **GATv2:** Brody et al. (2022) *ICLR 2022*
- **Ring current model:** Haigh & Mallion (1979) *Progress in NMR Spectroscopy*
- **Random coil SCS:** Kjaergaard & Poulsen (2011) *J. Biomol. NMR* 50, 157–165
- **SPARTA+:** Shen & Bax (2010) *J. Biomol. NMR* 48, 13–22
- **n→π* geometry:** Bartlett et al. (2010) *J. Am. Chem. Soc.*
- **¹⁵N preceding residue effect:** Wang & Jardetzky (2004) *J. Biomol. NMR* 28, 327–340

---

*Last updated: v9 — May 2025*
