# NMR HSQC Chemical Shift Predictor — Version History

> **How to use this document:**  Paste each version's section into the first Markdown cell of its corresponding notebook. The performance charts are rendered inline using Mermaid-compatible tables and ASCII charts — or simply read the numbers directly.

---

## Performance Overview (GNN Test Set)

| Version | ¹H MAE (ppm) | ¹⁵N MAE (ppm) | r ¹H | r ¹⁵N | GNN Params | Train Proteins |
|---------|-------------|--------------|------|-------|-----------|----------------|
| v5      | 0.470       | 3.065        | 0.43 | 0.31  | ~852k     | 39             |
| v6      | 0.318       | 1.801        | 0.72 | 0.77  | ~235k     | 4,337          |
| v7      | 0.361       | 1.988        | 0.63 | 0.73  | ~235k     | 4,337          |
| v7+     | 0.312       | 1.738 (val)  | —    | —     | ~235k     | 4,337          |
| v8      | 0.207       | 6.587 ⚠️     | 0.71 | 0.77  | ~861k     | 4,337          |
| **v9**  | **0.311**   | **1.775 ✓**  | **0.73** | **0.78** | **~861k** | **4,337** |
| Target  | < 0.30      | < 2.00       | > 0.90 | > 0.90 | —       | —              |

```
¹H MAE progress  (lower = better)
v5 ████████████████████████  0.470
v6 ████████████████          0.318
v7 ██████████████████        0.361
v9 ████████████████          0.311 ← current best balanced
v8 ███████████               0.207 ← ¹H only (¹⁵N broken)
   target: ──────────────── 0.300

¹⁵N MAE progress (lower = better)
v5 ██████████████████████████████  3.065
v6 ██████████████████              1.801 ✓
v7 ████████████████████            1.988 ✓
v9 █████████████████               1.775 ✓ best
v8 █████████████████████████████████████████████████████████████  6.587 ✗
   target: ████████████████████  2.000

Pearson r ¹H (higher = better, target > 0.90)
v5  ████████████████████████  0.43
v6  ████████████████████████████████████  0.72
v9  ████████████████████████████████████  0.73 ← best
    target: ██████████████████████████████████████████████  0.90

Pearson r ¹⁵N (higher = better, target > 0.90)
v5  ████████████████  0.31
v6  ██████████████████████████████████████  0.77
v9  ████████████████████████████████████████  0.78 ← best
    target: ██████████████████████████████████████████████  0.90
```

---

## v5 — Proof of Concept (Initial Build)

### What this version does
First end-to-end trainable pipeline. Demonstrates that raw atom-stacking features fed into a GCN can produce non-trivial chemical shift predictions.

### Architecture
- **Model:** `GCNConv` (basic graph convolution, no edge features)
- **Node features:** Per-residue atom neighbour stacking — every neighbouring atom's Cartesian coordinates converted to local frame, plus element one-hot → flattened to fixed `max_nb × FPB = 128 × 14 = 1792` dims
- **Edge features:** None
- **Output:** `[N, 2]` — absolute ¹H and ¹⁵N chemical shifts
- **Loss:** MSE, unweighted

### Data
- **BMRB entries processed:** ~100 (download cap)
- **Valid proteins built:** 56 (44 skipped — no residue-number alignment attempted)
- **Train / Val / Test split:** 39 / 11 / 6 proteins

### Key problems identified
1. ¹⁵N output collapsed to predict the training mean (~0 SCS) — model learned to ignore ¹⁵N
2. Val set of 11 proteins gave highly noisy MAE estimates → unreliable early stopping
3. `MAX_ENTRIES=100` gave far too few training samples for 852k-parameter network (param/sample ratio ≈ 426)
4. No BMRB↔PDB residue-number alignment → 44% skip rate

### Test metrics
| Metric | GNN | MLP |
|--------|-----|-----|
| MAE ¹H | 0.470 | 0.497 |
| MAE ¹⁵N | 3.065 | 3.369 |
| r ¹H | 0.43 | 0.25 |
| r ¹⁵N | 0.31 | 0.21 |

---

## v6 — Scale-Up + Physical Feature Engineering

### What this version does
The most impactful single upgrade. Replaces the ad-hoc atom-stacking representation with a principled physics-based feature set, scales the dataset from 56 to 4,337 training proteins, and fixes the ¹⁵N collapse.

### Architecture
- **Model:** `GATv2Conv` with edge features (4 layers, 4 attention heads, hidden=128)
- **Node features (73 dims):**
  - Geometry (21): aa one-hot(20) + sin/cos φψω(6) + sin/cos χ₁₋₄(8) + χ_mask(4) + Cβ_dir(3)
  - Physics HIGH (12): ring_current(1) + SASA(2) + H-bond(7) + ensemble_RMSD(2)
  - Physics MED (7): electrostatics(2) + n→π*(2) + AM1_charges(3)
  - Dynamics (9): S², Rex, τₑ, T1ρ, training_weight, B_bb, B_sc, disorder, rotamer_entropy
  - Dynamics mask (4): data-source availability flags
- **Edge features (30 dims):** delta_seq(1) + RBF(16) + r̂_local(3) + R_rel(6) + bond_type(4)
- **Output:** Secondary chemical shifts (SCS = obs − random_coil), standardised per training set
- **Loss:** Weighted Huber loss with per-residue dynamics weight

### Data
- `MAX_ENTRIES` raised to 5000
- **BMRB↔PDB offset search** `[-10, +10]` residue-number alignment — reduces skip rate from 44% → 28%
- **SCS outlier filter:** |Δδ_HN| > 3.0 ppm or |Δδ_¹⁵N| > 15.0 ppm → discard (removes mis-referenced entries)
- **H-atom completion** via `pdbfixer` (fast-path skips NMR PDB files that already contain H)
- **Valid proteins:** 6,197 built; 4,337 / 1,240 / 620 train/val/test split
- **y-target:** Secondary chemical shift (SCS), globally standardised

### Key improvements over v5
- ¹⁵N MAE: 3.065 → **1.801** (−41%) — target met
- ¹H r: 0.43 → **0.72** (+0.29)
- Dataset 78× larger; skip rate halved
- Physics-informed features replace raw atom coordinates
- Per-head normalised loss prevents ¹⁵N collapse
- EMA smoothed early stopping (α=0.25, patience=40)

### Test metrics
| Metric | GNN | MLP |
|--------|-----|-----|
| MAE ¹H | 0.318 | 0.433 |
| MAE ¹⁵N | 1.801 ✓ | 2.600 |
| r ¹H | 0.72 | 0.42 |
| r ¹⁵N | 0.77 | 0.49 |

---

## v7 — Regularisation + Neighbour Encoding Experiments

### What this version does
Applies 5 targeted optimisations based on convergence analysis of v6. Introduces P2 (explicit i±1 residue type) and tests regularisation strategies.

### Architecture changes from v6
- **P1 Regularisation:** dropout 0.1→0.2, weight_decay 1e-4→5e-4, **DropEdge p=0.1**
- **P2 Neighbour aa types:** aa_prev(20) + aa_next(20) appended to geometry → node_dim 73→113
- **P3 LR warmup:** 10-epoch linear warmup + cosine decay to 1e-5
- **P4 Deeper network:** NUM_LAYERS 3→4 (receptive field now covers i±4, matching α-helix H-bond range)
- **P5 Log-cosh loss:** replaces Huber loss (smoother gradient at large residuals)
- **Training params:** EMA_ALPHA=0.25, PATIENCE=40, EPOCHS=300

### Key findings
- **DropEdge (P1) caused significant regression:** ¹H r 0.72→0.63, MAE 0.318→0.361
  - Root cause: protein graph edges are already sparse; random deletion cuts the exact structural signal (H-bond contacts, helix topology) the GNN needs
  - Identified via ablation: disabling DropEdge recovered ¹H MAE to 0.312 (v7+)
- **P2 one-hot appending** increased node_dim to 113 but 40 of those dims are sparse binary → poor signal/noise ratio
- **GNN early-stopped at ep 97** due to EMA_ALPHA=0.25 being too aggressive (false early stop)
- **v7+ run** (DROP_EDGE_P=0, 250 epochs): val ¹H=0.312, val ¹⁵N=1.738 — best single-run results at this point

### Lessons learned
1. DropEdge is harmful for sparse protein graphs — confirmed removed in v8+
2. Sparse one-hot neighbour encoding is inefficient — replaced by learned embedding in v8
3. EMA_ALPHA must be lowered (0.25→0.1) for stable early stopping

### Test metrics (with DropEdge=0.1)
| Metric | GNN | MLP |
|--------|-----|-----|
| MAE ¹H | 0.361 | 0.410 |
| MAE ¹⁵N | 1.988 ✓ | 2.290 |
| r ¹H | 0.63 | 0.50 |
| r ¹⁵N | 0.73 | 0.62 |

---

## v8 — Learned Embeddings + Architectural Improvements

### What this version does
Replaces sparse one-hot neighbour encoding (P2) with a compact learned embedding, upgrades the model to 256 hidden dims, and introduces sinusoidal sequence distance encoding for edges.

### Architecture changes from v7
- **A1 aa_embed:** `Embedding(21, 8)` shared for residue types i, i-1, i+1; three embeddings concatenated → `[N, 24]` → `aa_embed_proj: Linear(24, hidden)` added to initial hidden state; node_dim 113→75
- **A3 Sinusoidal edge encoding:** scalar delta_seq(1) replaced by 8-dim sinusoidal basis; edge_dim 30→37
- **A4 Wider model:** HIDDEN_DIM 128→256 → ~861k parameters
- **Training:** DROP_EDGE_P=0.0, EMA_ALPHA=0.1, PATIENCE=60, EPOCHS=500
- **A2 Per-protein y centering (REVERTED in v9):** attempted to subtract per-protein SCS mean

### Key findings
- **¹H MAE 0.207 — best ever** (aa_embed + wider model combination works for ¹H)
- **¹⁵N MAE exploded to 6.587 ppm** — A2 per-protein centering was the cause
  - Centering amplified ¹⁵N target variance: σ_centred(¹⁵N) ≈ 4.2 ppm > σ_raw ≈ 3.1 ppm
  - HSQC predictions scattered into physically impossible range (70–155 ppm)
- Cache invalidation system (`CACHE_VERSION`) introduced to auto-detect and rebuild stale `graphs.pkl` when feature dims change

### Test metrics
| Metric | GNN | MLP |
|--------|-----|-----|
| MAE ¹H | **0.207 ✓** | 0.259 |
| MAE ¹⁵N | 6.587 ✗ | 8.750 ✗ |
| r ¹H | 0.71 | 0.53 |
| r ¹⁵N | 0.77 | 0.57 |

---

## v9 — A2 Revert + Stable Best Model (Current)

### What this version does
Reverts the broken A2 per-protein centering while retaining all working improvements from v8 (A1 aa_embed, A3 sinusoidal edges, A4 hidden=256). Achieves best **balanced** performance across both targets simultaneously.

### Changes from v8
- **A2 reverted:** `process_protein` stores raw SCS `Y` directly (no centering)
- **CACHE_VERSION bumped:** `v8_*` → `v9_*` triggers automatic stale cache invalidation and rebuild
- `y_mean` removed from `Data` attributes and `exclude_keys`
- `collect()` in `evaluate` restored to simple `DataLoader` loop

### Architecture (final state)
```
Input node features  [N, 75]:
  Continuous (73):
    aa_oh(20) + bb_enc sin/cos φψω(6) + χ_enc(8) + χ_mask(4) + Cβ_dir(3)   = 41
    ring_current(1) + SASA(2) + H-bond(7) + RMSD(2)                          = 12
    electrostatics(2) + n→π*(2) + AM1_charges(3)                              =  7
    dynamics S²/Rex/τₑ/T1ρ/w/B_bb/B_sc/disorder/rotamer(9) + mask(4)        = 13
  Index (2): aa_prev_idx, aa_next_idx  → aa_embed(21,8) → proj(24,256)

Input edge features  [E, 37]:
  sinusoidal_seq(8) + RBF_dist(16) + r̂_local(3) + R_rel_flat(6) + bond_type(4)

GATv2Conv × 4 layers, 4 heads, hidden=256
Output heads: Linear(256,64)→ReLU→Linear(64,1) × 2
aa_bias: Embedding(20,2) — per-residue-type output correction
```

### Training configuration
| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| LR | 1e-3 (10-ep warmup) → cosine → 1e-5 |
| Weight decay | 5e-4 |
| Dropout | 0.2 |
| DropEdge | 0.0 |
| Loss | Log-cosh, per-head, dynamics-weighted |
| EMA smoothing | α=0.1 |
| Patience | 60 |
| Epochs | 500 (early-stopped at 198) |
| Batch size | 32 |

### Test metrics
| Metric | GNN | MLP | Target |
|--------|-----|-----|--------|
| MAE ¹H | 0.311 | 0.416 | < 0.30 |
| MAE ¹⁵N | **1.775 ✓** | 2.333 | < 2.00 |
| r ¹H | 0.732 | 0.476 | > 0.90 |
| r ¹⁵N | 0.777 | 0.602 | > 0.90 |
| GNN advantage | ¹H −25%, ¹⁵N −24% | | |

### Remaining limitations
1. **Dynamic averaging:** single static PDB ≠ solution-state ensemble; fast motion (ps–ns) averages not captured
2. **Referencing noise:** ~15% of BMRB entries have unremediated systematic offsets; only extreme values (|SCS| > 3/15 ppm) filtered
3. **Random coil reference:** `_RC_SHIFTS` uses single values per residue type; pH/T/sequence-context corrections (POTENCI) not yet integrated
4. **Ligand/metal blind spot:** HETATM records fully ignored; paramagnetic metals (Fe, Cu, Co) in training set add uncorrectable noise
5. **r ceiling at ~0.73/0.78:** estimated 0.12–0.15 r gap attributable to (1)+(2)+(3) combined

