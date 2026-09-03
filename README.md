# NMR HSQC Chemical-Shift Predictor


<!-- V14_RELEASE_SUMMARY_START -->
## Current verified release: v14

The current frozen deployment release is the [v14 expanded-data five-seed ensemble](releases/v14_deployable_expanded_data/).

- Training: 69 independent entries, 6,013 residues
- Model selection: 10-entry validation cohort
- Internal test: 10 entries, opened once after selection
- Final external blind test: 40 entries and 5,389 paired targets
- External coverage: 100%
- Reference correction: none
- Hierarchical `b_j`: not used

Final external entry-macro MAE:

| Model | 1H MAE | 15N MAE | CSP 6.51 mean |
|---|---:|---:|---:|
| v13 | 0.469806 ppm | 2.842169 ppm | 0.704658 ppm |
| v14 | 0.462911 ppm | 2.792064 ppm | 0.693843 ppm |

The frozen external comparison supports a stable but modest v14 improvement. Ensemble SD remains an uncalibrated model-disagreement indicator.

See the [v14 release summary](docs/V14_RELEASE_SUMMARY.md) and [final external report](benchmarks/v14_external_blind/results/FINAL_EXTERNAL_REPORT.md).
<!-- V14_RELEASE_SUMMARY_END -->

Prediction of protein backbone-amide ¹H and ¹⁵N chemical shifts from protein
structure.

The current verified release is **v13 Step 33**, a five-seed,
nucleus-specific neural-network ensemble with a standalone PDB inference
pipeline.

> Earlier v5–v11 GNN results are retained as historical development records.
> They should not be compared directly with v13 because the datasets, targets,
> features, and evaluation protocols differ. Review of the earlier workflow
> identified risks from data splitting and repeated test-set use.

## Current release

- **Branch:** `codex_v13`
- **Primary deployment model:** Step 33
- **Release directory:** [`releases/v13_deployable_step33/`](releases/v13_deployable_step33/)
- **Release notes:** [`docs/V13_STEP33_RELEASE.md`](docs/V13_STEP33_RELEASE.md)
- **Model card:** [`MODEL_CARD.md`](releases/v13_deployable_step33/MODEL_CARD.md)
- **Design foundation:** [`docs/V13_DESIGN.md`](docs/V13_DESIGN.md)

Step 33 replaced the original Step 14 deployment candidate because Step 14
constructed some torsion-angle masks using BMRB assignment coverage. That
information is unavailable for an arbitrary new PDB and made the Step 14
feature pipeline unsuitable for deployment.

Step 33 was retrained with torsion angles and masks derived only from the full
PDB chain. It reproduces its saved predictions directly from PDB coordinates.

## Dataset and evaluation protocol

The current v13 benchmark contains:

| Split | BMRB/PDB entries | Residue rows |
|---|---:|---:|
| Train | 39 | 3,916 |
| Validation | 8 | 854 |
| Test | 9 | 854 |
| **Total** | **56** | **5,624** |

Splits were built from connected components at **40% sequence identity**.
No connected homology component crosses train, validation, or test.

Train-only preprocessing is used for:

- target means and standard deviations;
- structural-feature medians;
- structural-feature means and standard deviations.

The validation set was used for model selection. The test set was used only
for final reporting.

## v13 Step 33 results

### Final ensemble metrics

| Split | ¹H MAE | ¹H RMSE | ¹⁵N MAE | ¹⁵N RMSE | ¹H entry-macro MAE | ¹⁵N entry-macro MAE |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.5041 | 0.6712 | 3.2660 | 4.2610 | 0.4998 | 3.1979 |
| Test, reporting only | 0.4438 | 0.5731 | 3.0904 | 4.1759 | 0.4416 | 2.9643 |

All values are in ppm.

### Step 33 versus Step 14 on validation entries

The paired bootstrap resampled complete BMRB entries rather than individual
residues.

| Nucleus | Entry-macro MAE delta, Step 33 − Step 14 | 95% bootstrap CI | Interpretation |
|---|---:|---:|---|
| ¹H | −0.00491 ppm | [−0.00894, −0.00013] | Stable improvement |
| ¹⁵N | +0.00764 ppm | [−0.01836, +0.03716] | Inconclusive |

Negative delta means Step 33 has lower error.

The validation set contains only eight independent entries, so these intervals
should not be interpreted as a definitive population-level benchmark.

## Model architecture

The Step 33 ensemble contains five independently seeded models.

Each model uses:

- a shared 68-feature sequence and backbone-torsion trunk;
- a six-feature ¹H structural branch;
- a three-feature ¹⁵N local-contact branch;
- separate ¹H and ¹⁵N output heads.

Shared trunk:

    Linear(68, 96)
    ReLU
    LayerNorm(96)
    Dropout(0.15)
    Linear(96, 64)
    ReLU

¹H structural branch:

    Linear(6, 16)
    ReLU

¹⁵N structural branch:

    Linear(3, 8)
    ReLU

Each nucleus-specific head maps its concatenated representation through a
32-unit hidden layer to one standardized chemical-shift prediction.

The final prediction is the arithmetic mean of five seeds:

    20250830
    20250831
    20250832
    20250833
    20250834

## Input features

### Shared sequence and backbone features

- one-hot identity of residue \(i\);
- one-hot identity of residues \(i-1\) and \(i+1\);
- fractional position within the PDB chain;
- transformed chain length;
- sine, cosine, and availability mask for φ;
- sine, cosine, and availability mask for ψ.

Torsion features are derived from the complete PDB chain and do not depend on
chemical-shift assignment coverage.

### ¹H structure features

- Cα contact counts within 6 Å, 8 Å, and 10 Å;
- nearest eligible backbone-oxygen distance;
- eligible backbone-oxygen count within 3.5 Å;
- hydrogen-bond proxy availability mask.

### ¹⁵N structure features

- Cα contact counts within 6 Å, 8 Å, and 10 Å.

The geometric environment includes all protein chains in the first structural
model. The target residue is excluded from Cα contact counts. For the
backbone-oxygen proxy, same-chain residues \(i-1\), \(i\), and \(i+1\) are
excluded.

## Standalone prediction

Install the release dependencies:

    pip install -r releases/v13_deployable_step33/requirements.txt

Predict from a PDB or mmCIF structure:

    python releases/v13_deployable_step33/example_predict.py       my_structure.pdb       --chain A       --output hsqc_predictions.csv

The output includes:

- PDB chain and residue identifiers;
- predicted ¹H shift in ppm;
- predicted ¹⁵N shift in ppm;
- across-seed ensemble standard deviations;
- explicit flags confirming that no reference correction or \(b_j\) was used.

The ensemble standard deviation measures disagreement across five training
seeds. It is not a calibrated predictive interval.

## Deployment verification

For PDB `3MSP`, chain `A`:

- 117 structurally eligible residues were predicted;
- all 114 residues with stored Step 33 evaluation predictions matched;
- the standalone module ran in a fresh Python subprocess;
- the CPU-versus-CUDA maximum difference was approximately
  \(1.23×10^{-6}\) ppm for ¹H and \(1.02×10^{-5}\) ppm for ¹⁵N;
- the documented cross-device acceptance tolerance is \(2×10^{-5}\) ppm.

The additional three predictions correspond to structurally eligible residues
without deposited evaluation assignments. Their presence demonstrates that
the inference pipeline does not require BMRB assignment coverage.

## Chemical-shift referencing policy

v13 does **not** apply chemical-shift re-referencing.

Specifically:

- deposited target shifts were not corrected;
- no learned entry-specific offset was used;
- no hierarchical \(b_j\) term was used;
- reference-QC flags are retained as metadata and sensitivity analyses.

Two entries were flagged by the train-derived ¹⁵N reference screen:

- **BMRB 4129 / PDB 1Q80**, training split;
- **BMRB 4242 / PDB 3MSP**, test split.

BMRB 4129 was retained in primary training because a corrected exclusion
experiment did not provide convincing validation evidence for removing it.

BMRB 4242 was retained in the primary test report. A secondary sensitivity
analysis excluding it reduced test ¹⁵N residue MAE from approximately 3.09 ppm
to 2.54 ppm. This sensitivity result was not used for model selection.

See
[`quality_control/`](releases/v13_deployable_step33/quality_control/)
for the frozen policy and entry-level screen outputs.

## Reproducibility

The release contains:

- five PyTorch checkpoints;
- exact ordered feature manifests;
- train-only preprocessing parameters;
- target scaling;
- training histories;
- ensemble predictions;
- validation paired-bootstrap results;
- shift-list provenance;
- reference-QC decisions;
- PDB backtests;
- a SHA-256 manifest.

The curated release manifest must match the release directory exactly:

[`release_manifest.json`](releases/v13_deployable_step33/release_manifest.json)

Existing v13 core tests can be run with:

    PYTHONPATH=src python -m pytest -q

## Important limitations

- The dataset contains only 56 independent entries.
- Validation contains eight entries and test contains nine.
- The current metrics are not yet an external blind benchmark.
- The model does not explicitly use pH, temperature, ligand state, dynamics,
  or multiple structural conformers.
- Unusual chemistry, modified residues, missing backbone atoms, paramagnetic
  systems, and structures outside the training distribution may be unreliable.
- The current ensemble spread is not calibrated uncertainty.
- Comparison with SPARTA+, SHIFTX2, UCBShift, or other predictors requires the
  same proteins, targets, referencing policy, and evaluation protocol.

## Recommended next experiment

The next priority is an external blind benchmark rather than further tuning on
the current validation or test entries.

The benchmark should:

1. select new BMRB/PDB entries before evaluating the model;
2. exclude entries homologous to the existing 56-entry dataset;
3. use the frozen release without retraining;
4. report residue-level and entry-macro metrics;
5. report suspected reference outliers as sensitivity analyses without
   correcting deposited targets.

## v14 development status

`codex_v14` is the active research branch. It inherits the frozen v13 Step 33
release without modifying its checkpoints, preprocessing, or reported metrics.

The first v14 objective is an external blind benchmark of the frozen v13 model.
Candidate eligibility and homology rules must be locked before model
evaluation. External benchmark results may guide later v14 development, but
must not be used to revise the frozen v13 model.

See [`docs/V14_PLAN.md`](docs/V14_PLAN.md).

## Historical development

Older notebooks, plots, and GNN experiments remain available from the frozen
[`codex_v13`](https://github.com/wangqian2149185/nmr-hsqc-predictor/tree/codex_v13)
branch and the repository's Git history.

Their reported metrics are historical and are not the current verified v13
benchmark.

## Data sources

- [Biological Magnetic Resonance Data Bank](https://bmrb.io/)
- [RCSB Protein Data Bank](https://www.rcsb.org/)

Users are responsible for complying with the source databases' terms,
attribution requirements, and data-quality limitations.

## Citation

This project is under active research development. Until a versioned archival
release is published, cite the repository and the exact Git commit used.

For the current v13 Step 33 release:

    Branch: codex_v13
    Release commit: 6e9984d2b39c06fdbf6ce36837e4eabc2484f5f9
