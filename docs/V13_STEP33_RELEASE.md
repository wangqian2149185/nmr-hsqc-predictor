# v13 Step 33 deployable release

The primary v13 deployment model is the five-seed Step 33
nucleus-specific ensemble under:

`releases/v13_deployable_step33/`

## Selection decision

Step 33 replaces Step 14 as the deployment model because Step 14's
training-time torsion masks depended on BMRB assignment coverage.

Step 33 uses full-chain PDB torsion angles. On the validation entries:

- 1H entry-macro MAE improved by approximately 0.00491 ppm;
- the paired entry bootstrap classified the 1H improvement as stable;
- the 15N difference was small and statistically inconclusive.

The test set was retained for reporting and was not used for model selection.

## Standalone inference

The release contains:

- `inference.py`;
- `example_predict.py`;
- five PyTorch checkpoints;
- ordered feature and preprocessing metadata;
- model and QC documentation;
- validation and deployment backtests;
- SHA-256 checksums.

Example command:

    python releases/v13_deployable_step33/example_predict.py       structure.pdb       --chain A       --output predictions.csv

Dependencies are listed in the release-local `requirements.txt`.

## Deployment validation

The standalone module was imported in a fresh Python subprocess and used to
predict PDB 3MSP chain A.

All 114 stored evaluation rows matched. The maximum CPU-versus-CUDA numerical
difference was approximately:

- 1H: 1.23e-6 ppm;
- 15N: 1.02e-5 ppm.

The documented cross-device acceptance tolerance is 2e-5 ppm.

## Referencing policy

- No chemical-shift reference correction is applied.
- No entry-specific or hierarchical b_j term is used.
- BMRB 4129 and BMRB 4242 remain marked as suspected 15N reference outliers.
- Deposited target shifts were not corrected.

See the release `MODEL_CARD.md`, `release_decision.json`, and validation/QC
directories for details.
