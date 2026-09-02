# v14 expanded-data ensemble release candidate

## Status

This directory is a release candidate assembled from the
validation-selected v14 ensemble. It must pass an independent
clean-process reload test before it is designated deployable.

## Model

- Architecture: unchanged nucleus-specific Step 33 MLP
- Ensemble: arithmetic mean of five independently seeded models
- Seeds: 20250830, 20250831, 20250832, 20250833, 20250834
- Training rows: 6,013
- Training entries: 69
- Base features: 68
- 1H structure features: 6
- 15N structure features: 3
- Angle source: full-chain PDB/mmCIF geometry

## Validation selection

- Frozen v13 primary metric: 0.609763
- v14 primary metric: 0.598542
- v14-minus-v13 delta: -0.011221
- Evidence: stable_improvement
- Promotion supported: true

## One-time internal-test report

- 1H MAE: 0.462731 ppm
- 1H RMSE: 0.598490 ppm
- 15N MAE: 2.782977 ppm
- 15N RMSE: 3.701503 ppm

Internal-test results are reporting-only.

## Example inference command

    python example_predict.py structure.cif --chain A \
        --output hsqc_predictions.csv

## Referencing policy

- Reference correction: none
- Hierarchical b_j: not used
- Chemical-shift targets modified: no

## External evaluation

The frozen external-blind cohort remains unopened. It is reserved
for final reporting after clean-reload verification and release
freezing.
