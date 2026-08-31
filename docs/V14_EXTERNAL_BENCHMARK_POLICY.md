# v14 external benchmark policy

## Status

This policy was frozen after the v13 external-blind evaluation and before v14 training-data expansion.

External evaluation commit:

- Predictions: `0908fda`
- Formal evaluation: `d83e6e3`
- Generalization comparison: `bb28f0e`
- Known-QC sensitivity: `8e31c10`

## Frozen external cohort

The 40 BMRB/PDB entries in `benchmarks/v13_external_blind/` are evaluation-only.

They must not be used for:

- v14 training;
- validation or early stopping;
- feature selection;
- hyperparameter selection;
- architecture selection;
- reference-offset fitting;
- post-hoc prediction calibration.

The cohort may be evaluated again only after the v14 training recipe and model-selection decision are frozen.

## Main findings

- External 1H median absolute error was nearly unchanged, but upper-tail and extreme errors increased.
- External 15N degradation was distributed more broadly across the error distribution.
- Entry size showed little association with entry-level MAE.
- No external entry met the frozen entry-wide residual outlier rule.
- BMRB 4242 remains in the primary internal test report; its exclusion is sensitivity analysis only.

## v14 development priorities

1. Expand the development dataset while excluding both the original v13 entries and the frozen external cohort.
2. Preserve homology-separated connected-component splitting at the protein/entry level.
3. Use entry-balanced training or sampling so long chains do not dominate optimization.
4. Test targeted structural features before increasing MLP capacity.
5. Investigate 1H extreme-tail cases for missing atoms, alternate locations, chain breaks, termini, ligands, and multi-chain environments.

## Referencing policy

- No chemical shifts are corrected.
- No hierarchical b_j is used.
- Entry-wide residual flags are diagnostics and are not proof of misreferencing.
- Reference-related exclusions require independent metadata or experimental evidence.

## Model-selection policy

All v14 decisions must be made using newly constructed training and validation partitions.

The frozen 40-entry external benchmark is not eligible for model selection.
