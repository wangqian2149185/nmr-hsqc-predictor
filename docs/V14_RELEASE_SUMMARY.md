# v14 expanded-data release summary

## Status

- Branch: `codex_v14`
- Frozen deployable release: `releases/v14_deployable_expanded_data`
- Deployable release commit: `fdf0bf99301c92e8392ed986cada3a2735cd284c`
- Final external-evaluation commit: `3550e1fa4423c2dc3a96d4d0e0bcea8c0535a21d`
- Reference correction: none
- Hierarchical entry offset `b_j`: not used

## Model

v14 retains the deployment-safe five-seed nucleus-specific MLP architecture established in v13. Its full-chain PDB features can be generated without chemical-shift assignment coverage.

The main v14 change is the development dataset:

- 69 training entries and 6,013 training residues;
- 10 model-selection validation entries and 703 residues;
- 10 one-time internal-test entries and 698 residues;
- 40 permanently isolated external entries and 5,389 residues.

Preprocessing and target standardization were fitted using training data only.

## Model-selection validation

On the frozen 10-entry v14 validation cohort, v14 improved the combined standardized entry-level metric relative to the frozen v13 model:

- v13: 0.609763
- v14: 0.598542
- v14 minus v13: -0.011221
- paired-entry bootstrap 95% CI: [-0.017112, -0.004595]
- probability of improvement: 0.99932

The model was selected using validation data only.

## One-time internal test

| Metric | Value |
|---|---:|
| 1H residue MAE | 0.462731 ppm |
| 1H residue RMSE | 0.598490 ppm |
| 1H entry-macro MAE | 0.469330 ppm |
| 15N residue MAE | 2.782977 ppm |
| 15N residue RMSE | 3.701503 ppm |
| 15N entry-macro MAE | 2.757390 ppm |

The internal test was opened once after model selection and was not used to modify the model.

## Final external-blind evaluation

The final external protocol was locked before v14 inference. All 40 entries and all 5,389 paired targets were retained.

| Metric | v13 | v14 | v14 change |
|---|---:|---:|---:|
| 1H entry-macro MAE | 0.469806 | 0.462911 | -1.47% |
| 15N entry-macro MAE | 2.842169 | 2.792064 | -1.76% |
| CSP 6.51 entry-macro mean | 0.704658 | 0.693843 | -1.53% |

The paired-entry bootstrap classified the predeclared 1H MAE/RMSE, 15N MAE/RMSE, and CSP comparisons as stable v14 improvements.

These improvements are statistically consistent but modest in absolute magnitude. v14 should be regarded as a stronger reproducible baseline, not as a large accuracy breakthrough.

## CSP convention

The reporting-only combined chemical-shift error is

`CSP = sqrt(error_H^2 + (error_N / 6.51)^2)`.

The 6.51 factor is an empirical proton/nitrogen chemical-shift scale convention. It was not used to alter the locked v14 training loss.

## Uncertainty limitation

Ensemble standard deviation correlated only weakly with absolute external error:

- 1H Spearman correlation: 0.040
- 15N Spearman correlation: 0.110

Ensemble SD is therefore reported as model disagreement, not as a calibrated confidence interval.

## Recommended next research stage

Do not tune v14 using the external cohort. A later development version should prioritize:

1. expanding from 69 to hundreds of independent training entries;
2. preserving entry-level and homology-aware separation;
3. predeclaring comparisons of training-standard-deviation scaling, CSP-oriented 6.51 scaling, uncertainty-weighted loss, and separate H/N models;
4. reserving a new untouched external evaluation cohort.

## Reproducibility artifacts

- [Deployable release](../releases/v14_deployable_expanded_data/)
- [Final external protocol](../benchmarks/v14_external_blind/protocol/)
- [Final external report](../benchmarks/v14_external_blind/results/FINAL_EXTERNAL_REPORT.md)
- [Final result manifest](../benchmarks/v14_external_blind/results/evaluation_manifest.json)
