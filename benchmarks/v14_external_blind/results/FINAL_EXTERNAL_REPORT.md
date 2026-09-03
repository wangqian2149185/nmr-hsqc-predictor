# Final v14 external-blind evaluation

## Evaluation integrity

- Frozen external entries: 40
- Frozen paired H/N targets: 5,389
- Prediction coverage: 100%
- Post-result entry exclusions: none
- Post-result target exclusions: none
- External results used for model selection: no
- Frozen deployable model modified: no
- Reference correction applied: no
- Hierarchical b_j used: no

## v14 primary external metrics

| Metric | Value |
|---|---:|
| 1H residue-micro MAE | 0.465025 ppm |
| 1H residue-micro RMSE | 0.622658 ppm |
| 1H entry-macro MAE | 0.462911 ppm |
| 1H entry-macro RMSE | 0.612173 ppm |
| 15N residue-micro MAE | 2.800198 ppm |
| 15N residue-micro RMSE | 3.708013 ppm |
| 15N entry-macro MAE | 2.792064 ppm |
| 15N entry-macro RMSE | 3.627918 ppm |
| CSP_6.51 entry-macro mean | 0.693843 ppm |

## Same-cohort comparison with frozen v13

- 1H entry-macro MAE: 0.469806 to 0.462911 ppm (1.47% lower).
- 15N entry-macro MAE: 2.842169 to 2.792064 ppm (1.76% lower).
- CSP_6.51 entry-macro mean: 0.704658 to 0.693843 ppm (1.53% lower).

All five predeclared paired-entry bootstrap comparisons were classified as stable v14 improvements.

## Numerical verification

CSV round-trip residual reconstruction used a fixed 1e-05 ppm tolerance. The maximum observed reconstruction difference remained below this threshold.

## Uncertainty limitation

Ensemble SD has weak rank correlation with absolute error: Spearman rho=0.040 for 1H and rho=0.110 for 15N.
Ensemble SD is therefore a model-disagreement indicator, not a calibrated prediction-error confidence interval.

## Conclusion

The locked external evaluation supports a statistically stable but modest improvement of v14 over v13. It is not a large absolute accuracy breakthrough. Further progress should prioritize a substantially larger independent training-entry set and predeclared loss-scaling experiments.
