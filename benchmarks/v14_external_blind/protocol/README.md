# v14 final external-blind evaluation protocol

This directory freezes the one-time final external evaluation protocol before v14 inference.

Frozen inputs:

- external entries: 40
- paired H/N targets: 5389
- cached structures: 40
- deployable release commit: `fdf0bf99301c92e8392ed986cada3a2735cd284c`

Primary reporting includes residue-micro and entry-macro MAE/RMSE for 1H and 15N.

CSP is reporting-only and uses `sqrt(error_H^2 + (error_N / 6.51)^2)`.

No external entry or target may be removed after viewing v14 results. External results cannot be used to modify or select the frozen v14 model.
