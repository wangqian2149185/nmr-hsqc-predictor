# v15 buffered cohort assignment

The deterministic buffered allocation protocol was frozen before any
entry-level cohort assignment.

The frozen winner was then replayed exactly:

- winner trial: 300
- seed: 20560902
- role order: internal test, model-selection validation, new external blind

## New v15 candidate disposition

- training: 1,205
- model-selection validation: 30
- internal test: 30
- new external blind: 50
- excluded homology buffer: 913
- historical-anchor quarantine: 2,449

The 106 historical v14-development entries remain training-only, producing
1,311 total training entries.

No direct frozen-threshold homology edge crosses between evaluation roles or
between evaluation and training. Exact-sequence groups are indivisible.

The new external blind and internal-test cohorts are assigned but sealed.
Their targets have not been accessed and no metrics have been calculated.
No preprocessing was fitted and no model was run.
