# v14 baseline training protocol

This document freezes the first v14 training experiment before
any candidate model is trained.

## Objective

The baseline measures the effect of expanding the training
cohort while holding the deployable v13 Step 33 architecture
and optimization configuration fixed.

## Training data

The model will be trained from scratch on 6,013 residues from
69 entries:

- 3,916 rows from 39 original-v13 training entries;
- 2,097 rows from 30 expansion-v14 training entries.

All preprocessing and target-scaling statistics were fitted
only on these training rows.

## Model-selection data

The 703 rows from 10 expansion-validation entries are the only
data eligible for early stopping and model selection.

The primary selection metric is the equally weighted
standardized entry-macro MAE across 1H and 15N. Uncertainty is
estimated using a paired bootstrap over the 10 validation
entries.

## Isolated cohorts

The following data are excluded from training, preprocessing
fit, hyperparameter selection, and architecture selection:

- 698-row expansion internal test;
- original-v13 validation and test cohorts, retained only for
  legacy reporting;
- the frozen 40-entry external-blind cohort.

Internal-test evaluation is permitted only after the validation
decision. External evaluation is permitted only after v14
development is frozen.

## Controlled baseline

The baseline retains the v13 Step 33 architecture, five seeds,
standardized equal-nucleus MSE, AdamW optimizer, learning rate,
weight decay, batch size, epoch limit, and early-stopping
patience. It changes only the training cohort and its
training-only preprocessing statistics.

No reference correction is applied, targets are not modified,
and hierarchical b_j is not used.
