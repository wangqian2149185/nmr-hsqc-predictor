# v14 Development Plan

## Starting point

The `codex_v14` branch begins from the frozen v13 README commit:

`329acd145090bb3fffbb5cde805848efc704e0f2`

The immutable deployment baseline is:

`releases/v13_deployable_step33/`

The baseline contains the Step 33 five-seed nucleus-specific ensemble,
standalone PDB inference, preprocessing metadata, validation evidence,
reference-QC policy, and checksums.

## Frozen v13 constraints

v14 work must not silently modify the v13 baseline.

The following v13 decisions remain fixed:

- Step 33 is the primary v13 deployment model.
- Step 14 is a research benchmark only.
- Splitting uses 40%-identity connected components.
- Model selection uses validation entries only.
- The v13 test set is reporting-only.
- No target chemical shifts were corrected.
- No entry-specific or hierarchical b_j term was used.
- BMRB 4129 and 4242 retain their frozen reference-QC labels.

If the v13 release directory changes, it must be treated as a new release
rather than an in-place correction.

## Phase 1 — External blind benchmark

Before candidate discovery:

1. freeze eligibility rules;
2. freeze homology-exclusion rules;
3. freeze shift-list selection rules;
4. freeze reference-QC and reporting policies;
5. record the v13 release-manifest checksum.

Candidate selection must not use model predictions or prediction error.

The frozen v13 model will be evaluated without retraining or preprocessing
refitting.

Primary reporting will include:

- residue-level MAE and RMSE;
- entry-macro MAE;
- per-entry results;
- bootstrap intervals over complete entries;
- primary metrics retaining suspected reference outliers;
- separate reference-QC sensitivity analyses.

## Phase 2 — Dataset expansion

After the external benchmark is locked:

- build a deterministic BMRB–PDB candidate-discovery pipeline;
- select chemical-shift lists using explicit provenance rules;
- resolve chains and residue mappings;
- audit referencing metadata;
- generate immutable dataset manifests and checksums;
- cluster sequences before any train/validation/test assignment.

External benchmark entries must remain excluded from future training if they
are to serve as a continuing benchmark.

## Phase 3 — v14 model comparisons

New models will be compared against the frozen Step 33 baseline using identical
data, splits, targets, and metrics.

Candidate model families may include:

- expanded residue-window MLPs;
- one-dimensional sequence encoders;
- residue-level graph neural networks;
- structure-aware transformers.

Larger architecture alone is not considered progress. Improvements must be
supported by entry-level validation evidence and reproducible deployment.

## Referencing policy

Reference metadata and entry-level residual offsets remain QC signals.

v14 will not introduce target re-referencing, an entry-specific correction, or
a hierarchical b_j term unless a separate preregistered experiment provides
clear physical justification and validation evidence.

## Repository policy

The active branch keeps:

- production source;
- regression tests;
- frozen releases;
- current design and protocol documentation.

Legacy notebooks, obsolete plots, and superseded README files remain available
through `codex_v13` and Git history rather than the active v14 working tree.
