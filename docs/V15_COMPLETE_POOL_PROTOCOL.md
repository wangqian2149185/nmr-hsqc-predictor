# v15 complete-pool screening protocol

STEP 63 freezes the v15 data-discovery and cohort-allocation rules before new candidate screening or model training.

## Objective

- Screen the complete frozen official BMRB-PDB mapping snapshot.
- Reach at least 300 independent training entries; prefer 500.
- Preserve sequence-homology isolation across all dataset roles.
- Create a new untouched external blind cohort for v15.

## Historical-cohort policy

- The 106-entry v14 development dataset may contribute only to v15 training.
- The opened 40-entry v14 external cohort is historical reporting only and is excluded from every v15 development role.
- Exact BMRB, PDB, and homology-component overlap across roles is prohibited.

## Eligibility

- At least 40 mapped paired backbone-amide H/N targets.
- Protein entity-to-structure-chain identity at least 0.90.
- Sequence coverage at least 0.70.
- Exact residue mapping and deployment-safe feature generation.
- Shift-list provenance must be unambiguous or explicitly flagged.
- Non-polymer ligands and unsupported nonstandard codes are recorded and ignored rather than causing an entry-level crash.

## Cohort isolation

- Assignment unit: 40% sequence-identity connected component.
- Components cannot cross training, validation, internal test, or new external blind roles.
- Allocation seed: 20260902.
- New external blind minimum: 50 entries.
- Internal test minimum: 30 entries.
- Model-selection validation minimum: 30 entries.

## Referencing and loss

- No target correction.
- No reference correction.
- No hierarchical b_j.
- First baseline retains the fixed v14 architecture and train-only target-standardized loss.
- CSP 6.51 and alternative nitrogen weighting are predeclared future comparisons after the dataset is frozen.

The machine-readable protocol and checksum lock are stored in `development/v15_protocol/`.
