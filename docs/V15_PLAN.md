# v15 large-data development plan

## Starting point

v15 starts from the completed v14 commit `8d43a1598f46c0d5e2be7edba14aaddb2266b25f`.

The frozen v14 deployable release and its validation, internal-test, and external-blind results are immutable.

## Primary objective

Increase the effective number of independent protein entries before increasing model capacity.

The initial target is several hundred quality-controlled, structure-matched BMRB entries. A practical milestone is at least 300 training entries; 500 or more is preferred if data quality and homology constraints allow it.

## Development sequence

1. Freeze the complete-pool discovery and eligibility protocol.
2. Screen the complete BMRB-PDB mapping snapshot.
3. Audit shift-list provenance and referencing metadata.
4. Resolve entity, chain, residue, and structure mappings.
5. Remove exact identifier overlap with all earlier datasets.
6. Cluster sequences using homology-aware connected components.
7. Reserve a new untouched external cohort.
8. Freeze train, validation, internal-test, and external roles.
9. Generate deployment-safe full-chain PDB features.
10. Train the fixed v14 architecture as the v15 data-scale baseline.
11. Only then compare predeclared loss-scaling alternatives.

## Loss-scaling experiments

After the expanded dataset and splits are fixed, compare:

- train-only target-standard-deviation scaling;
- CSP-oriented nitrogen divisor 6.51;
- uncertainty-weighted multitask loss;
- separate nucleus-specific H and N models.

Model selection must report H and N separately. CSP with divisor 6.51 is an additional reporting metric and must not hide a nucleus-specific regression.

## Data isolation

- Protein/BMRB entry is the primary independent unit.
- Residues from one entry must never cross data roles.
- Homologous connected components must not cross roles.
- The new external cohort must remain untouched until the final v15 release is frozen.
- The v14 external cohort may be retained for historical reporting but must not be used for v15 selection.

## Referencing policy

The initial v15 baseline applies no chemical-shift correction and contains no hierarchical entry offset b_j.

Referencing metadata and entry-level residual patterns may be recorded as QC annotations. Any future correction model would require a separate predeclared experiment and version.

## Capacity policy

The fixed v14 architecture is the first v15 baseline. Architecture expansion is allowed only after demonstrating that the enlarged dataset has been assembled without leakage and that the baseline is not strongly underfitting.

## Success criteria

- several hundred independent training entries;
- complete provenance and deterministic reconstruction;
- homology-aware role separation;
- standalone PDB inference;
- separately reported H and N accuracy;
- calibrated uncertainty evaluated explicitly;
- one new untouched final external cohort.
