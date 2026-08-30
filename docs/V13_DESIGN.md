# v13 scientific and evaluation design

v13 treats the v11 numbers as a historical development baseline, not as a final
independent benchmark. No v13 performance is claimed until the data are rebuilt
and the model is retrained.

## Problems corrected in the v13 foundation

1. BMRB-entry random splitting is replaced by externally computed protein/sequence
   cluster grouping. Identical PDB, UniProt, exact-sequence, and homology-related
   entries must share one cluster ID.
2. BMRB–PDB mapping is chain-aware and sequence-alignment based. A single integer
   residue-number offset is no longer considered sufficient provenance.
3. Graphs must contain every structurally valid residue. H and N labels use separate
   masks; missing experimental assignments do not remove message-passing nodes.
4. BMRB entry offsets are explicit nuisance variables. H and N offsets are not
   forced to be numerically equal. A robust absolute loss is combined with a
   within-entry centered loss rather than relying on hard SCS clipping alone.
5. H/N model selection uses head-specific ppm scales. Reports must include both
   residue-micro and entry-macro errors with cluster-level confidence intervals.
6. Experimental BMRB relaxation values are excluded from the structure-only primary
   model. They may be evaluated only as a separately labelled experimental-data
   augmented model.

## Required split protocol

- Deduplicate by BMRB ID, PDB ID, UniProt ID, and exact sequence.
- Generate sequence clusters before splitting (30% identity for the strict test).
- Assign entire connected groups to train, validation, or test.
- Never use the final test for feature, architecture, or threshold selection.
- Keep a second, easier 50% identity benchmark and the legacy random-entry split only
  for comparison.

## Target model

For entry `j`, residue `i`, and nucleus `k`:

```
observed_shift = condition_aware_random_coil
               + structural_model(graph, residue)
               + entry_reference_offset
               + robust_residual
```

The recommended first v13 network remains a cleaned GATv2 baseline. An equivariant
network should be evaluated only after the corrected data pipeline and split are
stable.

## Required reports

- H and N MAE, RMSE, median absolute error, and Pearson/Spearman correlation;
- residue-micro and entry-macro aggregation;
- cluster bootstrap 95% confidence intervals;
- performance by sequence identity, residue type, structure method, resolution,
  assignment coverage, and predicted entry offset;
- matched-capacity MLP, sequence-only GNN, spatial-edge ablation, and physics-feature
  ablations;
- raw-label and re-referenced-label evaluations.
