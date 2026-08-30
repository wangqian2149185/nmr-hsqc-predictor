# NMR HSQC Predictor — v13 Deployable Step 33

## Selected model

The primary deployment model is the five-seed Step 33
nucleus-specific ensemble.

Step 33 uses full-chain PDB torsion angles and does not depend on BMRB
assignment coverage when constructing input features.

## Inputs and outputs

Input:

- a protein structure with backbone coordinates;
- an explicitly selected protein chain.

Output:

- predicted backbone-amide 1H chemical shift in ppm;
- predicted backbone-amide 15N chemical shift in ppm;
- across-seed ensemble standard deviations.

Proline residues are excluded by default because they generally do not have
a conventional backbone amide proton.

## Model architecture

A shared 68-feature sequence/torsion trunk is combined with separate
nucleus-specific structural branches:

- 1H branch: six geometric features;
- 15N branch: three local-contact features.

Five independently seeded neural networks are averaged for the final
prediction.

## Model selection

The model was selected using validation entries only.

Compared with the Step 14 research benchmark, Step 33 showed a stable
entry-level validation improvement for 1H. The 15N difference was small and
statistically inconclusive.

The held-out test set was used only for final reporting.

## Deployment verification

For PDB 3MSP chain A, the PDB-only pipeline reproduced all 114 saved Step 33
evaluation predictions within 1e-5 ppm.

The PDB pipeline also produced predictions for three structurally eligible
residues without deposited experimental assignments. Those extra predictions
are expected and demonstrate that inference does not require BMRB assignment
coverage.

## Referencing policy

No chemical-shift reference correction is applied.

No entry-specific or hierarchical b_j term is included.

BMRB 4129 and BMRB 4242 remain marked as suspected 15N referencing outliers,
but their deposited shifts were not corrected.

## Important limitations

The training set contains only 56 BMRB/PDB entries. Validation and test
uncertainty is therefore dominated by the small number of independent
proteins.

The ensemble standard deviation measures disagreement among five training
seeds. It is not a calibrated predictive uncertainty interval.

Predictions may be unreliable for unusual chemistry, missing backbone atoms,
ligand-induced states, non-standard residues, extreme conditions, or
structures outside the training distribution.

## Historical Step 14 model

Step 14 is retained only as a research benchmark. It should not be used as
the default PDB deployment model because some training-time torsion masks
depended on chemical-shift assignment coverage.
