# v15 entity, structure, and chain selection policy

This policy was frozen before downloading candidate coordinate files.

## Current sequence-screen state

- 5,259 shift-eligible BMRB entries
- 5,178 entries with at least one passing sequence alignment
- 5,005 entries with one scientific contender
- 173 entries tied after sequence and assignment evidence
- 81 entries without a passing sequence alignment

A unique sequence contender is provisional until its coordinate chain is
validated.

## Coordinate validation hierarchy

1. Preserve all passing scientific contenders.
2. Verify the declared author chain in the mmCIF file.
3. Align coordinate residues to the BMRB entity.
4. Require sequence identity of at least 0.90.
5. Require mapped-target coverage of at least 0.70.
6. Maximize unambiguously mapped paired H/N targets.
7. Maximize usable coordinate-residue coverage.
8. Minimize missing backbone atoms needed by deployment-safe features.
9. Keep remaining exact ties unresolved.

Row order, PDB identifier, and chain identifier are not scientific
tie-breakers.

No chemical-shift correction or hierarchical entry offset is permitted.
