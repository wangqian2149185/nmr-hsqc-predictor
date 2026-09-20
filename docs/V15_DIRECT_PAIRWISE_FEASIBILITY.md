# v15 direct pairwise-isolation feasibility

The frozen connected-component allocation is infeasible, but a diagnostic
using the unchanged 0.40 identity and 0.70 shorter-sequence-coverage edge
definition found that direct-edge isolation may retain a usable pool.

Exact-sequence co-membership with historical entries was treated as stronger
evidence than an ordinary homology edge.

## Corrected upper bounds

Among 4,677 new candidate nodes:

- 2,449 have exact-sequence or direct-edge evidence to opened v14 external
  anchors and must remain quarantined;
- 1,944 have development-anchor evidence, but no opened-external evidence,
  and are provisionally training-only;
- 284 have no exact or direct historical-anchor evidence and are provisionally
  eligible for new evaluation roles;
- 2,228 are provisionally training eligible.

These are upper bounds. They do not yet enforce zero direct homology edges
between newly assigned training, validation, internal-test, and external-blind
roles.

No buffered allocation policy has been adopted and no cohort has been
assigned or opened.
