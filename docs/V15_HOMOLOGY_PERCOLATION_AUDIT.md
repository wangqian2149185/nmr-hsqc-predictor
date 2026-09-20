# v15 homology percolation audit

## Result

The frozen v15 homology-allocation protocol cannot produce isolated
development and evaluation cohorts.

The exact all-pairs calculation evaluated 9,528,795 unordered pairs among
4,366 unique sequences. At the frozen thresholds of 0.40 identity and 0.70
shorter-sequence coverage, 335,423 homology edges were retained.

All 4,823 sequence nodes belong to one connected component.

## Historical-anchor consequence

The single component contains:

- 4,677 new v15 candidate nodes;
- 106 frozen v14-development nodes;
- 40 opened v14-external nodes.

Because the component contains opened-external anchors, the frozen isolation
rule quarantines the complete component. Zero new candidates remain freely
allocatable.

## This is not only an anchor effect

After removing all historical anchors, the new-only induced graph still
contains one connected component:

- 4,223 unique sequences;
- 4,662 new nodes;
- 318,433 retained edges.

Every unique sequence has at least seven retained edges. The median degree is
124 and the maximum degree is 768.

The graph also contains 4,385 direct new-to-opened-external edges.

## Frozen-protocol disposition

The predeclared minimum of 300 training entries cannot be satisfied.
Accordingly:

- cohort allocation must stop;
- no new internal test may be opened;
- no new external blind cohort may be opened;
- preprocessing must not be fitted;
- model training must not begin.

This is the protocol's predeclared insufficient-pool outcome. Homology
isolation must not be weakened silently.

## Counterfactual diagnostics

Stricter identity thresholds were evaluated only to characterize graph
percolation. These calculations do not change the frozen 0.40 threshold and
do not constitute cohort assignments.

Any replacement graph rule or threshold requires a separately frozen,
scientifically justified protocol before cohort allocation.

No reference correction or hierarchical entry offset was used.
