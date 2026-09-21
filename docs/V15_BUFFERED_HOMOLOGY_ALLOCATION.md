# v15 buffered homology allocation protocol

The original 0.40-identity connected-component allocation was exact but
infeasible because all 4,823 nodes formed one component.

The replacement allocation protocol retains the same pairwise homology
definition. It does not raise the identity threshold.

Instead, exact-sequence groups are indivisible and explicit buffer exclusions
ensure that no retained homology edge crosses between training, validation,
internal test, and new external blind roles.

Opened-external neighbors are quarantined. Development-anchor neighbors are
training-only. Evaluation selections remove all of their direct neighbors
from training.

The deterministic search evaluates six role orders and 100 seeded trials per
order. The winner maximizes retained training entries after satisfying all
minimum cohort sizes and zero-cross-edge constraints.

The frozen feasibility simulation found all 600 trials feasible. No
entry-level cohort assignment existed when this protocol was frozen.
