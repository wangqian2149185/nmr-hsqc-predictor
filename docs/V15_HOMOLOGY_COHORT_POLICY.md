# v15 homology-safe cohort policy

This operational policy was frozen before homology-edge calculation or cohort
assignment.

## Frozen scientific rules

- Sequence-identity threshold: 40%.
- Connected components are the indivisible allocation unit.
- No component may be split across roles.
- Allocation seed: `20260902`.
- Allocation priority:
  1. new external blind;
  2. internal test;
  3. model-selection validation;
  4. training.
- New external, internal-test, and validation target fractions are each 10%.
- Minimum entry counts are 50, 30, and 30 respectively.
- Training receives remaining eligible components and must contain at least
  300 entries before model training may proceed.

## Historical anchors

The 106 v14 development entries are training-only anchors. Components
homologous to them are training-only.

The 40 already opened v14 external entries are historical-external quarantine
anchors. Any new entry in such a component is excluded from every v15 model
role. Historical-external quarantine has precedence over a training anchor.

## Homology calculation

MMseqs2 is used only to discover candidate edges. Candidate edges are verified
against the frozen 40% identity and 70% shorter-sequence coverage definitions.
Exact duplicates are added independently. Every sequence containing `X` is
compared exhaustively against the combined sequence universe.

## Deterministic allocation

Complete unanchored components are allocated in the frozen priority order.
Entry count, paired H/N target count, and total sequence length receive equal
weight. Exact objective ties use a SHA-256 order derived from the frozen seed
and component membership.

No target values, predictions, residuals, or model metrics may influence
allocation.

## Sealed cohorts

The new external cohort is sealed before training and opened once for final
evaluation. The internal test is opened once after model selection. Only the
model-selection validation cohort may guide model selection.

No reference correction or hierarchical entry offset is permitted.
