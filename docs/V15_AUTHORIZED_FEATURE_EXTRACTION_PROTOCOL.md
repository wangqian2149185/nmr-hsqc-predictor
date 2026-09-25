# v15 authorized feature-extraction protocol

This protocol was frozen before reading any target-feature shard data rows.

## Identifier bridge

The cohort assignment uses `node_id`; the feature evidence uses
`candidate_id`. They are not directly interchangeable.

The frozen bridge joins the two tables one-to-one by unique `bmrb_id` and
requires normalized `pdb_id` equality for every record. All 4,677 records
passed and no identifier or row-order tie-break was used.

## Authorized new-v15 roles

- Training: 1,205 entries
- Model-selection validation: 30 entries

The 106 historical v14 development entries remain training-only and are
loaded from the already frozen v14 development table.

## Sealed and excluded roles

- Internal test: 30 entries, sealed
- New external blind: 50 entries, sealed
- Excluded buffer: 913 entries
- Historical-anchor quarantine: 2,449 entries

## Mixed-role shard handling

A complete mixed-role shard may not be materialized as a dataframe.
Records must be streamed and checked against the frozen candidate-ID role map.

Compressed bytes may be decompressed to locate record boundaries, but target
fields for forbidden roles may not be converted, materialized, printed,
aggregated, persisted, or used for metrics.

## Preprocessing

Imputation and target scaling remain unfitted. Future preprocessing must be
fitted using training rows only. Reference correction and hierarchical entry
offsets are prohibited.

Policy SHA256: `8a5003bf26b557bf7fdd7e85a4a2f3850e1bd28b1c6f4cdf44b75a91fcaa658a`
