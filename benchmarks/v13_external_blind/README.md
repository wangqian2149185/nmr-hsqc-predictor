# Frozen v13 external blind benchmark

This directory freezes the external benchmark protocol before candidate
discovery or model evaluation.

The evaluated model will be the immutable v13 Step 33 release under:

`releases/v13_deployable_step33/`

Files:

- `protocol.json` — eligibility, homology, QC, and reporting rules;
- `protocol_lock.json` — checksums locking the protocol;
- `existing_v13_entry_exclusion_set.csv` — the existing 56 entries;
- `existing_v13_chain_sequences.csv` — complete chain sequences and structure checksums;
- `existing_v13_chain_sequences.fasta` — homology-exclusion reference FASTA.

External candidates must not be chosen using model predictions or prediction
error.

No deposited chemical shifts will be corrected, and no entry-specific or
hierarchical b_j correction will be used.
