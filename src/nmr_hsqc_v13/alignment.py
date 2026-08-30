"""Chain-aware BMRB/PDB sequence alignment.

v11 collapsed chain identifiers and searched only a single integer residue offset.
That silently overwrote residues in multimers and could accept incorrect matches.
v13 aligns complete residue-name sequences per chain and returns explicit provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

@dataclass(frozen=True)
class ResidueRecord:
    chain: str
    seq_id: Hashable
    res_name: str


@dataclass(frozen=True)
class ResidueMatch:
    bmrb: ResidueRecord
    pdb: ResidueRecord


@dataclass(frozen=True)
class ChainAlignment:
    bmrb_chain: str
    pdb_chain: str
    matches: tuple[ResidueMatch, ...]
    identity: float
    coverage: float


def _by_chain(records: Iterable[ResidueRecord]) -> dict[str, list[ResidueRecord]]:
    out: dict[str, list[ResidueRecord]] = {}
    for record in records:
        out.setdefault(record.chain, []).append(record)
    return out


def _align_pair(
    bmrb: list[ResidueRecord], pdb: list[ResidueRecord]
) -> tuple[tuple[ResidueMatch, ...], float, float]:
    a = [r.res_name for r in bmrb]
    b = [r.res_name for r in pdb]
    if not a or not b:
        return (), 0.0, 0.0
    # Needleman-Wunsch over three-letter residue tokens. Keeping this small
    # implementation local makes the provenance layer usable before the heavier
    # notebook dependencies are installed.
    gap = -2
    score = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    trace = [[""] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        score[i][0], trace[i][0] = i * gap, "up"
    for j in range(1, len(b) + 1):
        score[0][j], trace[0][j] = j * gap, "left"
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            options = {
                "diag": score[i - 1][j - 1] + (2 if a[i - 1] == b[j - 1] else -1),
                "up": score[i - 1][j] + gap,
                "left": score[i][j - 1] + gap,
            }
            trace[i][j] = max(options, key=options.get)
            score[i][j] = options[trace[i][j]]
    aligned_pairs: list[tuple[int, int]] = []
    i, j = len(a), len(b)
    while i or j:
        direction = trace[i][j]
        if direction == "diag":
            aligned_pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif direction == "up":
            i -= 1
        else:
            j -= 1
    aligned_pairs.reverse()
    matches: list[ResidueMatch] = []
    aligned = identical = 0
    for ai, bi in aligned_pairs:
        left, right = bmrb[ai], pdb[bi]
        aligned += 1
        if left.res_name == right.res_name:
            identical += 1
            matches.append(ResidueMatch(left, right))
    identity = identical / aligned if aligned else 0.0
    coverage = identical / len(bmrb) if bmrb else 0.0
    return tuple(matches), identity, coverage


def align_bmrb_to_pdb(
    bmrb_records: Iterable[ResidueRecord],
    pdb_records: Iterable[ResidueRecord],
    *,
    min_identity: float = 0.90,
    min_coverage: float = 0.70,
) -> tuple[ChainAlignment, ...]:
    """Greedily match BMRB chains to distinct PDB chains by sequence quality."""
    bmrb_chains, pdb_chains = _by_chain(bmrb_records), _by_chain(pdb_records)
    candidates = []
    for bchain, bseq in bmrb_chains.items():
        for pchain, pseq in pdb_chains.items():
            matches, identity, coverage = _align_pair(bseq, pseq)
            candidates.append((identity * coverage, bchain, pchain, matches, identity, coverage))

    result: list[ChainAlignment] = []
    used_bmrb: set[str] = set()
    used_pdb: set[str] = set()
    for _, bchain, pchain, matches, identity, coverage in sorted(candidates, reverse=True):
        if bchain in used_bmrb or pchain in used_pdb:
            continue
        if identity < min_identity or coverage < min_coverage:
            continue
        result.append(ChainAlignment(bchain, pchain, matches, identity, coverage))
        used_bmrb.add(bchain)
        used_pdb.add(pchain)
    return tuple(result)
