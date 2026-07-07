"""MHC-I binding prior — used for (a) ligand sanity of the target peptide
and (b) building the off-target panel (which proteome peptides even bind
this allele).

Real backend: NetMHCpan-4.1 or MHCflurry.  This mock returns a %rank
derived from a crude anchor-residue heuristic so the demo behaves
plausibly (hydrophobic C-terminus / position-2 anchors score better on
A*02-like alleles).
"""
from __future__ import annotations

from .base import det_rng
from ..types import Peptide


# Very rough anchor preferences by allele family, for the MOCK only.
_ANCHOR_PREF = {
    "A*02": {2: set("LMIVAT"), -1: set("LIVMA")},
    "A*01": {2: set("TS"), -1: set("YF")},
    "A*03": {2: set("LIVM"), -1: set("KRY")},
    "C*07": {2: set("LFY"), -1: set("LFY")},
}


def _family(allele: str) -> str:
    for fam in _ANCHOR_PREF:
        if fam in allele:
            return fam
    return "A*02"


class NetMHCpanMock:
    name = "NetMHCpan-4.1 (mock)"
    is_mock = True

    def __init__(self, seed: int = 0):
        self.seed = seed

    def percent_rank(self, peptide: Peptide, allele: str) -> float:
        """Return %rank; lower means stronger predicted binding (<2 = binder)."""
        fam = _family(allele)
        pref = _ANCHOR_PREF[fam]
        seq = peptide.sequence
        score = 0.0
        for pos, residues in pref.items():
            aa = seq[pos - 1] if pos > 0 else seq[-1]
            score += 1.0 if aa in residues else 0.0
        # 0,1,2 anchors satisfied -> map to a plausible rank, plus noise.
        base = {0: 6.0, 1: 1.8, 2: 0.4}[int(score)]
        jitter = det_rng("netmhc", seq, allele, seed=self.seed).uniform(-0.15, 0.4)
        return round(max(0.01, base + jitter), 3)

    def is_ligand(self, peptide: Peptide, allele: str, max_rank: float = 2.0) -> bool:
        return self.percent_rank(peptide, allele) <= max_rank
