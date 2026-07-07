"""Off-target panel builder — the adversary the binder must reject.

This is one of the agent's most consequential decisions: *which* peptides
must the design discriminate against? The paper hand-picked 1-4 near
neighbors; the agent constructs a systematic panel:

  (a) proteome scan  — peptides that bind the same allele (NetMHCpan) and
      share the outward-facing residues;
  (b) point variants — single substitutions at the exposed positions the
      binder reads;
  (c) self-antigens  — known highly-expressed peptides with the same anchor
      motif (the Titin-vs-MAGE-A3 cross-reactivity risk).

In this scaffold the "proteome" is a small bundled list so the demo runs
offline. Replace `proteome_iter` with a real UniProt + viral proteome scan.
"""
from __future__ import annotations

from .tools.base import det_rng
from .types import Target, OffTarget, Peptide

# A tiny stand-in "proteome" of 9mers for the demo. Swap for a real scan.
_DEMO_PROTEOME = [
    ("Titin", "ESDPIVAQY"), ("HERV-K", "SLLQHLIGL"), ("PRAME", "SLLQHLIGL"),
    ("NY-ESO-1", "SLLMWITQC"), ("Survivin", "ELTLGEFLKL"), ("gp100", "IMDQVPFSV"),
    ("MART1", "AAGIGILTV"), ("WT1", "RMFPNAPYL"), ("CTNNB1", "SYLDSGIHF"),
    ("hTERT", "ILAKFLHWL"), ("MAGE-A1", "KVLEYVIKV"), ("CEA", "YLSGANLNL"),
    ("PAP", "FLFLLFFWL"), ("HPV-E7", "YMLDLQPET"), ("EBV-LMP2", "CLGGLLTMV"),
]

_AA = "ACDEFGHIKLMNPQRSTVWY"
# Positions typically outward-facing (1-indexed) for a canonical 9mer.
_EXPOSED = (4, 5, 6, 7, 8)


def _similarity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n if n else 0.0


def build_panel(target: Target, netmhc, size: int, seed: int = 7) -> list[OffTarget]:
    tseq = target.peptide.sequence
    allele = target.hla_allele
    candidates: list[OffTarget] = []

    # (a) proteome peptides that bind this allele and are confusable
    for protein, pep_seq in _DEMO_PROTEOME:
        if pep_seq == tseq:
            continue
        try:
            pep = Peptide(pep_seq)
        except ValueError:
            continue
        rank = netmhc.percent_rank(pep, allele)
        if rank > 3.0:        # doesn't meaningfully bind -> not a threat
            continue
        conf = _similarity(pep_seq, tseq)
        candidates.append(OffTarget(
            peptide=pep, origin="proteome", parent_protein=protein,
            binds_allele_rank=rank, confusability=round(conf, 3),
        ))

    # (b) single-residue variants at exposed positions (hardest neighbors)
    rng = det_rng("panel_variants", tseq, seed=seed)
    for pos in _EXPOSED:
        if pos > len(tseq):
            continue
        for _ in range(2):
            new_aa = rng.choice(_AA)
            if new_aa == tseq[pos - 1]:
                continue
            variant = tseq[:pos - 1] + new_aa + tseq[pos:]
            pep = Peptide(variant)
            rank = netmhc.percent_rank(pep, allele)
            if rank > 3.0:
                continue
            candidates.append(OffTarget(
                peptide=pep, origin="point_variant",
                parent_protein=f"{tseq}[{pos}{tseq[pos-1]}>{new_aa}]",
                binds_allele_rank=rank,
                confusability=round(_similarity(variant, tseq), 3),
            ))

    # Deduplicate by sequence, then keep the hardest (most confusable) N.
    seen: dict[str, OffTarget] = {}
    for ot in candidates:
        key = ot.peptide.sequence
        if key not in seen or ot.confusability > seen[key].confusability:
            seen[key] = ot
    ranked = sorted(seen.values(), key=lambda o: o.confusability, reverse=True)
    return ranked[:size]
