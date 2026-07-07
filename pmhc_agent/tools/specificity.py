"""The specificity engine — the core of the whole agent (§06 of the design).

Combines three signals into one *contrastive* verdict:
  1. Contrastive ProteinMPNN recovery: does the interface "read" the target
     peptide's side chains better than any off-target's?
  2. Fine-tuned AF2 margin: predicted binding confidence for on-target minus
     the best off-target (Motmaen et al., PNAS 2023 fine-tuned model).
  3. Interface partition (phi): fraction of binding energy from PEPTIDE vs
     conserved MHC framework contacts.

The verdict uses the WORST-CASE off-target (min over the panel), because a
binder is only as specific as its single worst cross-reaction. This directly
guards against the paper's mage-282 failure mode (bound tetramer specifically,
yet cross-activated on other proteome peptides on the same HLA).
"""
from __future__ import annotations

from .base import det_rng
from ..types import Design, Target, OffTarget, SpecificityResult


def _peptide_similarity(a: str, b: str) -> float:
    """Crude fraction of shared residues at aligned positions (0..1)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / n


class SpecificityEngineMock:
    name = "Contrastive MPNN + fine-tuned AF2 (mock)"
    is_mock = True

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _bind_score(self, design: Design, peptide_seq: str, target_seq: str,
                    is_on_target: bool) -> float:
        """Predicted binding confidence in [0, 1] for design vs a peptide.

        On-target designs that fold well score high. Off-targets score high
        only to the extent they resemble the on-target peptide at the
        outward-facing positions the binder reads (approximated by sequence
        similarity here).
        """
        rng = det_rng("spec", design.id, peptide_seq, seed=self.seed)
        fold = design.fold
        fold_signal = 0.0
        if fold is not None:
            fold_signal = max(0.0, min(1.0, (12.0 - fold.pae_interface) / 9.0))
        if is_on_target:
            base = 0.55 + 0.4 * fold_signal
        else:
            sim = _peptide_similarity(peptide_seq, target_seq)
            # A specific design suppresses off-targets; leakage grows with sim
            # and shrinks with how peptide-focused the interface is.
            phi = design.backbone.peptide_contact_fraction
            leak = sim * (1.2 - phi)
            base = 0.2 + 0.6 * fold_signal * leak
        return round(min(0.99, max(0.01, base + rng.uniform(-0.05, 0.05))), 3)

    def score(self, design: Design, target: Target,
              panel: list[OffTarget]) -> SpecificityResult:
        tseq = target.peptide.sequence
        on = self._bind_score(design, tseq, tseq, is_on_target=True)
        offs: dict[str, float] = {}
        for ot in panel:
            offs[ot.peptide.sequence] = self._bind_score(
                design, ot.peptide.sequence, tseq, is_on_target=False
            )
        worst_off = max(offs.values()) if offs else 0.0
        # margin scaled to ~unit range so theta ~1.0 is meaningful.
        margin = round((on - worst_off) * 5.0, 3)

        # phi: energy fraction from peptide contacts (proxy: contact fraction).
        phi = round(min(0.95, design.backbone.peptide_contact_fraction
                        + 0.05), 3)
        # MPNN recovery: on-target recovered above all off-targets?
        recovery_ok = on > worst_off + 0.05

        return SpecificityResult(
            on_target_score=on,
            off_target_scores=offs,
            margin=margin,
            peptide_energy_fraction=phi,
            mpnn_recovery_ok=recovery_ok,
        )
