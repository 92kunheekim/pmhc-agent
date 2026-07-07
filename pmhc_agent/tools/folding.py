"""Fold & dock prediction + a cross-check consensus.

Real backend: AF2 in initial-guess/complex mode for the primary prediction,
with AF3 and Chai-1 as orthogonal cross-checks (gate G8) — any single
predictor can be confidently wrong on de novo interfaces.

The mock derives pAE-interface / pLDDT / RMSD from design quality, and
occasionally flags predictor disagreement.
"""
from __future__ import annotations

from .base import det_rng
from ..types import Design, Target, FoldResult


class FoldPredictorMock:
    name = "AF2 initial-guess + AF3/Chai consensus (mock)"
    is_mock = True

    def __init__(self, seed: int = 0):
        self.seed = seed

    def predict(self, design: Design, target: Target) -> FoldResult:
        rng = det_rng("fold", design.id, target.peptide.sequence, seed=self.seed)
        # Normalize the two quality signals to ~[0,1] so a strong design
        # (peptide-focused backbone + tight packing) clears the gates and a
        # weak one does not — yielding a realistic "few survive" funnel.
        q = min(1.0, design.backbone.peptide_contact_fraction / 0.65)
        pack = max(0.0, min(1.0, (1.25 - design.mpnn_score) / 0.5))
        signal = 0.5 * q + 0.5 * pack
        pae = round(max(3.0, 14.0 - 12.0 * signal + rng.uniform(-1.2, 1.2)), 2)
        plddt = round(min(97.0, 55.0 + 42.0 * signal + rng.uniform(-3, 3)), 1)
        rmsd = round(max(0.3, 3.0 - 2.6 * signal + rng.uniform(-0.35, 0.35)), 2)
        # AF2/AF3 disagree ~8% of the time, more for marginal designs.
        disagree_p = 0.05 + 0.15 * max(0.0, 0.6 - signal)
        agree = rng.random() > disagree_p
        return FoldResult(
            pae_interface=pae, plddt=plddt, ca_rmsd_to_design=rmsd,
            predictors_agree=agree,
        )
