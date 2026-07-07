"""Backbone generation.

Real backend: RFdiffusion, hotspot-targeted to the OUTWARD-FACING peptide
residues so the backbone arcs over the groove and presents side chains to
the peptide (not the conserved MHC helices). `partial_diffuse` reuses a
privileged scaffold via partial noising/denoising.

The mock samples backbones with a `peptide_contact_fraction`; scaffolds
seeded from memory get a boost (they already solved a similar geometry).
"""
from __future__ import annotations

from .base import det_rng
from ..types import Target, Backbone


class RFdiffusionMock:
    name = "RFdiffusion (mock)"
    is_mock = True

    def __init__(self, seed: int = 0):
        self.seed = seed

    def generate(
        self,
        target: Target,
        n: int,
        round_index: int,
        seed_scaffold: str | None = None,
        contact_bias: float = 0.0,
    ) -> list[Backbone]:
        """Generate `n` backbones.

        contact_bias (0..~0.2) nudges the peptide_contact_fraction upward —
        the orchestrator raises it when a round produced too many MHC-leaning
        interfaces (Loop A refinement).
        """
        out: list[Backbone] = []
        for i in range(n):
            tag = f"r{round_index}_bb{i}"
            rng = det_rng("rfdiff", target.peptide.sequence, tag, seed=self.seed)
            if seed_scaffold:
                source = f"partial_diffusion:{seed_scaffold}"
                base = 0.55                      # privileged scaffolds start higher
            else:
                source = "de_novo"
                base = 0.42
            pcf = min(0.95, base + contact_bias + rng.uniform(-0.18, 0.22))
            length = rng.randint(55, 75)
            out.append(
                Backbone(
                    id=tag,
                    scaffold_source=source,
                    length=length,
                    peptide_contact_fraction=round(pcf, 3),
                )
            )
        return out
