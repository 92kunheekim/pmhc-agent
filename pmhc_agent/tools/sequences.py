"""Sequence design.

Real backend: ProteinMPNN (or LigandMPNN for explicit peptide atoms).
Designs sequences optimizing monomer stability + interface packing, several
per backbone. In a real run you would also fix/penalize residues that
contact conserved MHC positions.

The mock emits a pseudo-random amino-acid sequence plus an MPNN NLL score
and a Rosetta ddG proxy, both correlated with the backbone's peptide-contact
fraction so that "better" backbones tend to yield "better" designs.
"""
from __future__ import annotations

from .base import det_rng
from ..types import Backbone, Design

# ProteinMPNN is commonly run with cysteine excluded to avoid unpaired-Cys
# liabilities; the mock mirrors that so the demo funnel is driven by fold +
# specificity rather than a random-cysteine lottery at the developability gate.
_AA = "ADEFGHIKLMNPQRSTVWY"


class ProteinMPNNMock:
    name = "ProteinMPNN (mock)"
    is_mock = True

    def __init__(self, seed: int = 0):
        self.seed = seed

    def design(self, backbone: Backbone, n: int, round_index: int) -> list[Design]:
        out: list[Design] = []
        for j in range(n):
            rng = det_rng("mpnn", backbone.id, j, seed=self.seed)
            seq = "".join(rng.choice(_AA) for _ in range(backbone.length))
            # Better packing correlates with high peptide-contact backbones.
            quality = backbone.peptide_contact_fraction
            mpnn = round(1.25 - 0.4 * quality + rng.uniform(-0.12, 0.12), 3)
            ddg = round(-14.0 - 22.0 * quality + rng.uniform(-4, 4), 2)
            out.append(
                Design(
                    id=f"{backbone.id}_s{j}",
                    backbone=backbone,
                    sequence=seq,
                    mpnn_score=mpnn,
                    rosetta_ddg=ddg,
                )
            )
        return out
