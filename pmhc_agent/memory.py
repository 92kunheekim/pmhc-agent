"""Scaffold memory + campaign history.

Privileged scaffolds (backbones whose designs cleared the specificity gate)
are banked, indexed by allele, and transplanted to future targets via
partial diffusion — the campaign gets cheaper as the library of solved
geometries grows. This mock keeps everything in-memory; back it with a
vector + structural index (e.g. FAISS + Foldseek) for production.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .types import Design


@dataclass
class ScaffoldRecord:
    scaffold_id: str
    allele: str
    best_margin: float
    peptide_contact_fraction: float
    from_target: str


@dataclass
class Memory:
    scaffolds: list[ScaffoldRecord] = field(default_factory=list)
    assay_outcomes: list[dict] = field(default_factory=list)

    def bank(self, design: Design, allele: str, target_desc: str) -> None:
        rec = ScaffoldRecord(
            scaffold_id=design.backbone.id,
            allele=allele,
            best_margin=design.specificity.margin if design.specificity else 0.0,
            peptide_contact_fraction=design.backbone.peptide_contact_fraction,
            from_target=target_desc,
        )
        self.scaffolds.append(rec)

    def privileged_for(self, allele: str, top: int = 3) -> list[str]:
        """Return best scaffold ids for this allele family, if any."""
        fam = allele.split(":")[0]
        matches = [s for s in self.scaffolds if s.allele.startswith(fam)]
        matches.sort(key=lambda s: s.best_margin, reverse=True)
        return [s.scaffold_id for s in matches[:top]]

    def record_assay(self, target_desc: str, outcome: dict) -> None:
        self.assay_outcomes.append({"target": target_desc, **outcome})
