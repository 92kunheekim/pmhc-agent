"""Resolve a 3D pMHC-I structure for the target.

Real backend: query PDB/IEDB for a solved structure; else predict with
AlphaFold3 (or a peptide-MHC-specialized predictor). The paper explicitly
designed against *predicted* pMHC (e.g. PRAME), so unsolved targets are in
scope. This mock fabricates a structure id + confidence.
"""
from __future__ import annotations

from .base import det_rng
from ..types import Target, Peptide


class StructureResolverMock:
    name = "pMHC structure resolver (mock: PDB->AF3)"
    is_mock = True

    def __init__(self, seed: int = 0, solved_ids: dict[str, str] | None = None):
        self.seed = seed
        # Map "PEPTIDE|ALLELE" -> real PDB id, for known solved targets.
        self.solved_ids = solved_ids or {}

    def resolve(self, target: Target) -> Target:
        key = f"{target.peptide.sequence}|{target.hla_allele}"
        if key in self.solved_ids:
            target.pmhc_structure_id = self.solved_ids[key]
            target.structure_confidence = 1.0
            return target
        # Otherwise "predict" it.
        rng = det_rng("struct", key, seed=self.seed)
        target.pmhc_structure_id = f"AF3:{abs(hash(key)) % 10**6:06d}"
        # Peptide-level confidence; longer/charged peptides slightly lower.
        conf = 0.92 - 0.01 * max(0, len(target.peptide) - 9) + rng.uniform(-0.05, 0.05)
        target.structure_confidence = round(min(0.99, max(0.4, conf)), 3)
        return target
