"""Antibody target + negative-set types.

An antibody target is richer than a pMHC target: an antigen structure, the
epitope to hit (hotspot residues), the antibody framework to graft CDRs onto,
and the format (VHH / scFv / IgG). The negative set is a panel of
cross-reactive antigens the binder must reject (the antibody analogue of the
pMHC off-target peptide panel).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

FORMATS = ("VHH", "scFv", "IgG")


@dataclass
class AntibodyTarget:
    """What the agent designs an antibody *against*."""
    antigen: str                        # e.g. "influenza-HA"
    epitope_hotspots: tuple             # residues on the antigen, e.g. ("A45","A47","A49")
    framework: str = "hu-VHH-3-23"      # framework to graft CDRs onto
    fmt: str = "VHH"                    # VHH | scFv | IgG
    antigen_structure_id: Optional[str] = None    # PDB or "AF3:<hash>"
    structure_confidence: Optional[float] = None

    def __post_init__(self):
        if self.fmt not in FORMATS:
            raise ValueError(f"format {self.fmt!r} not in {FORMATS}")
        if not self.epitope_hotspots:
            raise ValueError("an epitope (hotspot residues) is required")

    @property
    def n_cdrs(self) -> int:
        return 3 if self.fmt == "VHH" else 6      # VHH: 3 CDRs; scFv/IgG: VH+VL


@dataclass
class CrossTarget:
    """A cross-reactive antigen the antibody must NOT bind (negative set)."""
    antigen: str
    origin: str = "homolog"             # homolog | self | polyreactivity-probe
    similarity: float = 0.0             # 0..1 epitope similarity to the target
