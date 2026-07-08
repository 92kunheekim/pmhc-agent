"""Antibody gate thresholds. ipTM cutoffs are format-dependent, following the
paper: ipTM >= 0.6 for VHHs, >= 0.85 for scFvs."""
from __future__ import annotations

from dataclasses import dataclass

IPTM_BY_FORMAT = {"VHH": 0.60, "scFv": 0.85, "IgG": 0.85}


@dataclass
class AntibodyGatePolicy:
    min_epitope_contact_fraction: float = 0.40   # paratope focused on epitope
    max_mpnn_score: float = 1.10                 # CDR packing
    min_iptm: float = 0.60                       # set per-format at intake
    max_self_consistency_rmsd: float = 2.0       # RF2 self-consistency (Angstrom)
    min_plddt: float = 75.0
    # developability
    min_humanness: float = 0.80
    max_polyreactivity: float = 0.30
    # diversity
    max_designs_per_scaffold: int = 3
