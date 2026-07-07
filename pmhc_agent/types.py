"""Core data types for the pMHC-Design Agent.

These dataclasses are the typed "campaign state" that flows through the
pipeline. They are deliberately model-agnostic: a `Design` produced by a
mock backbone generator has the same shape as one produced by real
RFdiffusion, so you can swap backends without touching the orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------
# Target definition
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Peptide:
    """An MHC-I ligand: a short (8-11mer) linear peptide."""
    sequence: str

    def __post_init__(self) -> None:
        if not (7 <= len(self.sequence) <= 12):
            raise ValueError(
                f"Peptide {self.sequence!r} has implausible length "
                f"{len(self.sequence)} for MHC-I (expected 8-11)."
            )

    def __len__(self) -> int:
        return len(self.sequence)


@dataclass
class Target:
    """A pMHC-I target: what the agent designs a binder *against*."""
    peptide: Peptide
    hla_allele: str                      # e.g. "HLA-A*02:01"
    source_antigen: str = "unknown"      # e.g. "MAGE-A3"
    pmhc_structure_id: Optional[str] = None   # PDB id or "AF3:<hash>"
    structure_confidence: Optional[float] = None  # pTM/pLDDT on the peptide


# --------------------------------------------------------------------------
# Off-target panel
# --------------------------------------------------------------------------
@dataclass
class OffTarget:
    """A peptide the binder must *reject* (same MHC, confusable)."""
    peptide: Peptide
    origin: str                # "proteome" | "point_variant" | "self_antigen"
    parent_protein: str = ""
    binds_allele_rank: float = 99.0   # NetMHCpan %rank (lower = binds better)
    confusability: float = 0.0        # 0..1, how hard it is to discriminate


# --------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------
@dataclass
class Backbone:
    """A generated protein backbone arcing over the peptide groove."""
    id: str
    scaffold_source: str               # "de_novo" | "partial_diffusion:<id>"
    length: int
    peptide_contact_fraction: float    # fraction of contacts on peptide vs MHC
    coords_ref: str = ""               # path/handle to real coordinates


@dataclass
class Design:
    """A backbone + a designed sequence: a candidate binder."""
    id: str
    backbone: Backbone
    sequence: str
    mpnn_score: float = 0.0            # lower is better (per-residue NLL)
    rosetta_ddg: Optional[float] = 0.0  # more negative better; None if not computed
    struct_ref: Optional[str] = None  # path to this design's complex PDB (real pipeline)

    # Populated by later stages:
    fold: Optional["FoldResult"] = None
    specificity: Optional["SpecificityResult"] = None
    composite_score: float = 0.0
    liabilities: list[str] = field(default_factory=list)
    # Engine generalization (additive, non-breaking):
    #  * metrics: a named-metric bag any backend populates and any domain's
    #    gates read by name (pmhc: pae_interaction/plddt/margin/phi; antibody:
    #    iptm/self_consistency/ddg). Lets one gate runner serve every domain.
    #  * chains: multi-chain designs (pmhc {"B": seq}; scFv {"H":.., "L":..}).
    #    `sequence` remains the single-chain view for back-compat.
    metrics: dict = field(default_factory=dict)
    chains: dict = field(default_factory=dict)


@dataclass
class FoldResult:
    """Output of the fold/dock predictor (AF2 initial-guess style)."""
    pae_interface: float               # lower is better
    plddt: float                       # higher is better
    ca_rmsd_to_design: float           # design vs prediction, lower is better
    predictors_agree: bool = True      # AF2 vs AF3/Chai consensus (gate G8)


@dataclass
class SpecificityResult:
    """Output of the contrastive specificity engine (§06 of the design)."""
    on_target_score: float                     # blended AF2+MPNN confidence
    off_target_scores: dict[str, float]        # peptide seq -> score
    margin: float                              # worst-case on - off (min)
    peptide_energy_fraction: float             # phi: energy from peptide vs MHC
    mpnn_recovery_ok: bool                     # on-target recovered above all off

    @property
    def worst_offender(self) -> Optional[str]:
        if not self.off_target_scores:
            return None
        return max(self.off_target_scores, key=self.off_target_scores.get)


# --------------------------------------------------------------------------
# Gate + campaign bookkeeping
# --------------------------------------------------------------------------
class Stage(str, Enum):
    INTAKE = "intake"
    PANEL = "panel"
    BACKBONE = "backbone"
    SEQUENCE = "sequence"
    FOLD = "fold"
    SPECIFICITY = "specificity"
    LIBRARY = "library"
    DONE = "done"
    STALLED = "stalled"


@dataclass
class GateResult:
    name: str
    passed: int
    rejected: int
    reason_hint: str = ""


@dataclass
class RoundReport:
    round_index: int
    generated: int
    gate_results: list[GateResult] = field(default_factory=list)
    survivors: int = 0
    diagnosis: str = ""


@dataclass
class Campaign:
    """The full mutable state of one design campaign for one target."""
    target: Target
    panel: list[OffTarget] = field(default_factory=list)
    library: list[Design] = field(default_factory=list)   # accepted designs
    rounds: list[RoundReport] = field(default_factory=list)
    stage: Stage = Stage.INTAKE
    # Per-target learned specificity threshold (theta). Adapts each round.
    theta: float = 1.0
    done: bool = False
    notes: list[str] = field(default_factory=list)
