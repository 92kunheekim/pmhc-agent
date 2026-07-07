"""Gate policies and campaign budget.

Thresholds here are *starting policies*. The orchestrator adapts the
specificity threshold (theta) per target from observed score distributions
(see orchestrator.adapt_theta). Everything is a plain dataclass so you can
serialize / override it from a YAML or CLI without touching code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatePolicy:
    # G1 ligand sanity (NetMHCpan %rank; lower binds better)
    max_ligand_rank: float = 2.0
    # G2 peptide-centric backbone
    min_peptide_contact_fraction: float = 0.40
    # G3 foldability / packing
    max_mpnn_score: float = 1.10          # per-residue NLL cutoff
    max_rosetta_ddg: float = -20.0        # keep ddG <= this (more negative better)
    # G4 on-target fold & dock
    max_pae_interface: float = 10.0
    min_plddt: float = 80.0
    max_ca_rmsd: float = 2.0
    # G5 contrastive recovery handled as boolean flag
    # G6 specificity margin uses campaign.theta (adaptive)
    # G7 interface partition
    min_peptide_energy_fraction: float = 0.50
    # G8 predictor consensus handled as boolean flag
    # G9 developability: forbid these liabilities
    forbid_free_cys: bool = True
    max_designs_per_scaffold: int = 3     # diversity cap


@dataclass
class Budget:
    max_rounds: int = 6
    backbones_per_round: int = 240
    sequences_per_backbone: int = 2
    target_library_size: int = 24         # stop when we have this many accepted
    stall_patience: int = 2               # rounds w/o margin gain before escalate


@dataclass
class GpuRequest:
    """GPUs requested per task at each fan-out stage (Ray `num_gpus`).

    Defaults are 0.0 so the mock/local runs need no GPU. On a real Ray/K8s
    cluster set these to route each stage onto GPU workers. Fractional values
    pack several light tasks onto one physical GPU (e.g. ProteinMPNN at 0.25).
    """
    sequence: float = 0.0      # ProteinMPNN — light; fractional GPU is fine
    fold: float = 0.0          # AlphaFold — heavy; usually a full GPU
    specificity: float = 0.0   # fine-tuned AF2 — heavy; usually a full GPU


@dataclass
class AgentConfig:
    gates: GatePolicy = None
    budget: Budget = None
    gpus: GpuRequest = None                # per-stage GPU requests (Ray)
    panel_size: int = 12                  # hardest N off-targets to defend against
    seed: int = 7                         # determinism for the mock backends

    def __post_init__(self) -> None:
        if self.gates is None:
            self.gates = GatePolicy()
        if self.budget is None:
            self.budget = Budget()
        if self.gpus is None:
            self.gpus = GpuRequest()
