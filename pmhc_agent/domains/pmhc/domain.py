"""PMHCDomain — the pMHC-I science as a DesignDomain plugin.

This wraps the existing, proven pMHC code (intake, panel builder, tool
registry, G2-G9 gate predicates, theta adaptation, composite scoring) behind
the domain interface. It reuses those functions verbatim, so an Engine driven
by PMHCDomain produces results identical to the legacy Orchestrator — which is
the regression proof that the generalization changed nothing.
"""
from __future__ import annotations

from ...types import Target, Campaign, Stage, Design
from ...config import GatePolicy
from ...panel import build_panel
from ...tools import build_registry, ToolRegistry
from ...interfaces import Gate
from ... import gates as G


class PMHCDomain:
    name = "pmhc-I"

    def __init__(self, seed: int = 7, policy: GatePolicy | None = None,
                 panel_size: int = 12, registry: ToolRegistry | None = None,
                 solved_ids: dict | None = None):
        self.seed = seed
        self.policy = policy or GatePolicy()
        self.panel_size = panel_size
        self.reg = registry or build_registry(seed=seed, solved_ids=solved_ids)

    # -- target intake + structure resolution ------------------------------
    def intake(self, target: Target) -> Campaign:
        camp = Campaign(target=target)
        rank = self.reg.netmhc.percent_rank(target.peptide, target.hla_allele)
        if rank > self.policy.max_ligand_rank:
            camp.stage = Stage.STALLED
            camp.done = True
            camp.notes.append(
                f"G1 fail: peptide %rank {rank} > {self.policy.max_ligand_rank}"
                " — not a credible ligand for this allele.")
            return camp
        self.reg.structure.resolve(target)
        if (target.structure_confidence or 0) < 0.5:
            camp.notes.append(
                f"Low pMHC structure confidence ({target.structure_confidence}); "
                "flagging for human review but proceeding.")
        camp.stage = Stage.PANEL
        camp.notes.append(
            f"Target ready: {target.source_antigen} {target.peptide.sequence} "
            f"on {target.hla_allele} (%rank {rank}, "
            f"struct {target.pmhc_structure_id}).")
        return camp

    # -- adversarial negative set ------------------------------------------
    def build_negative_set(self, camp: Campaign) -> None:
        camp.panel = build_panel(camp.target, self.reg.netmhc,
                                 size=self.panel_size, seed=self.seed)
        camp.notes.append(
            f"Off-target panel: {len(camp.panel)} confusable peptides "
            f"(hardest: {[o.peptide.sequence for o in camp.panel[:3]]}).")
        camp.stage = Stage.BACKBONE

    # -- tools --------------------------------------------------------------
    def tools(self) -> ToolRegistry:
        return self.reg

    # -- gate list (stage-tagged; reuses the proven G2-G9 predicates) -------
    def gates(self) -> list[Gate]:
        p = self.policy
        return [
            Gate("G2 peptide-centric", "after_sequence",
                 lambda d, c: G.g2_peptide_centric(d, c["policy"])),
            Gate("G3 foldable", "after_sequence",
                 lambda d, c: G.g3_foldable(d, c["policy"])),
            Gate("G4 fold&dock", "after_fold",
                 lambda d, c: G.g4_fold_dock(d, c["policy"])),
            Gate("G5 contrastive recovery", "after_score",
                 lambda d, c: G.g5_recovery(d)),
            Gate("G6 specificity margin", "after_score",
                 lambda d, c: G.g6_margin(d, c["theta"])),
            Gate("G7 interface partition", "after_score",
                 lambda d, c: G.g7_partition(d, c["policy"])),
            Gate("G8 consensus", "after_score",
                 lambda d, c: G.g8_consensus(d)),
            Gate("G9 developable", "after_score",
                 lambda d, c: G.g9_developable(d, c["policy"])),
        ]

    def gate_ctx(self, camp: Campaign) -> dict:
        return {"policy": self.policy, "theta": camp.theta}

    # -- adaptive threshold (theta), from the observed margin spread --------
    def adapt_thresholds(self, camp: Campaign, designs: list[Design]) -> None:
        margins = sorted(d.specificity.margin for d in designs
                         if d.specificity and d.specificity.margin > 0)
        if not margins:
            return
        idx = int(0.6 * (len(margins) - 1))
        adaptive = margins[idx]
        camp.theta = round(max(0.8, 0.5 * camp.theta + 0.5 * adaptive), 3)

    # -- composite ranking --------------------------------------------------
    def rank_score(self, d: Design) -> float:
        aff = max(0.0, 1.2 - d.mpnn_score)
        spec = max(0.0, d.specificity.margin) if d.specificity else 0.0
        dev = 0.6 if d.liabilities else 1.0
        return round(aff * spec * dev, 4)

    # -- memory key + gpu hints + brain context ----------------------------
    def target_key(self, target: Target) -> str:
        return target.hla_allele

    def gpus_per_stage(self) -> dict:
        return {}   # engine falls back to AgentConfig.gpus

    def describe(self) -> str:
        return (
            "Domain: high-specificity binders for peptide-MHC-I complexes. "
            "Backbones arc over the peptide (RFdiffusion), sequences via "
            "ProteinMPNN, fold/dock via AlphaFold2, specificity scored "
            "contrastively (worst-case margin over an off-target peptide "
            "panel). Gates: G2 peptide-centric backbone, G3 foldability, "
            "G4 fold&dock (pAE/pLDDT), G5 contrastive recovery, G6 margin>=theta, "
            "G7 interface partition, G8 predictor consensus, G9 developability.")
