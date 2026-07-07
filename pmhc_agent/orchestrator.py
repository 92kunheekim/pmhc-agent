"""The orchestrator — plans, routes, scores, decides to stop, re-plans.

This is where the "agent" lives. The heavy models are dumb tools; the value
is the policy encoded here: what to run, on what, with which off-targets, how
to score, when to stop, and what to try next. Mirrors the control-loop
pseudocode in §09 of the design doc.

Wire the LLM in where `diagnose()` is called if you want free-form reasoning
over the score distributions; the deterministic policy below is a strong,
inspectable default that runs with no API key.
"""
from __future__ import annotations

import logging
from statistics import mean

from .config import AgentConfig
from .tools import ToolRegistry, build_registry
from .types import (Target, Campaign, Design, Stage, RoundReport, GateResult)
from functools import partial

from . import gates as G
from .panel import build_panel
from .memory import Memory
from .diagnostics import Diagnoser, RuleBasedDiagnoser
from .execution import (Executor, LocalExecutor,
                        task_sequence, task_fold, task_specificity)

log = logging.getLogger("pmhc_agent")


class Orchestrator:
    def __init__(self, config: AgentConfig | None = None,
                 registry: ToolRegistry | None = None,
                 memory: Memory | None = None,
                 diagnoser: Diagnoser | None = None,
                 executor: Executor | None = None):
        self.cfg = config or AgentConfig()
        self.reg = registry or build_registry(seed=self.cfg.seed)
        self.mem = memory or Memory()
        # The swappable "brain": rule-based by default; pass an LLMDiagnoser
        # for model-driven reasoning. Both return the same ReplanAction.
        self.diagnoser = diagnoser or RuleBasedDiagnoser()
        # The swappable execution backend: in-process by default; pass a
        # RayExecutor to fan the heavy stages out onto GPU workers. Both
        # preserve order, so determinism holds either way.
        self.executor = executor or LocalExecutor()

    # -- Stage 1: intake + structure --------------------------------------
    def intake(self, target: Target) -> Campaign:
        camp = Campaign(target=target)
        rank = self.reg.netmhc.percent_rank(target.peptide, target.hla_allele)
        if rank > self.cfg.gates.max_ligand_rank:
            camp.stage = Stage.STALLED
            camp.done = True
            camp.notes.append(
                f"G1 fail: peptide %rank {rank} > {self.cfg.gates.max_ligand_rank}"
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

    # -- Stage 2: adversarial panel ---------------------------------------
    def build_panel(self, camp: Campaign) -> None:
        camp.panel = build_panel(camp.target, self.reg.netmhc,
                                 size=self.cfg.panel_size, seed=self.cfg.seed)
        camp.notes.append(
            f"Off-target panel: {len(camp.panel)} confusable peptides "
            f"(hardest: {[o.peptide.sequence for o in camp.panel[:3]]}).")
        camp.stage = Stage.BACKBONE

    # -- One generate->filter round ---------------------------------------
    def _round(self, camp: Campaign, round_index: int,
               contact_bias: float, use_scaffold_seed: bool) -> RoundReport:
        cfg, reg, p = self.cfg, self.reg, self.cfg.gates
        seed_scaffold = None
        if use_scaffold_seed:
            priv = self.mem.privileged_for(camp.target.hla_allele)
            seed_scaffold = priv[0] if priv else None

        # Generate backbones (RFdiffusion). In production this is itself a
        # per-trajectory fan-out; the mock returns them in one call.
        backbones = reg.backbones.generate(
            camp.target, cfg.budget.backbones_per_round, round_index,
            seed_scaffold=seed_scaffold, contact_bias=contact_bias)

        # Sequence design (ProteinMPNN) — fan out one task per backbone.
        seq_fn = partial(task_sequence, reg.sequences,
                         cfg.budget.sequences_per_backbone, round_index)
        design_lists = self.executor.map(seq_fn, backbones,
                                         gpus_per_task=cfg.gpus.sequence,
                                         label="mpnn")
        designs: list[Design] = [d for sub in design_lists for d in sub]

        report = RoundReport(round_index=round_index, generated=len(designs))

        # G2 peptide-centric backbone (cheap)
        designs, r = G.run_gate("G2 peptide-centric", designs,
                                lambda d: G.g2_peptide_centric(d, p))
        report.gate_results.append(r)
        # G3 foldability / packing (cheap)
        designs, r = G.run_gate("G3 foldable", designs,
                                lambda d: G.g3_foldable(d, p))
        report.gate_results.append(r)

        # G4 fold & dock (AlphaFold) — fan out one GPU task per survivor.
        fold_fn = partial(task_fold, reg.folding, camp.target)
        designs = self.executor.map(fold_fn, designs,
                                    gpus_per_task=cfg.gpus.fold, label="af2")
        designs, r = G.run_gate("G4 fold&dock", designs,
                                lambda d: G.g4_fold_dock(d, p))
        report.gate_results.append(r)

        # Specificity engine (fine-tuned AF2 + contrastive MPNN) — fan out
        # one GPU task per survivor; each scores against the full panel.
        spec_fn = partial(task_specificity, reg.specificity, camp.target,
                          camp.panel)
        designs = self.executor.map(spec_fn, designs,
                                    gpus_per_task=cfg.gpus.specificity,
                                    label="specificity")

        # Adapt theta from THIS round's margin distribution before G6.
        self._adapt_theta(camp, designs)

        designs, r = G.run_gate("G5 contrastive recovery", designs, G.g5_recovery)
        report.gate_results.append(r)
        designs, r = G.run_gate("G6 specificity margin", designs,
                                lambda d: G.g6_margin(d, camp.theta),
                                reason_hint=f"theta={camp.theta:.2f}")
        report.gate_results.append(r)
        designs, r = G.run_gate("G7 interface partition", designs,
                                lambda d: G.g7_partition(d, p))
        report.gate_results.append(r)
        # G8 predictor consensus
        designs, r = G.run_gate("G8 consensus", designs, G.g8_consensus)
        report.gate_results.append(r)
        # G9 developability
        designs, r = G.run_gate("G9 developable", designs,
                                lambda d: G.g9_developable(d, p))
        report.gate_results.append(r)

        # Diversity cap + composite score
        designs = G.enforce_diversity(designs, p)
        for d in designs:
            d.composite_score = self._composite(d)
        report.survivors = len(designs)
        camp.library.extend(designs)
        return report

    # -- Scoring helpers ---------------------------------------------------
    def _composite(self, d: Design) -> float:
        """affinity_proxy x specificity_margin x developability (higher better)."""
        aff = max(0.0, 1.2 - d.mpnn_score)            # packing as affinity proxy
        spec = max(0.0, d.specificity.margin)
        dev = 0.6 if d.liabilities else 1.0
        return round(aff * spec * dev, 4)

    def _adapt_theta(self, camp: Campaign, designs: list[Design]) -> None:
        """Set the specificity threshold from the observed margin spread.

        Policy: theta = 75th-percentile-ish of positive margins, floored, so
        we keep only clearly-discriminating designs but don't demand more than
        this target can offer.
        """
        margins = sorted(d.specificity.margin for d in designs
                         if d.specificity and d.specificity.margin > 0)
        if not margins:
            return
        idx = int(0.6 * (len(margins) - 1))
        adaptive = margins[idx]
        # Blend toward the current theta for stability; keep a sane floor.
        camp.theta = round(max(0.8, 0.5 * camp.theta + 0.5 * adaptive), 3)

    # -- Full campaign -----------------------------------------------------
    def run(self, target: Target) -> Campaign:
        camp = self.intake(target)
        if camp.done:
            log.info("Campaign halted at intake: %s", camp.notes[-1])
            return camp
        self.build_panel(camp)

        contact_bias = 0.0
        use_seed = bool(self.mem.privileged_for(target.hla_allele))
        prev_lib_size = 0
        stalled = 0

        for r in range(self.cfg.budget.max_rounds):
            report = self._round(camp, r, contact_bias, use_seed)

            # Stalled = a round added no new accepted designs (not merely that
            # the best margin plateaued while the library keeps growing).
            if len(camp.library) <= prev_lib_size:
                stalled += 1
            else:
                stalled = 0
            prev_lib_size = len(camp.library)

            if len(camp.library) >= self.cfg.budget.target_library_size:
                report.diagnosis = "Target library size reached."
                camp.rounds.append(report)
                camp.stage = Stage.LIBRARY
                camp.done = True
                break

            action = self.diagnoser.diagnose(report, camp.theta, stalled,
                                             self.cfg.budget.stall_patience)
            report.diagnosis = action.diagnosis
            camp.rounds.append(report)

            if action.escalate:
                camp.stage = Stage.STALLED
                camp.done = True
                camp.notes.append("ESCALATE: " + action.diagnosis)
                break
            contact_bias += action.contact_bias
            if action.lower_theta_by:
                camp.theta = max(0.8, camp.theta - action.lower_theta_by)
            use_seed = use_seed or action.use_scaffold_seed

        # Bank privileged scaffolds from the best designs.
        top = sorted(camp.library, key=lambda d: d.composite_score, reverse=True)
        for d in top[:5]:
            self.mem.bank(d, target.hla_allele,
                          f"{target.source_antigen}:{target.peptide.sequence}")
        if camp.stage != Stage.STALLED and not camp.done:
            camp.stage = Stage.LIBRARY
            camp.done = True
        return camp

    # -- Wet-lab handoff (human-gated) ------------------------------------
    def recommend_library(self, camp: Campaign, top_n: int = 10) -> list[Design]:
        """Return the ranked, synthesis-ready shortlist.

        NOTE: ordering DNA and running assays are HUMAN-APPROVED actions.
        This method only *recommends*; it never actuates anything.
        """
        ranked = sorted(camp.library, key=lambda d: d.composite_score,
                        reverse=True)
        return ranked[:top_n]
