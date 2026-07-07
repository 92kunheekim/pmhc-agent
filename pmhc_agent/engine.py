"""Engine — the domain-agnostic orchestrator.

Same control loop, memory, brain, executor, and gate runner as the original
`Orchestrator`, but driven by a `DesignDomain` instead of hardcoded pMHC calls.
Give it `PMHCDomain` and it reproduces the legacy behavior exactly (proven by
tests/test_engine_pmhc.py); give it an `AntibodyDomain` later and the same
engine runs antibody campaigns.

The legacy `Orchestrator` is left in place unchanged for back-compat; this is
the generalized path it will eventually be replaced by.
"""
from __future__ import annotations

import logging
from functools import partial

from .config import AgentConfig
from .types import Campaign, Design, Stage, RoundReport, Target
from .interfaces import DesignDomain
from .execution import (Executor, LocalExecutor,
                        task_sequence, task_fold, task_specificity)
from . import gates as G
from .memory import Memory
from .diagnostics import Diagnoser, RuleBasedDiagnoser

log = logging.getLogger("pmhc_agent.engine")


class Engine:
    def __init__(self, domain: DesignDomain, config: AgentConfig | None = None,
                 executor: Executor | None = None,
                 memory: Memory | None = None,
                 diagnoser: Diagnoser | None = None):
        self.domain = domain
        self.cfg = config or AgentConfig()
        self.executor = executor or LocalExecutor()
        self.mem = memory or Memory()
        self.diagnoser = diagnoser or RuleBasedDiagnoser()

    # -- per-stage GPU request (domain may override config) ----------------
    def _gpus(self, stage: str) -> float:
        override = self.domain.gpus_per_stage().get(stage)
        if override is not None:
            return override
        return getattr(self.cfg.gpus, stage, 0.0)

    @staticmethod
    def _populate_metrics(d: Design) -> None:
        """Mirror typed results into the named-metric bag (additive; domains
        whose gates read metrics use this — pMHC gates read typed fields, so
        this is a no-op for parity)."""
        m = d.metrics
        m.setdefault("mpnn_score", d.mpnn_score)
        m["peptide_contact_fraction"] = d.backbone.peptide_contact_fraction
        if d.rosetta_ddg is not None:
            m["rosetta_ddg"] = d.rosetta_ddg
        if d.fold is not None:
            m["pae_interaction"] = d.fold.pae_interface
            m["plddt"] = d.fold.plddt
            m["ca_rmsd"] = d.fold.ca_rmsd_to_design
        if d.specificity is not None:
            m["margin"] = d.specificity.margin
            m["phi"] = d.specificity.peptide_energy_fraction

    def _run_stage_gates(self, designs, stage, ctx, camp, report):
        for gate in self.domain.gates():
            if gate.stage != stage:
                continue
            hint = f"theta={camp.theta:.2f}" if "margin" in gate.name else ""
            designs, gr = G.run_gate(
                gate.name, designs,
                (lambda d, g=gate: g.predicate(d, ctx)), reason_hint=hint)
            report.gate_results.append(gr)
        return designs

    # -- one generate -> filter round --------------------------------------
    def _round(self, camp: Campaign, r: int, contact_bias: float,
               use_scaffold_seed: bool) -> RoundReport:
        dom, tools = self.domain, self.domain.tools()
        seed_scaffold = None
        if use_scaffold_seed:
            priv = self.mem.privileged_for(dom.target_key(camp.target))
            seed_scaffold = priv[0] if priv else None

        scaffolds = tools.backbones.generate(
            camp.target, self.cfg.budget.backbones_per_round, r,
            seed_scaffold=seed_scaffold, contact_bias=contact_bias)

        seq_fn = partial(task_sequence, tools.sequences,
                         self.cfg.budget.sequences_per_backbone, r)
        design_lists = self.executor.map(seq_fn, scaffolds,
                                         gpus_per_task=self._gpus("sequence"),
                                         label="mpnn")
        designs: list[Design] = [d for sub in design_lists for d in sub]
        for d in designs:
            self._populate_metrics(d)

        report = RoundReport(round_index=r, generated=len(designs))
        ctx = dom.gate_ctx(camp)
        designs = self._run_stage_gates(designs, "after_sequence", ctx, camp, report)

        fold_fn = partial(task_fold, tools.folding, camp.target)
        designs = self.executor.map(fold_fn, designs,
                                    gpus_per_task=self._gpus("fold"), label="af2")
        for d in designs:
            self._populate_metrics(d)
        designs = self._run_stage_gates(designs, "after_fold", ctx, camp, report)

        spec_fn = partial(task_specificity, tools.specificity, camp.target,
                          camp.panel)
        designs = self.executor.map(spec_fn, designs,
                                    gpus_per_task=self._gpus("specificity"),
                                    label="specificity")
        for d in designs:
            self._populate_metrics(d)

        dom.adapt_thresholds(camp, designs)
        ctx = dom.gate_ctx(camp)                      # theta may have changed
        designs = self._run_stage_gates(designs, "after_score", ctx, camp, report)

        policy = ctx["policy"]
        designs = G.enforce_diversity(designs, policy)
        for d in designs:
            d.composite_score = dom.rank_score(d)
        report.survivors = len(designs)
        camp.library.extend(designs)
        return report

    # -- full campaign -----------------------------------------------------
    def run(self, target: Target) -> Campaign:
        dom = self.domain
        camp = dom.intake(target)
        if camp.done:
            log.info("Campaign halted at intake: %s",
                     camp.notes[-1] if camp.notes else "")
            return camp
        dom.build_negative_set(camp)

        contact_bias = 0.0
        use_seed = bool(self.mem.privileged_for(dom.target_key(target)))
        prev_lib_size = 0
        stalled = 0

        for r in range(self.cfg.budget.max_rounds):
            report = self._round(camp, r, contact_bias, use_seed)

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

        top = sorted(camp.library, key=lambda d: d.composite_score, reverse=True)
        for d in top[:5]:
            self.mem.bank(d, dom.target_key(target),
                          f"{getattr(target, 'source_antigen', '?')}")
        if camp.stage != Stage.STALLED and not camp.done:
            camp.stage = Stage.LIBRARY
            camp.done = True
        return camp

    def recommend_library(self, camp: Campaign, top_n: int = 10) -> list[Design]:
        return sorted(camp.library, key=lambda d: d.composite_score,
                      reverse=True)[:top_n]
