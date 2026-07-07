"""Failure diagnosis + threshold adaptation (the agent's judgment).

When a round yields too few survivors, the agent classifies *why* and
returns a concrete adjustment for the next round instead of blindly
re-sampling. This is the taxonomy from §08 of the design doc.

This module is the agent's "brain". Two interchangeable implementations of
the `Diagnoser` interface are provided:

  * `RuleBasedDiagnoser` — the deterministic `diagnose()` policy below.
    Fast, free, reproducible, testable. The default.
  * `LLMDiagnoser` (in `pmhc_agent.llm`) — hands the same decision to a
    language model for open-ended reasoning, then clamps + validates its
    answer back into a `ReplanAction`.

Both return the SAME `ReplanAction`, so the orchestrator is agnostic to
which brain is in charge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import RoundReport


@dataclass
class ReplanAction:
    diagnosis: str
    contact_bias: float = 0.0        # bump RFdiffusion toward the peptide
    lower_theta_by: float = 0.0      # relax specificity margin (bounded)
    use_scaffold_seed: bool = False  # reseed from privileged scaffold
    escalate: bool = False           # give up / ask a human


def diagnose(report: RoundReport, theta: float, stalled_rounds: int,
             patience: int) -> ReplanAction:
    """Read the gate tallies of the last round and decide what to change."""
    gates = {g.name: g for g in report.gate_results}

    def rej(name: str) -> int:
        return gates[name].rejected if name in gates else 0

    def passed(name: str) -> int:
        return gates[name].passed if name in gates else 0

    if stalled_rounds >= patience:
        return ReplanAction(
            diagnosis=("Stalled: no specificity-margin improvement for "
                       f"{stalled_rounds} rounds. Escalating to human review "
                       "— target may present too few outward-facing residues."),
            escalate=True,
        )

    # Most designs died at the peptide-centric backbone gate -> bias contacts.
    if rej("G2 peptide-centric") > passed("G2 peptide-centric"):
        return ReplanAction(
            diagnosis="Backbones were MHC-leaning; biasing RFdiffusion toward "
                      "outward-facing peptide residues.",
            contact_bias=0.08, use_scaffold_seed=True,
        )

    # Folded & specific-recovering but margin gate too strict for this target.
    if passed("G5 contrastive recovery") > 0 and passed("G6 specificity margin") == 0:
        return ReplanAction(
            diagnosis="Designs recover the target but miss the margin cutoff; "
                      "relaxing theta slightly and reseeding from privileged "
                      "scaffolds.",
            lower_theta_by=0.15, use_scaffold_seed=True,
        )

    # Bound but not specific (paper's mage-282 mode): recovery failing broadly.
    if rej("G5 contrastive recovery") > passed("G5 contrastive recovery"):
        return ReplanAction(
            diagnosis="Designs bind but don't discriminate (cross-reactive); "
                      "hardening peptide focus before re-scoring.",
            contact_bias=0.06,
        )

    # Didn't fold well.
    if rej("G4 fold&dock") > passed("G4 fold&dock"):
        return ReplanAction(
            diagnosis="Poor fold/dock confidence; widening backbone + sequence "
                      "sampling and reseeding from privileged scaffolds.",
            use_scaffold_seed=True,
        )

    return ReplanAction(
        diagnosis="Marginal yield; resampling with privileged-scaffold seeding.",
        use_scaffold_seed=True,
    )


# --------------------------------------------------------------------------
# The Diagnoser interface — the swappable "brain"
# --------------------------------------------------------------------------
# Safe bounds the orchestrator will accept for any proposed action, whether
# it came from a rule or an LLM. Keeping these here (not just inside the LLM
# backend) means the guardrail applies no matter what produces the action.
CONTACT_BIAS_MAX = 0.20
LOWER_THETA_MAX = 0.50


def clamp_action(a: ReplanAction) -> ReplanAction:
    """Clamp a proposed action into safe ranges. Idempotent."""
    a.contact_bias = max(0.0, min(CONTACT_BIAS_MAX, float(a.contact_bias)))
    a.lower_theta_by = max(0.0, min(LOWER_THETA_MAX, float(a.lower_theta_by)))
    a.use_scaffold_seed = bool(a.use_scaffold_seed)
    a.escalate = bool(a.escalate)
    return a


class Diagnoser(Protocol):
    """A brain: read the last round, decide what to change next."""
    name: str

    def diagnose(self, report: RoundReport, theta: float,
                 stalled_rounds: int, patience: int) -> ReplanAction:
        ...


class RuleBasedDiagnoser:
    """Deterministic policy (the default). Wraps `diagnose()` above."""
    name = "rules"

    def diagnose(self, report: RoundReport, theta: float,
                 stalled_rounds: int, patience: int) -> ReplanAction:
        return clamp_action(diagnose(report, theta, stalled_rounds, patience))
