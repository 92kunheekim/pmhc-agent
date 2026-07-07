"""Engine interfaces — the seams that make the orchestration domain-agnostic.

A `DesignDomain` supplies everything science-specific (target parsing, the
negative set to screen against, the tool backends, the gate list, ranking, and
a self-description for the LLM brain). The engine (`engine.Engine`) holds the
domain-independent machinery (loop, executor, memory, brain, gate runner) and
drives whatever domain it is given — pMHC today, antibody next.

See docs/ARCHITECTURE-generalization.md for the full plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .types import Design, Campaign, Target


# --------------------------------------------------------------------------
# Gate: a named check, tagged with the stage it runs after, over a Design.
# --------------------------------------------------------------------------
STAGES = ("after_sequence", "after_fold", "after_score")


@dataclass
class Gate:
    """A filter gate. `predicate(design, ctx) -> bool`; `ctx` carries the
    domain policy + per-round state (e.g. the adaptive threshold theta).

    `stage` places the gate in the pipeline so expensive predictions only run
    on survivors of the cheap gates (the cheap-first funnel):
      * after_sequence — cheap checks on scaffold/sequence metrics
      * after_fold     — checks needing the structure prediction
      * after_score    — checks needing the specificity/negative-set score
    """
    name: str
    stage: str
    predicate: Callable[[Design, dict], bool]
    reason_hint: str = ""


# --------------------------------------------------------------------------
# Tool Protocols — the four backend roles (implementations live per-domain).
# --------------------------------------------------------------------------
class BackboneGenerator(Protocol):
    def generate(self, target, n, round_index, seed_scaffold=None,
                 contact_bias=0.0) -> list: ...


class SequenceDesigner(Protocol):
    def design(self, backbone, n, round_index) -> list: ...


class FoldPredictor(Protocol):
    def predict(self, design, target): ...


class NegativeScorer(Protocol):
    """Scores a design against the domain's negative set (pMHC: off-target
    peptides; antibody: cross-reactivity antigens)."""
    def score(self, design, target, panel): ...


# --------------------------------------------------------------------------
# DesignDomain — the plugin the engine is parameterized by.
# --------------------------------------------------------------------------
class DesignDomain(Protocol):
    name: str

    def intake(self, target: Target) -> Campaign:
        """Validate the target, resolve/predict its structure -> Campaign."""
        ...

    def build_negative_set(self, camp: Campaign) -> None:
        """Populate camp.panel with the adversarial set to screen against."""
        ...

    def tools(self):
        """Return the ToolRegistry (backbones/sequences/folding/specificity)."""
        ...

    def gates(self) -> list[Gate]:
        """Ordered, stage-tagged gate list."""
        ...

    def gate_ctx(self, camp: Campaign) -> dict:
        """Per-round context passed to gate predicates (policy + theta)."""
        ...

    def adapt_thresholds(self, camp: Campaign, designs: list[Design]) -> None:
        """Adapt per-target thresholds (e.g. theta) from observed scores."""
        ...

    def rank_score(self, design: Design) -> float:
        """Composite ranking score for the final library."""
        ...

    def target_key(self, target: Target) -> str:
        """Key under which privileged scaffolds are banked/reused."""
        ...

    def gpus_per_stage(self) -> dict:
        """Optional per-stage GPU requests (falls back to config)."""
        ...

    def describe(self) -> str:
        """Domain context string for the LLM brain's system prompt."""
        ...
