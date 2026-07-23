"""LLM-backed diagnoser — the agent's "brain" as a language model.

This is the LLM alternative to the rule-based `diagnose()` policy. It hands
the round's results to an Anthropic model and asks it to reason about what to
change next, forcing a structured answer (via tool use) that maps directly
onto the same `ReplanAction` the rules produce. The orchestrator cannot tell
the difference.

Design choices that matter:
  * DROP-IN: same interface (`.diagnose(...) -> ReplanAction`) as
    `RuleBasedDiagnoser`, so nothing else in the loop changes.
  * GUARDRAILED: whatever the model proposes is clamped into safe ranges
    (`clamp_action`) before the orchestrator acts on it. The model advises;
    it does not get unbounded control.
  * GRACEFUL: if the `anthropic` SDK isn't installed, no API key is set, or
    the call fails, it falls back to the deterministic rules and annotates
    the diagnosis so you can see it happened. The package therefore still
    runs with no key.
  * INJECTABLE: you can pass a `client` (any object with
    `.messages.create(...)`), which lets tests exercise it with a fake and
    no network.

Enable it:
    pip install anthropic         # or:  pip install -e ".[llm]"
    export ANTHROPIC_API_KEY=sk-...
    python -m pmhc_agent.run --brain llm
"""
from __future__ import annotations

import json
import logging
import os
import re

from .types import RoundReport
from .diagnostics import ReplanAction, RuleBasedDiagnoser, clamp_action

log = logging.getLogger("pmhc_agent.llm")

# Configurable — set to whatever Anthropic model you have access to.
# Model ids change over time and differ per account. List yours with
#   anthropic.Anthropic().models.list()
# and override without editing code via  export PMHC_LLM_MODEL=...
DEFAULT_MODEL = os.environ.get("PMHC_LLM_MODEL", "claude-sonnet-4-5")

# The structured-output contract we force the model to fill in. Its fields
# are exactly the fields of ReplanAction.
_REPLAN_TOOL = {
    "name": "propose_replan",
    "description": (
        "Propose how to steer the next design round of a pMHC-I binder-design "
        "campaign, based on the filter-gate results just observed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": "One or two sentences: what went wrong this "
                               "round and why this action addresses it.",
            },
            "contact_bias": {
                "type": "number",
                "description": "How much to bias RFdiffusion toward the "
                               "outward-facing peptide residues next round. "
                               "0.0 = no change; ~0.06-0.10 for a real nudge; "
                               "max 0.20. Raise when backbones are MHC-leaning "
                               "or designs bind but aren't specific.",
            },
            "lower_theta_by": {
                "type": "number",
                "description": "How much to relax the specificity margin "
                               "threshold theta. 0.0 = keep strict; max 0.50. "
                               "Only relax when designs clearly recover the "
                               "on-target peptide but narrowly miss the margin.",
            },
            "use_scaffold_seed": {
                "type": "boolean",
                "description": "Reseed backbone generation from a privileged "
                               "scaffold in memory instead of de novo.",
            },
            "escalate": {
                "type": "boolean",
                "description": "Stop and ask a human. Set true only if the "
                               "campaign is stuck (no improvement for >= "
                               "patience rounds) or the target looks "
                               "intrinsically undesignable.",
            },
        },
        "required": ["diagnosis", "contact_bias", "lower_theta_by",
                     "use_scaffold_seed", "escalate"],
    },
}

_SYSTEM = (
    "You are the planning brain of an autonomous agent that designs "
    "high-specificity protein binders for peptide-MHC-I complexes "
    "(cf. Baker lab, Science 2025). Each round the agent generates binder "
    "backbones (RFdiffusion), designs sequences (ProteinMPNN), predicts "
    "folding/docking (AlphaFold2), and scores specificity contrastively "
    "against a panel of off-target peptides. Designs are filtered through a "
    "gate stack (G2 peptide-centric backbone, G3 foldability, G4 fold&dock, "
    "G5 contrastive recovery, G6 specificity margin >= theta, G7 interface "
    "partition, G8 predictor consensus, G9 developability).\n\n"
    "Given the gate pass/reject tallies from the last round, decide how to "
    "steer the next one. Reason about WHERE designs are dying:\n"
    "- Many rejected at G2  -> backbones are MHC-leaning; raise contact_bias.\n"
    "- Many rejected at G4  -> poor fold; reseed from a privileged scaffold.\n"
    "- Pass G5 but fail G6  -> designs recognize the target but miss the "
    "margin; relax theta a little AND reseed.\n"
    "- Many rejected at G5  -> designs bind but don't discriminate "
    "(cross-reactive); raise contact_bias to sharpen peptide focus.\n"
    "- Stuck for >= patience rounds -> escalate to a human.\n\n"
    "Be conservative: prefer small, targeted nudges. Call the "
    "`propose_replan` tool with your decision."
)


def _summarize(report: RoundReport, theta: float, stalled_rounds: int,
               patience: int) -> str:
    gates = [
        {"gate": g.name, "passed": g.passed, "rejected": g.rejected,
         "note": g.reason_hint}
        for g in report.gate_results
    ]
    payload = {
        "round_index": report.round_index,
        "designs_generated": report.generated,
        "survivors_this_round": report.survivors,
        "current_theta": round(theta, 3),
        "stalled_rounds": stalled_rounds,
        "stall_patience": patience,
        "gate_results": gates,
    }
    return json.dumps(payload, indent=2)


def _clean_field_text(value: object) -> str:
    """Defensively strip tool-call/XML artifacts a model may leak into a
    free-text field. Some models occasionally emit closing tags or the next
    parameter as inline pseudo-XML *inside* a string value (e.g. a stray
    ``</diagnosis>`` or ``<parameter name=...>``). We cut at the first such
    artifact and drop any residual tags so the parsed text stays clean."""
    text = str(value or "")
    text = re.split(r"</?\s*(?:diagnosis|parameter|tool|invoke|function)\b",
                    text, maxsplit=1)[0]
    text = re.sub(r"<[^>]*>", "", text)           # remove any leftover tags
    return text.strip()


class LLMDiagnoser:
    """Diagnoser backed by an Anthropic model, with rule-based fallback."""
    name = "llm"

    def __init__(self, client: object | None = None,
                 model: str = DEFAULT_MODEL, max_tokens: int = 600,
                 fallback: RuleBasedDiagnoser | None = None):
        self._client = client                 # inject for tests; else lazy
        self.model = model
        self.max_tokens = max_tokens
        self._fallback = fallback or RuleBasedDiagnoser()

    # -- client management -------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic  # imported lazily so the SDK is optional
        except ImportError as e:  # pragma: no cover - depends on env
            raise RuntimeError(
                "anthropic SDK not installed. `pip install anthropic` or "
                "`pip install -e \".[llm]\"`, or use the rule-based brain."
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic()
        return self._client

    # -- the Diagnoser interface ------------------------------------------
    def diagnose(self, report: RoundReport, theta: float,
                 stalled_rounds: int, patience: int) -> ReplanAction:
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM,
                tools=[_REPLAN_TOOL],
                tool_choice={"type": "tool", "name": "propose_replan"},
                messages=[{
                    "role": "user",
                    "content": (
                        "Here are the results of the round I just ran. "
                        "Decide how to steer the next round.\n\n"
                        + _summarize(report, theta, stalled_rounds, patience)
                    ),
                }],
            )
            action = self._parse(resp)
            return clamp_action(action)
        except Exception as e:  # network, auth, SDK, parse — all fall back
            log.warning("LLM diagnose failed (%s); falling back to rules.", e)
            fb = self._fallback.diagnose(report, theta, stalled_rounds, patience)
            fb.diagnosis = f"[LLM unavailable -> rules] {fb.diagnosis}"
            return fb

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def _parse(resp: object) -> ReplanAction:
        """Pull the tool_use block out of an Anthropic response."""
        for block in getattr(resp, "content", []):
            if getattr(block, "type", None) == "tool_use":
                data = block.input
                return ReplanAction(
                    diagnosis="[LLM] " + _clean_field_text(data.get("diagnosis", "")),
                    contact_bias=float(data.get("contact_bias", 0.0) or 0.0),
                    lower_theta_by=float(data.get("lower_theta_by", 0.0) or 0.0),
                    use_scaffold_seed=bool(data.get("use_scaffold_seed", False)),
                    escalate=bool(data.get("escalate", False)),
                )
        raise ValueError("No tool_use block in model response.")


def make_diagnoser(brain: str = "rules", **kwargs):
    """Factory used by the CLI. `brain` is 'rules' or 'llm'."""
    if brain == "llm":
        return LLMDiagnoser(**kwargs)
    return RuleBasedDiagnoser()
