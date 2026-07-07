"""Tests for the LLM-backed diagnoser.

These use an INJECTED FAKE client (no `anthropic` install, no API key, no
network), so they run in CI exactly like the rest of the suite. They verify:
  * the model's structured answer is parsed into a ReplanAction,
  * out-of-range values are clamped (the guardrail),
  * a broken/absent client falls back to the deterministic rules,
  * the orchestrator runs end-to-end with an LLM brain.
"""
from __future__ import annotations

from pmhc_agent import (Orchestrator, AgentConfig, Budget, Target, Peptide,
                        LLMDiagnoser)
from pmhc_agent.types import RoundReport, GateResult


# --- a minimal fake of the Anthropic client -------------------------------
class _Block:
    def __init__(self, data):
        self.type = "tool_use"
        self.name = "propose_replan"
        self.input = data


class _Resp:
    def __init__(self, data):
        self.content = [_Block(data)]


class _Messages:
    def __init__(self, data, sink):
        self._data, self._sink = data, sink

    def create(self, **kwargs):
        self._sink.append(kwargs)      # record the call for assertions
        return _Resp(self._data)


class FakeClient:
    """Mimics anthropic.Anthropic(): exposes `.messages.create(**kwargs)`."""
    def __init__(self, data):
        self.calls: list = []
        self.messages = _Messages(data, self.calls)


class BrokenClient:
    class messages:  # noqa
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("simulated API failure")


def _report():
    r = RoundReport(round_index=0, generated=480, survivors=2)
    r.gate_results = [
        GateResult("G2 peptide-centric", passed=100, rejected=380),
        GateResult("G4 fold&dock", passed=40, rejected=60),
        GateResult("G6 specificity margin", passed=2, rejected=38,
                   reason_hint="theta=1.40"),
    ]
    return r


def test_llm_answer_is_parsed_into_replan_action():
    fake = FakeClient({
        "diagnosis": "Backbones look MHC-leaning; nudge toward the peptide.",
        "contact_bias": 0.08, "lower_theta_by": 0.0,
        "use_scaffold_seed": True, "escalate": False,
    })
    d = LLMDiagnoser(client=fake, model="fake-model")
    action = d.diagnose(_report(), theta=1.4, stalled_rounds=0, patience=2)
    assert action.use_scaffold_seed is True
    assert action.escalate is False
    assert abs(action.contact_bias - 0.08) < 1e-9
    assert "MHC-leaning" in action.diagnosis
    # It actually called the model with tool_choice forcing our tool.
    assert fake.calls and fake.calls[0]["tool_choice"]["name"] == "propose_replan"


def test_out_of_range_values_are_clamped():
    fake = FakeClient({
        "diagnosis": "over-aggressive proposal",
        "contact_bias": 5.0,        # -> clamp to 0.20
        "lower_theta_by": -3.0,     # -> clamp to 0.0
        "use_scaffold_seed": 1, "escalate": 0,
    })
    d = LLMDiagnoser(client=fake, model="fake-model")
    action = d.diagnose(_report(), theta=1.4, stalled_rounds=0, patience=2)
    assert action.contact_bias == 0.20
    assert action.lower_theta_by == 0.0
    assert action.use_scaffold_seed is True
    assert action.escalate is False


def test_broken_client_falls_back_to_rules():
    d = LLMDiagnoser(client=BrokenClient(), model="fake-model")
    action = d.diagnose(_report(), theta=1.4, stalled_rounds=0, patience=2)
    # Fallback annotates the diagnosis and still returns a usable action.
    assert "LLM unavailable -> rules" in action.diagnosis
    assert 0.0 <= action.contact_bias <= 0.20


def test_orchestrator_runs_with_llm_brain():
    fake = FakeClient({
        "diagnosis": "resample with scaffold seeding",
        "contact_bias": 0.05, "lower_theta_by": 0.0,
        "use_scaffold_seed": True, "escalate": False,
    })
    agent = Orchestrator(
        config=AgentConfig(seed=7, budget=Budget(target_library_size=20)),
        diagnoser=LLMDiagnoser(client=fake, model="fake-model"),
    )
    camp = agent.run(Target(Peptide("AAGIGILTV"), "HLA-A*02:01", "MART-1"))
    assert camp.done
    assert len(camp.library) > 0
    assert agent.diagnoser.name == "llm"
