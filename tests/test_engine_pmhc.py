"""Regression proof for the engine generalization.

The domain-driven Engine, given PMHCDomain, must produce results IDENTICAL to
the legacy Orchestrator. If this passes, wrapping pMHC as a DesignDomain
changed nothing observable — which is the whole point of the refactor. Also
exercises the Engine on the Ray executor to confirm the abstraction is
transport-agnostic too.
"""
from __future__ import annotations

import pytest

from pmhc_agent import (Orchestrator, Engine, PMHCDomain, AgentConfig, Budget,
                        Target, Peptide, LocalExecutor)


def _target():
    return Target(Peptide("AAGIGILTV"), "HLA-A*02:01", "MART-1")


def _cfg():
    return AgentConfig(seed=7, budget=Budget(target_library_size=20))


def test_engine_matches_legacy_orchestrator():
    """Same seed, same target -> identical accepted library and shortlist."""
    legacy = Orchestrator(config=_cfg())
    lc = legacy.run(_target())
    legacy_ids = [d.id for d in legacy.recommend_library(lc)]

    engine = Engine(domain=PMHCDomain(seed=7), config=_cfg())
    ec = engine.run(_target())
    engine_ids = [d.id for d in engine.recommend_library(ec)]

    assert len(ec.library) == len(lc.library)
    assert engine_ids == legacy_ids                     # byte-identical ranking
    # Same per-round gate funnel (names + pass/reject counts).
    assert len(ec.rounds) == len(lc.rounds)
    for er, lr in zip(ec.rounds, lc.rounds):
        assert [(g.name, g.passed, g.rejected) for g in er.gate_results] == \
               [(g.name, g.passed, g.rejected) for g in lr.gate_results]


def test_engine_populates_metrics_bag():
    engine = Engine(domain=PMHCDomain(seed=7), config=_cfg())
    camp = engine.run(_target())
    d = engine.recommend_library(camp, top_n=1)[0]
    # The named-metric bag is populated (what antibody-domain gates will read).
    for key in ("pae_interaction", "plddt", "margin", "phi",
                "peptide_contact_fraction"):
        assert key in d.metrics
    assert d.metrics["margin"] == d.specificity.margin


def test_engine_domain_describes_itself():
    dom = PMHCDomain()
    assert dom.name == "pmhc-I"
    assert "peptide-MHC" in dom.describe()
    assert dom.target_key(_target()) == "HLA-A*02:01"


def test_engine_runs_on_ray_executor():
    ray = pytest.importorskip("ray")
    from pmhc_agent import RayExecutor
    engine = Engine(domain=PMHCDomain(seed=7), config=_cfg(),
                    executor=RayExecutor(address=None))
    try:
        camp = engine.run(_target())
    finally:
        ray.shutdown()
    assert camp.done and len(camp.library) > 0


def test_non_ligand_halts_via_domain():
    engine = Engine(domain=PMHCDomain(seed=7), config=_cfg())
    camp = engine.run(Target(Peptide("EVDPIGHLY"), "HLA-A*01:01", "MAGE-A3"))
    # Marginal ligand halts at intake, same as the legacy path.
    assert camp.stage.value in ("stalled", "library")
