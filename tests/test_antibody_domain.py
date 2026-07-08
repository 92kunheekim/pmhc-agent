"""The antibody domain runs end-to-end on the shared engine.

Proves the generalization goal: a different science (RFantibody), different
metrics (ipTM / self-consistency / humanness), reusing the SAME Engine, loop,
executor, memory, brain, and gate runner — only the DesignDomain differs.
"""
from __future__ import annotations

import pytest

from pmhc_agent import (Engine, AntibodyDomain, AntibodyTarget, AgentConfig,
                        Budget, LocalExecutor)


def _target(fmt="VHH"):
    return AntibodyTarget(antigen="influenza-HA",
                          epitope_hotspots=("A45", "A47", "A49", "A51"), fmt=fmt)


def _engine(seed=7):
    return Engine(domain=AntibodyDomain(seed=seed),
                  config=AgentConfig(seed=seed, budget=Budget(target_library_size=20)))


def test_antibody_campaign_runs_and_produces_library():
    camp = _engine().run(_target())
    assert camp.done and len(camp.library) > 0
    for d in camp.library:                       # antibody metrics populated
        assert "iptm" in d.metrics
        assert "self_consistency_rmsd" in d.metrics
        assert "humanness" in d.metrics


def test_accepted_designs_clear_iptm_and_developability():
    dom = AntibodyDomain(seed=7)
    camp = _engine().run(_target("VHH"))
    for d in camp.library:
        assert d.metrics["iptm"] >= 0.60                 # VHH cutoff
        assert d.metrics["humanness"] >= dom.policy.min_humanness
        assert d.metrics["polyreactivity"] <= dom.policy.max_polyreactivity
        # worst-case cross-reactivity margin cleared
        worst = max(d.specificity.off_target_scores.values())
        assert d.specificity.on_target_score > worst


def test_scfv_uses_stricter_iptm_cutoff():
    dom = AntibodyDomain(seed=7)
    camp = dom.intake(_target("scFv"))
    assert dom.policy.min_iptm == 0.85               # scFv is stricter than VHH
    # n_cdrs reflects VH+VL
    assert _target("scFv").n_cdrs == 6 and _target("VHH").n_cdrs == 3


def test_determinism():
    a = [d.id for d in _engine(42).recommend_library(_engine(42).run(_target()))]
    b = [d.id for d in _engine(42).recommend_library(_engine(42).run(_target()))]
    assert a == b


def test_runs_on_ray_executor():
    ray = pytest.importorskip("ray")
    from pmhc_agent import RayExecutor
    eng = Engine(domain=AntibodyDomain(seed=7),
                 config=AgentConfig(seed=7, budget=Budget(target_library_size=20)),
                 executor=RayExecutor(address=None))
    try:
        camp = eng.run(_target())
    finally:
        ray.shutdown()
    assert camp.done and len(camp.library) > 0


def test_bad_format_and_missing_epitope_rejected():
    with pytest.raises(ValueError):
        AntibodyTarget(antigen="x", epitope_hotspots=("A1",), fmt="Fab")
    with pytest.raises(ValueError):
        AntibodyTarget(antigen="x", epitope_hotspots=(), fmt="VHH")
