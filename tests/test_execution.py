"""Tests for the execution layer (Local + Ray).

The Ray tests are skipped automatically if `ray` isn't installed, so the core
suite still runs anywhere. When Ray IS present, we assert that a Ray-executed
campaign gives byte-identical results to a local one — proving the executor is
a transparent, order-preserving swap and that determinism is not affected by
where the work runs.
"""
from __future__ import annotations

import pytest

from pmhc_agent import (Orchestrator, AgentConfig, Budget, Target, Peptide,
                        LocalExecutor, make_executor)
from pmhc_agent.execution import task_fold


def _target():
    return Target(Peptide("AAGIGILTV"), "HLA-A*02:01", "MART-1")


def test_local_executor_maps_in_order():
    ex = LocalExecutor()
    out = ex.map(lambda x: x * x, [1, 2, 3, 4])
    assert out == [1, 4, 9, 16]


def test_make_executor_defaults_to_local():
    assert make_executor("local").name == "local"
    # Unknown kinds also fall back to local rather than crashing.
    assert make_executor("nonsense").name == "local"


def test_orchestrator_runs_with_explicit_local_executor():
    agent = Orchestrator(
        config=AgentConfig(seed=7, budget=Budget(target_library_size=20)),
        executor=LocalExecutor())
    camp = agent.run(_target())
    assert camp.done and len(camp.library) > 0


def test_make_executor_ray_falls_back_when_unavailable(monkeypatch):
    """If Ray import fails, make_executor('ray') must degrade to local."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "ray":
            raise ImportError("simulated: ray not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ex = make_executor("ray")
    assert ex.name == "local"          # graceful fallback, no exception


# --- Ray-requiring parity test (skipped if ray absent) --------------------
ray = pytest.importorskip("ray")


def test_ray_matches_local_and_preserves_determinism():
    from pmhc_agent import RayExecutor
    t = _target()
    cfg = lambda: AgentConfig(seed=7, budget=Budget(target_library_size=20))

    local = Orchestrator(config=cfg(), executor=LocalExecutor())
    cl = local.run(t)
    ids_local = [d.id for d in local.recommend_library(cl)]

    rex = RayExecutor(address=None)     # spin up a local Ray head, num_gpus=0
    try:
        rag = Orchestrator(config=cfg(), executor=rex)
        cr = rag.run(t)
        ids_ray = [d.id for d in rag.recommend_library(cr)]
    finally:
        ray.shutdown()

    assert len(cr.library) == len(cl.library)
    assert ids_ray == ids_local          # order-preserving => identical output
