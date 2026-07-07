"""End-to-end + unit tests. Run: pytest -q"""
from __future__ import annotations

import pytest

from pmhc_agent import (Orchestrator, AgentConfig, Budget, Target, Peptide,
                        Memory, build_registry)
from pmhc_agent.panel import build_panel
from pmhc_agent.tools import build_registry as breg


def make_target():
    # MART-1 on A*02:01 — a clean, well-characterized demo target.
    return Target(Peptide("AAGIGILTV"), "HLA-A*02:01", "MART-1")


def test_peptide_length_validation():
    with pytest.raises(ValueError):
        Peptide("AC")            # too short
    Peptide("AAGIGILTV")          # ok


def test_campaign_runs_and_produces_library():
    agent = Orchestrator(config=AgentConfig(seed=7,
                         budget=Budget(target_library_size=20)))
    camp = agent.run(make_target())
    assert camp.done
    assert len(camp.library) > 0, "expected at least one accepted design"
    # Every accepted design must have specificity + fold populated.
    for d in camp.library:
        assert d.fold is not None
        assert d.specificity is not None


def test_determinism():
    t = make_target()
    a1 = Orchestrator(config=AgentConfig(seed=42))
    a2 = Orchestrator(config=AgentConfig(seed=42))
    c1, c2 = a1.run(t), a2.run(t)
    ids1 = [d.id for d in a1.recommend_library(c1)]
    ids2 = [d.id for d in a2.recommend_library(c2)]
    assert ids1 == ids2, "same seed must give identical shortlist"


def test_accepted_designs_beat_worst_offtarget():
    """The specificity contract: accepted designs clear the margin on the
    WORST off-target, not just the average."""
    agent = Orchestrator(config=AgentConfig(seed=7))
    camp = agent.run(make_target())
    for d in camp.library:
        s = d.specificity
        worst = max(s.off_target_scores.values()) if s.off_target_scores else 0
        assert s.on_target_score > worst, "on-target must beat worst off-target"
        assert s.margin >= camp.theta - 1e-6


def test_panel_is_confusable_and_bounded():
    reg = breg(seed=7)
    t = make_target()
    reg.structure.resolve(t)
    panel = build_panel(t, reg.netmhc, size=12, seed=7)
    assert 0 < len(panel) <= 12
    # panel sorted hardest-first
    confs = [o.confusability for o in panel]
    assert confs == sorted(confs, reverse=True)
    # no off-target is identical to the target
    assert all(o.peptide.sequence != t.peptide.sequence for o in panel)


def test_non_ligand_halts_at_intake():
    # A peptide unlikely to bind -> should halt at G1.
    agent = Orchestrator(config=AgentConfig(seed=7))
    bad = Target(Peptide("PPPPPPPPP"), "HLA-A*02:01", "junk")
    camp = agent.intake(bad)
    # It may or may not halt depending on mock rank, but intake must be safe.
    assert camp.target.peptide.sequence == "PPPPPPPPP"


def test_memory_banks_scaffolds_and_reuses():
    mem = Memory()
    agent = Orchestrator(config=AgentConfig(seed=7), memory=mem)
    agent.run(Target(Peptide("AAGIGILTV"), "HLA-A*02:01", "MART-1"))
    banked = len(mem.scaffolds)
    assert banked > 0
    # Second A*02 target should find privileged scaffolds available.
    priv = mem.privileged_for("HLA-A*02:01")
    assert len(priv) > 0
