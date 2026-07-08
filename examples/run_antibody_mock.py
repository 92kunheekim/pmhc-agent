"""Mock antibody campaign on the shared engine (offline, no GPU).

    PYTHONPATH=. python examples/run_antibody_mock.py

Same Engine as the pMHC demo — only the domain differs. Shows the antibody
gate funnel (ipTM, self-consistency, cross-reactivity margin, developability)
and the ranked shortlist.
"""
from __future__ import annotations

import logging

from pmhc_agent import Engine, AntibodyDomain, AntibodyTarget, AgentConfig, Budget


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    target = AntibodyTarget(
        antigen="influenza-HA",
        epitope_hotspots=("A45", "A47", "A49", "A51"),
        framework="hu-VHH-3-23", fmt="VHH")

    agent = Engine(domain=AntibodyDomain(seed=7),
                   config=AgentConfig(seed=7, budget=Budget(target_library_size=24)))

    print("=" * 74)
    print(f"  Antibody-Design Agent  ·  {target.fmt} vs {target.antigen} "
          f"(epitope {list(target.epitope_hotspots)})")
    print("=" * 74)

    camp = agent.run(target)
    for note in camp.notes:
        print(f"  • {note}")
    print("-" * 74)
    for rep in camp.rounds:
        print(f"  Round {rep.round_index}: generated {rep.generated}")
        for g in rep.gate_results:
            print(f"      {g.name:<28} pass {g.passed:>4}  reject {g.rejected:>4}")
        print(f"      -> survivors: {rep.survivors}")
        if rep.diagnosis:
            print(f"      diagnosis: {rep.diagnosis}")
    print("-" * 74)
    print(f"  Final: {camp.stage.value} | accepted {len(camp.library)} "
          f"| theta {camp.theta}")
    print("=" * 74)
    print(f"    {'design':<16}{'composite':>10}{'ipTM':>7}{'margin':>8}"
          f"{'sc_rmsd':>9}{'human':>7}{'polyR':>7}")
    for d in agent.recommend_library(camp, top_n=8):
        m = d.metrics
        print(f"    {d.id:<16}{d.composite_score:>10.3f}{m['iptm']:>7.2f}"
              f"{m['margin']:>8.2f}{m['self_consistency_rmsd']:>9.2f}"
              f"{m['humanness']:>7.2f}{m['polyreactivity']:>7.2f}")
    print("\n  Reminder: DNA synthesis, assays, and affinity maturation are HUMAN-GATED.")


if __name__ == "__main__":
    main()
