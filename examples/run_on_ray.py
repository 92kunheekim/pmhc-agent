"""Run a campaign on a Ray cluster (the KubeRay driver entrypoint).

This is what the RayJob in deploy/rayjob.yaml executes. It attaches to the
RayCluster it runs inside (`address="auto"`), requests GPUs per stage, and
runs the same agent — the heavy fold/specificity/sequence tasks fan out to
the GPU worker pods.

Locally you can smoke-test it with a local Ray head and zero GPUs:

    RAY_ADDRESS=local PMHC_STAGE_GPUS=0 python examples/run_on_ray.py
"""
from __future__ import annotations

import os

from pmhc_agent import (Orchestrator, AgentConfig, Budget, GpuRequest,
                        Target, Peptide, RayExecutor, make_diagnoser)


def main() -> None:
    peptide = os.environ.get("PMHC_PEPTIDE", "AAGIGILTV")
    allele = os.environ.get("PMHC_ALLELE", "HLA-A*02:01")
    antigen = os.environ.get("PMHC_ANTIGEN", "MART-1")
    library = int(os.environ.get("PMHC_LIBRARY", "48"))
    # On a real cluster leave RAY_ADDRESS unset -> "auto" attaches to KubeRay.
    address = os.environ.get("RAY_ADDRESS", "auto")
    if address in ("local", ""):
        address = None
    # GPUs per task per stage. On the cluster: full GPU for the AF-based
    # stages, a fraction for ProteinMPNN so several pack onto one GPU.
    g = float(os.environ.get("PMHC_STAGE_GPUS", "1"))
    gpus = GpuRequest(sequence=min(g, 0.25) if g else 0.0,
                      fold=g, specificity=g)
    brain = os.environ.get("PMHC_BRAIN", "rules")   # 'rules' or 'llm'

    cfg = AgentConfig(seed=7, budget=Budget(target_library_size=library),
                      gpus=gpus)
    agent = Orchestrator(
        config=cfg,
        executor=RayExecutor(address=address),
        diagnoser=make_diagnoser(brain),
    )
    camp = agent.run(Target(Peptide(peptide), allele, antigen))

    print(f"campaign {antigen} {peptide}/{allele}: "
          f"stage={camp.stage.value}, accepted={len(camp.library)}, "
          f"theta={camp.theta}")
    for d in agent.recommend_library(camp, top_n=10):
        s = d.specificity
        print(f"  {d.id:<16} composite={d.composite_score:.3f} "
              f"margin={s.margin:.2f} worst_off={s.worst_offender}")


if __name__ == "__main__":
    main()
