"""Run several targets in sequence, sharing one Memory so privileged
scaffolds banked on early targets accelerate later ones (partial diffusion
reuse). Demonstrates the memory / active-learning benefit.

    python examples/multi_target.py
"""
from __future__ import annotations

from pmhc_agent import Orchestrator, AgentConfig, Memory, Target, Peptide

TARGETS = [
    ("MART-1", "AAGIGILTV", "HLA-A*02:01"),
    ("gp100", "IMDQVPFSV", "HLA-A*02:01"),
    ("NY-ESO-1", "SLLMWITQC", "HLA-A*02:01"),
    ("Flu-M1", "GILGFVFTL", "HLA-A*02:01"),
]


def main() -> None:
    mem = Memory()
    agent = Orchestrator(config=AgentConfig(seed=11), memory=mem)
    print(f"{'target':<12}{'allele':<14}{'accepted':>9}{'theta':>8}"
          f"{'scaffolds_in_mem':>18}")
    for antigen, pep, allele in TARGETS:
        camp = agent.run(Target(Peptide(pep), allele, antigen))
        print(f"{antigen:<12}{allele:<14}{len(camp.library):>9}"
              f"{camp.theta:>8.2f}{len(mem.scaffolds):>18}")
    print(f"\nTotal privileged scaffolds banked: {len(mem.scaffolds)}")
    print("Later A*02 targets reuse scaffolds banked by earlier ones.")


if __name__ == "__main__":
    main()
