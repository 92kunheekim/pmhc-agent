"""CLI demo entrypoint.

    python -m pmhc_agent.run                          # default MAGE-A3 demo
    python -m pmhc_agent.run --peptide EVDPIGHLY --allele HLA-A*01:01 \
        --antigen MAGE-A3 --library 24

Runs a full in-silico campaign with mock backends and prints the funnel,
the adaptive threshold, the failure diagnoses per round, and the ranked
shortlist. No GPU or API key required.
"""
from __future__ import annotations

import argparse
import logging

from .types import Target, Peptide
from .config import AgentConfig, Budget, GatePolicy
from .orchestrator import Orchestrator
from .llm import make_diagnoser


def _fmt_gate_row(g) -> str:
    hint = f"  ({g.reason_hint})" if g.reason_hint else ""
    return f"      {g.name:<26} pass {g.passed:>4}  reject {g.rejected:>5}{hint}"


def run_demo(peptide: str, allele: str, antigen: str,
             library: int, seed: int, verbose: bool,
             brain: str = "rules", model: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(message)s")

    cfg = AgentConfig(
        seed=seed,
        budget=Budget(target_library_size=library),
        gates=GatePolicy(),
    )
    kwargs = {"model": model} if (brain == "llm" and model) else {}
    diagnoser = make_diagnoser(brain, **kwargs)
    agent = Orchestrator(config=cfg, diagnoser=diagnoser)
    target = Target(peptide=Peptide(peptide), hla_allele=allele,
                    source_antigen=antigen)

    print("=" * 74)
    print(f"  pMHC-Design Agent  ·  campaign: {antigen} {peptide} / {allele}")
    print(f"  backends: {'ALL MOCK' if agent.reg.all_mock else 'MIXED'}"
          f"   seed={seed}   brain: {diagnoser.name}")
    print("=" * 74)

    camp = agent.run(target)

    for note in camp.notes:
        print(f"  • {note}")
    print("-" * 74)

    for rep in camp.rounds:
        print(f"  Round {rep.round_index}:  generated {rep.generated} designs")
        for g in rep.gate_results:
            print(_fmt_gate_row(g))
        print(f"      -> survivors this round: {rep.survivors}"
              f"   |  library so far: "
              f"{sum(r.survivors for r in camp.rounds[:rep.round_index+1])}")
        if rep.diagnosis:
            print(f"      diagnosis: {rep.diagnosis}")
        print()

    print("-" * 74)
    print(f"  Final stage: {camp.stage.value}   "
          f"| accepted designs: {len(camp.library)}   "
          f"| adaptive theta: {camp.theta}")
    print("=" * 74)

    shortlist = agent.recommend_library(camp, top_n=10)
    if not shortlist:
        print("  No designs cleared the gates — see diagnosis above.")
    else:
        print("  RANKED SHORTLIST (human-approved synthesis required):\n")
        print(f"    {'design':<16}{'composite':>10}{'margin':>9}"
              f"{'pAE_i':>8}{'pLDDT':>8}{'phi':>7}  worst_off")
        for d in shortlist:
            s = d.specificity
            print(f"    {d.id:<16}{d.composite_score:>10.4f}{s.margin:>9.2f}"
                  f"{d.fold.pae_interface:>8.1f}{d.fold.plddt:>8.1f}"
                  f"{s.peptide_energy_fraction:>7.2f}  {s.worst_offender}")
    print()
    print("  Reminder: DNA ordering, assays, and spend are HUMAN-GATED.")
    print("  The agent recommends; a scientist commits.")


def main() -> None:
    ap = argparse.ArgumentParser(description="pMHC-Design Agent demo")
    ap.add_argument("--peptide", default="AAGIGILTV",
                    help="target peptide sequence (8-11mer)")
    ap.add_argument("--allele", default="HLA-A*02:01", help="HLA-I allele")
    ap.add_argument("--antigen", default="MART-1", help="source antigen name")
    ap.add_argument("--library", type=int, default=24,
                    help="target accepted-library size (stop criterion)")
    ap.add_argument("--seed", type=int, default=7, help="determinism seed")
    ap.add_argument("--brain", choices=["rules", "llm"], default="rules",
                    help="decision engine: deterministic rules (default) or "
                         "an Anthropic LLM (needs ANTHROPIC_API_KEY; falls "
                         "back to rules if unavailable)")
    ap.add_argument("--model", default=None,
                    help="Anthropic model id for --brain llm "
                         "(default: env PMHC_LLM_MODEL or a sane default)")
    ap.add_argument("--quiet", action="store_true", help="less logging")
    args = ap.parse_args()
    run_demo(args.peptide, args.allele, args.antigen, args.library,
             args.seed, verbose=not args.quiet,
             brain=args.brain, model=args.model)


if __name__ == "__main__":
    main()
