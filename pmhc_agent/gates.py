"""The filter-gate stack (G1-G9 in the design doc).

Each gate is a pure predicate over a Design (plus campaign context). Gates
are ordered cheap-first so the population is thinned by fast checks before
spending "GPU-minutes" on the contrastive panel. `run_gate` returns the
survivors plus a GateResult tally for the round report.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable

from .types import Design, Campaign, GateResult
from .config import GatePolicy


def run_gate(name: str, designs: list[Design],
             predicate: Callable[[Design], bool],
             reason_hint: str = "") -> tuple[list[Design], GateResult]:
    kept = [d for d in designs if predicate(d)]
    result = GateResult(name=name, passed=len(kept),
                        rejected=len(designs) - len(kept),
                        reason_hint=reason_hint)
    return kept, result


# --- Individual gate predicates -------------------------------------------
def g2_peptide_centric(d: Design, p: GatePolicy) -> bool:
    return d.backbone.peptide_contact_fraction >= p.min_peptide_contact_fraction


def g3_foldable(d: Design, p: GatePolicy) -> bool:
    # ddG is optional: the real ProteinMPNN backend does not produce a binding
    # ddG (that's a separate Rosetta/FastRelax task), so skip the ddG sub-check
    # when it's unavailable. The mock always sets it, so the mock path is
    # unaffected.
    ddg_ok = (d.rosetta_ddg is None) or (d.rosetta_ddg <= p.max_rosetta_ddg)
    return d.mpnn_score <= p.max_mpnn_score and ddg_ok


def g4_fold_dock(d: Design, p: GatePolicy) -> bool:
    f = d.fold
    return (f is not None and f.pae_interface <= p.max_pae_interface
            and f.plddt >= p.min_plddt and f.ca_rmsd_to_design <= p.max_ca_rmsd)


def g5_recovery(d: Design) -> bool:
    return d.specificity is not None and d.specificity.mpnn_recovery_ok


def g6_margin(d: Design, theta: float) -> bool:
    return d.specificity is not None and d.specificity.margin >= theta


def g7_partition(d: Design, p: GatePolicy) -> bool:
    return (d.specificity is not None
            and d.specificity.peptide_energy_fraction
            >= p.min_peptide_energy_fraction)


def g8_consensus(d: Design) -> bool:
    return d.fold is not None and d.fold.predictors_agree


def annotate_liabilities(d: Design, p: GatePolicy) -> Design:
    """Flag developability liabilities (free Cys, N-glyc sequons)."""
    liabilities = []
    if "C" in d.sequence:
        liabilities.append("free_cys")
    if _has_nglyc(d.sequence):
        liabilities.append("n_glyc_sequon")
    d.liabilities = liabilities
    return d


def _has_nglyc(seq: str) -> bool:
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 2] in ("S", "T") and seq[i + 1] != "P":
            return True
    return False


def g9_developable(d: Design, p: GatePolicy) -> bool:
    annotate_liabilities(d, p)
    if p.forbid_free_cys and "free_cys" in d.liabilities:
        return False
    return True


def enforce_diversity(designs: list[Design], p: GatePolicy) -> list[Design]:
    """Cap how many sequences from one backbone reach the library, best-margin
    first, so no single scaffold dominates.

    NOTE: in production, key this on a *structural cluster* (e.g. Foldseek /
    TM-score bins over the real backbones) rather than the backbone id, so
    near-identical scaffolds are collapsed. The id-based cap here is the mock
    stand-in for that clustering step.
    """
    ranked = sorted(
        designs,
        key=lambda d: (d.specificity.margin if d.specificity else -1e9),
        reverse=True,
    )
    per_scaffold: Counter = Counter()
    out = []
    for d in ranked:
        sc = d.backbone.id
        if per_scaffold[sc] < p.max_designs_per_scaffold:
            per_scaffold[sc] += 1
            out.append(d)
    return out
