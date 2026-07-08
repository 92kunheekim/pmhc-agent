"""AntibodyDomain — de novo antibody design (RFantibody) as a DesignDomain.

Runs on the same Engine as pMHC. The gates here read the generalized
`Design.metrics` bag by name (iptm, self_consistency_rmsd, humanness, ...) —
this is exactly the decoupling the metric bag was introduced for: a different
science, different metrics, same engine.
"""
from __future__ import annotations

from ...types import Campaign, Stage, Design
from ...interfaces import Gate
from .target import AntibodyTarget, CrossTarget
from .policy import AntibodyGatePolicy, IPTM_BY_FORMAT
from .tools import build_antibody_registry

# A tiny stand-in set of cross-reactive antigens for the mock negative set.
_HOMOLOGS = [
    ("influenza-HA-H3", 0.62), ("influenza-HA-H7", 0.41),
    ("self-cardiac-myosin", 0.28), ("polyreactivity-probe-dsDNA", 0.20),
    ("RSV-F", 0.15),
]


class AntibodyDomain:
    name = "antibody"

    def __init__(self, seed: int = 7, policy: AntibodyGatePolicy | None = None,
                 panel_size: int = 5, registry=None):
        self.seed = seed
        self.policy = policy or AntibodyGatePolicy()
        self.panel_size = panel_size
        self.reg = registry or build_antibody_registry(seed)

    # -- intake -------------------------------------------------------------
    def intake(self, target: AntibodyTarget) -> Campaign:
        camp = Campaign(target=target)
        # ipTM cutoff depends on the format (VHH 0.60 / scFv 0.85).
        self.policy.min_iptm = IPTM_BY_FORMAT.get(target.fmt, 0.85)
        self.reg.structure.resolve(target)
        camp.stage = Stage.PANEL
        camp.notes.append(
            f"Target ready: {target.fmt} vs {target.antigen} "
            f"(epitope {list(target.epitope_hotspots)}, {target.n_cdrs} CDRs, "
            f"struct {target.antigen_structure_id}); ipTM gate >= "
            f"{self.policy.min_iptm}.")
        return camp

    # -- negative set: cross-reactive antigens -----------------------------
    def build_negative_set(self, camp: Campaign) -> None:
        camp.panel = [CrossTarget(antigen=a, similarity=s)
                      for a, s in _HOMOLOGS[:self.panel_size]]
        camp.notes.append(
            f"Cross-reactivity panel: {len(camp.panel)} antigens "
            f"(hardest: {[c.antigen for c in camp.panel[:2]]}).")
        camp.stage = Stage.BACKBONE

    def tools(self):
        return self.reg

    # -- gates read the METRIC BAG (the generalization payoff) -------------
    def gates(self) -> list[Gate]:
        return [
            Gate("A2 epitope-focused", "after_sequence",
                 lambda d, c: d.metrics.get("peptide_contact_fraction", 0.0)
                 >= c["policy"].min_epitope_contact_fraction),
            Gate("A3 CDR packing", "after_sequence",
                 lambda d, c: d.metrics.get("mpnn_score", 9.9)
                 <= c["policy"].max_mpnn_score),
            Gate("A4 ipTM + self-consistency", "after_fold",
                 lambda d, c: d.metrics.get("iptm", 0.0) >= c["policy"].min_iptm
                 and d.metrics.get("self_consistency_rmsd", 9.9)
                 <= c["policy"].max_self_consistency_rmsd
                 and d.metrics.get("plddt", 0.0) >= c["policy"].min_plddt),
            Gate("A5 cross-reactivity margin", "after_score",
                 lambda d, c: d.metrics.get("margin", -9.9) >= c["theta"]),
            Gate("A6 developability", "after_score",
                 lambda d, c: d.metrics.get("humanness", 0.0)
                 >= c["policy"].min_humanness
                 and d.metrics.get("polyreactivity", 9.9)
                 <= c["policy"].max_polyreactivity),
        ]

    def gate_ctx(self, camp: Campaign) -> dict:
        return {"policy": self.policy, "theta": camp.theta}

    def adapt_thresholds(self, camp: Campaign, designs) -> None:
        margins = sorted(d.specificity.margin for d in designs
                         if d.specificity and d.specificity.margin > 0)
        if not margins:
            return
        idx = int(0.6 * (len(margins) - 1))
        camp.theta = round(max(0.8, 0.5 * camp.theta + 0.5 * margins[idx]), 3)

    def rank_score(self, d: Design) -> float:
        iptm = d.metrics.get("iptm", 0.0)
        margin = max(0.0, d.metrics.get("margin", 0.0))
        dev = 1.0
        if (d.metrics.get("humanness", 0) < self.policy.min_humanness
                or d.metrics.get("polyreactivity", 1) > self.policy.max_polyreactivity):
            dev = 0.6
        return round(iptm * margin * dev, 4)

    def target_key(self, target: AntibodyTarget) -> str:
        return target.antigen

    def gpus_per_stage(self) -> dict:
        return {}

    def describe(self) -> str:
        return (
            "Domain: de novo antibody design (RFantibody). RFdiffusion designs "
            "CDR loops + docking onto an epitope, ProteinMPNN designs the CDRs, "
            "RoseTTAFold2 self-consistency + AlphaFold3 ipTM filter (ipTM>=0.6 "
            "VHH / 0.85 scFv), cross-reactivity scored vs homologous antigens. "
            "Gates: A2 epitope-focused paratope, A3 CDR packing, A4 ipTM + "
            "self-consistency, A5 cross-reactivity margin>=theta, A6 "
            "developability (humanness, polyreactivity).")
