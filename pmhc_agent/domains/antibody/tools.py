"""Mock antibody backends (offline, deterministic) — the RFantibody pipeline.

Mirror the pMHC mocks so a full antibody campaign runs end-to-end in seconds
with no GPU. Each has the same method the engine calls, and the real backends
(RFantibody, RF2/AF3-ipTM) drop in later behind these exact signatures.

Antibody-specific metrics (ipTM, self-consistency, humanness, polyreactivity)
flow through the generalized metric bag: fold results carry them in
`FoldResult.extra`, sequence design writes developability metrics onto the
Design, and the engine merges everything into `Design.metrics` for the gates.
"""
from __future__ import annotations

from dataclasses import field

from ...tools.base import det_rng
from ...types import Backbone, Design, FoldResult, SpecificityResult
from ...tools import ToolRegistry

_AA = "ADEFGHIKLMNPQRSTVWY"        # cysteine excluded, as in ProteinMPNN runs


class AntibodyStructureMock:
    name = "antigen structure resolver (mock)"
    is_mock = True

    def __init__(self, seed=0):
        self.seed = seed

    def resolve(self, target):
        key = f"{target.antigen}|{','.join(target.epitope_hotspots)}"
        rng = det_rng("ab_struct", key, seed=self.seed)
        target.antigen_structure_id = f"AF3:{abs(hash(key)) % 10**6:06d}"
        target.structure_confidence = round(min(0.99, 0.88 + rng.uniform(-0.1, 0.1)), 3)
        return target


class RFantibodyMock:
    """Fine-tuned RFdiffusion for antibodies: CDR loops + docking onto the
    epitope. `peptide_contact_fraction` is reused as the epitope-contact
    fraction (fraction of the paratope's contacts on the epitope)."""
    name = "RFantibody (mock)"
    is_mock = True

    def __init__(self, seed=0):
        self.seed = seed

    def generate(self, target, n, round_index, seed_scaffold=None, contact_bias=0.0):
        out = []
        for i in range(n):
            tag = f"r{round_index}_ab{i}"
            rng = det_rng("rfab", target.antigen, tag, seed=self.seed)
            if seed_scaffold:
                source, base = f"partial_diffusion:{seed_scaffold}", 0.55
            else:
                source, base = "de_novo", 0.42
            ecf = min(0.95, base + contact_bias + rng.uniform(-0.18, 0.22))
            length = rng.randint(110, 130) if target.fmt != "VHH" else rng.randint(110, 125)
            out.append(Backbone(id=tag, scaffold_source=source, length=length,
                                peptide_contact_fraction=round(ecf, 3)))
        return out


class ProteinMPNNCDRMock:
    """Designs CDR sequences on the fixed framework; also emits developability
    metrics (humanness, polyreactivity) onto the Design."""
    name = "ProteinMPNN-CDR (mock)"
    is_mock = True

    def __init__(self, seed=0):
        self.seed = seed

    def design(self, backbone, n, round_index):
        out = []
        for j in range(n):
            rng = det_rng("abmpnn", backbone.id, j, seed=self.seed)
            q = backbone.peptide_contact_fraction
            seq = "".join(rng.choice(_AA) for _ in range(backbone.length))
            mpnn = round(1.25 - 0.4 * q + rng.uniform(-0.12, 0.12), 3)
            d = Design(id=f"{backbone.id}_s{j}", backbone=backbone, sequence=seq,
                       mpnn_score=mpnn, rosetta_ddg=None)
            # Developability, correlated with packing quality:
            d.metrics["humanness"] = round(min(0.99, 0.80 + 0.15 * q +
                                               rng.uniform(-0.08, 0.08)), 3)
            d.metrics["polyreactivity"] = round(max(0.0, 0.35 - 0.25 * q +
                                                    rng.uniform(-0.05, 0.15)), 3)
            out.append(d)
        return out


class RF2AF3Mock:
    """Fine-tuned RoseTTAFold2 self-consistency + AlphaFold3 ipTM (the paper's
    filter). Emits ipTM and self-consistency RMSD in FoldResult.extra."""
    name = "RF2 self-consistency + AF3 ipTM (mock)"
    is_mock = True

    def __init__(self, seed=0):
        self.seed = seed

    def predict(self, design, target):
        rng = det_rng("rf2af3", design.id, target.antigen, seed=self.seed)
        q = design.backbone.peptide_contact_fraction
        pack = max(0.0, min(1.0, (1.25 - design.mpnn_score) / 0.5))
        signal = 0.55 * q + 0.45 * pack
        iptm = round(min(0.98, 0.42 + 0.5 * signal + rng.uniform(-0.05, 0.05)), 3)
        sc_rmsd = round(max(0.3, 2.6 - 2.2 * signal + rng.uniform(-0.3, 0.3)), 2)
        plddt = round(min(97.0, 60.0 + 38.0 * signal + rng.uniform(-3, 3)), 1)
        return FoldResult(
            pae_interface=round(max(3.0, 14 - 11 * signal), 2),   # unused by AB gates
            plddt=plddt, ca_rmsd_to_design=sc_rmsd, predictors_agree=True,
            extra={"iptm": iptm, "self_consistency_rmsd": sc_rmsd})


class CrossReactivityMock:
    """Antibody 'specificity': on-target ipTM vs ipTM to cross-reactive
    antigens. Worst-case margin, same contract as the pMHC specificity engine."""
    name = "cross-reactivity (mock)"
    is_mock = True

    def __init__(self, seed=0):
        self.seed = seed

    def score(self, design, target, panel):
        on = design.fold.extra.get("iptm", 0.0) if design.fold else 0.0
        offs = {}
        q = design.backbone.peptide_contact_fraction
        for ct in panel:
            rng = det_rng("abxr", design.id, ct.antigen, seed=self.seed)
            # Cross-reactivity grows with epitope similarity, shrinks with a
            # tightly epitope-focused paratope.
            leak = ct.similarity * (1.2 - q)
            offs[ct.antigen] = round(min(0.98, max(0.02,
                                    on * leak + rng.uniform(-0.03, 0.03))), 3)
        worst = max(offs.values()) if offs else 0.0
        margin = round((on - worst) * 5.0, 3)
        return SpecificityResult(
            on_target_score=on, off_target_scores=offs, margin=margin,
            peptide_energy_fraction=round(min(0.95, q + 0.05), 3),
            mpnn_recovery_ok=on > worst + 0.05,
            extra={"cross_reactivity_margin": margin})


class _NetStub:
    """Unused-by-antibody registry slot (the engine never calls it)."""
    name = "n/a"
    is_mock = True


def build_antibody_registry(seed: int = 7) -> ToolRegistry:
    return ToolRegistry(
        netmhc=_NetStub(),
        structure=AntibodyStructureMock(seed),
        backbones=RFantibodyMock(seed),
        sequences=ProteinMPNNCDRMock(seed),
        folding=RF2AF3Mock(seed),
        specificity=CrossReactivityMock(seed),
    )
