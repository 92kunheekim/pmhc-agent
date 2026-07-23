"""FastAPI service for the pMHC-Design Agent — a live, synchronous demo.

`POST /campaign` runs a full mock design campaign (generate -> filter ->
diagnose -> re-design) and returns the per-round funnel and the ranked,
synthesis-ready shortlist in one response. Backends are mock, so it needs no
GPU and no API key; the campaign finishes in ~1 s.

Run locally:
    uvicorn pmhc_agent.service.api:app --host 0.0.0.0 --port 8000

The decision engine (the "brain") is the free, deterministic rule-based
diagnoser by default, so public traffic never spends API credits; the
Anthropic-backed brain remains available in the library (`--brain llm`).
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pmhc_agent import Target, Peptide, AgentConfig, Budget, Orchestrator
from pmhc_agent.config import GatePolicy
from pmhc_agent.llm import make_diagnoser

app = FastAPI(
    title="pMHC-Design Agent",
    version="0.2.0",
    description=(
        "Autonomous high-specificity pMHC-I binder design — a live mock demo "
        "of the generate -> filter -> diagnose -> re-design loop. POST /campaign "
        "with a peptide + HLA allele to watch the agent run and return a ranked "
        "shortlist. In-silico hypotheses only; all wet-lab actions are human-gated."
    ),
)

# Public endpoint runs mock backends only; cap the target library so a single
# request can't spin the loop for an unbounded number of rounds.
_MAX_LIBRARY = 200
_AA = set("ACDEFGHIKLMNPQRSTVWY")


class CampaignRequest(BaseModel):
    peptide: str = Field("AAGIGILTV", description="target peptide (8-11mer)")
    allele: str = Field("HLA-A*02:01", description="HLA-I allele")
    antigen: str = Field("MART-1", description="source antigen name (label only)")
    library: int = Field(24, ge=1, le=_MAX_LIBRARY,
                         description="target accepted-library size (stop criterion)")
    seed: int = Field(7, description="determinism seed — same seed => same run")


def _design_json(d) -> dict:
    s = d.specificity
    return {
        "id": d.id,
        "composite_score": round(d.composite_score, 4),
        "specificity_margin": round(s.margin, 2),
        "pae_interface": round(d.fold.pae_interface, 1),
        "plddt": round(d.fold.plddt, 1),
        "peptide_energy_fraction": round(s.peptide_energy_fraction, 2),
        "worst_off_target": s.worst_offender,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "service": "pMHC-Design Agent",
        "what": "Live mock demo of an autonomous pMHC-I binder-design agent.",
        "try": {
            "method": "POST",
            "path": "/campaign",
            "example_body": {"peptide": "AAGIGILTV", "allele": "HLA-A*02:01",
                             "antigen": "MART-1", "library": 24, "seed": 7},
        },
        "docs": "/docs",
        "note": "Backends are mock (no GPU); in-silico hypotheses only, human-gated.",
    }


@app.post("/campaign")
def campaign(req: CampaignRequest):
    pep = req.peptide.strip().upper()
    if not (8 <= len(pep) <= 11) or any(c not in _AA for c in pep):
        raise HTTPException(
            status_code=422,
            detail="peptide must be an 8-11mer of standard amino acids",
        )

    cfg = AgentConfig(seed=req.seed,
                      budget=Budget(target_library_size=req.library),
                      gates=GatePolicy())
    agent = Orchestrator(config=cfg, diagnoser=make_diagnoser("rules"))
    target = Target(peptide=Peptide(pep), hla_allele=req.allele,
                    source_antigen=req.antigen)

    camp = agent.run(target)

    rounds = [{
        "round": rep.round_index,
        "generated": rep.generated,
        "gates": [{"name": g.name, "passed": g.passed, "rejected": g.rejected,
                   "hint": g.reason_hint or None} for g in rep.gate_results],
        "survivors": rep.survivors,
        "diagnosis": rep.diagnosis or None,
    } for rep in camp.rounds]

    shortlist = [_design_json(d) for d in agent.recommend_library(camp, top_n=10)]

    return {
        "target": {"antigen": req.antigen, "peptide": pep, "allele": req.allele},
        "backends": "all_mock" if agent.reg.all_mock else "mixed",
        "brain": "rules",
        "seed": req.seed,
        "rounds": rounds,
        "accepted_designs": len(camp.library),
        "adaptive_theta": round(camp.theta, 3),
        "shortlist": shortlist,
        "safety": ("In-silico hypotheses only. DNA ordering, assays, and spend are "
                   "human-gated — the agent recommends, a scientist commits."),
    }
