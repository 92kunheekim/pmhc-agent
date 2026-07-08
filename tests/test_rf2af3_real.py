"""Tests for the REAL RF2 self-consistency + AF3 ipTM backend.

Verified without a GPU: both CLIs, the RF2 scorefile parse, the AF3 summary
parse, and the AF3 input-JSON builder, against fixtures. The two model runs are
injected so predict() runs fully without the tools; only they need a GPU.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

from pmhc_agent.domains.antibody.tools_real import RF2AF3Real
from pmhc_agent.types import Backbone, Design
from pmhc_agent import AntibodyTarget, GatePolicy  # noqa: F401
from pmhc_agent.domains.antibody.policy import AntibodyGatePolicy

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
RF2_SC = os.path.join(FIX, "ab_rf2.sc")
AF3_SUMMARY = os.path.join(FIX, "ab_af3_summary.json")


def _tool(**kw):
    base = dict(rfantibody_dir="/opt/RFantibody", af3_dir="/opt/alphafold3",
                af3_model_dir="/opt/af3/models", af3_db_dir="/opt/af3/db",
                binder_chains=("H",))
    base.update(kw)
    return RF2AF3Real(**base)


def _design(struct):
    bb = Backbone(id="r0_ab0", scaffold_source="de_novo", length=10,
                  peptide_contact_fraction=0.66, coords_ref="")
    d = Design(id="r0_ab0_s0", backbone=bb, sequence="MARTVDKLQH",
               struct_ref=struct)
    d.chains = {"H": "MARTVDKLQH"}
    return d


def _target():
    t = AntibodyTarget(antigen="influenza-HA", epitope_hotspots=("T305",), fmt="VHH")
    t.antigen_seq = "GLMWLSYFV"
    return t


def test_rf2_and_af3_commands():
    t = _tool()
    rf2 = t.build_rf2_command("/in", "/out")
    assert rf2[1].endswith(os.path.join("scripts", "rf2_predict.py"))
    assert "input.pdb_dir=/in" in rf2 and "output.pdb_dir=/out" in rf2
    af3 = t.build_af3_command("/w/x.json", "/w/af3out")
    assert af3[1].endswith("run_alphafold.py")
    assert "--json_path=/w/x.json" in af3
    assert "--model_dir=/opt/af3/models" in af3
    assert "--db_dir=/opt/af3/db" in af3


def test_parse_rf2_and_af3_outputs():
    rf2 = RF2AF3Real.parse_rf2_scorefile(RF2_SC)
    assert abs(rf2["r0_ab0_s0"]["self_consistency_rmsd"] - 0.842) < 1e-9
    assert abs(rf2["r0_ab0_s0"]["plddt"] - 91.20) < 1e-9
    summ = RF2AF3Real.parse_af3_summary(AF3_SUMMARY)
    assert summ["iptm"] == 0.82 and summ["ptm"] == 0.76


def test_af3_input_json_builder(tmp_path):
    t = _tool()
    jp = tmp_path / "in.json"
    t.write_af3_input(_design("x.pdb"), _target(), str(jp))
    doc = json.loads(jp.read_text())
    assert doc["dialect"] == "alphafold3"
    ids = [s["protein"]["id"] for s in doc["sequences"]]
    assert "H" in ids and "T" in ids            # antibody chain + target chain
    hseq = [s["protein"]["sequence"] for s in doc["sequences"]
            if s["protein"]["id"] == "H"][0]
    assert hseq == "MARTVDKLQH"


def test_predict_merges_rf2_and_af3(tmp_path):
    pdb = tmp_path / "r0_ab0_s0.pdb"
    pdb.write_text("ATOM      1  CA  ALA H   1      0.0 0.0 0.0\n")
    out = tmp_path / "out"

    def fake_runner(argv, **kw):
        prog = argv[1]
        if prog.endswith("rf2_predict.py"):
            outdir = [a.split("=", 1)[1] for a in argv
                      if a.startswith("output.pdb_dir=")][0]
            shutil.copy(RF2_SC, os.path.join(outdir, "out.sc"))
        elif prog.endswith("run_alphafold.py"):
            outdir = [a.split("=", 1)[1] for a in argv
                      if a.startswith("--output_dir=")][0]
            d = os.path.join(outdir, "r0_ab0_s0"); os.makedirs(d, exist_ok=True)
            shutil.copy(AF3_SUMMARY,
                        os.path.join(d, "r0_ab0_s0_summary_confidences.json"))

    t = _tool(runner=fake_runner)
    fr = t.predict(_design(str(pdb)), _target())
    assert fr.extra["iptm"] == 0.82                       # from AF3
    assert abs(fr.extra["self_consistency_rmsd"] - 0.842) < 1e-9   # from RF2
    assert fr.plddt == 91.20
    assert not t.is_mock


def test_predicted_metrics_drive_gate_A4():
    """A strong design (ipTM 0.82) passes the VHH ipTM gate; a weak one fails."""
    from pmhc_agent.domains.antibody.domain import AntibodyDomain
    dom = AntibodyDomain(seed=7)
    dom.policy = AntibodyGatePolicy(min_iptm=0.60)
    a4 = [g for g in dom.gates() if g.name.startswith("A4")][0]
    ctx = {"policy": dom.policy, "theta": 1.0}

    strong = _design("x")
    strong.metrics = {"iptm": 0.82, "self_consistency_rmsd": 0.84, "plddt": 91.2}
    weak = _design("y")
    weak.metrics = {"iptm": 0.41, "self_consistency_rmsd": 2.1, "plddt": 74.0}
    assert a4.predicate(strong, ctx) is True
    assert a4.predicate(weak, ctx) is False


def test_backend_is_ray_dispatchable():
    import pickle
    t = _tool()
    rt = pickle.loads(pickle.dumps(t))
    assert rt.build_rf2_command("/i", "/o")[0] == "python"
