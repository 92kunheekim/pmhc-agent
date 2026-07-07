"""Tests for the REAL AF2 initial-guess fold/dock backend.

AF2 can't run here (GPU + weights + real PDB), so we test the deterministic
parts that break integrations: the exact CLI and the scorefile parser +
metric mapping, against a captured real-format `.sc` fixture. The subprocess
is injected so the full predict() path runs with no tool installed.
"""
from __future__ import annotations

import os
import shutil

import pytest

from pmhc_agent.tools.alphafold_real import AF2InitialGuess
from pmhc_agent.types import Backbone, Design, Target, Peptide
from pmhc_agent import GatePolicy
from pmhc_agent.gates import g4_fold_dock

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "af2_initial_guess.sc")


def _tool(**kw):
    return AF2InitialGuess(af2ig_dir="/opt/dl_binder_design/af2_initial_guess",
                           **kw)


def _design(struct_path: str, did="r0_bb12_s1"):
    bb = Backbone(id="r0_bb12", scaffold_source="de_novo", length=48,
                  peptide_contact_fraction=0.6, coords_ref="")
    return Design(id=did, backbone=bb, sequence="SDEELRR", struct_ref=struct_path)


def _target():
    return Target(Peptide("GILGFVFTL"), "HLA-A*02:01", "Flu-M1")


def test_build_command_matches_real_cli():
    t = _tool(recycle=3)
    argv = t.build_command("/w/in", "/w/out", "/w/out/out.sc")
    assert argv[1].endswith("predict.py")
    assert argv[argv.index("-pdbdir") + 1] == "/w/in"
    assert argv[argv.index("-outpdbdir") + 1] == "/w/out"
    assert argv[argv.index("-scorefilename") + 1] == "/w/out/out.sc"
    assert argv[argv.index("-recycle") + 1] == "3"


def test_parse_scorefile_reads_all_records():
    rows = AF2InitialGuess.parse_scorefile(FIX)
    assert set(rows) == {"r0_bb12_s1", "r0_bb44_s0", "r0_bb07_s2"}
    rec = rows["r0_bb12_s1"]
    assert abs(rec["pae_interaction"] - 5.204) < 1e-9
    assert abs(rec["plddt_binder"] - 91.63) < 1e-9
    assert abs(rec["binder_aligned_rmsd"] - 0.842) < 1e-9
    assert rec["description"] == "r0_bb12_s1"      # stays a string


def test_metric_mapping_to_foldresult():
    rows = AF2InitialGuess.parse_scorefile(FIX)
    fr = AF2InitialGuess._to_fold_result(rows["r0_bb12_s1"])
    # pae_interaction -> pae_interface, plddt_binder -> plddt, rmsd -> rmsd
    assert fr.pae_interface == 5.204
    assert fr.plddt == 91.63
    assert fr.ca_rmsd_to_design == 0.842
    # This good design (pae 5.2, plddt 91.6, rmsd 0.84) passes G4.
    good = Design(id="d", backbone=_design("x").backbone, sequence="A", fold=fr)
    assert g4_fold_dock(good, GatePolicy()) is True


def test_weak_design_fails_g4():
    rows = AF2InitialGuess.parse_scorefile(FIX)
    fr = AF2InitialGuess._to_fold_result(rows["r0_bb44_s0"])  # pae 11.9, plddt 74
    d = Design(id="d", backbone=_design("x").backbone, sequence="A", fold=fr)
    assert g4_fold_dock(d, GatePolicy()) is False


def test_predict_end_to_end_with_injected_runner(tmp_path):
    pdb = tmp_path / "r0_bb12_s1.pdb"
    pdb.write_text("ATOM      1  CA  ALA B   1      0.0 0.0 0.0\n")

    def fake_runner(argv, **kw):
        out = argv[argv.index("-outpdbdir") + 1]
        # predict.py writes the scorefile named by -scorefilename.
        sf = argv[argv.index("-scorefilename") + 1]
        shutil.copy(FIX, sf if os.path.isabs(sf) else os.path.join(out, sf))

    t = _tool(runner=fake_runner)
    fr = t.predict(_design(str(pdb), "r0_bb12_s1"), _target())
    assert fr.pae_interface == 5.204 and fr.plddt == 91.63
    assert not t.is_mock


def test_predict_requires_real_structure():
    t = _tool()
    d = _design("", "r0_bb12_s1")          # no struct_ref, mock backbone
    with pytest.raises(FileNotFoundError):
        t.predict(d, _target())


def test_backend_is_ray_dispatchable():
    import pickle
    t = AF2InitialGuess(af2ig_dir="/opt/af2ig")   # runner=None default
    rt = pickle.loads(pickle.dumps(t))
    assert rt.build_command("/i", "/o", "/o/s.sc")[0] == "python"
