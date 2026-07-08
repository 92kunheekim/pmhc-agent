"""Tests for the REAL RFantibody backbone backend.

RFantibody needs a GPU + weights + real PDBs, absent in dev. So we test the
deterministic parts: the exact antibody CLI, and the HLT-geometry epitope-focus
computation against a fixture with known coordinates. The subprocess is
injected so generate() runs fully without the tool. Only the diffusion itself
needs a GPU.
"""
from __future__ import annotations

import os
import shutil

import pytest

from pmhc_agent.domains.antibody.tools_real import RFantibodyReal
from pmhc_agent import AntibodyTarget

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "ab_hlt_complex.pdb")


def _tool(**kw):
    base = dict(rfantibody_dir="/opt/RFantibody",
                framework_pdb="/opt/frameworks/hu-VHH.pdb",
                ckpt_path="/opt/weights/RFdiffusion_Ab.pt",
                design_loops="H1:7,H2:6,H3:5-13",
                binder_chains=("H",), target_chain="T")
    base.update(kw)
    return RFantibodyReal(**base)


def _target(pdb=""):
    t = AntibodyTarget(antigen="influenza-HA",
                       epitope_hotspots=("T305", "T456"), fmt="VHH")
    t.antigen_structure_id = pdb
    return t


def test_build_command_matches_real_cli():
    t = _tool()
    argv = t.build_command("/d/antigen.pdb", "/o/ab", 20, ("T305", "T456"))
    assert argv[1].endswith(os.path.join("scripts", "rfdiffusion_inference.py"))
    assert "--config-name" in argv and argv[argv.index("--config-name") + 1] == "antibody"
    assert "antibody.target_pdb=/d/antigen.pdb" in argv
    assert "antibody.framework_pdb=/opt/frameworks/hu-VHH.pdb" in argv
    assert "inference.ckpt_override_path=/opt/weights/RFdiffusion_Ab.pt" in argv
    assert "ppi.hotspot_res=[T305,T456]" in argv
    assert "antibody.design_loops=[H1:7,H2:6,H3:5-13]" in argv
    assert "inference.num_designs=20" in argv
    assert "inference.output_prefix=/o/ab" in argv


def test_epitope_contact_fraction_from_geometry():
    """H1->hotspot, H2->off-epitope, H3->hotspot => 2/3 focused on epitope."""
    t = _tool()
    frac = t.epitope_contact_fraction(FIX, ("T305", "T456"))
    assert abs(frac - 0.667) < 1e-3
    assert t.binder_length(FIX) == 3               # 3 heavy-chain residues (VHH)


def test_epitope_focus_drops_if_hotspots_wrong():
    # If the intended epitope were elsewhere, the same paratope scores as
    # off-epitope -> fraction 0.0.
    t = _tool()
    assert t.epitope_contact_fraction(FIX, ("T400",)) == pytest.approx(1/3, abs=1e-3)


def test_generate_end_to_end_with_injected_runner(tmp_path):
    antigen = tmp_path / "antigen.pdb"
    shutil.copy(FIX, antigen)
    out = tmp_path / "rfab_out"

    def fake_runner(argv, **kw):
        prefix = [a.split("=", 1)[1] for a in argv
                  if a.startswith("inference.output_prefix=")][0]
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        n = int([a.split("=", 1)[1] for a in argv
                 if a.startswith("inference.num_designs=")][0])
        for i in range(n):
            shutil.copy(FIX, f"{prefix}_{i}.pdb")

    t = _tool(runner=fake_runner, out_dir=str(out))
    bbs = t.generate(_target(str(antigen)), n=3, round_index=0)
    assert len(bbs) == 3
    for bb in bbs:
        assert os.path.exists(bb.coords_ref)
        assert abs(bb.peptide_contact_fraction - 0.667) < 1e-3   # epitope focus
        assert bb.length == 3
        assert bb.scaffold_source == "de_novo"
    assert not t.is_mock


def test_requires_framework_and_antigen():
    with pytest.raises(ValueError):                       # no framework
        RFantibodyReal(rfantibody_dir="/opt/RFantibody").generate(
            _target("/x.pdb"), 1, 0)
    with pytest.raises(FileNotFoundError):                # no antigen PDB
        _tool().generate(_target(""), 1, 0)


def test_backend_is_ray_dispatchable():
    import pickle
    t = _tool()
    rt = pickle.loads(pickle.dumps(t))                    # runner=None default
    assert rt.design_loops == "H1:7,H2:6,H3:5-13"
    assert rt.build_command("/a.pdb", "/o", 2, ("T1",))[0] == "python"
