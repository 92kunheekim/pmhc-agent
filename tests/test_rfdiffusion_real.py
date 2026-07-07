"""Tests for the REAL RFdiffusion backbone backend.

RFdiffusion needs a GPU + weights + a real target PDB, absent in dev. So we
test the deterministic parts: the exact Hydra CLI (incl. hotspots on the
peptide and partial-diffusion), and the REAL geometry computation
(peptide-contact fraction + binder length) against a PDB fixture with known
coordinates. The subprocess is injected so generate() runs fully without the
tool. Only the diffusion computation itself needs a GPU.
"""
from __future__ import annotations

import os
import shutil

import pytest

from pmhc_agent.tools.rfdiffusion_real import RFdiffusionReal
from pmhc_agent.types import Target, Peptide

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "rf_binder_complex.pdb")


def _tool(**kw):
    base = dict(
        rfdiffusion_dir="/opt/RFdiffusion",
        target_contig="A1-275/0 C1-9/0",
        hotspot_res=("C4", "C5", "C6", "C7", "C8"),
        binder_length_range="70-100",
        binder_chain="B", peptide_chain="C", mhc_chains=("A",),
    )
    base.update(kw)
    return RFdiffusionReal(**base)


def _target(pdb=""):
    t = Target(Peptide("GILGFVFTL"), "HLA-A*02:01", "FluM1")
    t.pmhc_structure_id = pdb
    return t


def test_build_command_matches_real_cli():
    t = _tool()
    argv = t.build_command("/data/pmhc.pdb", "/out/design", num_designs=8)
    assert argv[1].endswith(os.path.join("scripts", "run_inference.py"))
    assert "inference.input_pdb=/data/pmhc.pdb" in argv
    assert "inference.output_prefix=/out/design" in argv
    assert "inference.num_designs=8" in argv
    # contigs keep the target and append the binder length range
    contig = [a for a in argv if a.startswith("contigmap.contigs=")][0]
    assert "A1-275/0 C1-9/0" in contig and "70-100" in contig
    # hotspots target the peptide's outward-facing residues
    hot = [a for a in argv if a.startswith("ppi.hotspot_res=")][0]
    assert hot == "ppi.hotspot_res=[C4,C5,C6,C7,C8]"
    # de novo (not partial) => no partial_T arg
    assert not any(a.startswith("diffuser.partial_T=") for a in argv)


def test_build_command_partial_diffusion():
    t = _tool(partial_T=12)
    argv = t.build_command("/s.pdb", "/o/d", 4, partial=True)
    assert "diffuser.partial_T=12" in argv


def test_peptide_contact_fraction_from_geometry():
    """Fixture geometry: B1~peptide, B2~MHC, B3~peptide => 2/3."""
    t = _tool()
    frac = t.peptide_contact_fraction(FIX)
    assert abs(frac - 0.667) < 1e-3
    assert t.binder_length(FIX) == 3        # three residues in chain B


def test_contact_fraction_respects_cutoff():
    # With a tiny cutoff, nothing is in contact -> fraction 0.0.
    t = _tool(contact_cutoff=1.0)
    assert t.peptide_contact_fraction(FIX) == 0.0


def test_generate_end_to_end_with_injected_runner(tmp_path):
    target_pdb = tmp_path / "pmhc.pdb"
    shutil.copy(FIX, target_pdb)
    out_dir = tmp_path / "rf_out"

    def fake_runner(argv, **kw):
        # RFdiffusion writes <output_prefix>_<i>.pdb; emulate 3 designs.
        prefix = [a.split("=", 1)[1] for a in argv
                  if a.startswith("inference.output_prefix=")][0]
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        n = int([a.split("=", 1)[1] for a in argv
                 if a.startswith("inference.num_designs=")][0])
        for i in range(n):
            shutil.copy(FIX, f"{prefix}_{i}.pdb")

    t = _tool(runner=fake_runner, out_dir=str(out_dir))
    bbs = t.generate(_target(str(target_pdb)), n=3, round_index=0)

    assert len(bbs) == 3
    for bb in bbs:
        assert bb.scaffold_source == "de_novo"
        assert os.path.exists(bb.coords_ref)             # real PDB persists
        assert abs(bb.peptide_contact_fraction - 0.667) < 1e-3
        assert bb.length == 3
    assert not t.is_mock


def test_generate_partial_diffusion_from_scaffold(tmp_path):
    target_pdb = tmp_path / "pmhc.pdb"; shutil.copy(FIX, target_pdb)
    scaffold = tmp_path / "priv_scaffold.pdb"; shutil.copy(FIX, scaffold)
    out_dir = tmp_path / "out"

    seen = {}
    def fake_runner(argv, **kw):
        seen["argv"] = argv
        prefix = [a.split("=", 1)[1] for a in argv
                  if a.startswith("inference.output_prefix=")][0]
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        shutil.copy(FIX, f"{prefix}_0.pdb")

    t = _tool(runner=fake_runner, out_dir=str(out_dir))
    bbs = t.generate(_target(str(target_pdb)), n=1, round_index=1,
                     seed_scaffold=str(scaffold))
    # partial diffusion used the scaffold + set partial_T
    assert any(a.startswith("diffuser.partial_T=") for a in seen["argv"])
    assert bbs[0].scaffold_source.startswith("partial_diffusion:")


def test_generate_requires_config_and_pdb():
    # Missing hotspots/contig -> ValueError.
    with pytest.raises(ValueError):
        RFdiffusionReal(rfdiffusion_dir="/opt/RFdiffusion").generate(
            _target("/nope.pdb"), 1, 0)
    # Missing target PDB -> FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        _tool().generate(_target(""), 1, 0)


def test_backend_is_ray_dispatchable():
    import pickle
    t = _tool()
    rt = pickle.loads(pickle.dumps(t))     # runner=None default is picklable
    assert rt.hotspot_res == ("C4", "C5", "C6", "C7", "C8")
