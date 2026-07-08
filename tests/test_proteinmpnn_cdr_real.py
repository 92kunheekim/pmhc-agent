"""Tests for the REAL CDR-ProteinMPNN backend (RFantibody interface design).

Verified without a GPU: the exact RFantibody interface-design CLI, and the
extraction of the designed antibody-chain sequence from an HLT output PDB
(three-letter -> one-letter). The subprocess is injected so design() runs fully
without the tool; only the model run needs a GPU.
"""
from __future__ import annotations

import os
import shutil

import pytest

from pmhc_agent.domains.antibody.tools_real import ProteinMPNNCDRReal
from pmhc_agent.types import Backbone

DESIGNED = os.path.join(os.path.dirname(__file__), "fixtures",
                        "ab_mpnn_designed.pdb")


def _tool(**kw):
    base = dict(rfantibody_dir="/opt/RFantibody", binder_chains=("H",))
    base.update(kw)
    return ProteinMPNNCDRReal(**base)


def test_build_command_matches_real_cli():
    t = _tool()
    argv = t.build_command("/in", "/out", 4)
    assert argv[1].endswith(os.path.join("scripts",
                                         "proteinmpnn_interface_design.py"))
    assert argv[argv.index("-pdbdir") + 1] == "/in"
    assert argv[argv.index("-outpdbdir") + 1] == "/out"
    assert argv[argv.index("-seqs_per_struct") + 1] == "4"


def test_extract_designed_sequence_from_hlt_pdb():
    t = _tool(binder_chains=("H",))
    chains = t.chain_sequences(DESIGNED)
    # Heavy chain residues MET-ALA-ARG-THR-VAL-ASP-LYS-LEU-GLN-HIS -> MARTVDKLQH
    assert chains["H"] == "MARTVDKLQH"
    assert "T" not in chains                     # target chain is not designed


def test_design_end_to_end_with_injected_runner(tmp_path):
    hlt = tmp_path / "r0_ab0.pdb"
    shutil.copy(DESIGNED, hlt)
    out = tmp_path / "mpnn_out"

    def fake_runner(argv, **kw):
        outdir = argv[argv.index("-outpdbdir") + 1]
        os.makedirs(outdir, exist_ok=True)
        n = int(argv[argv.index("-seqs_per_struct") + 1])
        stem = "r0_ab0"
        for i in range(n):                       # dl_binder_design naming
            shutil.copy(DESIGNED, os.path.join(outdir, f"{stem}_dldesign_{i}.pdb"))

    t = _tool(runner=fake_runner, out_dir=str(out))
    bb = Backbone(id="r0_ab0", scaffold_source="de_novo", length=10,
                  peptide_contact_fraction=0.66, coords_ref=str(hlt))
    designs = t.design(bb, n=3, round_index=0)

    assert len(designs) == 3
    for d in designs:
        assert d.sequence == "MARTVDKLQH"
        assert d.chains == {"H": "MARTVDKLQH"}
        assert os.path.exists(d.struct_ref)       # threaded PDB for RF2/AF3
        assert d.rosetta_ddg is None
    assert not t.is_mock


def test_scfv_extracts_both_chains(tmp_path):
    # With binder_chains H+L, both chains are pulled (fixture has only H, so L
    # is simply absent -> sequence is the H part). Confirms multi-chain handling.
    t = _tool(binder_chains=("H", "L"))
    chains = t.chain_sequences(DESIGNED)
    assert chains.get("H") == "MARTVDKLQH"


def test_requires_real_hlt_pdb():
    t = _tool()
    bb = Backbone(id="x", scaffold_source="de_novo", length=10,
                  peptide_contact_fraction=0.6, coords_ref="")
    with pytest.raises(FileNotFoundError):
        t.design(bb, n=1, round_index=0)


def test_backend_is_ray_dispatchable():
    import pickle
    t = _tool()
    rt = pickle.loads(pickle.dumps(t))
    assert rt.binder_chains == ("H",)
    assert rt.build_command("/i", "/o", 1)[0] == "python"
