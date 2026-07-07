"""Tests for the REAL ProteinMPNN backend.

We cannot run ProteinMPNN here (needs a GPU + PyTorch + a real PDB), so we
test the two deterministic, environment-independent parts that are the usual
source of integration bugs:

  1. build_command()  -> the exact real CLI (argv), incl. designing only the
     binder chain and holding MHC+peptide fixed.
  2. parse_fasta()    -> parsing ProteinMPNN's real FASTA output format,
     picking the DESIGNED chain out of the '/'-joined multi-chain sequence,
     skipping the native record, reading the per-sample score.

The subprocess is injected (`runner=`) with a fake that writes the captured
fixture, so we also exercise the full `design()` path and Ray dispatch without
the tool installed. The only thing NOT covered here is the model computation
itself, which requires a GPU worker.
"""
from __future__ import annotations

import os
import shutil

import pytest

from pmhc_agent.tools.proteinmpnn_real import ProteinMPNNReal
from pmhc_agent.types import Backbone
from pmhc_agent import GatePolicy
from pmhc_agent.gates import g3_foldable

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "pmhc_complex.fa")
BINDER_SEQS = {  # designed chain B, per sample, from the fixture
    1: "SDEELRRLAEEAERLLKKAEEAGDPELRRRALELLERLGDPEEAIRRLG",
    2: "SPEELRKLAEELERKLGGADPETARRAAEELIRRLGDPEEALRRLGGG",
    3: "SEEELKRLAEELRRLLGGSDPETARRAAEELLRRLGDPEELIRRLGGD",
}


def _tool(**kw):
    return ProteinMPNNReal(mpnn_repo="/opt/ProteinMPNN", binder_chain="B",
                           context_chains=["A", "C"], **kw)


def test_build_command_matches_real_cli():
    t = _tool(sampling_temp=0.1, batch_size=1)
    argv = t.build_command("/data/x.pdb", "/tmp/out", n=8, seed=40)
    assert argv[1].endswith("protein_mpnn_run.py")
    # Designs ONLY chain B; everything else stays fixed context.
    assert argv[argv.index("--pdb_path_chains") + 1] == "B"
    assert argv[argv.index("--pdb_path") + 1] == "/data/x.pdb"
    assert argv[argv.index("--num_seq_per_target") + 1] == "8"
    assert argv[argv.index("--sampling_temp") + 1] == "0.1"
    assert argv[argv.index("--seed") + 1] == "40"


def test_parse_fasta_picks_designed_chain_and_score():
    t = _tool()
    recs = t.parse_fasta(FIX, chain_order=["A", "B", "C"])
    # 4 records in the fixture: 1 native (skipped) + 3 samples.
    assert len(recs) == 3
    assert [r["sample"] for r in recs] == [1, 2, 3]
    # It extracts chain B (the binder), NOT chain A (MHC) or C (peptide).
    for r in recs:
        assert r["seq"] == BINDER_SEQS[r["sample"]]
        assert "/" not in r["seq"]           # single chain, not the concat
    # scores parsed from the sample headers (lower is better).
    assert abs(recs[0]["score"] - 1.0512) < 1e-9


def test_design_end_to_end_with_injected_runner(tmp_path):
    """Full design() path: fake runner writes the fixture where MPNN would."""
    pdb = tmp_path / "pmhc_complex.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1      0.0 0.0 0.0\n")

    def fake_runner(argv, **kw):
        out = argv[argv.index("--out_folder") + 1]
        seqs = os.path.join(out, "seqs")
        os.makedirs(seqs, exist_ok=True)
        # ProteinMPNN names the file after the input pdb stem.
        shutil.copy(FIX, os.path.join(seqs, "pmhc_complex.fa"))

    t = _tool(runner=fake_runner)
    bb = Backbone(id="r0_bb0", scaffold_source="de_novo", length=48,
                  peptide_contact_fraction=0.6, coords_ref=str(pdb))
    designs = t.design(bb, n=3, round_index=0)

    assert len(designs) == 3
    assert not t.is_mock
    for d in designs:
        assert d.sequence == BINDER_SEQS[int(d.id.split("_s")[1])]
        assert d.mpnn_score > 0
        assert d.rosetta_ddg is None                  # deferred to Rosetta
        # G3's ddG sub-check is skipped when ddG is None, so a good MPNN score
        # alone can pass G3 (threshold 1.15 clears all three fixture scores).
        assert g3_foldable(d, GatePolicy(max_mpnn_score=1.15)) is True
    # And the mpnn_score sub-check still bites: a strict cutoff rejects the
    # weakest sample even though ddG is None.
    weak = max(designs, key=lambda d: d.mpnn_score)
    assert g3_foldable(weak, GatePolicy(max_mpnn_score=1.10)) is False


def test_design_requires_real_pdb():
    t = _tool()
    bb = Backbone(id="x", scaffold_source="de_novo", length=48,
                  peptide_contact_fraction=0.6, coords_ref="")   # mock has none
    with pytest.raises(FileNotFoundError):
        t.design(bb, n=2, round_index=0)


def test_real_backend_is_ray_dispatchable():
    """The tool + a picklable runner must survive serialization for Ray."""
    import pickle
    t = ProteinMPNNReal(mpnn_repo="/opt/ProteinMPNN")   # runner=None (default)
    round_trip = pickle.loads(pickle.dumps(t))
    assert round_trip.binder_chain == "B"
    assert round_trip.build_command("/a.pdb", "/o", 2, 37)[0] == "python"
