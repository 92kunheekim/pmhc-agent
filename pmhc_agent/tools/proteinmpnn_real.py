"""REAL ProteinMPNN backend (shells out to the actual tool).

This replaces `ProteinMPNNMock` with a wrapper around the real ProteinMPNN
(https://github.com/dauparas/ProteinMPNN). It has the SAME method the
orchestrator calls — `.design(backbone, n, round_index) -> list[Design]` — so
it drops into `build_registry(sequences=ProteinMPNNReal(...))` with no other
change, and it is dispatched onto GPU workers by the same `RayExecutor`.

WHAT IT DOES
------------
Given a backbone whose `coords_ref` points to a real complex PDB (MHC chain +
peptide chain + the binder chain to design), it:
  1. builds the exact `protein_mpnn_run.py` command line,
  2. runs it (on the GPU worker), designing only the binder chain while the
     MHC and peptide chains are held fixed as context,
  3. parses ProteinMPNN's FASTA output (`<out>/seqs/<stem>.fa`),
  4. returns one `Design` per sampled sequence, carrying the real designed
     binder sequence and ProteinMPNN's per-sequence `score` (mean negative
     log-likelihood; lower is better) as `mpnn_score`.

VERIFIABILITY (important, read this)
------------------------------------
Running the model needs a GPU worker with ProteinMPNN + PyTorch installed and
a real input PDB — none of which exist in the mock/dev environment. So this
module is split so the deterministic, environment-independent parts ARE unit
tested (see tests/test_proteinmpnn_real.py):
  * `build_command()`  — the exact argv is asserted against the real CLI.
  * `parse_fasta()`    — parsed against a captured real-format `.fa` fixture.
The subprocess call itself is injectable (`runner=`) so tests drive it with a
fake that drops a fixture in place. The ONLY part that requires a GPU +
installed ProteinMPNN is the actual `runner` invocation in production.

`rosetta_ddg` is left as None here: ProteinMPNN does not produce a binding
ddG. A real deployment computes that in a separate Rosetta/FastRelax task; the
G3 gate skips the ddG sub-check when it is None (see gates.g3_foldable).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field

from ..types import Backbone, Design


# ProteinMPNN sample-record header carries: T=, sample=, score=, global_score=,
# seq_recovery=.  The native (first) record has no "sample=".
_SCORE_RE = re.compile(r"score=([-\d.eE]+)")
_SAMPLE_RE = re.compile(r"sample=(\d+)")


@dataclass
class ProteinMPNNReal:
    """Wrapper around dauparas/ProteinMPNN's protein_mpnn_run.py.

    Parameters
    ----------
    mpnn_repo : path to the cloned ProteinMPNN repo (contains
        protein_mpnn_run.py).
    binder_chain : chain id in the PDB that ProteinMPNN should DESIGN
        (e.g. "B"). All other chains are held fixed as context.
    context_chains : the fixed chains (e.g. ["A", "C"] for MHC + peptide).
        Used only to compute the chain order for parsing; if None, chains are
        read from the PDB.
    sampling_temp, batch_size, base_seed : standard ProteinMPNN knobs.
    python_exe : interpreter used to launch the run script.
    runner : callable(argv:list[str]) -> None that executes the command.
        Defaults to subprocess.run(check=True). Injected in tests.
    """
    mpnn_repo: str
    binder_chain: str = "B"
    context_chains: list[str] | None = None
    sampling_temp: float = 0.1
    batch_size: int = 1
    base_seed: int = 37
    python_exe: str = "python"
    runner: object = None                      # None -> subprocess (picklable)
    name: str = "ProteinMPNN (real)"
    is_mock: bool = field(default=False)

    # -- command construction (VERIFIED by tests) --------------------------
    def build_command(self, pdb_path: str, out_folder: str, n: int,
                      seed: int) -> list[str]:
        """The exact real ProteinMPNN CLI. Designs only `binder_chain`."""
        return [
            self.python_exe,
            os.path.join(self.mpnn_repo, "protein_mpnn_run.py"),
            "--pdb_path", pdb_path,
            "--pdb_path_chains", self.binder_chain,   # chains to DESIGN
            "--out_folder", out_folder,
            "--num_seq_per_target", str(n),
            "--sampling_temp", str(self.sampling_temp),
            "--seed", str(seed),
            "--batch_size", str(self.batch_size),
        ]

    # -- output parsing (VERIFIED by tests) --------------------------------
    def _chain_order(self, pdb_path: str) -> list[str]:
        """Alphabetical chain order = order chains are concatenated in the
        ProteinMPNN FASTA (chains joined by '/'). Prefer the configured
        chains; otherwise read chain ids from the PDB."""
        if self.context_chains is not None:
            return sorted(set(self.context_chains) | {self.binder_chain})
        chains = set()
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                    chains.add(line[21])
        return sorted(chains)

    def parse_fasta(self, fa_path: str, chain_order: list[str]) -> list[dict]:
        """Parse ProteinMPNN seqs/<stem>.fa -> [{seq, score, sample}, ...].

        Skips the native (first) record; keeps sampled designs. The sequence
        line concatenates all chains as `chainA/chainB/...` in `chain_order`;
        we return the designed (binder) chain's sequence.
        """
        idx = chain_order.index(self.binder_chain)
        records: list[dict] = []
        header, seqlines = None, []

        def flush():
            if header is None:
                return
            if _SAMPLE_RE.search(header):        # a sampled design, not native
                seq = "".join(seqlines)
                chains = seq.split("/")
                m = _SCORE_RE.search(header)
                records.append({
                    "seq": chains[idx] if idx < len(chains) else chains[-1],
                    "score": float(m.group(1)) if m else float("nan"),
                    "sample": int(_SAMPLE_RE.search(header).group(1)),
                })

        with open(fa_path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    flush()
                    header, seqlines = line, []
                elif line:
                    seqlines.append(line)
        flush()
        return records

    # -- the orchestrator-facing method ------------------------------------
    def design(self, backbone: Backbone, n: int, round_index: int) -> list[Design]:
        pdb_path = backbone.coords_ref
        if not pdb_path or not os.path.exists(pdb_path):
            raise FileNotFoundError(
                f"ProteinMPNNReal needs a real PDB at backbone.coords_ref; "
                f"got {pdb_path!r}. (The mock RFdiffusion does not emit PDBs — "
                f"pair this with a real backbone generator.)")

        seed = self.base_seed + round_index          # reproducible per round
        run = self.runner or subprocess.run
        with tempfile.TemporaryDirectory(prefix="mpnn_") as out_folder:
            argv = self.build_command(pdb_path, out_folder, n, seed)
            # Production: this line runs ProteinMPNN on the GPU worker.
            if self.runner is None:
                run(argv, check=True)
            else:
                run(argv)                              # injected fake in tests

            stem = os.path.splitext(os.path.basename(pdb_path))[0]
            fa_path = os.path.join(out_folder, "seqs", f"{stem}.fa")
            if not os.path.exists(fa_path):
                raise RuntimeError(
                    f"ProteinMPNN produced no output at {fa_path}. "
                    f"Check the run logs on the worker.")
            recs = self.parse_fasta(fa_path, self._chain_order(pdb_path))

        designs: list[Design] = []
        for r in recs:
            designs.append(Design(
                id=f"{backbone.id}_s{r['sample']}",
                backbone=backbone,
                sequence=r["seq"],
                mpnn_score=r["score"],       # real ProteinMPNN NLL
                rosetta_ddg=None,            # computed by a separate Rosetta task
            ))
        return designs
