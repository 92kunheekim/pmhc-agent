"""REAL AlphaFold2 fold/dock backend (AF2 "initial guess").

Replaces `FoldPredictorMock` with a wrapper around AF2 initial guess
(Bennett et al., Nat. Commun. 2023; github.com/nrbennett3/dl_binder_design),
the AF2 variant the paper uses to validate de novo binders. Same method the
orchestrator calls — `.predict(design, target) -> FoldResult` — so it drops in
via `build_registry(folding=AF2InitialGuess(...))` and is dispatched onto GPU
workers by the same `RayExecutor`.

WHAT IT DOES
------------
Given a design whose `struct_ref` (or its backbone's `coords_ref`) points to a
complex PDB (designed binder chain + peptide + MHC), it:
  1. builds the real `af2_initial_guess/predict.py` command line,
  2. runs AF2 initial guess on the GPU worker (uses the input as the initial
     guess, then predicts the complex),
  3. parses the Rosetta-style scorefile it writes,
  4. maps the binder-design metrics onto `FoldResult`:
        pae_interaction   -> pae_interface   (lower is better; the key filter)
        plddt_binder      -> plddt
        binder_aligned_rmsd -> ca_rmsd_to_design

VERIFIABILITY
-------------
Running AF2 needs a GPU worker with the model weights + JAX/PyTorch installed
and a real complex PDB — absent in dev. So the deterministic parts ARE unit
tested (tests/test_alphafold_real.py): `build_command()` (exact argv) and
`parse_scorefile()` (against a captured real-format `.sc` fixture). The
subprocess is injectable (`runner=`). Only the model computation itself needs
a GPU.

BATCHING NOTE
-------------
AF2 amortizes best when run over a directory of many PDBs at once (the model
loads once). `predict()` here is the per-design drop-in that matches the
orchestrator's per-item fan-out; for large campaigns route the fold stage
through `predict_batch()` (below) or host this as a warm Ray actor so the
weights load once per worker rather than once per design.

CONSENSUS NOTE
--------------
AF2 initial guess is a single model, so `predictors_agree` is set True here;
the AF3/Chai cross-check (gate G8) is a separate backend that would override
it. See the design doc, §07 G8.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from ..types import Design, Target, FoldResult


@dataclass
class AF2InitialGuess:
    """Wrapper around dl_binder_design/af2_initial_guess/predict.py.

    Parameters
    ----------
    af2ig_dir : path to the af2_initial_guess directory (contains predict.py).
    python_exe : interpreter for the run script (often a dedicated conda env).
    recycle : AF2 recycles (predict.py `-recycle`); 3 is typical.
    scorefile_name : name of the scorefile predict.py writes into the out dir.
    runner : callable(argv) -> None that executes the command; defaults to
        subprocess.run(check=True). Injected in tests.
    """
    af2ig_dir: str
    python_exe: str = "python"
    recycle: int = 3
    scorefile_name: str = "out.sc"
    runner: object = None
    name: str = "AF2 initial guess (real)"
    is_mock: bool = field(default=False)

    # -- command construction (VERIFIED) -----------------------------------
    def build_command(self, pdbdir: str, outdir: str, scorefile: str) -> list[str]:
        """The real af2_initial_guess CLI over a directory of complex PDBs."""
        return [
            self.python_exe,
            os.path.join(self.af2ig_dir, "predict.py"),
            "-pdbdir", pdbdir,
            "-outpdbdir", outdir,
            "-scorefilename", scorefile,
            "-recycle", str(self.recycle),
        ]

    # -- scorefile parsing (VERIFIED) --------------------------------------
    @staticmethod
    def parse_scorefile(path: str) -> dict[str, dict]:
        """Parse a Rosetta-style `SCORE:` scorefile -> {description: {col: val}}.

        The first `SCORE:` line is the header; each subsequent `SCORE:` line is
        a record whose last column is the design `description`.
        """
        header: list[str] | None = None
        rows: dict[str, dict] = {}
        with open(path) as fh:
            for line in fh:
                if not line.startswith("SCORE:"):
                    continue
                toks = line.split()[1:]        # drop the leading "SCORE:"
                if header is None:
                    header = toks
                    continue
                rec: dict = {}
                for col, val in zip(header, toks):
                    if col == "description":
                        rec[col] = val
                    else:
                        try:
                            rec[col] = float(val)
                        except ValueError:
                            rec[col] = val
                rows[rec.get("description", str(len(rows)))] = rec
        return rows

    @staticmethod
    def _to_fold_result(rec: dict) -> FoldResult:
        return FoldResult(
            pae_interface=rec["pae_interaction"],
            plddt=rec["plddt_binder"],
            ca_rmsd_to_design=rec.get("binder_aligned_rmsd", 0.0),
            predictors_agree=True,     # single model; AF3/Chai gate is separate
        )

    # -- structure resolution ----------------------------------------------
    @staticmethod
    def _pdb_for(design: Design) -> str:
        pdb = design.struct_ref or design.backbone.coords_ref
        if not pdb or not os.path.exists(pdb):
            raise FileNotFoundError(
                f"AF2InitialGuess needs a real complex PDB at design.struct_ref "
                f"(or backbone.coords_ref); got {pdb!r}. The mock tools emit no "
                f"coordinates — pair with real RFdiffusion + ProteinMPNN.")
        return pdb

    # -- orchestrator-facing per-design method -----------------------------
    def predict(self, design: Design, target: Target) -> FoldResult:
        pdb = self._pdb_for(design)
        run = self.runner or subprocess.run
        with tempfile.TemporaryDirectory(prefix="af2ig_") as work:
            pdbdir = os.path.join(work, "in"); os.makedirs(pdbdir)
            outdir = os.path.join(work, "out"); os.makedirs(outdir)
            stem = os.path.splitext(os.path.basename(pdb))[0]
            staged = os.path.join(pdbdir, f"{stem}.pdb")
            shutil.copy(pdb, staged)
            scorefile = os.path.join(outdir, self.scorefile_name)
            argv = self.build_command(pdbdir, outdir, scorefile)
            # Production: this runs AF2 initial guess on the GPU worker.
            if self.runner is None:
                run(argv, check=True)
            else:
                run(argv)
            if not os.path.exists(scorefile):
                raise RuntimeError(f"AF2 produced no scorefile at {scorefile}.")
            rows = self.parse_scorefile(scorefile)
        rec = rows.get(stem) or next(iter(rows.values()))
        return self._to_fold_result(rec)

    # -- efficient batch path (recommended for real runs) ------------------
    def predict_batch(self, designs: list[Design],
                      target: Target) -> list[FoldResult]:
        """Run AF2 once over all designs' PDBs (model loads once). Returns
        results aligned to `designs`. Route the orchestrator's fold stage
        through this for large campaigns instead of per-design predict()."""
        run = self.runner or subprocess.run
        with tempfile.TemporaryDirectory(prefix="af2ig_batch_") as work:
            pdbdir = os.path.join(work, "in"); os.makedirs(pdbdir)
            outdir = os.path.join(work, "out"); os.makedirs(outdir)
            stems: list[str] = []
            for d in designs:
                pdb = self._pdb_for(d)
                stem = os.path.splitext(os.path.basename(pdb))[0]
                shutil.copy(pdb, os.path.join(pdbdir, f"{stem}.pdb"))
                stems.append(stem)
            scorefile = os.path.join(outdir, self.scorefile_name)
            argv = self.build_command(pdbdir, outdir, scorefile)
            if self.runner is None:
                run(argv, check=True)
            else:
                run(argv)
            rows = self.parse_scorefile(scorefile)
        return [self._to_fold_result(rows[s]) for s in stems]
