"""REAL RFdiffusion backbone backend (binder design mode).

Replaces `RFdiffusionMock` with a wrapper around RFdiffusion (Watson et al.,
Nature 2023; github.com/RosettaCommons/RFdiffusion). Same method the
orchestrator calls — `.generate(target, n, round_index, seed_scaffold, ...)
-> list[Backbone]` — so it drops in via `build_registry(backbones=...)` and is
dispatched onto GPU workers by the same `RayExecutor`. Its output PDBs are the
complex structures that the real ProteinMPNN and AF2 backends consume, so this
completes the generate -> design -> fold chain.

WHAT IT DOES
------------
Given the target pMHC complex PDB (from `target.pmhc_structure_id`), it:
  1. builds the real `run_inference.py` binder-design command line — keeping
     the MHC + peptide as context (contigs) and steering the binder onto the
     OUTWARD-FACING PEPTIDE RESIDUES via `ppi.hotspot_res`,
  2. runs RFdiffusion on the GPU worker (or partial-diffusion from a seed
     scaffold when reusing a privileged backbone),
  3. reads each output complex PDB into a `Backbone`, computing a REAL
     `peptide_contact_fraction` from the geometry (fraction of the binder's
     interface residues closer to the peptide than to the MHC) — the value
     gate G2 filters on.

VERIFIABILITY
-------------
RFdiffusion needs a GPU + weights + a real target PDB, absent in dev. So the
deterministic parts ARE unit tested (tests/test_rfdiffusion_real.py):
`build_command()` (exact Hydra argv, incl. hotspots and partial-diffusion) and
`peptide_contact_fraction()` / `binder_length()` (real geometry, against a PDB
fixture with known coordinates). The subprocess is injectable (`runner=`).
Only the diffusion computation itself needs a GPU.

NOTE on `contact_bias`: that was a mock knob. Real RFdiffusion steers peptide
focus through hotspot selection, not a scalar — so `contact_bias` is accepted
(interface compatibility) but not used here; the agent's Loop-A "bias toward
the peptide" instead maps to expanding/adjusting `hotspot_res`.
"""
from __future__ import annotations

import math
import os
import glob
import subprocess
import tempfile
from dataclasses import dataclass, field

from ..types import Target, Backbone


@dataclass
class RFdiffusionReal:
    """Wrapper around RFdiffusion/scripts/run_inference.py (binder design).

    Parameters
    ----------
    rfdiffusion_dir : path to the RFdiffusion repo (contains scripts/run_inference.py).
    target_contig : the target region to keep as context, in RFdiffusion contig
        syntax, e.g. "A1-275/0 C1-9/0" (MHC chain A residues 1-275, chain break,
        peptide chain C residues 1-9). Structure-specific; required.
    hotspot_res : outward-facing peptide residues to target, e.g.
        ["C4","C5","C6","C7","C8"]. Required — this is how specificity focus is
        encoded at generation time.
    binder_length_range : contig spec for the generated binder, e.g. "70-100"
        (range) or "60-60" (fixed).
    binder_chain / peptide_chain / mhc_chains : chain ids in the OUTPUT PDB, for
        the contact-fraction geometry.
    contact_cutoff : Angstrom cutoff for an interface contact.
    partial_T : denoising steps for partial diffusion when reusing a scaffold.
    out_dir : where output PDBs are written and kept (so coords_ref persists
        for the downstream ProteinMPNN/AF2 tools). In production point this at a
        PVC / object-store mount. If None, a persistent temp dir is created.
    runner : callable(argv) -> None; defaults to subprocess.run(check=True).
    """
    rfdiffusion_dir: str
    target_contig: str = ""
    hotspot_res: tuple = ()
    binder_length_range: str = "70-100"
    binder_chain: str = "B"
    peptide_chain: str = "C"
    mhc_chains: tuple = ("A",)
    contact_cutoff: float = 5.0
    partial_T: int = 10
    out_dir: str | None = None
    python_exe: str = "python"
    runner: object = None
    name: str = "RFdiffusion (real)"
    is_mock: bool = field(default=False)

    # -- command construction (VERIFIED) -----------------------------------
    def build_command(self, input_pdb: str, output_prefix: str,
                      num_designs: int, partial: bool = False) -> list[str]:
        """The real RFdiffusion binder-design CLI (Hydra key=value args).

        Note: each list element is one argv token; subprocess does NOT
        shell-split, so a contig element may contain spaces safely.
        """
        contigs = f"contigmap.contigs=[{self.target_contig} {self.binder_length_range}]"
        hotspots = "ppi.hotspot_res=[" + ",".join(self.hotspot_res) + "]"
        argv = [
            self.python_exe,
            os.path.join(self.rfdiffusion_dir, "scripts", "run_inference.py"),
            f"inference.input_pdb={input_pdb}",
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={num_designs}",
            contigs,
            hotspots,
        ]
        if partial:
            # Partial diffusion reuses the input structure as a starting point.
            argv.append(f"diffuser.partial_T={self.partial_T}")
        return argv

    # -- PDB geometry (VERIFIED) -------------------------------------------
    @staticmethod
    def _read_ca(pdb_path: str) -> list[tuple]:
        """Return [(chain, resseq, x, y, z), ...] for CA atoms."""
        out = []
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                out.append((
                    line[21], line[22:26].strip(),
                    float(line[30:38]), float(line[38:46]),
                    float(line[46:54]),
                ))
        return out

    def binder_length(self, pdb_path: str) -> int:
        cas = self._read_ca(pdb_path)
        return len({r for (c, r, *_ ) in cas if c == self.binder_chain})

    def peptide_contact_fraction(self, pdb_path: str) -> float:
        """Fraction of the binder's interface residues that sit closer to the
        peptide than to the MHC. 1.0 = purely peptide-focused; 0.0 = all MHC.
        Returns 0.0 if the binder has no interface contacts."""
        cas = self._read_ca(pdb_path)
        binder = [(x, y, z) for (c, r, x, y, z) in cas if c == self.binder_chain]
        pep = [(x, y, z) for (c, r, x, y, z) in cas if c == self.peptide_chain]
        mhc = [(x, y, z) for (c, r, x, y, z) in cas if c in self.mhc_chains]

        def nearest(p, pts):
            best = math.inf
            for q in pts:
                d = math.dist(p, q)
                if d < best:
                    best = d
            return best

        pep_ct = mhc_ct = 0
        for b in binder:
            dp = nearest(b, pep)
            dm = nearest(b, mhc)
            if min(dp, dm) <= self.contact_cutoff:      # an interface residue
                if dp <= dm:
                    pep_ct += 1
                else:
                    mhc_ct += 1
        total = pep_ct + mhc_ct
        return round(pep_ct / total, 3) if total else 0.0

    # -- orchestrator-facing method ----------------------------------------
    def generate(self, target: Target, n: int, round_index: int,
                 seed_scaffold: str | None = None,
                 contact_bias: float = 0.0) -> list[Backbone]:
        if not self.target_contig or not self.hotspot_res:
            raise ValueError(
                "RFdiffusionReal needs target_contig and hotspot_res set "
                "(structure-specific). See the class docstring.")

        partial = bool(seed_scaffold and os.path.exists(str(seed_scaffold)))
        input_pdb = seed_scaffold if partial else target.pmhc_structure_id
        if not input_pdb or not os.path.exists(str(input_pdb)):
            raise FileNotFoundError(
                f"RFdiffusionReal needs a real target PDB; got "
                f"{input_pdb!r}. In production a real StructureResolver sets "
                f"target.pmhc_structure_id to a PDB path.")

        run = self.runner or subprocess.run
        source = (f"partial_diffusion:{os.path.basename(str(seed_scaffold))}"
                  if partial else "de_novo")

        # Persistent output dir so coords_ref stays valid for downstream tools.
        out_dir = self.out_dir or tempfile.mkdtemp(prefix="rfdiff_out_")
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.join(out_dir, f"{target.source_antigen}_r{round_index}")
        argv = self.build_command(input_pdb, prefix, n, partial=partial)
        # Production: this runs RFdiffusion on the GPU worker.
        if self.runner is None:
            run(argv, check=True)
        else:
            run(argv)

        backbones: list[Backbone] = []
        for i in range(n):
            pdb = f"{prefix}_{i}.pdb"       # RFdiffusion writes <prefix>_<i>.pdb
            if not os.path.exists(pdb):
                continue
            backbones.append(Backbone(
                id=f"r{round_index}_bb{i}",
                scaffold_source=source,
                length=self.binder_length(pdb),
                peptide_contact_fraction=self.peptide_contact_fraction(pdb),
                coords_ref=pdb,
            ))
        return backbones
