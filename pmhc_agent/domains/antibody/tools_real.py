"""REAL RFantibody backbone backend (antibody de novo design).

Wraps the generative step of RFantibody (Bennett et al., Nature 2025;
github.com/RosettaCommons/RFantibody) — the antibody-fine-tuned RFdiffusion
that designs CDR loops and docks the antibody onto a chosen epitope. Same
method the engine calls — `.generate(target, n, round_index, ...) -> list[Backbone]`
— so it drops into `build_antibody_registry`/the domain with no other change,
and it is dispatched onto GPU workers by the same `RayExecutor`.

Output PDBs are HLT format (Heavy / Light / Target chains); for a VHH only H+T.
They are the complex structures the CDR-ProteinMPNN and RF2/AF3 backends
consume, completing the antibody generate -> design -> filter chain.

VERIFIABILITY (same discipline as the pMHC real backends)
--------------------------------------------------------
RFantibody needs a GPU + weights + real target/framework PDBs, absent in dev.
So the deterministic parts ARE unit tested (tests/test_rfantibody_real.py):
`build_command()` (the exact rfdiffusion_inference.py antibody CLI incl.
hotspots + design_loops) and `epitope_contact_fraction()` / `binder_length()`
(real geometry, against an HLT PDB fixture with known coordinates). The
subprocess is injectable (`runner=`). Only the diffusion itself needs a GPU.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field

from ...types import Backbone


@dataclass
class RFantibodyReal:
    """Wrapper around RFantibody/scripts/rfdiffusion_inference.py (--config-name antibody).

    Parameters
    ----------
    rfantibody_dir : path to the RFantibody repo (contains scripts/rfdiffusion_inference.py).
    framework_pdb : the antibody framework to graft CDRs onto (VHH or scFv).
    ckpt_path : RFdiffusion antibody weights (inference.ckpt_override_path).
    design_loops : CDR loop spec, RFantibody syntax. VHH: "H1:7,H2:6,H3:5-13";
        scFv adds "L1:8-13,L2:7,L3:9-11".
    binder_chains : antibody chains in the HLT output ("H",) for VHH; ("H","L") scFv.
    target_chain : the antigen chain in HLT output (default "T").
    contact_cutoff : Angstrom cutoff for an interface contact.
    out_dir : where output PDBs are kept (point at a PVC/scratch in production).
    runner : callable(argv) -> None; defaults to subprocess.run(check=True).
    """
    rfantibody_dir: str
    framework_pdb: str = ""
    ckpt_path: str = ""
    design_loops: str = "H1:7,H2:6,H3:5-13"
    binder_chains: tuple = ("H",)
    target_chain: str = "T"
    contact_cutoff: float = 5.0
    out_dir: str | None = None
    python_exe: str = "python"
    runner: object = None
    name: str = "RFantibody (real)"
    is_mock: bool = field(default=False)

    # -- command construction (VERIFIED) -----------------------------------
    def build_command(self, target_pdb: str, output_prefix: str,
                      num_designs: int, hotspots: tuple) -> list[str]:
        """The real RFantibody antibody-design CLI (Hydra key=value args)."""
        argv = [
            self.python_exe,
            os.path.join(self.rfantibody_dir, "scripts", "rfdiffusion_inference.py"),
            "--config-name", "antibody",
            f"antibody.target_pdb={target_pdb}",
            f"antibody.framework_pdb={self.framework_pdb}",
        ]
        if self.ckpt_path:
            argv.append(f"inference.ckpt_override_path={self.ckpt_path}")
        argv += [
            "ppi.hotspot_res=[" + ",".join(hotspots) + "]",
            "antibody.design_loops=[" + self.design_loops + "]",
            f"inference.num_designs={num_designs}",
            f"inference.output_prefix={output_prefix}",
        ]
        return argv

    # -- HLT geometry (VERIFIED) -------------------------------------------
    @staticmethod
    def _read_ca(pdb_path: str) -> list[tuple]:
        out = []
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                out.append((line[21], line[22:26].strip(),
                            float(line[30:38]), float(line[38:46]),
                            float(line[46:54])))
        return out

    def binder_length(self, pdb_path: str) -> int:
        cas = self._read_ca(pdb_path)
        return len({(c, r) for (c, r, *_ ) in cas if c in self.binder_chains})

    def epitope_contact_fraction(self, pdb_path: str, hotspots: tuple) -> float:
        """Fraction of the antibody's interface residues whose nearest target
        residue is an epitope HOTSPOT (vs an off-epitope target residue).
        1.0 = paratope perfectly focused on the intended epitope; 0.0 = off."""
        cas = self._read_ca(pdb_path)
        hot = set(hotspots)
        binder = [(x, y, z) for (c, r, x, y, z) in cas if c in self.binder_chains]
        target = [(f"{c}{r}", x, y, z) for (c, r, x, y, z) in cas
                  if c == self.target_chain]
        epi = off = 0
        for (bx, by, bz) in binder:
            best_id, best_d = None, math.inf
            for (tid, tx, ty, tz) in target:
                d = math.dist((bx, by, bz), (tx, ty, tz))
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is not None and best_d <= self.contact_cutoff:
                if best_id in hot:
                    epi += 1
                else:
                    off += 1
        total = epi + off
        return round(epi / total, 3) if total else 0.0

    # -- orchestrator-facing method ----------------------------------------
    def generate(self, target, n: int, round_index: int,
                 seed_scaffold: str | None = None,
                 contact_bias: float = 0.0) -> list[Backbone]:
        if not self.framework_pdb:
            raise ValueError("RFantibodyReal needs framework_pdb set.")
        hotspots = tuple(target.epitope_hotspots)
        target_pdb = target.antigen_structure_id
        if not target_pdb or not os.path.exists(str(target_pdb)):
            raise FileNotFoundError(
                f"RFantibodyReal needs a real antigen PDB; got {target_pdb!r}. "
                "In production a real structure resolver sets "
                "target.antigen_structure_id to a PDB path.")

        run = self.runner or __import__("subprocess").run
        out_dir = self.out_dir or tempfile.mkdtemp(prefix="rfab_out_")
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.join(out_dir, f"{target.antigen}_r{round_index}")
        argv = self.build_command(target_pdb, prefix, n, hotspots)
        # Production: this runs RFantibody diffusion on the GPU worker.
        if self.runner is None:
            run(argv, check=True)
        else:
            run(argv)

        source = ("partial_diffusion:" + os.path.basename(str(seed_scaffold))
                  if seed_scaffold else "de_novo")
        backbones: list[Backbone] = []
        for i in range(n):
            pdb = f"{prefix}_{i}.pdb"
            if not os.path.exists(pdb):
                continue
            backbones.append(Backbone(
                id=f"r{round_index}_ab{i}",
                scaffold_source=source,
                length=self.binder_length(pdb),
                peptide_contact_fraction=self.epitope_contact_fraction(pdb, hotspots),
                coords_ref=pdb))
        return backbones
