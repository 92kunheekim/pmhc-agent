# pMHC-Design Agent

[![CI](https://github.com/92kunheekim/pmhc-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/92kunheekim/pmhc-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**▶ Live demo:** [`/docs`](https://kunheekim--pmhc-agent-demo-fastapi-app.modal.run/docs) — a public, scale-to-zero FastAPI endpoint on Modal. POST a target and watch the agent run:

```bash
curl -X POST https://kunheekim--pmhc-agent-demo-fastapi-app.modal.run/campaign \
  -H 'content-type: application/json' \
  -d '{"peptide":"AAGIGILTV","allele":"HLA-A*02:01","antigen":"MART-1"}'
```

It returns the per-round gate funnel and a ranked shortlist (mock backends, no GPU; in-silico hypotheses, human-gated).

Runnable Python scaffolding for an AI agent that autonomously designs
**high-specificity binders to peptide–MHC-I complexes** — an implementation
skeleton of the pipeline in Lam, Motmaen et al., *Design of high-specificity
binders for peptide–MHC-I complexes*, **Science** 2025
([doi:10.1126/science.adv0185](https://www.science.org/doi/10.1126/science.adv0185)).

It **runs end-to-end today** with mock model backends (no GPU, no API key),
so you can watch the full generate → filter → diagnose → re-design loop work.
Each backend is a drop-in interface you replace with the real tool
(RFdiffusion, ProteinMPNN, AlphaFold2/3, NetMHCpan) **without touching the
orchestrator**.

> This is engineering scaffolding and a design aid, not a validated protocol.
> All wet-lab actions (DNA synthesis, assays, spend) are **human-gated** by
> design — the agent recommends; a scientist commits.

---

## What's original here (vs. the paper)

Lam & Motmaen *et al.* contribute the **science** — the specificity-design method and its wet-lab validation. This repository is the **engineering** around that idea, and that engineering is the contribution:

- **Agentic orchestration loop** — plan → gate → diagnose → adapt → re-design — turning a one-shot pipeline into a closed loop with a failure taxonomy and an adaptive specificity threshold (`theta`).
- **Pluggable "brain"** — deterministic rules vs. an Anthropic-SDK LLM behind one `Diagnoser` interface, with tool-use structured output, guardrail clamping, and graceful fallback to rules when no key/SDK is present.
- **Domain-agnostic `Engine`** — a `DesignDomain` seam so the same loop drives pMHC-I binder design today and de novo antibody design (RFantibody) unchanged.
- **Scale-out layer** — `LocalExecutor`/`RayExecutor` behind one interface, a KubeRay RayJob with autoscale-to-zero GPU workers, and a real institutional Kubernetes Job template.
- **Fixture-verified real-backend pattern** — every real tool wrapper (RFdiffusion, ProteinMPNN, AF2 initial-guess, RFantibody, RF2+AF3) has its exact CLI and output parsing unit-tested **without a GPU**.

The science is the paper's; the agent, the architecture, and the infrastructure are mine.

## Quick start

```bash
python -m pmhc_agent.run                       # default MART-1 / A*02:01 demo
python -m pmhc_agent.run --peptide EVDPIGHLY --allele "HLA-A*01:01" --antigen MAGE-A3
pip install -e ".[dev]"                        # makes `pmhc_agent` importable
python examples/multi_target.py                # memory reuse across targets
pytest -q                                      # run the tests (58 pass, 3 Ray tests skip without a cluster)
```

No third-party dependencies are needed for the mock demo — it runs on the
Python standard library alone (Python ≥ 3.10).

The demo prints the funnel per round (how many designs each gate passes /
rejects), the adaptively-tuned specificity threshold `theta`, the failure
diagnosis that steers the next round, and a ranked, synthesis-ready shortlist.

---

## What maps to what in the paper

| Paper step | Module | Tool interface (swap here) |
|---|---|---|
| RFdiffusion backbones over the peptide | `tools/backbones.py` | `RFdiffusionMock.generate()` |
| ProteinMPNN sequence design | `tools/sequences.py` | `ProteinMPNNMock.design()` |
| AF2 fold/dock + AF3/Chai cross-check | `tools/folding.py` | `FoldPredictorMock.predict()` |
| Fine-tuned AF2 + contrastive MPNN specificity | `tools/specificity.py` | `SpecificityEngineMock.score()` |
| pMHC structure (PDB or AF3) | `tools/structure.py` | `StructureResolverMock.resolve()` |
| MHC binding prior (panel + ligand sanity) | `tools/netmhc.py` | `NetMHCpanMock.percent_rank()` |
| Off-target panel (systematic, not hand-picked) | `panel.py` | `build_panel()` |
| Filter-gate stack G1–G9 | `gates.py` | pure predicates |
| Privileged-scaffold reuse | `memory.py` | `Memory` |
| Plan / route / stop / re-plan | `orchestrator.py` | `Orchestrator` |
| Failure taxonomy + threshold adaptation | `diagnostics.py` | `diagnose()` |

## The specificity contract (the core idea)

A design is accepted only if its **worst-case** margin over the entire
off-target panel clears the (adaptive) threshold:

```
margin(d) = min over p⁻ in panel [ score(d, on_target) − score(d, p⁻) ]
accept(d) ⇔ margin(d) ≥ theta_target  AND  peptide_energy_fraction ≥ phi
```

Using `min` (worst offender), not the mean, is deliberate: a binder is only
as specific as its single worst cross-reaction. This is the guardrail against
the paper's own failure mode (designs that bound their tetramer specifically
yet cross-activated on other self-peptides presented by the same HLA).

---

## Going from mock to real

Each mock exposes the exact method the orchestrator calls. To use a real
backend, implement the same signature and register it:

```python
from pmhc_agent import Orchestrator, AgentConfig
from pmhc_agent.tools import build_registry

class RealRFdiffusion:
    name = "RFdiffusion"
    is_mock = False
    def generate(self, target, n, round_index, seed_scaffold=None, contact_bias=0.0):
        # ... call RFdiffusion, return list[pmhc_agent.types.Backbone] ...
        ...

reg = build_registry(seed=7, backbones=RealRFdiffusion())   # mix real + mock
agent = Orchestrator(config=AgentConfig(), registry=reg)
camp = agent.run(target)
shortlist = agent.recommend_library(camp)   # human reviews before synthesis
```

Recommended real backends: RFdiffusion / RFdiffusion-AA, ProteinMPNN /
LigandMPNN, AlphaFold2 (initial-guess) + AlphaFold3 + Chai-1, the Motmaen
fine-tuned AF2 specificity model, NetMHCpan-4.1 / MHCflurry, Rosetta
FastRelax for the interface-energy partition (`phi`), and a FAISS + Foldseek
index behind `Memory`.

#### Worked example: the real ProteinMPNN backend

`tools/proteinmpnn_real.py` is a complete real backend — it shells out to the
actual [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), designing only
the binder chain while the MHC and peptide chains are held fixed:

```python
from pmhc_agent import Orchestrator, AgentConfig, RayExecutor, build_registry
from pmhc_agent.tools.proteinmpnn_real import ProteinMPNNReal

mpnn = ProteinMPNNReal(mpnn_repo="/opt/ProteinMPNN",
                       binder_chain="B", context_chains=["A", "C"])
reg = build_registry(seed=7, sequences=mpnn)          # real MPNN, mocks elsewhere
agent = Orchestrator(config=AgentConfig(), registry=reg,
                     executor=RayExecutor(address="auto"))
```

It requires backbones whose `coords_ref` points to real complex PDBs, so pair
it with a real RFdiffusion backend (the mock emits no coordinates). What IS
unit-tested without a GPU (`tests/test_proteinmpnn_real.py`): the exact CLI it
builds and its parsing of ProteinMPNN's real FASTA output (via a captured
fixture and an injected runner). Only the model computation itself needs a GPU
worker. `rosetta_ddg` is left `None` (ProteinMPNN gives no ddG; compute it in a
separate Rosetta task — G3 skips the ddG sub-check when it's absent).

#### Worked example: the real RFdiffusion backbone backend

`tools/rfdiffusion_real.py` wraps RFdiffusion in binder-design mode. It keeps
the MHC + peptide as context (contigs) and steers the binder onto the
outward-facing peptide residues via `ppi.hotspot_res`, then reads each output
complex PDB into a `Backbone` — computing a **real** `peptide_contact_fraction`
from the geometry (the value gate G2 filters on). Its output PDBs are what the
ProteinMPNN and AF2 backends consume, so the three complete the
generate → design → fold chain.

```python
from pmhc_agent.tools.rfdiffusion_real import RFdiffusionReal
rf = RFdiffusionReal(rfdiffusion_dir="/opt/RFdiffusion",
                     target_contig="A1-275/0 C1-9/0",       # MHC + peptide
                     hotspot_res=("C4","C5","C6","C7","C8"), # peptide residues
                     binder_length_range="70-100")
reg = build_registry(seed=7, backbones=rf, sequences=mpnn, folding=af2)
```

Verified without a GPU (`tests/test_rfdiffusion_real.py`): the exact Hydra CLI
(incl. hotspots and partial-diffusion for scaffold reuse) and the geometry
computation against a PDB fixture with known coordinates. Only the diffusion
itself needs a GPU. `contact_bias` (a mock knob) is accepted but unused — real
peptide focus is set through `hotspot_res`.

#### Worked example: the real AlphaFold2 fold/dock backend

`tools/alphafold_real.py` wraps **AF2 initial guess** (Bennett et al. 2023,
`dl_binder_design`) — the AF2 variant the paper uses to validate de novo
binders. It runs `af2_initial_guess/predict.py`, parses the Rosetta-style
scorefile, and maps `pae_interaction -> pae_interface`, `plddt_binder -> plddt`,
`binder_aligned_rmsd -> ca_rmsd_to_design`:

```python
from pmhc_agent.tools.alphafold_real import AF2InitialGuess
af2 = AF2InitialGuess(af2ig_dir="/opt/dl_binder_design/af2_initial_guess")
reg = build_registry(seed=7, sequences=mpnn, folding=af2)   # real MPNN + AF2
```

Verified without a GPU (`tests/test_alphafold_real.py`): the exact CLI and the
scorefile parse + metric mapping, against a captured `.sc` fixture, incl. that
a strong design passes G4 and a weak one (pae 11.9, pLDDT 74) fails. It expects
each design's `struct_ref` to point at a real complex PDB (populated upstream
by the RFdiffusion→ProteinMPNN handoff). For large campaigns use
`predict_batch()` (one AF2 load over many PDBs) or host it as a warm Ray actor.
`predictors_agree` is `True` (single model); the AF3/Chai consensus (G8) is a
separate backend.

### The pluggable "brain": rules vs. LLM

The agent's decision-making (what to change after each round) lives behind a
`Diagnoser` interface with two interchangeable backends — both return the
same `ReplanAction`, so the orchestrator is agnostic to which is in charge:

| Backend | Where | Cost / determinism | Use it for |
|---|---|---|---|
| `RuleBasedDiagnoser` (default) | `diagnostics.py` | free, deterministic, testable | reliable reflexes; reproducible runs |
| `LLMDiagnoser` | `llm.py` (Anthropic SDK) | per-call cost, non-deterministic | open-ended reasoning over the round's score distributions |

```bash
# rule-based (default) — no key, fully reproducible
python -m pmhc_agent.run

# LLM brain — reasons about each round with an Anthropic model
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=sk-...
export PMHC_LLM_MODEL=claude-3-5-sonnet-latest      # or any model you have
python -m pmhc_agent.run --brain llm --library 600  # forces several rounds
```

Or in code:

```python
from pmhc_agent import Orchestrator, LLMDiagnoser
agent = Orchestrator(diagnoser=LLMDiagnoser())     # model-driven brain
camp = agent.run(target)
```

Three properties worth noting:

- **Guardrailed.** Whatever the model proposes is clamped into safe ranges
  (`clamp_action` in `diagnostics.py`) before the orchestrator acts on it —
  the LLM advises, it does not get unbounded control.
- **Graceful.** If the `anthropic` SDK isn't installed, `ANTHROPIC_API_KEY`
  is unset, or a call fails, `LLMDiagnoser` falls back to the deterministic
  rules and annotates the diagnosis (`[LLM unavailable -> rules] ...`). The
  package still runs with no key.
- **Structured output.** The model is forced (via tool use) to fill in the
  exact fields of `ReplanAction`, so its answer is parsed, not guessed at.

`llm.py` is a complete, readable reference for wiring an LLM into an agent
loop: the system prompt, the tool schema, the call, the parse, the clamp, and
the fallback.

**Seeing it actually run.** The tests exercise this path with an *injected fake client* (deterministic, no key, no network — so CI stays green), and a captured transcript of the brain driving a real campaign against an Anthropic model is checked in at [`docs/llm-brain-run.md`](docs/llm-brain-run.md) — the per-round `[LLM]` diagnoses are the model reasoning over each round's gate funnel. Reproduce it with your own key:

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=sk-ant-...
# pick a model your account has: python -c "import anthropic; [print(m.id) for m in anthropic.Anthropic().models.list().data]"
python -m pmhc_agent.run --brain llm --library 600 --model <model-id> | tee docs/llm-brain-run.md
```

The parser also defensively strips any tool-call/XML artifacts a model may leak into a free-text field, so the diagnosis stays clean regardless of model quirks (regression-tested in `tests/test_llm_diagnoser.py`).

## Engine + domains (generalization)

The orchestration is being generalized into a domain-agnostic **engine** so the
same loop can drive pMHC-I binder design today and antibody design (RFantibody)
next. The seam is a `DesignDomain` (in `interfaces.py`) that supplies the
target intake, the negative set, the tool backends, the stage-tagged `Gate`
list, ranking, and a self-description for the LLM brain. `Engine` (in
`engine.py`) holds the domain-independent machinery and runs whatever domain it
is given:

```python
from pmhc_agent import Engine, PMHCDomain, AgentConfig
agent = Engine(domain=PMHCDomain(seed=7), config=AgentConfig())
camp = agent.run(target)          # identical results to the legacy Orchestrator
```

`Design` now carries a named-metric bag (`metrics`) and a `chains` dict so gates
never hardcode a score field and multi-chain (scFv VH+VL) designs fit where a
single binder chain did. The legacy `Orchestrator` remains for back-compat;
`test_engine_pmhc.py` proves `Engine(PMHCDomain)` is byte-identical to it. See
`docs/ARCHITECTURE-generalization.md` for the full plan.

**A second domain already runs on the engine.** `domains/antibody/` implements
de novo antibody design (RFantibody) as an `AntibodyDomain` — RFdiffusion CDR
loops, ProteinMPNN CDR design, RF2 self-consistency + AF3 **ipTM** filtering
(ipTM ≥ 0.6 VHH / 0.85 scFv), cross-reactivity screening, and developability
gates — all reading the same metric bag, driven by the same engine:

```bash
PYTHONPATH=. python examples/run_antibody_mock.py   # full mock antibody campaign
```

Only the domain differs from the pMHC run; the loop, executor, memory, brain,
and Ray/K8s layer are shared. Real RFantibody/RF2-AF3 backends drop in behind
the mock tool signatures the same fixture-verified way as the pMHC backends.

The first real antibody backend is already written:
`domains/antibody/tools_real.py::RFantibodyReal` wraps RFantibody's
`rfdiffusion_inference.py` (antibody mode) — framework + `ppi.hotspot_res` +
`antibody.design_loops`, parsing HLT-format output PDBs and computing a **real
epitope-focus fraction** from the geometry (paratope residues nearest an
epitope hotspot vs off-epitope). Verified without a GPU
(`tests/test_rfantibody_real.py`): the exact CLI and the HLT geometry against a
known-coordinate fixture; only the diffusion needs a GPU.

`ProteinMPNNCDRReal` (same file) wraps RFantibody's
`proteinmpnn_interface_design.py` — it designs the CDR loops on the fixed
framework of each HLT complex and threads them back into the structure that
RF2/AF3 then predict. Verified (`tests/test_proteinmpnn_cdr_real.py`): the CLI
and the extraction of the designed antibody-chain sequence from the output HLT
PDB (three-letter → one-letter, multi-chain aware for scFv).

`RF2AF3Real` (same file) finishes the chain: RoseTTAFold2 self-consistency
(`rf2_predict.py`) + AlphaFold3 **ipTM** (`run_alphafold.py`) — the paper's most
predictive filter. It merges both into one `FoldResult` so gate A4 reads ipTM,
self-consistency, and pLDDT from the metric bag. Verified
(`tests/test_rf2af3_real.py`): both CLIs, the RF2 scorefile parse, the AF3
summary-JSON parse, and the AF3 input-JSON builder, against fixtures; only the
model runs need a GPU. **All three antibody backends are now real** — the full
RFantibody generate → design → filter chain — with mocks retained for offline
runs. Wire them with `ToolRegistry(backbones=RFantibodyReal(...),
sequences=ProteinMPNNCDRReal(...), folding=RF2AF3Real(...), ...)` and pass it to
`AntibodyDomain(registry=...)`.

## Scaling out: Ray on Kubernetes (GPU fan-out)

The agent's loop runs in one place; the thousands of per-design model calls
fan out to GPU workers. That split lives behind an `Executor` interface
(`execution.py`), exactly like the pluggable brain:

| Executor | Runs where | Use it for |
|---|---|---|
| `LocalExecutor` (default) | in-process | mocks, tests, single-box runs |
| `RayExecutor` | Ray tasks on GPU workers | real RFdiffusion / AlphaFold / ProteinMPNN at scale |

```python
from pmhc_agent import Orchestrator, AgentConfig, GpuRequest, RayExecutor
cfg = AgentConfig(gpus=GpuRequest(sequence=0.25, fold=1, specificity=1))
agent = Orchestrator(config=cfg, executor=RayExecutor(address="auto"))
camp = agent.run(target)     # fold/specificity/sequence tasks land on GPU pods
```

Both executors **preserve order**, so a seeded campaign gives identical
results whether it runs locally or across a cluster (asserted by
`test_ray_matches_local_and_preserves_determinism`). Per-stage GPU requests
come from `GpuRequest`; fractional values pack light tasks onto one GPU.

Kubernetes deployment (KubeRay `RayJob` with an autoscale-to-zero GPU worker
group, Dockerfile, and driver) is in **[`deploy/`](deploy/README.md)**. Smoke-test
the whole Ray path locally with no GPU:

```bash
pip install -e ".[ray]"
RAY_ADDRESS=local PMHC_STAGE_GPUS=0 python examples/run_on_ray.py
```

## Layout

```
pmhc_agent/
  types.py          # typed campaign state
  config.py         # gate policies + budget
  tools/            # swappable model backends (mocks + real RFdiff/MPNN/AF2)
    rfdiffusion_real.py   # REAL RFdiffusion backbone backend
    proteinmpnn_real.py   # REAL ProteinMPNN backend
    alphafold_real.py     # REAL AF2 initial-guess fold/dock backend
  panel.py          # adversarial off-target panel builder
  gates.py          # G1–G9 filter stack
  specificity ->    # in tools/specificity.py (the core engine)
  memory.py         # privileged-scaffold memory
  diagnostics.py    # failure taxonomy + Diagnoser interface + RuleBasedDiagnoser
  llm.py            # LLMDiagnoser: Anthropic-backed brain (optional)
  execution.py      # Executor interface: LocalExecutor + RayExecutor
  interfaces.py     # DesignDomain + Gate + tool Protocols (engine seam)
  engine.py         # Engine: domain-agnostic orchestrator
  domains/pmhc/     # PMHCDomain: pMHC-I science as a DesignDomain
  domains/antibody/ # AntibodyDomain: de novo antibody design (RFantibody)
  orchestrator.py   # legacy pMHC orchestrator (back-compat; == Engine+PMHCDomain)
  run.py            # CLI demo (--brain rules|llm)
examples/multi_target.py
examples/run_on_ray.py        # RayJob driver entrypoint
deploy/                       # KubeRay RayJob, Dockerfile, deploy guide
tests/test_pipeline.py
tests/test_llm_diagnoser.py   # LLM brain, via an injected fake client
tests/test_execution.py       # Local + Ray executors (Ray test auto-skips)
```

## Safety

No autonomous wet-lab actuation, DNA ordering, or spend. Specificity claims
from any in-silico model are **hypotheses to test** — final off-target safety
rests on wet-lab profiling. Screen any real ordered sequences through standard
biosecurity/synthesis-screening norms.
