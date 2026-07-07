# Generalizing `pmhc-agent` into a shared protein-design engine

Goal: fork `pmhc-agent` into (a) a **domain-agnostic engine** and (b) thin
**domain plugins** — `pmhc` (existing) and `antibody` (RFantibody). Not a
rewrite: ~75% of the current code is already domain-agnostic and moves
unchanged; the rest is generalized behind one new interface (`DesignDomain`)
plus a metrics-bag change to `Design`.

The guiding principle stays the same as the original design: **the models are
dumb tools; the intelligence is the orchestration.** That orchestration —
the loop, the gate runner, the executor, the memory, the brain — doesn't care
whether the binder is a mini-protein or an antibody. Only the *tools*, the
*gates' thresholds*, the *negative set*, and the *target definition* do.

---

## 1. Module disposition

Every current module falls into one of three buckets.

| Module | Disposition | Notes |
|---|---|---|
| `execution.py` (Local/Ray executors) | **SHARED — as-is** | 100% domain-agnostic. No change. |
| `orchestrator.py` (control loop) | **SHARED — parameterized** | Loop stays; stops hardcoding pMHC tools/gates — takes a `DesignDomain`. |
| `memory.py` (scaffold memory) | **SHARED — tiny generalize** | Key scaffolds by `target.key()` instead of `hla_allele`. |
| `diagnostics.py` (Diagnoser, ReplanAction, clamp) | **SHARED — generalize** | Interface + LLM + clamp unchanged; rule text keys on standardized gate names. |
| `llm.py` (LLMDiagnoser) | **SHARED — parameterized** | System prompt built from `domain.describe()` instead of a hardcoded pMHC string. |
| `gates.py::run_gate`, `enforce_diversity` | **SHARED — as-is** | The gate *runner* is generic. |
| `gates.py::g2..g9` predicates | **DOMAIN** | The specific checks move into each domain's gate list. |
| `config.py::Budget, GpuRequest, AgentConfig` | **SHARED — as-is** | |
| `config.py::GatePolicy` | **DOMAIN** | Thresholds differ (pae/plddt/margin vs ipTM/self-consistency). |
| `panel.py::build_panel` | **DOMAIN** (shared concept) | Becomes a `NegativeSetBuilder` the domain implements. |
| `types.py::Design, RoundReport, Campaign, Stage, GateResult` | **SHARED — generalize `Design`** | Metrics bag + multi-chain (below). |
| `types.py::Target, Peptide, OffTarget, FoldResult, SpecificityResult` | **DOMAIN / generalize** | `Target` becomes a base; results fold into the metrics bag. |
| `tools/*` interfaces (generate/design/predict/score) | **SHARED (Protocols)** | Interfaces to core; implementations to domains. |
| `tools/*` implementations (mocks + real) | **DOMAIN** | RFdiffusion-minibinder vs RFantibody, AF2-initial-guess vs RF2/AF3-ipTM, etc. |
| `run.py`, `examples/*` | **DOMAIN** | One entrypoint per domain. |

Rough split: **shared engine ≈ 75%**, domain plugins ≈ 25% (mostly the tool
implementations and threshold constants you already isolated).

---

## 2. The two abstractions that make it work

### 2a. `DesignDomain` — the plugin interface (the linchpin)

Everything domain-specific is reachable through one object the orchestrator
holds. A domain answers: *how do I read a target, what do I screen against,
what tools run, what gates apply, how do I rank, and how do I describe myself
to the LLM brain.*

```python
# core/interfaces.py
class DesignDomain(Protocol):
    name: str

    def intake(self, request: dict) -> Campaign: ...
    #   parse+validate the target, resolve/predict its structure -> Campaign

    def build_negative_set(self, camp: Campaign) -> list: ...
    #   pMHC: proteome off-target peptides on the same allele
    #   antibody: cross-reactive / polyreactivity antigens for the epitope

    def tools(self) -> ToolRegistry: ...
    #   generative + sequence + fold + score backends (mock or real)

    def gates(self) -> list[Gate]: ...
    #   ORDERED, cheap-first list of (name, cost, predicate)

    def rank_score(self, d: Design) -> float: ...          # composite ranking
    def adapt_thresholds(self, camp: Campaign, designs) -> None: ...  # e.g. theta
    def describe(self) -> str: ...                          # domain context for the brain
```

The orchestrator becomes domain-driven and otherwise unchanged:

```python
agent = Orchestrator(domain=PMHCDomain(...),     config=cfg, executor=ex, diagnoser=brain)
agent = Orchestrator(domain=AntibodyDomain(...), config=cfg, executor=ex, diagnoser=brain)
camp  = agent.run(request)
```

### 2b. Metrics bag — decouple gates from hardcoded score fields

Today `Design` has typed `FoldResult(pae_interface, plddt, ...)` and
`SpecificityResult(margin, ...)`. Antibodies use *different* metrics (ipTM,
self-consistency RMSD). Instead of a second typed result, generalize to a
**named-metric bag** that any backend populates and any gate reads:

```python
@dataclass
class Design:
    id: str
    scaffold: Scaffold                       # was Backbone
    chains: dict[str, str]                    # was `sequence: str`  -> multi-chain
                                              #   pMHC {"B": ...}; scFv {"H": ..., "L": ...}
    struct_ref: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    #   pmhc:     pae_interaction, plddt, margin, phi, peptide_contact_fraction
    #   antibody: iptm, self_consistency_rmsd, ddg, plddt, humanness, polyreactivity
    liabilities: list[str] = field(default_factory=list)
    composite: float = 0.0
```

A `Gate` is then mostly data — a named check over the metrics bag with a
threshold from the domain's policy — with an escape hatch for custom logic:

```python
@dataclass
class Gate:
    name: str
    cost: str                                  # "cheap" | "gpu" — for ordering
    predicate: Callable[[Design, dict], bool]  # reads design.metrics + policy dict

# pMHC fold gate:      lambda d,p: d.metrics["pae_interaction"] <= p["max_pae"] and d.metrics["plddt"] >= p["min_plddt"]
# antibody fold gate:  lambda d,p: d.metrics["iptm"] >= p["min_iptm"]   # 0.6 VHH / 0.85 scFv
```

This one change is what lets the *same* gate runner, orchestrator, and
diagnostics serve both domains — the loop never names a metric; the domain does.

---

## 3. Proposed package layout

```
protein_design_agent/            # THE ENGINE (rename of pmhc_agent core)
  orchestrator.py                #  control loop, domain-driven
  execution.py                   #  Local/Ray executors            (unchanged)
  memory.py                      #  scaffold memory, keyed generically
  diagnostics.py                 #  Diagnoser, ReplanAction, clamp  (generalized)
  llm.py                         #  LLMDiagnoser (prompt from domain.describe())
  gates.py                       #  Gate dataclass + run_gate + enforce_diversity
  config.py                      #  Budget, GpuRequest, AgentConfig (no GatePolicy)
  types.py                       #  Design, Scaffold, Campaign, RoundReport, Stage, GateResult
  interfaces.py                  #  DesignDomain + tool Protocols

domains/
  pmhc/                          # EXISTING, refactored behind PMHCDomain
    target.py  panel.py  gates.py  policy.py
    tools/ (mocks + rfdiffusion_real, proteinmpnn_real, alphafold_real, specificity)
    domain.py                    #  class PMHCDomain(DesignDomain)
  antibody/                      # NEW (RFantibody)
    target.py  panel.py  gates.py  policy.py
    tools/ (mocks + rfantibody_real, proteinmpnn_cdr_real, rf2_af3_real)
    domain.py                    #  class AntibodyDomain(DesignDomain)
```

The engine has zero domain imports. Each domain depends on the engine, never
on the other domain.

---

## 4. What is genuinely antibody-specific (the new work)

Everything below lives only in `domains/antibody/` — the engine doesn't change
for any of it:

- **Target** = antigen structure + **epitope hotspots** + **framework** +
  **format** (VHH / scFv / IgG) + CDR contig spec. (pMHC target was just
  peptide + allele.)
- **Multi-chain designs** — scFv has VH+VL and **6 CDRs**; the `chains` dict
  and multi-CDR contigs handle this. IgG conversion is a post-step.
- **Fold/score backend** = fine-tuned **RoseTTAFold2 self-consistency** +
  **AlphaFold3 ipTM**, gated at **ipTM ≥ 0.6 (VHH) / ≥ 0.85 (scFv)** — a new
  real backend parsing ipTM, written the same fixture-verified way as
  `alphafold_real.py`.
- **Negative set** = cross-reactivity / polyreactivity antigens for the
  epitope, not proteome peptides on an allele.
- **Developability gates** = humanness, aggregation, polyreactivity — a
  first-class version of the current G9.
- **Low success rate (0–2%)** — the funnel and the ipTM gate carry more
  weight, so budget bigger fan-out and lean on the diagnosis/re-plan loop.
- **Affinity maturation** (OrthoRep, wet-lab) — an orchestrator Loop-C variant:
  the agent proposes and ingests, humans run it.

RFdiffusion and ProteinMPNN barely change — same tools, fine-tuned weights and
antibody-shaped inputs (framework template track, CDR contigs). Your existing
`rfdiffusion_real.py` / `proteinmpnn_real.py` are ~80% reusable.

---

## 5. Refactor order (keeps the 36 tests green the whole way)

1. **Lift-and-shift.** Rename `pmhc_agent` → `protein_design_agent`; add
   `interfaces.py` with the Protocols. No behavior change; tests still pass.
2. **Metrics bag.** Move `FoldResult`/`SpecificityResult` fields into
   `Design.metrics` (keep read-only properties for back-compat), and switch the
   gate predicates to read `metrics[...]`. Run tests — pure refactor.
3. **Wrap pMHC as a domain.** Bundle the existing target/panel/gates/tools
   behind `PMHCDomain`; make `Orchestrator` take a `domain`. **Green tests here
   are your regression proof** that the generalization changed nothing.
4. **Scaffold `AntibodyDomain` with mocks** mirroring pMHC's mocks (offline,
   deterministic) — a full mock antibody campaign runs end-to-end in seconds,
   exactly like the pMHC demo did. Add antibody tests.
5. **Real antibody backends later**, fixture-verified (RFantibody CLI + output,
   RF2/AF3 ipTM parse) — same discipline as the three real backends you built.

Do 1–3 first: at that point you have a proven engine and the pMHC agent running
on it, with zero regressions, and the antibody work becomes purely additive.

---

## 6. One-line summary

Pull the loop, executor, memory, brain, gate-runner, and Ray/K8s layer into an
engine; express each science as a `DesignDomain` that supplies a target, a
negative set, tools, gates, and a self-description; make predictions carry a
**named-metric bag** so gates never hardcode a score. Then "antibody design" is
a new folder, not a new project.
