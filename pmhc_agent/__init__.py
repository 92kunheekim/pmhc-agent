"""pMHC-Design Agent — autonomous design of high-specificity binders to
peptide-MHC-I complexes.

Grounded on Lam, Motmaen et al., *Science* 2025 (doi:10.1126/science.adv0185).
This package is runnable scaffolding: it executes end-to-end today with mock
model backends, and each backend is swappable for the real tool
(RFdiffusion / ProteinMPNN / AF2 / NetMHCpan) without changing the
orchestrator.
"""
from .types import Target, Peptide, Campaign, Design
from .config import AgentConfig, GatePolicy, Budget
from .orchestrator import Orchestrator
from .tools import build_registry, ToolRegistry
from .memory import Memory
from .diagnostics import Diagnoser, RuleBasedDiagnoser, ReplanAction
from .llm import LLMDiagnoser, make_diagnoser
from .config import GpuRequest
from .execution import (Executor, LocalExecutor, RayExecutor, make_executor)
from .interfaces import DesignDomain, Gate
from .engine import Engine
from .domains.pmhc.domain import PMHCDomain

__version__ = "0.2.0"
__all__ = [
    "Target", "Peptide", "Campaign", "Design",
    "AgentConfig", "GatePolicy", "Budget", "GpuRequest",
    "Orchestrator", "build_registry", "ToolRegistry", "Memory",
    "Diagnoser", "RuleBasedDiagnoser", "LLMDiagnoser", "ReplanAction",
    "make_diagnoser",
    "Executor", "LocalExecutor", "RayExecutor", "make_executor",
    # Engine generalization:
    "Engine", "DesignDomain", "Gate", "PMHCDomain",
]
