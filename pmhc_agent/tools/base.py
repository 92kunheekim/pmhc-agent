"""Tool layer foundations.

Every heavy model (RFdiffusion, ProteinMPNN, AF2, ...) is wrapped as a
`Tool`. The orchestrator only ever sees these interfaces, so a mock backend
and a real GPU backend are interchangeable. Swap by registering a different
implementation in tools/__init__.py::build_registry().

`det_rng` gives each tool a *deterministic* stream keyed on its inputs, so a
whole campaign is reproducible from a single seed — essential for debugging
an agent whose behavior depends on score distributions.
"""
from __future__ import annotations

import hashlib
import random
from typing import Protocol


def det_rng(*parts: object, seed: int = 0) -> random.Random:
    """A reproducible RNG seeded on the tool's semantic inputs.

    Using a hash of the inputs (rather than a global mutable RNG) means the
    score for a given (design, peptide) pair is stable no matter what order
    the orchestrator evaluates things in.
    """
    key = "|".join(str(p) for p in parts) + f"|seed={seed}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


class Tool(Protocol):
    """Marker protocol. Concrete tools expose plain methods (see below)."""
    name: str
    is_mock: bool
