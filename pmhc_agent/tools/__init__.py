"""Tool registry.

`build_registry` wires up a set of backends. To go from the mock demo to a
real run, implement the same method signatures against RFdiffusion /
ProteinMPNN / AF2 / NetMHCpan and swap them in here — the orchestrator does
not change.

    from pmhc_agent.tools import build_registry
    reg = build_registry(seed=7)                 # all mocks
    reg = build_registry(backbones=MyRealRFdiffusion(), ...)  # mix & match
"""
from __future__ import annotations

from dataclasses import dataclass

from .netmhc import NetMHCpanMock
from .structure import StructureResolverMock
from .backbones import RFdiffusionMock
from .sequences import ProteinMPNNMock
from .folding import FoldPredictorMock
from .specificity import SpecificityEngineMock


@dataclass
class ToolRegistry:
    netmhc: object
    structure: object
    backbones: object
    sequences: object
    folding: object
    specificity: object

    @property
    def all_mock(self) -> bool:
        return all(
            getattr(t, "is_mock", False)
            for t in (self.netmhc, self.structure, self.backbones,
                      self.sequences, self.folding, self.specificity)
        )


def build_registry(seed: int = 7, solved_ids: dict | None = None,
                   **overrides) -> ToolRegistry:
    defaults = dict(
        netmhc=NetMHCpanMock(seed),
        structure=StructureResolverMock(seed, solved_ids=solved_ids),
        backbones=RFdiffusionMock(seed),
        sequences=ProteinMPNNMock(seed),
        folding=FoldPredictorMock(seed),
        specificity=SpecificityEngineMock(seed),
    )
    defaults.update(overrides)
    return ToolRegistry(**defaults)
