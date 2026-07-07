"""Execution layer — WHERE the per-design work runs.

The agent's control loop is one thing; the fan-out of thousands of designs
across GPUs is another. This module separates them behind an `Executor`
interface so the orchestrator says "apply this function to these items" and
stays agnostic about whether that happens in-process or across a Ray cluster
of GPU workers on Kubernetes.

  * `LocalExecutor` — a plain in-process map. Default. No dependencies. Used
    by the mock demo and the test suite.
  * `RayExecutor` — dispatches each item as a Ray task with a GPU resource
    request, so the heavy stages (RFdiffusion / AlphaFold / ProteinMPNN)
    land on GPU workers. Falls back to local execution if Ray isn't
    installed, mirroring the LLM diagnoser's graceful degradation.

Determinism note: both executors PRESERVE INPUT ORDER, so a seeded campaign
gives identical results whether it runs locally or on Ray.

The functions dispatched (below) are module-level and picklable — a hard
requirement for Ray, which serializes the callable and its bound args to the
worker. `functools.partial(task_fold, tool, target)` is picklable because
`task_fold`, the mock tool, and the target dataclass all are. For real tools,
the "tool" passed here is a lightweight CLIENT (holds paths/config, not model
weights) that shells out to the tool binary already installed in the worker
container — so it stays cheap to serialize.
"""
from __future__ import annotations

import logging
from typing import Callable, Protocol

log = logging.getLogger("pmhc_agent.execution")


# --------------------------------------------------------------------------
# Dispatchable unit-of-work functions (picklable, module-level)
# --------------------------------------------------------------------------
def task_sequence(tool, n, round_index, backbone):
    """ProteinMPNN: design n sequences for one backbone. Returns list[Design]."""
    return tool.design(backbone, n, round_index)


def task_fold(tool, target, design):
    """AlphaFold: predict fold/dock for one design. Returns the design."""
    design.fold = tool.predict(design, target)
    return design


def task_specificity(tool, target, panel, design):
    """Specificity engine: score one design vs the off-target panel."""
    design.specificity = tool.score(design, target, panel)
    return design


# --------------------------------------------------------------------------
# Executor interface + backends
# --------------------------------------------------------------------------
class Executor(Protocol):
    name: str

    def map(self, fn: Callable, items: list, gpus_per_task: float = 0.0,
            label: str = "") -> list:
        ...


class LocalExecutor:
    """In-process map. gpus_per_task is ignored (nothing is scheduled)."""
    name = "local"

    def map(self, fn, items, gpus_per_task: float = 0.0, label: str = "") -> list:
        return [fn(x) for x in items]


class RayExecutor:
    """Dispatch each item as a Ray task with a GPU resource request.

    `gpus_per_task` maps to Ray's `num_gpus`. Fractional values (e.g. 0.25)
    let several light tasks pack onto one physical GPU — useful for cheap
    stages like ProteinMPNN so you don't waste a whole A100 on each.
    """
    name = "ray"

    def __init__(self, address: str = "auto", namespace: str | None = None,
                 max_in_flight: int = 512):
        import ray  # lazy: Ray is optional
        if not ray.is_initialized():
            # address="auto" attaches to the RayCluster the driver runs in
            # (the KubeRay head). Use address=None for a local Ray for tests.
            ray.init(address=address, namespace=namespace,
                     ignore_reinit_error=True)
        self._ray = ray
        self.max_in_flight = max_in_flight

    def map(self, fn, items, gpus_per_task: float = 0.0, label: str = "") -> list:
        ray = self._ray
        remote = ray.remote(_apply)
        # Submit in bounded waves so a huge round doesn't flood the scheduler.
        results: list = []
        for start in range(0, len(items), self.max_in_flight):
            chunk = items[start:start + self.max_in_flight]
            refs = [remote.options(num_gpus=gpus_per_task).remote(fn, x)
                    for x in chunk]
            results.extend(ray.get(refs))   # ray.get preserves order
        if label:
            log.info("ray stage '%s': %d tasks @ %.2f gpu each",
                     label, len(items), gpus_per_task)
        return results


def _apply(fn, x):
    """Trivial remote shim: apply a picklable fn to one item."""
    return fn(x)


def make_executor(kind: str = "local", **kwargs) -> Executor:
    """Factory. `kind` is 'local' or 'ray'. Falls back to local if Ray is
    unavailable, so the package always runs."""
    if kind == "ray":
        try:
            return RayExecutor(**kwargs)
        except Exception as e:  # ImportError, no cluster, etc.
            log.warning("Ray executor unavailable (%s); using LocalExecutor.", e)
            return LocalExecutor()
    return LocalExecutor()
