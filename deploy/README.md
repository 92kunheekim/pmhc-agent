# Deploying the pMHC-Design Agent on Ray + Kubernetes

The agent's control loop and brain run on the Ray **head** (CPU); the heavy
per-design stages fan out to autoscaling **GPU workers**. The split lives in
`pmhc_agent/execution.py` — swap `LocalExecutor` for `RayExecutor` and nothing
else in the agent changes.

## How the pieces map

| Layer | Runs where | What |
|---|---|---|
| Orchestrator + brain (`Diagnoser`) | Ray head (CPU) | plan, gate, adapt theta, decide/stop |
| Sequence design (ProteinMPNN) | GPU workers, `num_gpus` from `GpuRequest.sequence` | one task per backbone |
| Fold/dock (AlphaFold) | GPU workers, `GpuRequest.fold` | one task per surviving design |
| Specificity (fine-tuned AF2 + MPNN) | GPU workers, `GpuRequest.specificity` | one task per design, vs the panel |

Fractional `num_gpus` (e.g. `sequence: 0.25`) packs several light tasks onto
one physical GPU; heavy AF stages typically take a whole GPU each.

## Prerequisites

1. A Kubernetes cluster **with GPU nodes** (Kubernetes doesn't provide GPUs —
   the node pool must have them).
2. **KubeRay operator** — `helm repo add kuberay https://ray-project.github.io/kuberay-helm/ && helm install kuberay-operator kuberay/kuberay-operator`.
3. **NVIDIA GPU Operator** (or device plugin) so pods can request
   `nvidia.com/gpu`.
4. The image built and pushed to a registry your cluster can pull.

## Build & push the image

```bash
# from the package root (the dir containing pyproject.toml)
docker build -f deploy/Dockerfile -t REGISTRY/pmhc-agent:latest .
docker push REGISTRY/pmhc-agent:latest
```

The provided Dockerfile is CPU-only + mock tools — it proves the full Ray/K8s
path (dispatch, GPU requests, autoscaling) without the heavy models. For real
runs, base it on `rayproject/ray:<ver>-gpu` and add RFdiffusion / ProteinMPNN /
AlphaFold and their weights (or mount weights from a PVC).

## Run a campaign

Edit `deploy/rayjob.yaml` — set `REGISTRY/pmhc-agent:latest`, the `nodeSelector`
for your GPU node pool, and the `PMHC_*` env vars — then:

```bash
kubectl apply -f deploy/rayjob.yaml
kubectl get rayjob pmhc-campaign -w
kubectl logs -l job-name=pmhc-campaign -f
```

The GPU worker group starts at `replicas: 0`, autoscales up as the agent
submits tasks, and (with `shutdownAfterJobFinishes`) tears down when the
campaign ends — so an idle cluster costs no GPU.

## Smoke-test the path locally (no GPU, no K8s)

Because `RayExecutor` requests `num_gpus=0` when GPUs aren't configured, you
can run the exact same driver against a local Ray head:

```bash
pip install "ray>=2.9"
RAY_ADDRESS=local PMHC_STAGE_GPUS=0 python examples/run_on_ray.py
```

This exercises real Ray task dispatch and confirms results are identical to the
in-process `LocalExecutor` (the test `test_ray_matches_local_and_preserves_determinism`
asserts exactly that).

## Plain Kubernetes Job (e.g. MD Anderson `yn-gpu-workload`)

If your cluster uses plain `batch/v1` Jobs with a single GPU and **no Ray
operator** (as the provided institution template does), skip KubeRay entirely
and run the whole agent in one pod with the in-process `LocalExecutor`:

- **`deploy/job.mdanderson.gpu.yaml`** — the agent as a plain Job, adapted from
  the institution template (namespace, `k8s-user` label, `securityContext`
  uid/gid, `nodeSelector`, `/dev/shm`, home PVC, and the AlphaFold 2.3.1 DB
  mount all preserved). First run is the mock smoke test — it proves the pod,
  package, and GPU scheduling work with no model weights.
- **`deploy/Dockerfile.mdanderson`** — builds `FROM` your existing
  `tcrmodel2_cuda12.2` image (already has AF2 + CUDA), adding RFdiffusion,
  ProteinMPNN, and the package. Push to your Harbor project.

Submit:

```bash
kubectl apply -f deploy/job.mdanderson.gpu.yaml
kubectl -n yn-gpu-workload logs -f job/kkim14-pmhc-agent
```

Notes specific to that environment:
- The AlphaFold 2.3.1 databases are already mounted (`AF_DB`), so the AF2
  fold backend needs no DB download — point it at that path.
- Keep the `securityContext` uid/gid exactly as given, or writes to your home
  PVC will fail.
- The template mounts `data` and `alphafold_db` but only defines `home` + `shm`
  volumes; confirm whether a webhook injects the rest, else add the PVC names
  your admins provide (placeholders are in the Job file).
- **Ray on this cluster:** multi-pod fan-out needs the KubeRay operator
  installed cluster-wide — ask your admins. Until then, `LocalExecutor` in one
  GPU pod is the right choice; a single campaign still runs end to end, just
  without cross-pod parallelism.

## Cost & resilience notes

- **Scale to zero** (`minReplicas: 0`) so idle time is free.
- **Spot/preemptible GPU nodes** cut cost sharply; enable the commented
  toleration. Ray retries tasks whose node is reclaimed — keep stages
  idempotent so retries are safe.
- **Weights on a PVC**, not baked into the image, keep images small and node
  cold-starts fast.
- **`maxReplicas`** is your hard ceiling on both fan-out and spend.
- Prefer **object storage (S3/GCS)** over a single shared volume for the
  thousands of small structure files passing between stages at scale.
