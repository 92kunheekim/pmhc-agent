# Image for the pMHC-Design Agent on Ray/Kubernetes.
#
# The base is an official Ray image that matches your cluster's Ray version.
# For REAL model backends you would build FROM a CUDA-enabled Ray GPU image
# (e.g. rayproject/ray:2.56.0-gpu) and add RFdiffusion / ProteinMPNN /
# AlphaFold + their weights here (or mount weights from a PVC at runtime).
#
# This mock image is CPU-only and self-contained: it proves the orchestration
# path (Ray dispatch, GPU resource requests, autoscaling) end to end without
# needing the heavy tools installed.

FROM rayproject/ray:2.56.0

# --- the agent package ---
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY pmhc_agent /app/pmhc_agent
COPY examples /app/examples
RUN pip install --no-cache-dir -e .

# Optional extras:
#   RUN pip install --no-cache-dir -e ".[llm]"     # LLM diagnoser
#
# --- REAL backends would be added here, typically in separate stages ---
# COPY tools/proteinmpnn /opt/proteinmpnn
# RUN pip install --no-cache-dir torch ...         # per-tool deps
# ENV ALPHAFOLD_WEIGHTS=/weights/af2               # mounted from a PVC
#
# The driver entrypoint (invoked by the RayJob):
CMD ["python", "examples/run_on_ray.py"]
