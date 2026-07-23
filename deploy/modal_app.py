"""Deploy the pMHC-Design Agent demo API on Modal (serverless, scale-to-zero).

    pip install modal
    modal token new                      # one-time browser OAuth (no card)
    modal deploy deploy/modal_app.py

Modal prints a public URL, e.g.
    https://<workspace>--pmhc-agent-demo-fastapi-app.modal.run
Try it:
    curl <url>/health
    open <url>/docs
    curl -X POST <url>/campaign -H 'content-type: application/json' \\
         -d '{"peptide":"AAGIGILTV","allele":"HLA-A*02:01","antigen":"MART-1"}'

The image is tiny — the mock agent runs on the standard library, so only the
web stack is installed (no torch, no GPU) and cold starts are ~1-2 s.
"""
import modal

app = modal.App("pmhc-agent-demo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("fastapi==0.111.0", "uvicorn[standard]==0.29.0", "pydantic==2.7.1")
    # the agent package itself has NO third-party runtime deps (mock = stdlib)
    .run_commands(
        "pip install --no-deps 'git+https://github.com/92kunheekim/pmhc-agent.git@main'"
    )
)


@app.function(image=image, memory=512, timeout=60)
@modal.asgi_app()
def fastapi_app():
    from pmhc_agent.service.api import app as web_app
    return web_app
