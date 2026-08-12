# OligoGym benchmark reproduction image — one image for both CPU (c6id/r6id/m6id)
# and GPU (g4dn / Tesla T4, sm_75) AWS Batch tasks.
#
# Derived from Roche/oligogym @ 97f5b9f (Apache-2.0) plus the Phase-1 patched
# benchmark harness. The CPU-vs-CUDA divergence is confined to the torch wheel,
# so a single CUDA-enabled image serves both compute classes: on a CPU-only
# instance torch simply reports cuda.is_available() == False and every model
# class still runs.
#
# Build (x86_64 only — g4dn and c6id/r6id/m6id are all amd64):
#   docker buildx build --platform linux/amd64 -t <ref> .
FROM python:3.11-slim-bookworm

ARG OLIGOGYM_COMMIT=97f5b9f58d9e445a8ecb0218037af7465c3708c0
ARG TORCH_VERSION=2.13.0
# cu126 is the newest CUDA index that still ships torch 2.13.0 AND compiles
# sm_75 (Tesla T4). The build asserts sm_75 is present in torch.cuda.get_arch_list()
# below, so a future base bump that silently drops Turing fails the build.
ARG TORCH_CUDA=cu126
ARG RNAFM_SHA256=5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # fm.pretrained.rna_fm_t12() resolves the checkpoint under $TORCH_HOME/checkpoints
    TORCH_HOME=/opt/torch-hub \
    # Each fold-fit is one process; BLAS threads are packed by the scheduler, not
    # by the process. Overridden per task by the Nextflow process directive.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# procps: Nextflow requires `ps` for task metrics. git: pip install from the pinned
# commit. curl/ca-certificates: RNA-FM checkpoint fetch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 1. torch first, from the CUDA index, so nothing later pulls a CPU wheel over it.
#    Pinned exact: the Phase-1 environment validated torch 2.13.0.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
        --extra-index-url https://pypi.org/simple \
        torch==${TORCH_VERSION}+${TORCH_CUDA}

# ---------------------------------------------------------------------------
# 2. Everything else from PyPI, exact pins from the Phase-1 validated env.
#    torch_geometric 2.8 is PURE PYTHON: torch_scatter / torch_sparse /
#    torch_cluster / pyg_lib are deliberately NOT installed. Phase 1 verified all
#    four are absent from the working environment and every GNN test still passes
#    (oligogym touches only GCNConv + global_{max,mean,add}_pool). Adding the
#    compiled companions is the usual source of CUDA-version build failures and
#    buys nothing here.
# ---------------------------------------------------------------------------
COPY requirements-image.txt /tmp/requirements-image.txt
RUN pip install --no-cache-dir -r /tmp/requirements-image.txt

# ---------------------------------------------------------------------------
# 3. oligogym itself, at the pinned commit, --no-deps (the pins above are the
#    validated closure; upstream pyproject.toml under-declares by 10 packages).
#    This also installs the 18 MB of processed datasets that ship inside the
#    package at oligogym/resources/pkg_dataset/ — no dataset is fetched at
#    runtime despite DatasetDownloader's docstring.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir --no-deps \
        "git+https://github.com/Roche/oligogym.git@${OLIGOGYM_COMMIT}"

# ---------------------------------------------------------------------------
# 4. RNA-FM pretrained weights, baked in.
#    The URL hardcoded inside the `fm` package
#    (proj.cse.cuhk.edu.hk/rnafm/api/download?...) returns nginx 403 for every
#    path, so the checkpoint comes from the RNA-FM authors' own HF org. Baking it
#    avoids a 1.2 GB fetch per container start and removes a runtime dependency
#    on an unreliable host.
#    TRAP guarded here: if the checkpoint is missing/corrupt, upstream
#    RNAFMEmbeddings silently falls back to a 6-dim _get_simple_features
#    representation with NO error. The sha256 check makes that impossible.
# ---------------------------------------------------------------------------
#    PATH TRAP (verified empirically, not assumed): torch.hub.get_dir() is
#    $TORCH_HOME/**hub**, not $TORCH_HOME. The checkpoint must land in
#    $TORCH_HOME/hub/checkpoints/ or `fm` re-downloads from the dead CUHK URL.
RUN mkdir -p ${TORCH_HOME}/hub/checkpoints \
    && curl -fsSL -o ${TORCH_HOME}/hub/checkpoints/RNA-FM_pretrained.pth \
        https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth \
    && echo "${RNAFM_SHA256}  ${TORCH_HOME}/hub/checkpoints/RNA-FM_pretrained.pth" \
        | sha256sum -c - \
    && chmod -R a+rX ${TORCH_HOME}

# ---------------------------------------------------------------------------
# 5. Patched harness + calibration runner.
# ---------------------------------------------------------------------------
ENV OLIGOGYM_BENCH=/opt/oligogym-bench
COPY train_model_patched.py oligogym_patch.py calibrate.py gpu_pack.py \
     validate_image.py ${OLIGOGYM_BENCH}/
ENV PYTHONPATH=${OLIGOGYM_BENCH}
ENV PATH=${OLIGOGYM_BENCH}:${PATH}

# ---------------------------------------------------------------------------
# 6. Build-time validation. Fails the build on: a missing import, a torch wheel
#    without sm_75, or an RNA-FM checkpoint that does not load into a real
#    99.5M-parameter model. GPU-execution checks are skipped here (no GPU on a
#    build runner) and run on the first g4dn task instead.
# ---------------------------------------------------------------------------
RUN python ${OLIGOGYM_BENCH}/validate_image.py --build-time

WORKDIR /work
