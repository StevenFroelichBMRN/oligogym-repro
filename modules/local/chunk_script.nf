/*
 * The chunk command line, shared verbatim by the CPU and GPU processes so the two
 * queues cannot drift apart in behaviour.  The only differences between them are
 * the resource label and --require-cuda.
 *
 * This lives in a module (not a params closure and not a top-level def) for two
 * measured reasons: Nextflow >=25 rejects top-level statements mixed with script
 * declarations, and Seqera Platform serializes every param into the run record --
 * a closure in params aborts the head job with
 * "Unable to serialize key=workflow.params.chunk_script".
 */
def chunkScript(meta, chunk_json, configs_csv, task, require_cuda, params) {

    def seed_flag  = params.seed != null ? "--seed ${params.seed}" : ''
    def kfold_flag = params.proper_kfold ? '--proper-kfold' : ''
    def cuda_flag  = require_cuda ? '--require-cuda' : ''
    """
    set -euo pipefail

    # Thread caps: a chunk runs ${meta.procs} fold-fits concurrently, so an
    # unconstrained BLAS would oversubscribe the box ${meta.procs}-fold.  One
    # thread per worker is what the Phase 2 packing measurement was made under.
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    # Provenance, written into every output row rather than inferred later.
    export OLIGOGYM_IMAGE_DIGEST='${params.image_digest}'
    export OLIGOGYM_GIT_SHA='${params.git_sha}'
    export OLIGOGYM_UPSTREAM_SHA='${params.upstream_sha}'

    # The RNA-FM checkpoint is baked into the image under this TORCH_HOME, so
    # fm.pretrained.rna_fm_t12() loads from cache; its canonical host returns
    # 403 and a 1.2 GB fetch per container start would be wasteful.
    export TORCH_HOME=\${TORCH_HOME:-/opt/torch-hub}

    LOG='${meta.chunk_id}.log'
    {
      echo "[task] chunk=${meta.chunk_id} arm=${params.arm} queue=${meta.queue}"
      echo "[task] tier=${meta.tier} mem_class=${meta.mem_class} procs=${meta.procs}"
      echo "[task] n_configs=${meta.n_configs} est_minutes=${meta.est_min}"
      echo "[task] attempt=${task.attempt} cpus=${task.cpus} memory=${task.memory}"
      echo "[task] hostname=\$(hostname) uname=\$(uname -srm)"
      echo "[task] FUSION_GPU_USED=\${FUSION_GPU_USED:-unset}"
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \\
        || echo "[task] nvidia-smi unavailable"
    } >> "\$LOG" 2>&1

    run_chunk.py \\
        --chunk ${chunk_json} \\
        --configs ${configs_csv} \\
        --out '${meta.chunk_id}.parquet' \\
        --summary '${meta.chunk_id}_summary.json' \\
        --fold-indices '${meta.chunk_id}_folds.csv' \\
        --cache-dir "./feature_cache_${meta.chunk_id}" \\
        --procs ${meta.procs} \\
        --arm ${params.arm} \\
        ${seed_flag} ${kfold_flag} ${cuda_flag} \\
        >> "\$LOG" 2>&1

    echo "[task] done rc=0" >> "\$LOG"
    """
}
