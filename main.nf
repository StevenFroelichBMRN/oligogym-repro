#!/usr/bin/env nextflow
/*
 * OligoGym benchmark reproduction -- PRODUCTION pipeline (Phase 3).
 *
 * Reproduces the 9,190-config / 45,950-fold-fit published target (arm=primary)
 * and the 2,334-config defect-corrected re-run of TLR7/TLR8/Neurotox MOE
 * (arm=corrective).  The two arms are separate launches with separate output
 * prefixes -- never one undifferentiated blob -- because they answer different
 * questions: primary asks "do we get the published numbers", corrective asks
 * "what changes once the data defects are fixed".
 *
 * Shape:
 *   PARTITION  one task, CPU, cheap: manifest -> chunk descriptors
 *   RUN_CHUNK  N tasks: featurize once per (feature group, fold), fit every
 *              config in the group against it.  cpu chunks -> CPU Batch queue,
 *              gpu chunks -> GPU Batch queue, via two process aliases.
 *   COLLECT    one task: concatenate to one table, join the published CSV
 *
 * Why chunks and not one task per fold-fit: featurization is 55.2 % of per-fold
 * wall time (measured), and the primary target's 9,190 configs share only 239
 * distinct feature matrices -- the largest shared by 87 configs.  Grouping by
 * feature matrix and caching within the task eliminates 97.6 % of featurization
 * work.  See pipeline_notes.md for the full chunking arithmetic.
 *
 * Idempotence / resume:
 *   * every output is keyed by chunk_id, and chunk_id is a digest over the arm,
 *     the group key, the resource class and the sorted config_hash list -- so an
 *     unchanged chunk has an unchanged task hash and `-resume` skips it;
 *   * per-chunk publishing means a failed chunk loses only itself; siblings are
 *     already on S3;
 *   * `errorStrategy` retries OOM (137) and preemption up the memory ladder,
 *     then ignores, so one bad chunk cannot fail the run.
 */

nextflow.enable.dsl = 2

// ---- inputs ---------------------------------------------------------------
params.manifest     = "${projectDir}/assets/run_manifest.parquet"
params.calibration  = "${projectDir}/assets/calibration.csv"
params.published    = "${projectDir}/assets/oligogym_benchmarks.csv"

// pre-computed chunk manifests; when set, PARTITION is skipped entirely so a
// full sweep runs against exactly the chunk set that was reviewed beforehand.
params.chunks       = null
params.assignments  = null

// ---- what to run ----------------------------------------------------------
params.arm            = 'primary'      // 'primary' | 'corrective'
params.compute_class  = 'all'          // 'all' | 'cpu' | 'gpu'
params.only_datasets  = null           // comma-separated dataset_config_key
params.only_chunks    = null           // comma-separated chunk_id (smoke runs)
params.max_chunks     = null           // cap the number of chunks (smoke runs)
params.folds          = 5

// ---- chunk sizing (measured defaults; see pipeline_notes.md) --------------
params.target_chunk_s = 1800           // aim for 30 min of work per task
params.max_chunk_s    = 12600          // 3.5 h hard cap, under the 4 h timeout
params.min_chunk_s    = 600            // below this, whole groups are bin-packed
params.skip_rnafm_xl  = true           // RNA-FM x sherwood: expected-infeasible

// ---- determinism ----------------------------------------------------------
// UNSEEDED by default, matching the published methodology: upstream seeds
// nothing, and seeding changes which folds and nucleobase clusters are drawn.
// --seed makes a run repeatable while preserving the published resampling
// scheme; --proper_kfold additionally replaces the per-fold reshuffle with a
// true partition and is a DOCUMENTED DIVERGENCE, for sensitivity analysis only.
params.seed         = null
params.proper_kfold = false

// ---- output ---------------------------------------------------------------
params.outdir       = 's3://r6333-pep-nppc-oi-bmn333-dev/oligogym-repro'
params.version      = 'v1'
params.publish_mode = 'copy'

// ---- provenance ----------------------------------------------------------
params.image_digest = 'ghcr.io/stevenfroelichbmrn/oligogym-bench@sha256:c1bb023c3c5f317e1b071b97af0e0d5608d571bbaf541172b689bab7333ab8a8'
params.git_sha      = 'unset'
params.upstream_sha = '97f5b9f58d9e445a8ecb0218037af7465c3708c0'

// Output prefix and the shared chunk command line live in params/closures rather
// than top-level statements: Nextflow >=25 rejects statements mixed with script
// declarations, and a closure in params is visible to every process body.
params.results_prefix = "${params.outdir}/${params.version}/${params.arm}"

/*
 * The chunk command line, shared verbatim by the CPU and GPU processes so the
 * two queues cannot drift apart in behaviour.  The only differences between them
 * are the resource label and --require-cuda.
 */
params.chunk_script = { meta, chunk_json, configs_csv, task, require_cuda ->
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

// ==========================================================================
process PARTITION {
    label 'partition'
    tag "${params.arm}"
    publishDir "${params.results_prefix}/chunks", mode: params.publish_mode

    input:
    path manifest
    path calibration

    output:
    path "chunks_${params.arm}.csv",             emit: chunks
    path "chunk_assignments_${params.arm}.csv",  emit: assignments
    path "skips_${params.arm}.csv",              emit: skips, optional: true

    script:
    def only_ds  = params.only_datasets ? "--only-datasets '${params.only_datasets}'" : ''
    def only_ch  = params.only_chunks   ? "--only-chunks '${params.only_chunks}'"     : ''
    def max_ch   = params.max_chunks    ? "--max-chunks ${params.max_chunks}"         : ''
    def rnafm_xl = params.skip_rnafm_xl ? '' : '--no-skip-rnafm-xl'
    """
    partition_manifest.py \\
        --manifest ${manifest} \\
        --calibration ${calibration} \\
        --arm ${params.arm} \\
        --outdir . \\
        --folds ${params.folds} \\
        --target-chunk-s ${params.target_chunk_s} \\
        --max-chunk-s ${params.max_chunk_s} \\
        --min-chunk-s ${params.min_chunk_s} \\
        ${rnafm_xl} ${only_ds} ${only_ch} ${max_ch}
    """
}

// One process definition, two aliases: identical script, different resource
// labels, so the CPU and GPU Batch queues are selected by process selector in
// nextflow.config rather than by branching inside the script.
process RUN_CHUNK_CPU {
    label 'chunk_cpu'
    tag "${meta.chunk_id}:${meta.tier}:${meta.n_configs}cfg"
    publishDir "${params.results_prefix}/results",   mode: params.publish_mode, pattern: '*.parquet'
    publishDir "${params.results_prefix}/summaries", mode: params.publish_mode, pattern: '*_summary.json'
    publishDir "${params.results_prefix}/folds",     mode: params.publish_mode, pattern: '*_folds.csv'
    publishDir "${params.results_prefix}/logs",      mode: params.publish_mode, pattern: '*.log'

    input:
    tuple val(meta), path(chunk_json), path(configs_csv)

    output:
    tuple val(meta), path("${meta.chunk_id}.parquet"),      emit: results,   optional: true
    path "${meta.chunk_id}_summary.json",                   emit: summaries, optional: true
    path "${meta.chunk_id}_folds.csv",                      emit: folds,     optional: true
    path "${meta.chunk_id}.log",                            emit: logs,      optional: true

    script:
    params.chunk_script(meta, chunk_json, configs_csv, task, false)
}

process RUN_CHUNK_GPU {
    label 'chunk_gpu'
    tag "${meta.chunk_id}:${meta.tier}:${meta.n_configs}cfg"
    publishDir "${params.results_prefix}/results",   mode: params.publish_mode, pattern: '*.parquet'
    publishDir "${params.results_prefix}/summaries", mode: params.publish_mode, pattern: '*_summary.json'
    publishDir "${params.results_prefix}/folds",     mode: params.publish_mode, pattern: '*_folds.csv'
    publishDir "${params.results_prefix}/logs",      mode: params.publish_mode, pattern: '*.log'

    input:
    tuple val(meta), path(chunk_json), path(configs_csv)

    output:
    tuple val(meta), path("${meta.chunk_id}.parquet"),      emit: results,   optional: true
    path "${meta.chunk_id}_summary.json",                   emit: summaries, optional: true
    path "${meta.chunk_id}_folds.csv",                      emit: folds,     optional: true
    path "${meta.chunk_id}.log",                            emit: logs,      optional: true

    script:
    params.chunk_script(meta, chunk_json, configs_csv, task, true)
}

process COLLECT {
    label 'collect'
    tag "${params.arm}"
    publishDir "${params.results_prefix}/collected", mode: params.publish_mode

    input:
    path parquets
    path summaries
    path published

    output:
    path "oligogym_repro_${params.arm}.parquet", emit: table
    path "oligogym_repro_${params.arm}.csv",     emit: csv
    path "published_shape_${params.arm}.csv",    emit: published_shape
    path "collection_report_${params.arm}.json", emit: report

    script:
    def pub = params.arm == 'primary' ? "--published-csv ${published}" : ''
    """
    collect_results.py \\
        --inputs ${parquets} \\
        --summaries ${summaries} \\
        --out oligogym_repro_${params.arm}.parquet \\
        --out-csv oligogym_repro_${params.arm}.csv \\
        --published-shape published_shape_${params.arm}.csv \\
        --report collection_report_${params.arm}.json \\
        --arm ${params.arm} \\
        ${pub}
    """
}

// ==========================================================================
workflow {

    if (!(params.arm in ['primary', 'corrective'])) {
        error "params.arm must be 'primary' or 'corrective', got '${params.arm}'"
    }
    if (params.proper_kfold && params.seed == null) {
        error "--proper_kfold requires --seed (it needs a fixed permutation)"
    }

    // ---- chunk descriptors: staged in, or computed here ------------------
    if (params.chunks && params.assignments) {
        ch_chunks      = Channel.fromPath(params.chunks,      checkIfExists: true)
        ch_assignments = Channel.fromPath(params.assignments, checkIfExists: true)
    }
    else {
        PARTITION(
            Channel.fromPath(params.manifest,    checkIfExists: true),
            Channel.fromPath(params.calibration, checkIfExists: true)
        )
        ch_chunks      = PARTITION.out.chunks
        ch_assignments = PARTITION.out.assignments
    }

    // ---- materialise one descriptor + one config CSV per chunk -----------
    // Done in ONE task rather than one per chunk: 175 extra Batch tasks to write
    // 175 small CSVs would cost more in scheduling than the split itself, and
    // the single task's output files are content-addressed, so `-resume` still
    // skips a downstream chunk whose inputs are byte-identical.
    SPLIT_CONFIGS(ch_chunks, ch_assignments)

    // Scheduling metadata comes from SPLIT_CONFIGS's slim schedule.csv, which
    // has no free-text or JSON columns.  The full chunk manifest cannot be read
    // here: Nextflow's splitCsv does not handle RFC-4180 doubled quotes, and the
    // featurizer_args JSON column contains them ("{""flatten"":false}").
    ch_meta = SPLIT_CONFIGS.out.schedule
        .splitCsv(header: true)
        .map { row ->
            [
                row.chunk_id,
                [
                    chunk_id  : row.chunk_id,
                    arm       : row.arm,
                    queue     : row.queue,
                    tier      : row.dataset_size_tier,
                    mem_class : row.mem_class,
                    mem_gb    : (row.mem_gb_start as Integer),
                    procs     : (row.procs as Integer),
                    n_configs : (row.n_configs as Integer),
                    est_min   : (row.est_minutes as Double)
                ]
            ]
        }

    // key each emitted pair of files by chunk_id, then join to the metadata
    ch_files = SPLIT_CONFIGS.out.descriptors
        .flatten()
        .map { f -> [f.simpleName.replaceFirst(/^chunk_/, ''), f] }
        .join(
            SPLIT_CONFIGS.out.configs
                .flatten()
                .map { f -> [f.simpleName.replaceFirst(/^configs_/, ''), f] }
        )

    ch_sliced = ch_meta.join(ch_files).map { id, meta, cj, cfg -> [meta, cj, cfg] }

    ch_cpu = ch_sliced.filter { meta, cj, cfg -> meta.queue == 'cpu' }
    ch_gpu = ch_sliced.filter { meta, cj, cfg -> meta.queue == 'gpu' }

    RUN_CHUNK_CPU(ch_cpu)
    RUN_CHUNK_GPU(ch_gpu)

    ch_results = RUN_CHUNK_CPU.out.results.mix(RUN_CHUNK_GPU.out.results).map { it[1] }
    ch_sums    = RUN_CHUNK_CPU.out.summaries.mix(RUN_CHUNK_GPU.out.summaries)

    COLLECT(
        ch_results.collect(),
        ch_sums.collect(),
        Channel.fromPath(params.published, checkIfExists: true)
    )
}

process SPLIT_CONFIGS {
    label 'partition'
    tag "${params.arm}"

    input:
    path chunks_csv
    path assign_csv

    output:
    path "schedule.csv",   emit: schedule
    path "chunk_*.json",   emit: descriptors
    path "configs_*.csv",  emit: configs

    script:
    def only_ch = params.only_chunks ? "--only-chunks '${params.only_chunks}'" : ''
    def max_ch  = params.max_chunks  ? "--max-chunks ${params.max_chunks}"     : ''
    """
    split_chunks.py \\
        --chunks ${chunks_csv} \\
        --assignments ${assign_csv} \\
        --outdir . \\
        --compute-class ${params.compute_class} \\
        ${only_ch} ${max_ch}
    """
}
