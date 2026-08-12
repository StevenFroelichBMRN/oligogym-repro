# Smoke run report — OligoGym reproduction pipeline

Phase 3 validation. Every number below is measured from the Seqera run records
of real AWS Batch tasks or from the pipeline's own published outputs; nothing is
projected. Where something was **not** verified it is stated plainly and listed
in §7.

Pipeline repo: `github.com/StevenFroelichBMRN/oligogym-repro`
Workspace: `seqera-poc` (`234253664513050`), us-west-2.

---

## 1. What was launched

Four chunks were selected to span the required axes at ~1 % of the sweep
(450 fold-fits of the 45,940 the pipeline will run).

**On the two totals used in this report**, which are different quantities and are
not interchangeable: the published CSV holds **9,190 configs / 45,950 rows**, and
the pipeline *runs* **9,188 configs / 45,940 fold-fits** because 2 configs
(RNA-FM x sherwood) are skipped by default as expected-infeasible and recorded in
`skips_primary.csv`. Join figures in §5 are against the published 9,190/45,950;
chunk and workload figures are against the 9,188/45,940 actually executed.

| role | chunk_id | queue | tier | dataset | featurizer | cv | models | configs | fold-fits |
|---|---|---|---|---|---|---|---|---|---|
| cpu | `prim_68992813921575a9` | cpu | L | asoptimizer_cleaned | KMersCounts | random | Linear, KNN, RF, XGB | 23 | 115 |
| gpu non-RNA-FM | `prim_8d24dcf3f154c775` | gpu | M | acute_neurotox_lna | OneHotEncoder | nucleobase | CNN, GRU, MLP, Transformer | 64 | 320 |
| gpu RNA-FM | `prim_e30b16c0498d005d` | gpu | L | asoptimizer_cleaned | RNAFMEmbeddings | random | Transformer | 1 | 5 |
| XL | `prim_4163cf3ef1fc0327` | cpu | XL | sherwood | KMersCounts | nucleobase | XGBoost | 2 | 10 |

Both `cross_validation` modes are covered (`random`, `nucleobase`), both queues,
both memory classes of interest (`rnafm16`, `xl_tree_hi`).

## 2. Runs

| workflowId | arm/queue | outcome | note |
|---|---|---|---|
| `otGYSdCXpXJuw` | gpu | FAILED (head) | params closure not serializable — fixed |
| `1Sx6GFmNaJD4u9` | cpu | FAILED (head) | same |
| `3WMwe1C8wdznnU` / `3zAVXDoBh67Xaq` | gpu/cpu | FAILED (head) | relative asset paths — fixed with `${projectDir}` |
| `2vY2ztJUN5Bifu` / `34gNJEHFkF1TRW` | gpu/cpu | FAILED | Wave rejects digest-pinned image + Fusion — fixed |
| **`3ex9nFsLPH6632`** | **gpu** | **SUCCEEDED** | RNA-FM chunk completed on a real T4; one chunk exit-3 |
| **`20IvkF1n5EegXY`** | **cpu** | **SUCCEEDED** | 4/4 tasks, incl. XL/sherwood; join verified |
| `5SFypwqJEWjFhw`, `2dKGKKpI4or2WI`, `2gc75tQ5OPUGAP` | gpu | partial | successive fixes to the DL-chunk failure (§4) |
| **`56f9n797FOE8GW`** | **gpu** | **SUCCEEDED** | spawn fix: DL chunk `fitted=64 errors=0` × 5 folds on a real T4 |
| **`2hBCAjn6QWweSq`** | **gpu** | **SUCCEEDED** | clean `-resume`: `cached=3`, nothing re-executed |
| `5afjI3qeYTO5QT` | cpu | inconclusive | resume with an intervening commit — see §6 |

## 3. Per-task evidence (measured)

| chunk | status | instance | realtime | peak RSS | GPU |
|---|---|---|---|---|---|
| `prim_e30b16c0498d005d` (RNA-FM) | COMPLETED | **g4dn.2xlarge** | 1,417 s | **9,051.7 MB** | `FUSION_GPU_USED=true` |
| `prim_68992813921575a9` (cpu L) | COMPLETED | c6id.24xlarge | 1,535 s | 16,002.2 MB | n/a |
| `prim_4163cf3ef1fc0327` (XL sherwood) | COMPLETED | c6id.24xlarge | 3,967 s | 7,447.4 MB | n/a |
| `prim_8d24dcf3f154c775` (gpu DL) | **COMPLETED** after fix | g4dn.2xlarge | **895 s** | **14,251.6 MB** | `Tesla T4`, `fitted=64 errors=0` × 5 folds |

The GPU DL chunk — 64 CNN/GRU/MLP/Transformer configs at `procs=8` — completed **320/320 fold-fits with zero
errors** on a Tesla T4 after the §4 fixes (`56f9n797FOE8GW`). Its task peaked at
**14,251.6 MB RSS against a 12 GB request**, i.e. it exceeded its own memory
declaration and was placed only because Batch chose a much larger instance. See
§7 — this is the most important resource correction for Phase 4.

**CUDA device confirmed.** The chunk runner's own probe, from inside a task on
the GPU queue:

```json
{"cuda_available": true, "device_name": "Tesla T4", "torch_version": "2.13.0+cu126",
 "capability": [7, 5], "device_total_mb": 14911.7,
 "arch_list": ["sm_50","sm_60","sm_70","sm_75","sm_80","sm_86","sm_90"]}
```

`FUSION_GPU_USED=true` appears in the GPU task logs, and Fusion reported
`upload: 9624.08MB, sync:8` for the RNA-FM task — outputs reached S3.

**Two measurements worth carrying into Phase 4:**

* RNA-FM peaked at **9,051.7 MB RSS**, closely matching Phase 2's 9.66 GB
  prediction at the L tier. The ≥16 GB allocation is correct and not padded.
* The XL/sherwood chunk ran **3,967 s** against a predicted 41.5 min (2,490 s) —
  1.6× over. Sub-linear scaling still holds versus the 8.94× row ratio, but the
  XL tail is underestimated; see §7.

**Instance sizes were NOT pinned.** GPU tasks landed on `g4dn.2xlarge` as
intended, but CPU tasks landed on **`c6id.24xlarge`** — the compute environment
lists families only, so Batch chose the size. This is exactly the attribution
problem Phase 2 flagged. The pipeline requests 8 cpus and a memory ladder, which
constrains but does not pin. **Phase 4 should pin sizes in the compute
environments themselves before the sweep**, or per-chunk unit costs will again be
unattributable.

**How representative is that one GPU chunk?** Of the 175 primary chunks, 111 are
GPU-queue and **93 are multi-worker GPU** (`procs=8`: 66, `procs=7`: 27; the
other 18 are RNA-FM at `procs=1`), covering 4,414 configs. The tested chunk is
one of the **20** multi-worker GPU chunks at the **M** tier. The 44 L-tier and 27
XL-tier multi-worker GPU chunks have **not** been exercised, and under `spawn`
their per-worker matrix copies are larger — see §7 item 1.

## 4. The exit-3 chunk — root cause

`prim_8d24dcf3f154c775` (64 CNN/GRU/MLP/Transformer configs, `procs=8`) reported
`fitted=64 errors=64` on all five folds and exited 3 — the runner's own "every
config in this chunk failed" code. **It is a pipeline bug, not a config-level or
data failure**, and it had two sequential causes — the second only became visible
once the first was fixed:

1. **Daemonic workers.** `multiprocessing.Pool` marks its workers daemonic, and a
   daemonic process may not have children. Every DL model here trains through
   `pl.Trainer`, which starts child processes, so under `Pool` every DL config
   dies with `AssertionError: daemonic processes are not allowed to have
   children`. Reproduced locally and confirmed by grep of the model source.
   Fixed by replacing `Pool` with plain non-daemonic `mp.Process` workers.
2. **CUDA after fork.** With that fixed, the same chunk failed on the T4 with
   `AcceleratorError: CUDA error: initialization error` ×64 — a forked child
   cannot use CUDA if the parent ever initialised a context. **The specific import
   that initialises it in the parent was never identified**, so unlike cause 1
   this diagnosis rests on the error text and on the fix working, not on an
   isolated reproduction of the mechanism. Moving the device probe out-of-process
   was not sufficient. Rather than spend further cloud round-trips bisecting
   imports, **GPU chunks now use the `spawn` start method** (children
   re-import and receive state pickled, so there is no inherited context), while
   **CPU chunks keep `fork`** and its copy-on-write sharing of the feature matrix.
   Cost of spawn is one pickle of the cached matrices per worker — ~14.6 MB for
   this chunk.

Note the failure mode was *quiet*: `errorStrategy = ignore` meant the chunk lost
only itself and its siblings completed, which is the designed behaviour, but the
per-config error text lived only in the unpublished parquet. Two diagnosability
fixes were made and are now in the pipeline: the task script **tees** its log
(a failed task publishes nothing, so a redirect-only log is unreadable), and the
runner prints **distinct error signatures with counts** per fold.

**Status of the fix: VERIFIED on a real T4.** Run `56f9n797FOE8GW`, same chunk,
same 64 configs, `procs=8`:

```
[device] {"cuda_available": true, "device_name": "Tesla T4", ...}
[... fold 0] fitted=64 errors=0
[... fold 1] fitted=64 errors=0
[... fold 2] fitted=64 errors=0
[... fold 3] fitted=64 errors=0
[... fold 4] fitted=64 errors=0
RUN_CHUNK_GPU COMPLETED exit=0 rt=895s
```

320/320 fold-fits, zero errors, and the chunk's COLLECT step joined 64/64 configs
to the published CSV at 100 %. The daemonic-pool fix is additionally verified
locally on the fork path (115/115 rows, 23 configs × 5 folds, zero errors).

## 5. Join verification — the critical result

Verified on **both** cloud arms' own COLLECT steps, joining reproduced output to
`benchmarks/oligogym_benchmarks.csv` on a recomputed `config_hash`.

**GPU run (`56f9n797FOE8GW`), 64 DL configs:**

```json
{"reproduced_rows": 320, "reproduced_configs": 64,
 "configs_matched": 64, "configs_unmatched": 0, "join_rate_pct": 100.0,
 "worker_hash_agrees_pct": 100.0, "schema_matches_published_prefix": true,
 "value_comparison": {"mean_abs_delta_test_pearson": 0.07322,
                      "max_abs_delta_test_pearson": 0.44372}}
```

**CPU run (`20IvkF1n5EegXY`), 25 classical configs incl. XL/sherwood:**

```json
{"published_rows": 45950, "published_configs": 9190,
 "reproduced_rows": 125, "reproduced_configs": 25,
 "configs_matched": 25, "configs_unmatched": 0, "join_rate_pct": 100.0,
 "worker_hash_agrees_pct": 100.0, "worker_hash_disagreements": 0,
 "value_comparison": {"configs": 25,
   "mean_abs_delta_test_pearson": 0.00956,
   "max_abs_delta_test_pearson": 0.07231,
   "folds_per_config_pub": [5], "folds_per_config_rep": [5]}}
```

* **`schema_matches_published_prefix: true`** — the first 18 columns are exactly
  the published schema, in order.
* **100.0 % join rate**, zero unmatched configs.
* **`worker_hash_agrees_pct: 100.0`** — the hash the worker wrote and the hash
  the collector recomputes from the published-schema columns agree on every row,
  so the manifest's canonicalization and the collector's have not drifted.
* Value agreement: mean |Δ| in test Pearson of **0.0096** for the classical
  models (max 0.072) and **0.0732** for the DL models (max 0.444). Per-fold
  equality is *not* expected — the reproduction is unseeded by design, matching
  the published methodology — so distribution-level agreement is the correct test.
  The classical result is strong. **The DL spread is wider and Phase 4 should
  treat it as an open question**, not a pass: unseeded DL training varies over
  initialisation as well as folds, so a larger sample is needed to separate
  ordinary run-to-run variance from a systematic offset. One 64-config chunk on
  one dataset cannot make that call.

Independently verified offline on the full tables: **9,190/9,190** manifest
hashes reproduced exactly, **9,190/9,190** published configs and
**45,950/45,950** published rows join with zero on either side unmatched, and
**100/100** distinct arg strings round-trip byte-exact.

## 6. Resume — verified on AWS Batch

Relaunching the succeeded GPU workflow with `resume: true`, its `sessionId`, its
`workDir` and **no intervening commit** (`pullLatest: false`):

```
workflow 2hBCAjn6QWweSq  ->  SUCCEEDED, cached=3, succeeded=0
  SPLIT_CONFIGS  CACHED
  RUN_CHUNK_GPU  CACHED   (895 s of GPU training NOT repeated)
  COLLECT        CACHED
```

All three tasks were served from cache and nothing re-executed. Also verified
locally, where a second `-resume` launch likewise reported `cached: 1` for every
process.

**One operational caveat worth knowing.** An earlier Batch resume attempt
(`5afjI3qeYTO5QT`) reported `cached=0` and re-ran its chunks. That was *correct*:
`pullLatest=true` had fetched a changed `run_chunk.py`, so the task hash
legitimately differed. **Resume only skips work when the pipeline code is
unchanged** — expect a full re-run of a sweep if you push a commit mid-flight.

## 7. Not verified — read this before Phase 4

1. **GPU task memory under spawn — re-measure.** The one observed spawn-mode GPU
   chunk peaked at **14,251.6 MB against a 12 GB request** and was placed only
   because Batch chose a larger instance. The first rung is now **24 GB**, but
   with spawn each worker holds its own copy of the feature matrices, so GPU task
   RSS scales with `procs × matrix size` rather than being nearly flat in it as
   the fork-based Phase 2 measurement assumed. **A large-matrix GPU chunk at
   `procs=8` has never been run.** Phase 4 should watch the first XL/L GPU chunks
   for exit 137 and be ready to reduce `procs` for the GPU queue.
2. **Cache hit-rate in production.** Every smoke chunk was a cold first run, so
   the observed hit rate is 0 % by construction — the cache *stores* correctly
   (`bytes_written` non-zero, `write_errors: 0`) and re-hit was verified locally
   (5/5 folds `cache_hit=True`, `featurize_s=0.00`). The 97.6 % projection is
   Phase 2's, not re-measured here.
3. **Instance-size pinning** (§3) — requires editing the compute environments.
4. **XL cost model.** The one XL chunk ran 1.6× its estimate. The XL fit+predict
   extrapolation rests on a median ratio of 1.43 with a wide measured spread
   (0.97–2.84), and **RandomForest × XL is both unmeasured and a prior OOM
   casualty** — the least reliable cell in the plan.
5. **RNA-FM × sherwood** is skipped by default as expected-infeasible; not tested.
6. **The corrective arm was not run on Batch.** Its three transform paths were
   verified locally end-to-end with correct provenance (§8).
7. **120 GB retry rung** exceeds the pinned instance sizes; never exercised.
8. **Memory-ladder escalation on exit 137** was never triggered, so the retry
   ladder itself is untested end-to-end. The `ignore`-after-retries path WAS
   exercised (the exit-3 chunk) and behaved correctly: siblings completed and
   published.

## 8. Corrective arm — verified locally

All three paths ran end-to-end with correct `correction_applied` /
`correction_level` provenance on every output row:

| dataset / featurizer | applied | level | evidence |
|---|---|---|---|
| TLR7 × KMers | `tlr_key_swap` | `complete` | logged `from_key=TLR7 → to_key=tlr8`; RMSE ≈ 8.6, consistent with the 17.46–126.5 label range, not 0.57–5.44 |
| MOE × KMers | `moe_fasta_base_sequence` | `partial_base_composition_only` | 200/200 rows differ from faithful; per-row k-mer total rises by exactly 3 (one monomer × k∈{1,2,3}); modification columns byte-identical |
| MOE × OneHot | `none` | `none_helm_truncated` | no transform applied; runs identically to the faithful arm as the sensitivity-analysis control |

Implemented option **(c)**: the KMers base-composition partial correction *plus*
documented sensitivity analysis. No HELM extension is fabricated. Full reasoning
and limitations in `pipeline_notes.md` §9.

## 9. Launch invocation for the full sweep

```bash
# PRIMARY arm — 9,188 configs / 45,940 fold-fits / 175 chunks
nextflow run StevenFroelichBMRN/oligogym-repro -r main -profile batch \
  --arm primary \
  --chunks      '${projectDir}/assets/chunks_primary.csv' \
  --assignments '${projectDir}/assets/chunk_assignments_primary.csv' \
  --outdir  s3://r6333-pep-nppc-oi-bmn333-dev/oligogym-repro \
  --version v1 \
  --git_sha <commit> \
  -resume

# CORRECTIVE arm — 2,334 configs / 11,670 fold-fits / 10 chunks
nextflow run StevenFroelichBMRN/oligogym-repro -r main -profile batch \
  --arm corrective \
  --chunks      '${projectDir}/assets/chunks_corrective.csv' \
  --assignments '${projectDir}/assets/chunk_assignments_corrective.csv' \
  --outdir  s3://r6333-pep-nppc-oi-bmn333-dev/oligogym-repro \
  --version v1 \
  -resume
```

Via the Seqera API, the equivalent `paramsText` plus
`configProfiles: ["batch"]`, `computeEnvId` = **`5ZEM2WRyxXMWtRFVwpxwaz`**
(`batch-gpu_copy` — never `fJ1Tu2lZEwf2cSF54nm2v`, which Phase 2 found wedged).

**Asset paths must use `${projectDir}`** — a bare relative path resolves against
the launch directory and fails at `checkIfExists`. Measured, twice.

Start with `OLIGOGYM_QUEUE_SIZE=32` (the default). Raise only after tasks are
observed starting: the g4dn quota (`L-DB2E81BA`) is unconfirmed and 16 ×
`g4dn.2xlarge` needs 128 vCPU. **No quota block was observed** in this smoke run —
GPU tasks were placed on `g4dn.2xlarge` within ~3–5 minutes of submission — but
the smoke run never asked for more than two GPU instances at once, so the ceiling
is untested.

Recommended order: run the **corrective arm first** (10 chunks, ~2.8 h packed,
cheap) as a production shakedown, then the primary arm.
