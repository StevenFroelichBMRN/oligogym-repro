# OligoGym reproduction — production pipeline design notes

Phase 3 deliverable. Every constraint encoded in the pipeline is listed here with
the measurement that justifies it. Measurements come from Phase 2
(`calibration.csv`, 2,017 successful fold-fits on this project's own AWS Batch
hardware) and from Phase 1's audit of the pinned tree
(`97f5b9f58d9e445a8ecb0218037af7465c3708c0`). Where a number is an
extrapolation, it says so.

---

## 1. What is being reproduced

The authoritative target is the **published CSV**, not the paper's run count.
Phase 1 established that the paper's claimed 10,657 runs is not reconstructible
(the number is prime, is odd while the grid is symmetric in the two
`cross_validation` values, and of 208 arithmetic subset solutions none is
consistent with the CSV's own evidence). `benchmarks/oligogym_benchmarks.csv`
contains 45,950 rows = **9,190 unique configs × exactly 5 folds**, and the
manifest joins it at 100.0000 %.

| arm | configs | fold-fits | question it answers |
|---|---|---|---|
| `primary` | 9,188 run + 2 skipped | 45,940 | do we recover the published numbers? |
| `corrective` | 2,334 | 11,670 | what changes once the verified data defects are fixed? |

The two arms are **separate launches with separate output prefixes**
(`{outdir}/{version}/{arm}/…`). They are not run as one blob because they are not
comparable: the corrective arm deliberately changes the input data.

---

## 2. The chunking rule, and the arithmetic behind it

### 2.1 Why not one task per fold-fit

45,940 Batch tasks, each pulling a container, to run work whose S-tier unit cost
is **0.8–4.6 s** (measured). Scheduling and image pull would dominate. Worse, it
would discard the largest available lever.

### 2.2 The lever

Measured: **featurization is 55.2 % of per-fold wall time**. The redundancy is
*across configs*, not across folds — the 9,190-config target contains only
**239 distinct `(dataset, featurizer, featurizer_args, cross_validation)`
groups**, the largest shared by 87 configs, because every
Transformer/CNN/GRU hyperparameter variant re-featurizes an identical matrix.

So the chunk key is exactly the feature-cache key minus the fold:

```
(dataset_downloader_key, featurizer, featurizer_args_json, cross_validation)
```

A chunk featurizes once per (group, fold) and fits every config in the group
against that matrix. Measured A/B at replay depth 12 saved **91.5–91.9 %** of
featurization time; the analytical projection at the true sharing depth is
**97.6 %**, i.e. ~54 % of total wall-clock.

### 2.3 Sizing

Per-chunk wall time is modelled as two components with **different scaling**:

```
fixed_s = n_folds × featurize_s(dataset, featurizer, cv)     # paid ONCE per chunk
var_s   = n_folds × Σ fitpred_s(model_class, tier) / speedup(procs)
```

`fixed_s` is paid *again* by every chunk a group is split into. Sherwood ×
OneHotEncoder measures **889.4 s per fold** = 74 minutes of pure featurization
per chunk regardless of how few configs it holds. So the cap is not a constant:

```
cap = clamp(2 × fixed_s, target_chunk_s = 1800 s, max_chunk_s = 12600 s)
```

— aim for 30 minutes, never let featurization exceed ~50 % of a chunk, never
exceed 3.5 h (under the 4 h task timeout, with margin over the longest single
measured config, **3,526 s** XGBoost on Sherwood). Configs are packed
largest-first so the leftover bin is the small one.

**Undersized chunks are then bin-packed** (`--min-chunk-s`, default 600 s).
Merging combines *whole groups* and never splits one, so it cannot reduce cache
hits — a task holding three groups featurizes three matrices, each exactly once.
What it buys is task count: measured Batch cold start is ~5 minutes, so a
1-minute task is almost entirely overhead. Before merging, the primary arm's
median chunk was 1.3 min with 204 S-tier tasks; after, the median is ~29 min.

### 2.4 The XL extrapolation — corrected component-wise

The scaling ladder reports a measured whole-fold XL/L wall ratio of **4.21**.
That figure is a **blend and must not be applied to either cost component
alone.** Recomputed from `calibration.csv`, the components diverge by ~7×:

| component | per-class / per-featurizer XL/L | median |
|---|---|---|
| fit + predict | CNN 0.97, GRU 1.28, MLP 1.57, XGBoost 2.84 | **1.43** |
| featurization | KMersCounts 9.91, OneHotEncoder 10.48 | **10.20** |

Fit+predict is strongly **sub-linear** in rows (1.43 where the row ratio is
8.94 — per-epoch cost grows while epochs-to-early-stop falls), while
featurization is **essentially linear** (10.20 vs 8.94). Since featurization is
~50 % of XL wall time, the 4.21 blend sits between them — which is exactly why
using it for both would overestimate fit+predict ~3× and underestimate
featurization ~2.4×. The partitioner therefore carries two constants,
`XL_OVER_L_FITPRED = 1.43` and `XL_OVER_L_FEATURIZE = 10.20`, and every
estimate records its provenance (`measured`, `extrapolated_from_L_x1`,
`measured_tier_median`, …) in the chunk manifest.

Caveat carried forward: the fit+predict spread is wide (0.97–2.84) and
**tree-building sits at the top of it**. RandomForest × XL is both unmeasured
*and* an OOM casualty, so it is the least reliable cell in the plan.

### 2.5 Resulting chunk counts

| arm | chunks | cpu | gpu | median min | p95 min | max min |
|---|---|---|---|---|---|---|
| primary | **175** | 64 | 111 | 29.1 | 143.3 | 207.8 |
| corrective | **10** | 2 | 8 | 14.8 | 26.9 | 29.4 |

The p95/max tail is entirely Sherwood: a chunk there is ~74 min of
featurization before any model fits, which the cap deliberately permits rather
than paying that cost repeatedly.

---

## 3. Routing

`compute_class` comes from the manifest — `cpu` for Linear/KNN/RF/XGB, `gpu` for
MLP/CNN/GRU/Transformer/GNN — following paper §4.5, which puts MLP on the GPU
nodes despite it being tabular. GNN is assigned `gpu` as an extension of that
statement, not a claim from it (§4.5 names only the four).

**One measured override:** any `RNAFMEmbeddings` config routes to the **GPU
queue regardless of its model class**, including `LinearModel`. Embedding
extraction dominates its cost and is **10.8× faster on a T4** (measured 14.0 s
vs 150.6 s per fold at the M tier).

---

## 4. Resources — sized by host RAM, never by VRAM

**VRAM is not the constraint and sizing by it would be wrong.** Measured peak
VRAM is 17–47 MB for every non-RNA-FM model class — the maximum, GRU at
46.9 MB, is **0.31 % of a T4's 15,360 MB** — and it is **row-count invariant**
(CNN 17.5 MB at 192 rows, 17.9 MB at 291,551). VRAM is set by parameters plus one
mini-batch; only host RAM scales with rows.

So worker count is the measured packing optimum capped by the measured
per-process peak RSS against the instance's usable RAM:

```
procs = min(8, floor(30 GB / p95_rss(tier)))
```

| tier | measured p95 peak RSS, non-RNA-FM | procs |
|---|---|---|
| S | 1.6 GB | 8 |
| M | 1.7 GB | 8 |
| L | 3.2 GB | 8 |
| XL | 3.9 GB | 7 |

8 is the measured GPU packing optimum (CNN 1.94 → 12.50 fits/min at 8
concurrent, device still 60–65 % idle) and also matches 8 single-threaded fits on
a 16-vCPU `c6id.4xlarge`. `OMP/MKL/OPENBLAS_NUM_THREADS=1` is set in every task,
because the packing measurement was made under that condition and an
unconstrained BLAS would oversubscribe the box `procs`-fold.

### Two measured exceptions

* **RNA-FM: `procs = 1`, ≥16 GB.** Hard capacity wall — it peaks at
  **8,107.5 MB = 52.8 % of a T4**, and at concurrency 2 only one of two
  processes completes while at 8 none do. Host RSS is also large because
  embeddings are materialised dense `(n, L, 640)` float32: measured **9.66 GB at
  the L tier**.
* **XL tree models: `procs = 2`, 30 GB.** RandomForest `n_estimators ≥ 500` on
  Sherwood was **exit-137 OOM-killed at 8 GB**, so its requirement is a measured
  *lower bound*, not a measurement.

### Worker start method, and the memory consequence discovered in the smoke run

CPU chunks fork their workers; **GPU chunks spawn them**. A forked child cannot
use CUDA if the parent ever initialised a context, and on a real T4 the 64-config
DL chunk failed 64/64 with `AcceleratorError: CUDA error: initialization error`
even after the device probe was moved out-of-process. Spawn removes that failure
class outright — the child re-imports and receives state pickled, so it has no
inherited context. (Before that, the same chunk failed for a different reason:
`multiprocessing.Pool` workers are *daemonic*, and `pl.Trainer` starts child
processes, so every DL config died on `daemonic processes are not allowed to have
children`. Both fixes are in `run_chunk.py`; the history is in
`smoke_run_report.md` §4.)

**Spawn changes the memory model, and the measurement says so.** A forked worker
inherits the feature matrices copy-on-write; a spawned worker gets its **own
copy**. Measured: the 64-config GPU chunk at `procs=8` peaked at
**14,251.6 MB against its 12 GB request** and was placed only because the compute
environment chose a larger instance. The GPU first rung is therefore **24 GB**,
not 12 GB.

Carry this forward as a live risk: GPU task RSS now scales with
`procs × matrix size` rather than being nearly flat in it, and only one
spawn-mode GPU chunk has been observed — on an M-tier dataset. A large-matrix GPU
chunk at `procs=8` has never been run. If XL/L GPU chunks OOM in the sweep, the
fix is to reduce `procs` for the GPU queue rather than to climb the ladder.

### Retry ladder

CPU `8 → 30 → 120 GB`, GPU `24 → 30 → 120 GB`, on exit 137 (plus
104/134/139/143/247), `maxRetries = 3`, then
**`ignore`** — a chunk that still fails loses only itself, because results are
published per chunk and the collector records the gap.

**Stated limitation:** the third rung (120 GB) exceeds a `g4dn.2xlarge` /
`c6id.4xlarge` (32 GiB), so a task reaching it can only be placed if the compute
environment permits a larger size in the family. That is deliberate — the
alternative is a rung that can never succeed — but it means the instance-size
pin is a preference for attempts 1–2 and an escalation may land on a bigger box.
Every output row records the instance it actually ran on, so this is visible
rather than hidden.

### Instance pinning

Phase 2 found the compute environments list *families* (`['g4dn']`,
`['c6id','r6id','m6id']`), so Batch chose sizes itself and mixed families across
tasks, making unit costs unattributable. The intended sizes are
**`g4dn.2xlarge`** (8 vCPU + 1 T4) and **`c6id.4xlarge`** (16 vCPU).
**Not yet verified:** pinning a size requires editing the compute environments
themselves, which this pipeline cannot do from a launch config. The task
resource requests (8 cpus, the memory ladder) constrain placement but do not pin
a size. Recommend the user pin sizes in both environments before the full sweep.

---

## 5. Determinism

Phase 1 established three defects that make exact per-fold reproduction
impossible, and the pipeline **reproduces upstream behaviour by default** rather
than silently fixing it:

* `prepare_data_fold` reshuffles the **global unseeded RNG inside** the per-fold
  loop, so the five test sets are five independent random holdouts, not a
  partition — measured at n=1000, **32.6 % of samples are never in any test
  fold** and all 10 fold pairs overlap.
* The `nucleobase` branch performs five independent unseeded clustered splits;
  measured, two consecutive calls give test sets of **different sizes** (48 vs
  62 on TLR7).
* Nothing in the benchmark path seeds numpy/torch/lightning.

Therefore: **the primary reproduction runs UNSEEDED**, matching the published
methodology, and comparison against published values must be
**distribution-level (mean ± sd over folds), never per-fold**. `--seed N` is
threaded through as an optional flag and preserves the published resampling
scheme while making a run repeatable; `--proper_kfold` additionally replaces the
per-fold reshuffle with a true disjoint partition and is a **documented
methodological divergence** for sensitivity analysis only (it changes what is
being estimated). Every output row records `seed_used` (−1 = unseeded) and
`proper_kfold`.

### One methodological consequence of chunking, recorded not hidden

Upstream draws a fresh shuffle per fold **per config**, so no two configs see the
same holdout. In a chunk the split is drawn once per (group, fold) and shared by
every config in that group. Each config still sees five random 80/20 holdouts
from the same distribution, so per-config mean ± sd is distributionally
unchanged — but fold noise is now *shared* within a group, which makes
between-model comparisons within a group **paired** (lower variance) rather than
independent. Output column `fold_split_scope = 'chunk'` records this so the
reproduction report can state it.

---

## 6. The feature cache

`feature_cache.py` from Phase 2, used unmodified. Key is
`(dataset, downloader_key, featurizer, canonical(args), cv, n_folds, seed, fold)`
plus a `corrective` flag added here — **the corrective transform changes the
features, so it must change the key**, or a faithful-arm matrix would be served
to a corrective run.

Scope is **task-local disk**, deliberately. A shared cross-task cache over S3 was
not attempted: the grouping already captures the win, and an S3-backed cache
would add a network round-trip and a coherence problem to save re-doing work the
chunking has already eliminated.

**The cache stores RAW featurizer output.** The model-specific reshape (3-D
one-hot → flat for MLP and the tree models) is applied per config afterwards by
the harness's own `featurize()`. Caching post-reshape matrices would silently
hand MLP the CNN's matrix. The runner passes a stub featurizer holding the cached
matrices into `H.featurize()` so that reshape logic runs identically on a hit and
a miss — there is no second implementation of it to drift.

**Verified locally:** a re-run of the same chunk against a warm cache logged
`cache_hit=True` on 5/5 folds with `featurize_s=0.00`.  On Batch every smoke
chunk was a cold first run, so the observed hit rate there is 0 % by
construction; what the cloud runs verify is that the cache *stores* correctly
(`bytes_written` non-zero, `write_errors: 0`).  The 97.6 % elimination figure is
Phase 2's projection, not re-measured in Phase 3.

---

## 7. Output schema and the join

Each task writes a tidy parquet whose **first 18 columns are exactly the
published CSV's schema, in its order** — `cross_validation, dataset, featurizer,
featurizer_args, model, model_args`, then `train_*` metrics, `train_fold`,
`test_*` metrics, `test_fold` — so `df[PUBLISHED_COLUMNS]` is byte-comparable to
`oligogym_benchmarks.csv`. All 5 folds, train **and** test metrics. Fold indices
are published separately per chunk (`*_folds.csv`) rather than widened into the
metrics table.

Metrics are **not recomputed** anywhere in this pipeline: they come from
`oligogym.metrics.regression_metrics` through the patched harness, so they are
comparable to the published numbers by construction.

Provenance columns appended: `config_hash, arm, chunk_id, seed_used,
proper_kfold, fold_split_scope, instance, wall_s, peak_rss_mb, peak_vram_mb,
cuda_available, device_name, torch_version, image_digest, git_sha,
oligogym_commit, dataset_downloader_key_used, correction_applied,
correction_level, status, error, input_dim, seq_len, dataset_size_tier,
compute_class, queue`.

### Canonicalization — verified, not assumed

`featurizer_args` / `model_args` in the published CSV are Python **repr** strings
with sorted keys, e.g. `"{'k': [1, 2, 3], 'modification_abundance': True}"`.
**List order is semantic** (`k`, `hidden_dims`) and is preserved; only dict key
order is normalised. `config_hash` = first 16 hex of SHA-256 over
`cv|dataset|featurizer|canonical(fargs)|model|canonical(margs)`, using
`ast.literal_eval` (no code execution, and correct for Python reprs where
`json.loads` fails).

Verified in this session, on the real files:

* **9,190 / 9,190** manifest `config_hash` values reproduced exactly by the
  collector's independent implementation;
* **9,190 / 9,190** published configs and **45,950 / 45,950** published rows
  join; zero manifest-only and zero published-only configs;
* **100 / 100** distinct arg strings round-trip byte-exact through
  `literal_eval → sorted-key repr`;
* on real smoke output: **23/23** configs matched, `worker_hash_agrees_pct =
  100.0`, 5 folds per config both sides.

---

## 8. Idempotence and resume

* `chunk_id` is a blake2b digest over the arm, the group key, the resource class
  and the **sorted config_hash list**, so a chunk's identity is a function of its
  content. Change the config set and you get a new chunk; change nothing and
  `-resume` skips it.
* Both task inputs (the descriptor JSON and the config CSV) are files, hence
  content-addressed by Nextflow. `process.cache = 'lenient'` (size + timestamp)
  is set because strict hashing of S3-staged inputs is fragile.
* Results are published **per chunk**, so a failure loses only that chunk.
* The collector de-duplicates on `(arm, config_hash, train_fold)` keeping the
  last occurrence, so re-running collection over a partially rewritten prefix is
  safe.
* **Verified locally:** a second `-resume` launch reported `cached: 1` for all
  three processes and re-executed nothing.

---

## 9. Corrective arm — what is fixed, what is not

Implemented in `corrective_transform.py`, imported **only** when
`--arm corrective`; the faithful arm never touches it.

### Defect 1 — TLR7/TLR8 keys swapped. Fixed, completely.

Re-verified here against the shipped resources: `download('tlr7')` returns
`Alharbi_2020_1`, whose own metadata reads *"2'OMe gapmer screen of **TLR8**
potentiation"* with labels 0.57–5.44; `download('tlr8')` returns
`Alharbi_2020_2`, *"screen of **TLR7** inhibition"*, labels 17.46–126.5. Both
files hold the **same 192 compounds in the same row order** over the same 4
targets (x-sets equal, x-order equal, `fasta` identical), and `y` differs in
192/192 rows. Since only the label column differs, the fix is a clean, complete
relabeling of the key → file mapping. Features unchanged; endpoint corrected.
**Verified in a live corrective run:** the TLR7 chunk logged
`from_key=TLR7 → to_key=tlr8` and produced RMSE ≈ 8.6, consistent with the
17.46–126.5 label range rather than 0.57–5.44.

### Defect 2 — Neurotox MOE HELM/SMILES truncated. **Option (c): partial correction + documented sensitivity analysis.**

Re-verified over all 2,437 records: the HELM-derived base sequence is a **strict
prefix** of the `fasta` column in 2437/2437 and exactly **one base shorter** in
2437/2437; the dropped base **varies** — C 764, T 711, A 621, G 341 — so it is
not an overhang convention; SMILES phosphorus count equals
`n_monomers − 1` in 2437/2437 and the `fasta` length in 0/2437, i.e. SMILES is
truncated consistently with HELM. Cytotox LNA as a control shows a zero
discrepancy in 768/768.

**What the featurizers actually read** — established by reading `features.py`,
which corrected an earlier oversimplification. *Neither featurizer reads the
`fasta` column;* both derive the sequence from the HELM string in column `x`:

| featurizer | consumes | correctable from `fasta`? |
|---|---|---|
| `KMersCounts` | `monomers['base'].str[-1].str.cat()` for k-mers; `monomers['sugar']`/`['phosphate']` for modification counts | **partially** — k-mers yes, modification counts no |
| `RNAFMEmbeddings` | `_helm_to_fasta` → bases only, then T→U | **yes, fully** |
| `OneHotEncoder` | per-position `phosphate`/`sugar`/`base` | **no** |
| `HELMGraph` | per-monomer sugar/base/phosphate nodes | **no** |

So "run the corrective arm on fasta-derived featurizers" is not a drop-in
switch. What is implemented:

* `KMersCounts` → **`partial_base_composition_only`**. k-mer counts are rebuilt
  from the full-length `fasta` base sequence, recovering the 20th base. The
  `modification_abundance` sugar/phosphate counts are left exactly as upstream
  produces them from the truncated HELM. **Verified:** on 200 MOE records the
  corrected features differ in 200/200 rows, the per-row total k-mer count rises
  by exactly **3** (one each for k=1,2,3 — precisely one added monomer), and with
  `modification_abundance=True` the sugar/phosphate columns are **byte-identical**
  to the faithful arm, confirming the correction is exactly as narrow as claimed.
* `RNAFMEmbeddings` → **`full_base_sequence`**. This featurizer consumes only the
  base sequence, so substituting `fasta` corrects everything it reads.
* `OneHotEncoder`, `HELMGraph`, `SMILESGraph` → **`none_helm_truncated`**. **No
  transform is applied.** These run on the truncated HELM exactly as the faithful
  arm does and serve as the control half of the sensitivity analysis.

**No HELM extension is fabricated.** Appending a monomer would require inventing
the dropped nucleoside's sugar *and* phosphate chemistry. The dropped base varies
across all four bases, the 2'-MOE-vs-DNA sugar pattern is position-dependent in
these gapmers, and the linkage (`[sp]` vs `p`) varies too — there is no
defensible default for any of the three, and a guess would put invented
chemistry into a benchmark undetectably.

**What this does not fix, plainly:** the molecules are one nucleoside longer than
any HELM- or SMILES-derived feature can represent. For `OneHotEncoder` and
`HELMGraph` the corrective arm is a **replicate of the faithful arm, not a
correction** — those models see a 19-mer where the molecule is a 20-mer. For
`KMersCounts` only base composition is corrected, not the modification profile.
The defect can only be repaired upstream by re-releasing the MOE HELM strings at
full length. Every output row carries `correction_applied` and
`correction_level`, so no consumer can mistake a control for a correction.

### Defect 3 — Ichihara target mis-assignment. Noted, no fix.

Affects target-grouped splits only. This benchmark uses `random` and
`nucleobase`; neither groups by target, so target labels never enter a split or a
feature.

---

## 10. Harness integration and one production trap

The pipeline uses **`train_model_patched.py` + `oligogym_patch.py` unmodified**
and does not re-patch oligogym. All model-specific behaviour —
`prepare_model`, `featurize` (which owns the reshape and sets `input_dim` /
Transformer's `seq_len`), `predict`, `check_config`, `_metrics_frame` — is
delegated to the harness.

**RNA-FM silent-degradation guard (added here).**
`RNAFMEmbeddings._load_model` catches every exception and merely *warns*, so a
missing or unreachable checkpoint leaves `self.model = None` and the run
continues. In this version the failure then surfaces as
`AttributeError: 'RNAFMEmbeddingsFixed' object has no attribute
'_get_simple_features'` deep inside fold 0 — a traceback that does not name the
real cause — and were that fallback method ever added upstream, the featurizer
would instead return plausible 6-dimensional features and the run would
"succeed" with meaningless numbers. **Encountered live** while testing in this
sandbox (the checkpoint is baked into the image, not present here; the canonical
host returns 403). `preflight_rnafm()` now asserts, before the first fold, that a
model object exists and carries >9×10⁷ parameters (the real `rna_fm_t12`
checkpoint has 99,521,546), and names the expected path in the failure message.

Also set per task: `TORCH_HOME=/opt/torch-hub` so `fm.pretrained.rna_fm_t12()`
loads the baked-in checkpoint from cache and never attempts the network.

Lightning's progress bar is suppressed to `ERROR` level: at 8 workers × 5 folds ×
hundreds of epochs it produced megabytes of interleaved carriage-return noise per
task log, burying the per-task evidence. Metrics are unaffected — they come from
returned predictions.

---

## 11. Two Nextflow issues found by actually running it

Both were real failures fixed against Nextflow 26.04.6, not theoretical:

1. **Top-level statements are rejected** (`Statements cannot be mixed with
   script declarations`) in DSL ≥25. `results_prefix` and the shared chunk
   script became `params.results_prefix` and a `params.chunk_script` closure.
2. **`splitCsv` cannot parse RFC-4180 doubled quotes.** The chunk manifest's
   `featurizer_args_json` column contains `"{""flatten"":false}"`, which made
   `splitCsv` either shift every later column by one (so integer fields arrived
   holding text like `measured_tier_median`) or fail outright with
   `Invalid CSV value`. Fixed by having `split_chunks.py` emit a slim
   **`schedule.csv`** with only scalar scheduling columns for the driver to read,
   while the full descriptor travels to the task as `chunk_<id>.json` where a
   real JSON parser reads it.

A third, quieter bug: `timeline`/`report`/`trace` paths that interpolate
`${params.outdir}` resolve to `null/` because config is evaluated **before**
command-line params are applied. They now write to `runinfo/` in the launch
directory.

---

## 12. Concurrency and the unconfirmed quota

Concurrency is a parameter, not a constant: `OLIGOGYM_QUEUE_SIZE` (default 32)
sets `executor.queueSize`, with `submitRateLimit = '20/1min'`.

The **g4dn quota was not confirmed** ("Running On-Demand G and VT instances",
`L-DB2E81BA`, us-west-2). 16 × `g4dn.2xlarge` needs **128 vCPU** and this quota
commonly defaults to 0–64, which would cap the run at 8 instances. The
recommendation is to start at the default 32 and raise only after tasks are
observed actually starting. If tasks sit in `SUBMITTED` with zero started, treat
it as a probable quota cap and check `L-DB2E81BA` before waiting.

GPU work targets **`batch-gpu_copy` (`5ZEM2WRyxXMWtRFVwpxwaz`)**, never
`batch-gpu` (`fJ1Tu2lZEwf2cSF54nm2v`): Phase 2 found the latter accepted three
runs and left all of them `SUBMITTED` with **zero tasks for ~35 minutes**, while
the identical launch on `batch-gpu_copy` started in ~5 minutes and ran to
completion on real T4s.

---

## 13. Skips, recorded rather than silently dropped

| what | count | why |
|---|---|---|
| RNA-FM × sherwood | 2 configs | The dense `(n, L, 640)` float32 embedding array alone is ~25 GB at 291,551 rows; measured 9.66 GB peak RSS at 32,602 rows and this cell is unmeasured. Expected-infeasible; re-enable with `--no_skip_rnafm_xl`. |

Written to `skips_primary.csv` with the reason attached. Config accounting is
asserted: `assigned + skipped == selected` (9,188 + 2 = 9,190), each config
appears in exactly one chunk, and chunk ids are unique.

---

## 14. Cost expectation

Phase 2's projection, with the cache and packing this pipeline implements:
**~283 CPU-h + ~501 GPU-h device time → ~79 `g4dn.2xlarge` + ~35
`c6id.4xlarge` instance-hours ≈ $88** at on-demand list for both arms. The
independent cross-check from Seqera's own billed figures, scaled *unpacked and
uncached*, gives $282. Budget **$100–300**, expecting the low end if the cache
hits as measured. This pipeline's own chunk-level estimates total 8,119 packed
chunk-minutes for the primary arm across its **175 chunks** (111 gpu + 64 cpu,
per §2.5) — 135 h of packed task time — which is the figure to compare against
the trace after the full sweep. Note this is *task* time, not device time: each
task runs up to 8 fold-fits concurrently, so it is not directly comparable to the
CPU-h/GPU-h figures above.
