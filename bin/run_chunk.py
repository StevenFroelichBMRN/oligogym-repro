#!/usr/bin/env python
"""Run one chunk of the OligoGym reproduction: featurize once, fit many.

A chunk is a set of configs that share a feature matrix, i.e. the same
(dataset, featurizer, featurizer_args, cross_validation).  For each fold this
script builds the train/test split once, featurizes once (via the disk-backed
FeatureCache so a re-run or a sibling chunk on the same matrix pays nothing),
then fits every config in the chunk against that matrix -- optionally several
configs at a time, since Phase 2 measured 8 concurrent trainers per T4 with the
device still 60-65 % idle.

Everything model-specific is delegated to the PATCHED HARNESS
(`train_model_patched.py`): `prepare_model`, `featurize` (which owns the
model-specific reshape and sets `input_dim` / Transformer's `seq_len`),
`predict`, `_metrics_frame`, `check_config`.  Nothing in oligogym itself is
re-patched here, and no metric is recomputed by this file -- the numbers come
from `oligogym.metrics.regression_metrics` through the harness, so they are
comparable to the published CSV by construction.

Two consequences of chunking that are methodological, not cosmetic, and are
recorded in the output rather than hidden:

1.  **Configs in the same chunk share fold splits.**  Upstream draws a fresh
    unseeded shuffle per fold *per config*, so no two configs see the same
    holdout.  Here the split is drawn once per (chunk, fold).  Each config still
    sees five random 80/20 holdouts from the same distribution, so per-config
    mean +/- sd is distributionally unchanged; what changes is that fold noise is
    now *shared* across configs within a chunk, which makes between-model
    comparisons within a chunk paired (lower variance) rather than independent.
    The output column `fold_split_scope` records this as `chunk` so the
    reproduction report can say so.
2.  **Featurization is shared, so it must be model-independent.**  The cache
    stores the RAW featurizer output; the model-specific reshape (3-D one-hot ->
    flat for MLP/tree models) is applied per config by the harness afterwards.
    Caching post-reshape matrices would silently give MLP the CNN's matrix.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import multiprocessing as mp
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_model_patched as H  # noqa: E402  the patched harness
from feature_cache import FeatureCache, cache_key  # noqa: E402

N_FOLDS = 5

# Lightning's progress bar emits a carriage-return-laden line per batch per epoch.
# At 8 concurrent workers x 5 folds x hundreds of epochs that is megabytes of
# interleaved noise per task log, which makes the per-task evidence (device,
# cache hits, fold shapes) unreadable and inflates S3 log storage for no benefit.
# The metrics themselves are unaffected -- they come from the returned predictions.
os.environ.setdefault("PYTHONWARNINGS", "ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

#: metric names produced by oligogym.metrics.regression_metrics, in the order
#: the published CSV lists them.
METRICS = [
    "r2_score",
    "root_mean_squared_error",
    "mean_absolute_error",
    "pearson_correlation",
    "spearman_correlation",
]

PUBLISHED_COLUMNS = (
    ["cross_validation", "dataset", "featurizer", "featurizer_args", "model", "model_args"]
    + [f"train_{m}" for m in METRICS]
    + ["train_fold"]
    + [f"test_{m}" for m in METRICS]
    + ["test_fold"]
)


# ---------------------------------------------------------------- utilities
def _rss_mb() -> float:
    """Peak RSS of THIS process, MB.  ru_maxrss is KB on Linux, bytes on macOS."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1024.0 if platform.system() != "Darwin" else v / (1024.0 * 1024.0)


def _instance_id() -> str:
    """Best-effort instance type from the environment; '' if unavailable.

    Batch does not inject the instance type, so this reads what Nextflow/ECS do
    expose.  It is recorded verbatim rather than inferred: Phase 2 found that
    unpinned compute environments let Batch pick sizes, which made unit costs
    unattributable, and the fix is to record what actually ran.
    """
    for var in ("OLIGOGYM_INSTANCE", "ECS_CONTAINER_INSTANCE_TYPE", "AWS_BATCH_JQ_NAME"):
        if os.environ.get(var):
            return os.environ[var]
    return ""


_PROBE_SRC = """
import json
out = {"cuda_available": False, "device_name": "cpu", "torch_version": ""}
try:
    import torch
    out["torch_version"] = torch.__version__
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        out["device_name"] = torch.cuda.get_device_name(0)
        out["capability"] = list(torch.cuda.get_device_capability(0))
        out["device_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1048576, 1
        )
        out["arch_list"] = list(torch.cuda.get_arch_list())
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
"""


def _torch_device_info() -> Dict[str, Any]:
    """Probe the GPU in a SHORT-LIVED SUBPROCESS, never in this process.

    This is not defensive style, it is a measured bug fix.  `get_device_name` /
    `get_device_capability` create a CUDA primary context in the calling
    process.  This runner then forks a worker pool (fork is what makes the
    feature matrix shared copy-on-write rather than pickled per worker), and
    **CUDA cannot be re-initialised in a forked child** -- every child dies with
    "Cannot re-initialize CUDA in forked subprocess".

    Observed live on the smoke run: chunk prim_8d24dcf3f154c775 (64 CNN/GRU/MLP/
    Transformer configs, 8 workers, real T4) failed 64/64 configs this way, and
    exited 3 ("every config failed") rather than crashing, so the shape of the
    failure was total-but-quiet.

    Probing out-of-process keeps this process CUDA-clean, so each forked child
    creates its own context on first use.  That preserves BOTH the Phase 2
    packing measurement (8 concurrent processes per T4) and copy-on-write
    sharing of the feature matrix, which `spawn` would have cost us.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SRC],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip().splitlines()[-1])
        return {
            "cuda_available": False, "device_name": "cpu", "torch_version": "",
            "error": f"probe rc={proc.returncode}: {proc.stderr.strip()[-300:]}",
        }
    except Exception as exc:
        return {
            "cuda_available": False, "device_name": "cpu", "torch_version": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _map_nondaemonic(rows: List[dict], nprocs: int, init_args: tuple) -> List[dict]:
    """Fit `rows` concurrently in NON-DAEMONIC forked workers.

    `multiprocessing.Pool` marks its workers **daemonic**, and a daemonic process
    is forbidden from having children:

        AssertionError: daemonic processes are not allowed to have children

    Every DL model here trains through `pl.Trainer`, which starts child processes
    (DataLoader workers / Lightning's launcher), so under `Pool` *every* DL config
    fails.  Measured live on a real T4: chunk prim_8d24dcf3f154c775 (64
    CNN/GRU/MLP/Transformer configs, procs=8) reported `fitted=64 errors=64` on
    all five folds and exited 3, while the RNA-FM chunk in the same wave
    (procs=1, no pool) completed normally — the tell that the pool, not the GPU,
    was the problem.

    Plain `mp.Process` children are non-daemonic by default, so they may spawn
    their own children.  Fork is retained deliberately: the feature matrix is
    inherited copy-on-write rather than pickled to each worker, which is the
    whole point of grouping configs by feature matrix.  Results come back over a
    Queue; a worker that dies without reporting is recorded as a failed config
    rather than silently dropped, so the row count stays exact.
    """
    ctx = mp.get_context("fork")
    q: Any = ctx.Queue()

    def _worker(idx_rows: List[tuple]) -> None:
        _init_worker(*init_args)
        for i, row in idx_rows:
            try:
                q.put((i, _fit_one(row)))
            except Exception as exc:  # pragma: no cover
                q.put((i, {**row, "status": "error",
                           "error": f"{type(exc).__name__}: {exc}"}))

    # Round-robin the configs so each worker gets a mix of cheap and expensive
    # models rather than one worker inheriting every large one.
    shards: List[List[tuple]] = [[] for _ in range(nprocs)]
    for i, row in enumerate(rows):
        shards[i % nprocs].append((i, row))

    procs_list = [
        ctx.Process(target=_worker, args=(shard,), daemon=False)
        for shard in shards if shard
    ]
    for p in procs_list:
        p.start()

    out: Dict[int, dict] = {}
    # Drain while workers run: a full Queue pipe would otherwise deadlock a
    # worker in put() and the join() below would never return.
    while len(out) < len(rows) and any(p.is_alive() for p in procs_list):
        try:
            i, rec = q.get(timeout=5)
            out[i] = rec
        except Exception:
            continue
    for p in procs_list:
        p.join(timeout=60)
    while len(out) < len(rows):
        try:
            i, rec = q.get_nowait()
            out[i] = rec
        except Exception:
            break

    recs: List[dict] = []
    for i, row in enumerate(rows):
        if i in out:
            recs.append(out[i])
        else:
            # A worker died without reporting (OOM kill, segfault).  Record it as
            # a failed config so the parquet's row count still matches the chunk.
            recs.append({
                **row, "status": "error",
                "error": "worker died without reporting (possible OOM or crash)",
            })
    return recs


def _assert_cuda_clean(procs: int) -> None:
    """Refuse to fork a pool if this process already holds a CUDA context.

    Belt-and-braces for the failure above: if some future edit initialises CUDA
    in the parent, fail with the real reason instead of losing every config in
    the chunk to an opaque per-worker error.
    """
    if procs <= 1:
        return
    try:
        import torch

        if torch.cuda.is_initialized():
            raise RuntimeError(
                "CUDA is already initialised in the parent process; forking "
                f"{procs} workers from here would make every worker fail with "
                "'Cannot re-initialize CUDA in forked subprocess'. Probe the "
                "device out-of-process (see _torch_device_info)."
            )
    except ImportError:
        pass


def preflight_rnafm(featurizer) -> Dict[str, Any]:
    """Assert RNA-FM loaded REAL pretrained weights before any fold runs.

    Two failure modes this guards, both documented in harness_repair_notes.md:

    * `RNAFMEmbeddings._load_model` catches every exception and merely warns, so
      a missing or unreachable checkpoint leaves `self.model = None` and the run
      continues.  Upstream's `_get_embeddings_batch` then calls
      `self._get_simple_features(...)` -- a method that does not exist on the
      class -- so in this version the failure surfaces as an AttributeError deep
      inside fold 0, minutes into a task, with a traceback that does not name the
      real cause.  Were that method ever added, the featurizer would instead
      return plausible-looking 6-dimensional features and the run would "succeed"
      with meaningless numbers.
    * The canonical weight host (proj.cse.cuhk.edu.hk) returns 403 for every
      path, so a container without the checkpoint baked in cannot recover at
      run time.

    Failing here, before the first fold, converts both into one clear message
    naming the expected checkpoint path.
    """
    info: Dict[str, Any] = {"rnafm_model_loaded": False}
    model = getattr(featurizer, "model", None)
    if model is None:
        try:
            import torch

            hub = torch.hub.get_dir()
        except Exception:
            hub = "<torch unavailable>"
        raise RuntimeError(
            "RNA-FM weights did not load, so this chunk would either crash mid-fold "
            "or (if upstream's fallback path existed) silently produce 6-dimensional "
            "placeholder features instead of 640-dimensional embeddings. "
            f"Expected the checkpoint at {hub}/checkpoints/RNA-FM_pretrained.pth "
            "(1194.4 MB, sha256 5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435"
            "f776c52e93d5e99); the production image bakes it in under "
            "TORCH_HOME=/opt/torch-hub. The canonical download host returns 403, "
            "so a runtime fetch will not recover this."
        )
    n_params = sum(p.numel() for p in model.parameters())
    info["rnafm_model_loaded"] = True
    info["rnafm_n_params"] = int(n_params)
    # Measured in Phase 1: the real rna_fm_t12 checkpoint is 99,521,546 params.
    # A wildly different count means a different model was loaded.
    assert n_params > 9e7, (
        f"RNA-FM loaded but has only {n_params:,} parameters; the pretrained "
        "rna_fm_t12 checkpoint has 99,521,546. Wrong or truncated weights."
    )
    return info


class _CachedFeaturizer:
    """Stands in for a real featurizer so the harness applies its own logic.

    `H.featurize` owns the model-specific reshape and the `input_dim`/`seq_len`
    assignment.  Handing it a stub whose `fit_transform`/`transform` return the
    already-computed matrices means that logic runs identically whether the
    matrix came from the cache or from a real featurization -- there is no second
    implementation of it to drift.
    """

    def __init__(self, X_train, X_test):
        self._X_train, self._X_test = X_train, X_test

    def fit_transform(self, X):
        return self._X_train

    def transform(self, X):
        return self._X_test


# ---------------------------------------------------------------- worker side
_W: Dict[str, Any] = {}


def _init_worker(X_train, X_test, y_train, y_test, fold, seed, corrective):
    _W.update(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        fold=fold, seed=seed, corrective=corrective,
    )
    # A worker that hangs must leave a traceback rather than a silent timeout.
    faulthandler.enable()


def _fit_one(cfg_row: dict) -> dict:
    """Fit one config against the shared, already-featurized fold."""
    t0 = time.perf_counter()
    rec: Dict[str, Any] = {
        "config_hash": cfg_row["config_hash"],
        "fold": _W["fold"],
        "status": "ok",
        "error": "",
    }
    vram_mb = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None  # noqa: F841

    try:
        config = {
            "dataset": cfg_row["dataset_config_key"],
            "featurizer": cfg_row["featurizer"],
            "featurizer_args": json.loads(cfg_row["featurizer_args_json"] or "{}"),
            "model": cfg_row["model"],
            "model_args": json.loads(cfg_row["model_args_json"] or "{}"),
            "cross_validation": cfg_row["cross_validation"],
        }
        H.check_config(config)

        stub = _CachedFeaturizer(_W["X_train"], _W["X_test"])
        X_tr, X_te = H.featurize(_W["X_train"], _W["X_test"], stub, config)
        model = H.prepare_model(config)
        y_pred_train, y_pred_test = H.predict(
            model, X_tr, X_te, _W["y_train"], fit_kwargs=None
        )
        m_tr = H._metrics_frame(_W["y_train"], y_pred_train, _W["fold"]).iloc[0]
        m_te = H._metrics_frame(_W["y_test"], y_pred_test, _W["fold"]).iloc[0]
        for m in METRICS:
            rec[f"train_{m}"] = float(m_tr[m])
            rec[f"test_{m}"] = float(m_te[m])
        # input_dim / seq_len are set by the harness during featurize; recording
        # them makes a shape bug visible in the output instead of only in a log.
        rec["input_dim"] = config["model_args"].get("input_dim")
        rec["seq_len"] = config["model_args"].get("seq_len")
    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-2000:]
        for m in METRICS:
            rec[f"train_{m}"] = np.nan
            rec[f"test_{m}"] = np.nan

    try:
        import torch

        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        pass

    rec["wall_s"] = round(time.perf_counter() - t0, 4)
    rec["peak_rss_mb"] = round(_rss_mb(), 1)
    rec["peak_vram_mb"] = None if vram_mb is None else round(vram_mb, 1)
    return rec


# ---------------------------------------------------------------- fold driver
def build_fold(data, cv: str, fold: int, rng, seed, proper_kfold: bool):
    if cv == "random":
        return H.prepare_data_fold(
            data, fold, rng=rng, proper_kfold=proper_kfold, seed=seed
        )
    random_state = None if rng is None else int(rng.integers(0, 2**31 - 1))
    return H.prepare_data(data, split_strategy=cv, random_state=random_state)


def featurize_fold(
    chunk: dict,
    cfg_rows: List[dict],
    X_train_raw,
    X_test_raw,
    cache: FeatureCache,
    fold: int,
    seed: Optional[int],
    corrective: bool,
    featurizer_override=None,
):
    """Featurize one fold for the whole chunk, through the cache.

    Returns (X_train, X_test, featurize_s, cache_hit).
    """
    proto = dict(cfg_rows[0])
    key = cache_key(
        {
            "dataset": proto["dataset_config_key"],
            "dataset_downloader_key": proto["dataset_downloader_key"],
            "featurizer": proto["featurizer"],
            "featurizer_args": json.loads(proto["featurizer_args_json"] or "{}"),
            "cross_validation": proto["cross_validation"],
        },
        fold=fold,
        n_folds=N_FOLDS,
        seed=-1 if seed is None else seed,
    )
    # The corrective transform changes the FEATURES, so it must change the cache
    # key -- otherwise a faithful-arm matrix would be served to a corrective run.
    key["corrective"] = bool(corrective)

    got = cache.get(key)
    if got is not None:
        Xtr, Xte, _extra = got
        return Xtr, Xte, 0.0, True

    t0 = time.perf_counter()
    config = {
        "featurizer": proto["featurizer"],
        "featurizer_args": json.loads(proto["featurizer_args_json"] or "{}"),
    }
    featurizer = (
        featurizer_override
        if featurizer_override is not None
        else H.prepare_featurizer(config)
    )
    if proto["featurizer"] == "RNAFMEmbeddings":
        preflight_rnafm(featurizer)
    Xtr = featurizer.fit_transform(X_train_raw)
    Xte = featurizer.transform(X_test_raw)
    dt = time.perf_counter() - t0
    cache.put(key, Xtr, Xte)
    return Xtr, Xte, dt, False


GROUP_COLS = (
    "dataset_downloader_key",
    "featurizer",
    "featurizer_args_json",
    "cross_validation",
)


def run_chunk(args) -> int:
    chunk = json.loads(open(args.chunk).read()) if args.chunk.endswith(".json") else None
    if chunk is None:
        raise SystemExit("--chunk must be a .json chunk descriptor")
    cfg_df = pd.read_csv(args.configs)
    if not len(cfg_df):
        raise SystemExit(f"no configs in {args.configs}")
    cfg_rows = cfg_df.to_dict("records")

    seed = args.seed
    rng = None
    if seed is not None:
        H.seed_everything(seed)
        rng = np.random.default_rng(seed)

    dev = _torch_device_info()
    if args.require_cuda:
        # Fail loudly rather than silently training a "GPU" chunk on CPU: Phase 2
        # found the GPU env can accept work and never start a task, and a silent
        # CPU fallback would look like success at 8x the cost.
        assert dev["cuda_available"], (
            f"--require-cuda set but torch.cuda.is_available() is False; "
            f"device info: {dev}"
        )
    print(f"[device] {json.dumps(dev)}", flush=True)

    cache = FeatureCache(args.cache_dir)
    corrective = args.arm == "corrective"
    if corrective:
        import corrective_transform as CT

    records: List[dict] = []
    fold_meta: List[dict] = []
    idx_records: List[dict] = []
    group_meta: List[dict] = []
    corrections: Dict[str, dict] = {}
    keys_used: Dict[str, str] = {}
    load_s_total = 0.0
    # One dataset load serves every group on that dataset in this chunk; upstream
    # re-read the package resource 5x per config (Phase 1 gap #12).
    data_cache: Dict[str, Any] = {}

    # A merged chunk holds several WHOLE feature groups (the merge pass never
    # splits one), so each group gets its own folds, its own featurization and
    # its own cache entries -- the sharing that matters is inside a group.
    for gkey, gdf in cfg_df.groupby(list(GROUP_COLS), sort=True):
        g_rows = gdf.to_dict("records")
        proto = g_rows[0]
        downloader_key = proto["dataset_downloader_key"]
        correction = {"applied": "none", "level": "n/a", "notes": ""}
        if corrective:
            downloader_key, correction = CT.plan(
                dataset_config_key=proto["dataset_config_key"],
                downloader_key=downloader_key,
                featurizer=proto["featurizer"],
            )
            print(f"[corrective] {json.dumps(correction)}", flush=True)
        for r in g_rows:
            corrections[r["config_hash"]] = correction
            keys_used[r["config_hash"]] = downloader_key

        if downloader_key not in data_cache:
            t_load = time.perf_counter()
            data_cache[downloader_key] = H.DatasetDownloader().download(downloader_key)
            load_s_total += time.perf_counter() - t_load
        data = data_cache[downloader_key]

        featurizer_override = None
        if corrective:
            featurizer_override = CT.make_featurizer(
                featurizer=proto["featurizer"],
                featurizer_args=json.loads(proto["featurizer_args_json"] or "{}"),
                data=data,
                correction=correction,
            )

        cv = proto["cross_validation"]
        gid = "|".join(str(k) for k in gkey)
        for fold in range(N_FOLDS):
            Xtr_raw, Xte_raw, ytr, yte, idx_tr, idx_te = build_fold(
                data, cv, fold, rng, seed, args.proper_kfold
            )
            Xtr, Xte, feat_s, hit = featurize_fold(
                chunk, g_rows, Xtr_raw, Xte_raw, cache, fold, seed, corrective,
                featurizer_override=featurizer_override,
            )
            fold_meta.append(
                {
                    "group": gid,
                    "fold": fold,
                    "n_train": int(len(idx_tr)),
                    "n_test": int(len(idx_te)),
                    "featurize_s": round(feat_s, 4),
                    "cache_hit": bool(hit),
                    "n_configs": len(g_rows),
                }
            )
            idx_records.append(
                {
                    "group": gid,
                    "dataset": proto["dataset_config_key"],
                    "cross_validation": cv,
                    "fold": fold,
                    "train_indices": json.dumps([int(i) for i in np.asarray(idx_tr)]),
                    "test_indices": json.dumps([int(i) for i in np.asarray(idx_te)]),
                }
            )
            print(
                f"[{gid} fold {fold}] cache_hit={hit} featurize_s={feat_s:.2f} "
                f"n_train={len(idx_tr)} n_test={len(idx_te)}",
                flush=True,
            )

            procs = max(1, int(args.procs))
            if procs == 1 or len(g_rows) == 1:
                _init_worker(Xtr, Xte, ytr, yte, fold, seed, corrective)
                fold_recs = [_fit_one(r) for r in g_rows]
            else:
                _assert_cuda_clean(procs)
                fold_recs = _map_nondaemonic(
                    g_rows, min(procs, len(g_rows)),
                    (Xtr, Xte, ytr, yte, fold, seed, corrective),
                )
            records.extend(fold_recs)
            n_err = sum(1 for r in fold_recs if r["status"] != "ok")
            if n_err:
                # Print the DISTINCT error signatures. Without this the only copy
                # of the reason lives in the parquet's `error` column, which is
                # not published when a task fails -- measured cost: three API
                # round-trips to diagnose one broken chunk.
                sigs: Dict[str, int] = {}
                for r in fold_recs:
                    if r["status"] != "ok":
                        sigs[str(r.get("error"))[:300]] = (
                            sigs.get(str(r.get("error"))[:300], 0) + 1
                        )
                for sig, cnt in sorted(sigs.items(), key=lambda kv: -kv[1])[:5]:
                    print(f"[error x{cnt}] {sig}", flush=True)
            print(
                f"[{gid} fold {fold}] fitted={len(fold_recs)} errors={n_err}",
                flush=True,
            )
        group_meta.append(
            {
                "group": gid,
                "dataset": proto["dataset_config_key"],
                "downloader_key_used": downloader_key,
                "featurizer": proto["featurizer"],
                "cross_validation": cv,
                "n_configs": len(g_rows),
                "correction": correction,
            }
        )

    load_s = load_s_total
    # ---- assemble the published-schema table ------------------------------
    meta = {r["config_hash"]: r for r in cfg_rows}
    rows = []
    for rec in records:
        c = meta[rec["config_hash"]]
        row = {
            "cross_validation": c["cross_validation"],
            "dataset": c["dataset_config_key"],
            "featurizer": c["featurizer"],
            "featurizer_args": c["featurizer_args_repr"],
            "model": c["model"],
            "model_args": c["model_args_repr"],
            "train_fold": float(rec["fold"]),
            "test_fold": float(rec["fold"]),
        }
        for m in METRICS:
            row[f"train_{m}"] = rec.get(f"train_{m}")
            row[f"test_{m}"] = rec.get(f"test_{m}")
        row.update(
            {
                "config_hash": rec["config_hash"],
                "arm": args.arm,
                "chunk_id": chunk["chunk_id"],
                "seed_used": -1 if seed is None else int(seed),
                "proper_kfold": bool(args.proper_kfold),
                "fold_split_scope": "chunk",
                "instance": _instance_id(),
                "wall_s": rec["wall_s"],
                "peak_rss_mb": rec["peak_rss_mb"],
                "peak_vram_mb": rec["peak_vram_mb"],
                "cuda_available": dev["cuda_available"],
                "device_name": dev["device_name"],
                "torch_version": dev["torch_version"],
                "image_digest": os.environ.get("OLIGOGYM_IMAGE_DIGEST", ""),
                "git_sha": os.environ.get("OLIGOGYM_GIT_SHA", ""),
                "oligogym_commit": os.environ.get("OLIGOGYM_UPSTREAM_SHA", ""),
                "dataset_downloader_key_used": keys_used[rec["config_hash"]],
                "correction_applied": corrections[rec["config_hash"]]["applied"],
                "correction_level": corrections[rec["config_hash"]]["level"],
                "status": rec["status"],
                "error": rec["error"],
                "input_dim": rec.get("input_dim"),
                "seq_len": rec.get("seq_len"),
                "dataset_size_tier": c["dataset_size_tier"],
                "compute_class": c["compute_class"],
                "queue": c["queue"],
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    # Column order: published schema first, then provenance.  A consumer can do
    # df[PUBLISHED_COLUMNS] and get exactly the published CSV's shape.
    ordered = PUBLISHED_COLUMNS + [c for c in df.columns if c not in PUBLISHED_COLUMNS]
    df = df[ordered]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)

    stats = cache.stats()
    summary = {
        "chunk_id": chunk["chunk_id"],
        "arm": args.arm,
        "queue": chunk.get("queue"),
        "n_configs": len(cfg_rows),
        "n_rows": int(len(df)),
        "n_ok": int((df.status == "ok").sum()),
        "n_error": int((df.status != "ok").sum()),
        "datasets": sorted({r["dataset_config_key"] for r in cfg_rows}),
        "downloader_keys_used": sorted(set(keys_used.values())),
        "featurizers": sorted({r["featurizer"] for r in cfg_rows}),
        "cross_validations": sorted({r["cross_validation"] for r in cfg_rows}),
        "n_groups": len(group_meta),
        "groups": group_meta,
        "dataset_load_s": round(load_s, 3),
        "featurize_s_total": round(sum(f["featurize_s"] for f in fold_meta), 3),
        "cache": stats,
        "cache_hits": stats["hits"],
        "cache_misses": stats["misses"],
        "folds": fold_meta,
        "device": dev,
        "instance": _instance_id(),
        "seed_used": -1 if seed is None else int(seed),
        "task_peak_rss_mb": round(_rss_mb(), 1),
        "procs": int(args.procs),
    }
    with open(args.summary, "w") as fh:
        json.dump(summary, fh, indent=2)
    pd.DataFrame(idx_records).to_csv(args.fold_indices, index=False)

    print("=====CHUNK_SUMMARY_BEGIN=====", flush=True)
    print(json.dumps(summary), flush=True)
    print("=====CHUNK_SUMMARY_END=====", flush=True)

    # A chunk in which every config failed is a failed chunk; a chunk with some
    # failures still publishes its successes (siblings must not be lost).
    if summary["n_ok"] == 0:
        return 3
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one OligoGym benchmark chunk")
    ap.add_argument("--chunk", required=True, help="chunk descriptor JSON")
    ap.add_argument("--configs", required=True, help="CSV of this chunk's configs")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--summary", default="chunk_summary.json")
    ap.add_argument("--fold-indices", default="fold_indices.csv")
    ap.add_argument("--cache-dir", default="feature_cache")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--arm", default="primary", choices=["primary", "corrective"])
    ap.add_argument(
        "--seed", type=int, default=None,
        help="OFF by default: the published methodology seeds nothing, so "
             "seeding changes which folds and clusters are produced.",
    )
    ap.add_argument("--proper-kfold", action="store_true")
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args()
    sys.exit(run_chunk(args))


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
