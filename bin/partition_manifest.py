#!/usr/bin/env python
"""Partition the OligoGym run manifest into feature-matrix-grouped chunks.

Why chunk at all
----------------
One Nextflow task per fold-fit is 45,950 tasks for the primary arm; AWS Batch
scheduling plus container start dominates the compute for the S tier, where a
measured fold-fit is 0.8-4.6 s (calibration.csv).  Worse, it throws away the
single largest lever Phase 2 measured: featurization is **55.2 % of per-fold
wall time**, and the 9,190-config primary target contains only **239 distinct
(dataset, featurizer, featurizer_args, cross_validation) groups** -- the largest
shared by 87 configs.  Chunking *by shared feature matrix* means the matrix is
built once per chunk per fold and every model in the chunk fits against it.

The grouping key is therefore exactly the feature-cache key minus the fold:

    (dataset_downloader_key, featurizer, featurizer_args_json, cross_validation)

Sizing rule
-----------
Within a group, per-chunk wall time is modelled as

    fixed_s = n_folds * featurize_s(dataset, featurizer, cv)      # paid ONCE
    var_s   = n_folds * sum(fitpred_s(model_class, tier)) / speedup(P)

`fixed_s` is paid by every chunk a group is split into, so splitting a group with
expensive featurization is actively expensive: sherwood x OneHotEncoder measures
889.4 s per fold, i.e. 74 minutes of featurization per chunk however few configs
it holds.  The cap on a chunk is therefore not a constant but

    cap = clamp(2 * fixed_s, target_chunk_s, max_chunk_s)

which says: aim for `target_chunk_s` (default 1800 s = 30 min), but never let
featurization exceed ~50 % of a chunk, and never exceed `max_chunk_s` (default
12600 s = 3.5 h, under the 4 h task timeout with margin over the measured
3,526 s longest single config).  Small groups stay whole.

Unit costs are the MEASURED medians from Phase 2 (calibration.csv, 2,017
fold-fits on real Batch hardware).  Where a (model_class, tier) cell was not
measured -- XL for GNN/Transformer/Linear/KNN/RF -- the L value is scaled by the
measured XL/L ratio of 4.21 (median over the four classes measured at both
tiers), never linearly in rows: the linear model was tested against the Sherwood
measurements and was wrong by 2-2.5x.

Routing
-------
`compute_class` comes from the manifest (cpu = Linear/KNN/RF/XGB, gpu =
MLP/CNN/GRU/Transformer/GNN, following paper section 4.5 which puts MLP on the
GPU nodes), with one measured override: **any RNAFMEmbeddings config goes to the
GPU queue regardless of its model class**, because embedding extraction dominates
its cost and is 10.8x faster on a T4 (measured: 14.0 s vs 150.6 s per fold at the
M tier).  RNA-FM chunks carry `procs=1` -- at concurrency 2 only one of two
processes completes and at 8 none do (it peaks at 8,107.5 MB = 52.8 % of a T4).

Memory classes
--------------
Measured peak RSS at XL for configs that completed was 2.7-3.9 GB, so 8 GB is
adequate for most work.  Two measured exceptions:
  * XL tree models (RandomForest n_estimators>=500, XGBoost) start at 32 GB --
    RF x sherwood was exit-137 OOM-killed at 8 GB;
  * RNA-FM starts at 16 GB -- measured 9.66 GB peak RSS at the L tier because
    embeddings are materialised dense (n, L, 640) float32.
Those configs are split into their own chunks so a chunk is homogeneous in memory
class and the 8/32/120 GB retry ladder escalates only what needs it.

RNA-FM x sherwood is emitted as a SKIP with a reason, not as a task: the dense
embedding array alone would be ~25 GB at 291,551 rows and the cell is unmeasured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, Tuple

import pandas as pd

# --------------------------------------------------------------------------
# Measured constants from Phase 2.  Each is a median over real Batch fold-fits
# in calibration.csv; nothing here is a guess.
# --------------------------------------------------------------------------

# --- XL extrapolation: the two cost components scale DIFFERENTLY -----------
# The scaling ladder reports a measured whole-fold XL/L wall ratio of 4.21
# (recomputed here: CNN 3.48, MLP 3.97, XGBoost 4.45, GRU 4.69, median 4.21).
# That figure is a BLEND and must not be applied to either component alone --
# recomputed from calibration.csv, the components diverge by 7x:
#
#   fit+predict XL/L : CNN 0.97, GRU 1.28, MLP 1.57, XGBoost 2.84  median 1.43
#   featurize   XL/L : KMersCounts 9.91, OneHotEncoder 10.48       median 10.20
#
# So fit+predict is strongly sub-linear in rows (1.43 where the row ratio is
# 8.94) while featurization is essentially LINEAR in rows (10.20 vs 8.94, i.e.
# slightly super-linear).  That also explains the 4.21 blend: featurization is
# ~50 % of XL wall time, so the blended ratio sits between the two components.
# Using 4.21 for both -- as an earlier revision of this file did -- overestimates
# fit+predict by ~3x and underestimates featurization by ~2.4x.
#
#: measured median XL/L ratio for fit+predict, over the four model classes
#: measured at both tiers.  Spread is wide (0.97-2.84) and tree-building is at
#: the top of it, which is why RF x XL is additionally flagged unmeasured below.
XL_OVER_L_FITPRED = 1.43

#: measured median XL/L ratio for featurization, over the two featurizers
#: measured at XL.  Close to the 8.94 row ratio: featurization cost tracks rows.
XL_OVER_L_FEATURIZE = 10.20

#: measured GPU packing speedup at N concurrent trainers on one T4 (CNN
#: 1.94 -> 12.50 fits/min = 6.44x at 8; Transformer 2.11 -> 10.23 = 4.85x).
#: The conservative end of that range is used for sizing.
SPEEDUP = {1: 1.0, 2: 1.9, 4: 3.6, 8: 4.85}

#: RandomForest is only a memory risk with many trees; measured OOM (exit 137)
#: was at n_estimators >= 500 on sherwood.
RF_HEAVY_N_ESTIMATORS = 500

#: Measured p95 peak RSS per fold-fit process, non-RNA-FM, by size tier (GB).
#: From calibration.csv over 1,844 non-RNA-FM fold-fits.  Worker count is set
#: from THESE, not from VRAM: measured peak VRAM is 17-47 MB (0.31 % of a T4)
#: and is row-count invariant (CNN 17.5 MB at 192 rows, 17.9 MB at 291,551), so
#: the device is never the binding constraint.  Host RAM is.
P95_RSS_GB = {"S": 1.6, "M": 1.7, "L": 3.2, "XL": 3.9}

#: Measured RNA-FM peak RSS by tier (GB): the (n, L, 640) float32 embedding
#: array is materialised dense.  9.66 GB at 32,602 rows is measured.
RNAFM_RSS_GB = {"S": 3.5, "M": 8.1, "L": 9.7}

#: Usable RAM of the pinned instance sizes, GB.  g4dn.2xlarge and c6id.4xlarge
#: both advertise 32 GiB; ~1.5 GB is reserved for the ECS agent and the Fusion
#: client, so a task requesting the full 32 GB would never be schedulable.
INSTANCE_USABLE_GB = 30

#: Desired worker count before the memory envelope is applied.  8 is the
#: measured GPU packing optimum (CNN 1.94 -> 12.50 fits/min; device still 60-65 %
#: idle) and matches 8 single-threaded fits on a 16-vCPU c6id.4xlarge.
PROCS_DESIRED = 8

CORRECTIVE_DATASETS = (
    "immune_modulation_TLR7",
    "immune_modulation_TLR8",
    "acute_neurotox_moe_cleaned",
)

#: What the corrective transform can and cannot repair for Neurotox MOE, per
#: featurizer.  Derived by reading the featurizer source, not assumed:
#:   * KMersCounts._extract_features -> monomers['base'].str.cat() : base-only,
#:     so a full-length base sequence from `fasta` recovers the 20th base for the
#:     k-mer counts, but the modification_abundance counts still come from HELM.
#:   * RNAFMEmbeddings._helm_to_fasta -> also base-only (T->U), so it is fully
#:     correctable from `fasta`.
#:   * OneHotEncoder / HELMGraph need per-position sugar + phosphate monomer
#:     identity, which exists ONLY in HELM.  The dropped 3' nucleoside's sugar
#:     and phosphate chemistry is absent from the shipped data, so these are
#:     NOT correctable and are not pretended to be.
MOE_CORRECTION_LEVEL = {
    "RNAFMEmbeddings": "full_base_sequence",
    "KMersCounts": "partial_base_composition_only",
    "OneHotEncoder": "none_helm_truncated",
    "HELMGraph": "none_helm_truncated",
    "SMILESGraph": "none_helm_truncated",
}


def _model_class(featurizer: str, model: str) -> str:
    """Cost-model class name: RNA-FM configs are costed separately."""
    return f"RNA-FM + {model}" if featurizer == "RNAFMEmbeddings" else model


def build_cost_tables(calibration_csv: str) -> Tuple[Dict, Dict, Dict]:
    """Measured unit costs, keyed for lookup with documented fallbacks."""
    cal = pd.read_csv(calibration_csv)
    cal = cal[cal.status == "ok"].copy()
    cal["fitpred_s"] = cal.fit_s.fillna(0) + cal.predict_s.fillna(0)
    cal["model_class"] = [_model_class(f, m) for f, m in zip(cal.featurizer, cal.model)]

    featurize_exact = (
        cal.groupby(["dataset", "featurizer", "cross_validation"])
        .featurize_s.median()
        .to_dict()
    )
    featurize_tier = cal.groupby(["tier", "featurizer"]).featurize_s.median().to_dict()
    fitpred = cal.groupby(["model_class", "tier"]).fitpred_s.median().to_dict()

    # Fill unmeasured XL cells from the measured L value, each component with ITS
    # OWN measured ratio (see the constants above; they differ by 7x).
    for (mc, tier), v in list(fitpred.items()):
        if tier == "L" and (mc, "XL") not in fitpred:
            fitpred[(mc, "XL")] = v * XL_OVER_L_FITPRED
    for (tier, feat), v in list(featurize_tier.items()):
        if tier == "L" and ("XL", feat) not in featurize_tier:
            featurize_tier[("XL", feat)] = v * XL_OVER_L_FEATURIZE
    return featurize_exact, featurize_tier, fitpred


def featurize_seconds(row, featurize_exact: Dict, featurize_tier: Dict):
    """Per-fold featurization seconds for a group, plus provenance label."""
    k = (row["dataset_config_key"], row["featurizer"], row["cross_validation"])
    if k in featurize_exact:
        return float(featurize_exact[k]), "measured_exact"
    # cv is not a strong driver of featurization cost (measured: asoptimizer OHE
    # 89.7 s random vs 83.0 s nucleobase = 8 %), so falling back across cv
    # before falling back across dataset is the better approximation.
    for cv in ("random", "nucleobase"):
        k2 = (row["dataset_config_key"], row["featurizer"], cv)
        if k2 in featurize_exact:
            return float(featurize_exact[k2]), "measured_other_cv"
    k3 = (row["dataset_size_tier"], row["featurizer"])
    if k3 in featurize_tier:
        return float(featurize_tier[k3]), "measured_tier_median"
    return 5.0, "unmeasured_default"


def fitpred_seconds(row, fitpred: Dict):
    """Per-fold fit+predict seconds for one config, plus provenance label."""
    mc = _model_class(row["featurizer"], row["model"])
    tier = row["dataset_size_tier"]
    if (mc, tier) in fitpred:
        return float(fitpred[(mc, tier)]), "measured"
    order = ["S", "M", "L", "XL"]
    have = [t for t in order if (mc, t) in fitpred]
    if have:
        src = have[-1]
        steps = order.index(tier) - order.index(src)
        # Per-step scaling uses the measured fit+predict ratio.  Applying it
        # across more than one tier step compounds an already-wide measured
        # spread (0.97-2.84), so the provenance label carries the step count and
        # such estimates should be read as order-of-magnitude only.
        val = fitpred[(mc, src)] * (
            XL_OVER_L_FITPRED**steps if steps > 0 else 1.0
        )
        return float(val), f"extrapolated_from_{src}_x{max(steps, 0)}"
    return 30.0, "unmeasured_default"


def _rf_heavy(model: str, model_args_json: str) -> bool:
    if model == "XGBoostModel":
        return True
    if model != "RandomForestModel":
        return False
    try:
        args = json.loads(model_args_json)
    except (TypeError, ValueError):
        return True
    return int(args.get("n_estimators", 0)) >= RF_HEAVY_N_ESTIMATORS


def classify(row):
    """(queue, procs, mem_gb_start, mem_class) for one config.

    Worker count is the measured packing optimum capped by the measured
    per-process peak RSS against the pinned instance's usable RAM -- never by
    VRAM, which is 0.12-0.31 % of a T4 and row-count invariant.
    """
    tier = row["dataset_size_tier"]

    if row["featurizer"] == "RNAFMEmbeddings":
        # Two measured constraints, both hard:
        #  * exactly ONE RNA-FM process fits per T4 (8,107.5 MB peak VRAM =
        #    52.8 % of the device; at concurrency 2 only 1 of 2 processes
        #    completes, at 8 none do);
        #  * host RSS is 9.7 GB at the L tier because embeddings are dense.
        # It still routes to the GPU queue even for LinearModel, because
        # extraction dominates and is 10.8x faster on a T4 (14.0 s vs 150.6 s
        # per fold, measured at the M tier).
        rss = RNAFM_RSS_GB.get(tier, 9.7)
        mem = int(min(max(16, round(rss * 1.6)), INSTANCE_USABLE_GB))
        return "gpu", 1, mem, "rnafm16"

    queue = row["compute_class"]

    if tier == "XL" and _rf_heavy(row["model"], row["model_args_json"]):
        # RandomForest n_estimators>=500 on sherwood was exit-137 OOM-killed at
        # 8 GB, so the per-process requirement is a measured LOWER BOUND (>8 GB)
        # and the true figure is unknown.  These chunks therefore get the whole
        # instance envelope split over only 2 workers, and escalate from there.
        return queue, 2, INSTANCE_USABLE_GB, "xl_tree_hi"

    rss = P95_RSS_GB.get(tier, 3.9)
    procs = max(1, min(PROCS_DESIRED, int(INSTANCE_USABLE_GB // rss)))
    mem = int(min(max(8, round(procs * rss * 1.25)), INSTANCE_USABLE_GB))
    return queue, procs, mem, f"std_{tier.lower()}_p{procs}"


def partition(
    manifest: pd.DataFrame,
    arm: str,
    featurize_exact: Dict,
    featurize_tier: Dict,
    fitpred: Dict,
    n_folds: int = 5,
    target_chunk_s: float = 1800.0,
    max_chunk_s: float = 12600.0,
    min_chunk_s: float = 600.0,
    skip_rnafm_xl: bool = True,
):
    """Return (chunks, assignments, skips)."""
    df = manifest.copy()

    est = [fitpred_seconds(r, fitpred) for _, r in df.iterrows()]
    df["fitpred_s"] = [e[0] for e in est]
    df["fitpred_src"] = [e[1] for e in est]
    cls = [classify(r) for _, r in df.iterrows()]
    df["queue"] = [c[0] for c in cls]
    df["procs"] = [c[1] for c in cls]
    df["mem_gb_start"] = [c[2] for c in cls]
    df["mem_class"] = [c[3] for c in cls]

    # ---- documented skips -------------------------------------------------
    skip_mask = pd.Series(False, index=df.index)
    if skip_rnafm_xl:
        skip_mask |= (df.featurizer == "RNAFMEmbeddings") & (
            df.dataset_size_tier == "XL"
        )
    skips = df[skip_mask].copy()
    skips["skip_reason"] = (
        "RNA-FM x XL (sherwood, 291,551 rows): the dense (n, L, 640) float32 "
        "embedding array alone is ~25 GB at this row count; measured 9.66 GB "
        "peak RSS at 32,602 rows and this cell is unmeasured. "
        "Expected-infeasible; re-enable with --no-skip-rnafm-xl."
    )
    df = df[~skip_mask]

    group_cols = [
        "dataset_downloader_key",
        "featurizer",
        "featurizer_args_json",
        "cross_validation",
    ]
    chunk_rows, assign_rows = [], []

    for gkey, g in df.groupby(group_cols, sort=True):
        first = g.iloc[0]
        feat_s, feat_src = featurize_seconds(first, featurize_exact, featurize_tier)
        fixed_s = n_folds * feat_s
        # A chunk must be homogeneous in queue / memory class / worker count,
        # because those are Nextflow task-level resource declarations.
        for (queue, mem_class, procs, mem_gb), sub in g.groupby(
            ["queue", "mem_class", "procs", "mem_gb_start"], sort=True
        ):
            cap = min(max(2.0 * fixed_s, target_chunk_s), max_chunk_s)
            speed = SPEEDUP.get(procs, float(procs) * 0.6)
            budget = max(cap - fixed_s, 60.0)  # never emit a zero-config chunk

            # Largest-first packing: expensive configs go in first so the tail
            # chunk is the small one rather than one that overruns the cap.
            sub = sub.sort_values("fitpred_s", ascending=False)
            cur, cur_var, packed = [], 0.0, []
            for _, r in sub.iterrows():
                w = n_folds * r["fitpred_s"] / speed
                if cur and (cur_var + w) > budget:
                    packed.append((cur, cur_var))
                    cur, cur_var = [], 0.0
                cur.append(r)
                cur_var += w
            if cur:
                packed.append((cur, cur_var))

            for part, var_s in packed:
                ds = first["dataset_config_key"]
                sig = "|".join(
                    [arm, str(gkey[0]), str(gkey[1]), str(gkey[2]), str(gkey[3]),
                     queue, mem_class, str(procs)]
                    + sorted(r["config_hash"] for r in part)
                )
                chunk_id = (
                    f"{arm[:4]}_"
                    f"{hashlib.blake2b(sig.encode(), digest_size=8).hexdigest()}"
                )
                chunk_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "arm": arm,
                        "dataset_config_key": ds,
                        "dataset_downloader_key": gkey[0],
                        "dataset_rows": int(first["dataset_rows"]),
                        "dataset_size_tier": first["dataset_size_tier"],
                        "featurizer": gkey[1],
                        "featurizer_args_json": gkey[2],
                        "cross_validation": gkey[3],
                        "queue": queue,
                        "mem_class": mem_class,
                        "mem_gb_start": int(mem_gb),
                        "procs": int(procs),
                        "n_configs": len(part),
                        "n_fold_fits": len(part) * n_folds,
                        "models": ",".join(sorted({r["model"] for r in part})),
                        "featurize_s_per_fold": round(feat_s, 3),
                        "featurize_cost_source": feat_src,
                        "est_fixed_s": round(fixed_s, 1),
                        "est_var_s": round(var_s, 1),
                        "est_minutes": round((fixed_s + var_s) / 60.0, 2),
                        "est_serial_minutes": round((fixed_s + var_s * speed) / 60.0, 2),
                        "featurize_share": round(
                            fixed_s / max(fixed_s + var_s, 1e-9), 3
                        ),
                        "moe_correction_level": (
                            MOE_CORRECTION_LEVEL.get(gkey[1], "none_helm_truncated")
                            if (arm == "corrective" and ds == "acute_neurotox_moe_cleaned")
                            else "n/a"
                        ),
                    }
                )
                for r in part:
                    assign_rows.append(
                        {
                            "chunk_id": chunk_id,
                            "arm": arm,
                            "config_hash": r["config_hash"],
                            "dataset_config_key": r["dataset_config_key"],
                            "dataset_downloader_key": r["dataset_downloader_key"],
                            "dataset_size_tier": r["dataset_size_tier"],
                            "featurizer": r["featurizer"],
                            "featurizer_args_json": r["featurizer_args_json"],
                            "featurizer_args_repr": r["featurizer_args_repr"],
                            "model": r["model"],
                            "model_args_json": r["model_args_json"],
                            "model_args_repr": r["model_args_repr"],
                            "cross_validation": r["cross_validation"],
                            "compute_class": r["compute_class"],
                            "queue": queue,
                            "est_fitpred_s_per_fold": round(r["fitpred_s"], 3),
                            "fitpred_cost_source": r["fitpred_src"],
                        }
                    )

    chunks = pd.DataFrame(chunk_rows)
    assign = pd.DataFrame(assign_rows)
    chunks, assign = merge_small_chunks(
        chunks, assign, arm, target_chunk_s, min_chunk_s
    )
    chunks = chunks.sort_values(["queue", "est_minutes"], ascending=[True, False])
    return chunks, assign, skips


def merge_small_chunks(
    chunks: pd.DataFrame,
    assign: pd.DataFrame,
    arm: str,
    target_chunk_s: float,
    min_chunk_s: float,
):
    """Bin-pack undersized chunks so a Batch task is never mostly overhead.

    Splitting a feature group across tasks destroys the cache win, but *combining
    whole groups into one task does not*: the cache is keyed per
    (dataset, featurizer, args, cv, fold), so a task holding three whole groups
    featurizes three matrices, each exactly once, and every config in each group
    still reuses its matrix.  What merging buys is task count: measured Batch
    cold start is ~5 minutes and each task pulls the container, so a 1-minute
    task is almost entirely overhead.

    Only chunks with identical (queue, mem_class, procs) are merged -- those are
    task-level resource declarations and must stay homogeneous -- and merging
    stops at `target_chunk_s`.  Groups are never split here, so this pass cannot
    reduce cache hits.
    """
    if not len(chunks):
        return chunks, assign

    small = chunks[chunks.est_minutes * 60.0 < min_chunk_s]
    big = chunks[chunks.est_minutes * 60.0 >= min_chunk_s]
    if not len(small):
        chunks = chunks.assign(n_groups=1, merged=False)
        return chunks, assign

    merged_rows, remap = [], {}
    for (queue, mem_class, procs), grp in small.groupby(
        ["queue", "mem_class", "procs"], sort=True
    ):
        # Descending size, so the largest pieces are placed first and the
        # leftover bin is the small one.
        grp = grp.sort_values("est_minutes", ascending=False)
        bins: list[list] = []
        loads: list[float] = []
        for _, r in grp.iterrows():
            w = r["est_minutes"] * 60.0
            placed = False
            for i, load in enumerate(loads):
                if load + w <= target_chunk_s:
                    bins[i].append(r)
                    loads[i] = load + w
                    placed = True
                    break
            if not placed:
                bins.append([r])
                loads.append(w)

        for members in bins:
            if len(members) == 1:
                merged_rows.append(dict(members[0], n_groups=1, merged=False))
                continue
            sig = "|".join([arm, queue, mem_class, str(procs)]
                           + sorted(m["chunk_id"] for m in members))
            new_id = (
                f"{arm[:4]}m_"
                f"{hashlib.blake2b(sig.encode(), digest_size=8).hexdigest()}"
            )
            for m in members:
                remap[m["chunk_id"]] = new_id
            uniq = lambda col: sorted({str(m[col]) for m in members})  # noqa: E731
            ds = uniq("dataset_config_key")
            feats = uniq("featurizer")
            cvs = uniq("cross_validation")
            merged_rows.append(
                {
                    "chunk_id": new_id,
                    "arm": arm,
                    "dataset_config_key": ds[0] if len(ds) == 1 else "MIXED",
                    "dataset_downloader_key": "|".join(uniq("dataset_downloader_key")),
                    "dataset_rows": int(max(m["dataset_rows"] for m in members)),
                    "dataset_size_tier": members[0]["dataset_size_tier"],
                    "featurizer": feats[0] if len(feats) == 1 else "MIXED",
                    "featurizer_args_json": "MIXED",
                    "cross_validation": cvs[0] if len(cvs) == 1 else "MIXED",
                    "queue": queue,
                    "mem_class": mem_class,
                    "mem_gb_start": int(members[0]["mem_gb_start"]),
                    "procs": int(procs),
                    "n_configs": int(sum(m["n_configs"] for m in members)),
                    "n_fold_fits": int(sum(m["n_fold_fits"] for m in members)),
                    "models": ",".join(
                        sorted({x for m in members for x in str(m["models"]).split(",")})
                    ),
                    "featurize_s_per_fold": round(
                        sum(m["featurize_s_per_fold"] for m in members), 3
                    ),
                    "featurize_cost_source": "|".join(uniq("featurize_cost_source")),
                    "est_fixed_s": round(sum(m["est_fixed_s"] for m in members), 1),
                    "est_var_s": round(sum(m["est_var_s"] for m in members), 1),
                    "est_minutes": round(sum(m["est_minutes"] for m in members), 2),
                    "est_serial_minutes": round(
                        sum(m["est_serial_minutes"] for m in members), 2
                    ),
                    "featurize_share": round(
                        sum(m["est_fixed_s"] for m in members)
                        / max(sum(m["est_minutes"] for m in members) * 60.0, 1e-9),
                        3,
                    ),
                    "moe_correction_level": "|".join(uniq("moe_correction_level")),
                    "n_groups": len(members),
                    "merged": True,
                }
            )

    out = pd.concat(
        [big.assign(n_groups=1, merged=False), pd.DataFrame(merged_rows)],
        ignore_index=True,
    )
    if remap:
        assign = assign.assign(
            chunk_id=assign.chunk_id.map(lambda c: remap.get(c, c))
        )
    return out, assign


def select_arm(manifest: pd.DataFrame, arm: str) -> pd.DataFrame:
    """The two reproduction arms, per the gap audit's authoritative target."""
    pub = manifest[manifest.in_published_csv == True]  # noqa: E712
    if arm == "primary":
        return pub
    if arm == "corrective":
        return pub[pub.dataset_config_key.isin(CORRECTIVE_DATASETS)]
    raise SystemExit(f"unknown arm {arm!r}: expected 'primary' or 'corrective'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Partition the OligoGym run manifest")
    ap.add_argument("--manifest", required=True, help="run_manifest.parquet or .csv")
    ap.add_argument("--calibration", required=True, help="calibration.csv from Phase 2")
    ap.add_argument("--arm", default="primary", choices=["primary", "corrective"])
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--target-chunk-s", type=float, default=1800.0)
    ap.add_argument("--max-chunk-s", type=float, default=12600.0)
    ap.add_argument(
        "--min-chunk-s", type=float, default=600.0,
        help="chunks estimated below this are bin-packed together (whole groups "
             "only, so cache hits are unaffected); 0 disables merging",
    )
    ap.add_argument("--no-skip-rnafm-xl", action="store_true")
    ap.add_argument(
        "--only-datasets", default=None,
        help="comma-separated dataset_config_key filter (smoke runs)",
    )
    ap.add_argument(
        "--only-chunks", default=None,
        help="comma-separated chunk_id filter, applied after partitioning",
    )
    ap.add_argument("--max-chunks", type=int, default=None)
    args = ap.parse_args()

    read = pd.read_parquet if args.manifest.endswith(".parquet") else pd.read_csv
    manifest = read(args.manifest)
    sub = select_arm(manifest, args.arm)
    if args.only_datasets:
        keep = [s.strip() for s in args.only_datasets.split(",")]
        sub = sub[sub.dataset_config_key.isin(keep)]

    fe, ft, fp = build_cost_tables(args.calibration)
    chunks, assign, skips = partition(
        sub, args.arm, fe, ft, fp,
        n_folds=args.folds,
        target_chunk_s=args.target_chunk_s,
        max_chunk_s=args.max_chunk_s,
        min_chunk_s=args.min_chunk_s,
        skip_rnafm_xl=not args.no_skip_rnafm_xl,
    )

    # Invariants that must hold before any filtering, or the partition is wrong.
    assert assign.config_hash.is_unique, "a config was assigned to two chunks"
    assert len(assign) + len(skips) == len(sub), (
        f"config accounting broken: {len(assign)} assigned + {len(skips)} skipped "
        f"!= {len(sub)} selected"
    )
    assert chunks.chunk_id.is_unique, "duplicate chunk_id"
    assert set(assign.chunk_id) == set(chunks.chunk_id), "chunk/assignment mismatch"

    if args.only_chunks:
        keep = {s.strip() for s in args.only_chunks.split(",")}
        chunks = chunks[chunks.chunk_id.isin(keep)]
        assign = assign[assign.chunk_id.isin(keep)]
    if args.max_chunks:
        chunks = chunks.head(args.max_chunks)
        assign = assign[assign.chunk_id.isin(set(chunks.chunk_id))]

    os.makedirs(args.outdir, exist_ok=True)
    cpath = os.path.join(args.outdir, f"chunks_{args.arm}.csv")
    apath = os.path.join(args.outdir, f"chunk_assignments_{args.arm}.csv")
    chunks.to_csv(cpath, index=False)
    assign.to_csv(apath, index=False)
    if len(skips):
        skips[
            ["config_hash", "dataset_config_key", "featurizer", "model",
             "cross_validation", "dataset_size_tier", "skip_reason"]
        ].to_csv(os.path.join(args.outdir, f"skips_{args.arm}.csv"), index=False)

    print(json.dumps({
        "arm": args.arm,
        "configs_selected": int(len(sub)),
        "configs_assigned": int(len(assign)),
        "configs_skipped": int(len(skips)),
        "fold_fits": int(len(assign) * args.folds),
        "feature_groups": int(sub.groupby(group_cols_public()).ngroups),
        "chunks": int(len(chunks)),
        "chunks_by_queue": chunks.queue.value_counts().to_dict(),
        "chunks_by_mem_class": chunks.mem_class.value_counts().to_dict(),
        "configs_by_queue": assign.queue.value_counts().to_dict(),
        "est_packed_minutes_total": round(float(chunks.est_minutes.sum()), 1),
        "est_minutes_p50": round(float(chunks.est_minutes.median()), 2),
        "est_minutes_p95": round(float(chunks.est_minutes.quantile(0.95)), 2),
        "est_minutes_max": round(float(chunks.est_minutes.max()), 2),
        "merged_chunks": int(chunks.merged.sum()),
        "groups_per_chunk_max": int(chunks.n_groups.max()),
        "chunks_csv": cpath,
        "assignments_csv": apath,
    }, indent=2))


def group_cols_public():
    return [
        "dataset_downloader_key",
        "featurizer",
        "featurizer_args_json",
        "cross_validation",
    ]


if __name__ == "__main__":
    main()
