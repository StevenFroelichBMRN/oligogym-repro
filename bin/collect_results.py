#!/usr/bin/env python
"""Concatenate per-chunk parquets into one results table and join the published CSV.

Two jobs:

1.  **Concatenate** every per-chunk parquet into one table whose first 18 columns
    are exactly `benchmarks/oligogym_benchmarks.csv`'s schema, in its order, with
    provenance columns appended.  `--published-shape` writes the 18-column view
    alone, byte-comparable to the published file.

2.  **Verify the join.**  `config_hash` is recomputed here from the reproduced
    rows using the Phase 1 canonicalization (verified byte-exact both directions:
    100/100 distinct arg strings in the published CSV round-trip), and the
    published CSV is hashed the same way.  The join is then checked as a
    measurement, not an assumption -- the report states how many reproduced
    configs matched a published config, and any mismatch is written out rather
    than swallowed.

The canonicalization matters and is easy to get wrong.  `featurizer_args` and
`model_args` in the published CSV are Python **repr** strings with sorted keys,
e.g. "{'k': [1, 2, 3], 'modification_abundance': True}".  List order is
SEMANTIC (`k`, `hidden_dims`) and is preserved; only dict key order is
normalised.  Hashing the raw string would fail on whitespace or key order, and
sorting list elements would silently merge `k=[1,2,3]` with `k=[3,2,1]`.
"""

from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

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

HASH_COLUMNS = [
    "cross_validation",
    "dataset",
    "featurizer",
    "featurizer_args",
    "model",
    "model_args",
]


# ------------------------------------------------------------------ hashing
def canonical_args(s: Any) -> str:
    """Python-repr arg string -> sorted-key JSON, list order preserved.

    Phase 1's normalizer, reused verbatim so `config_hash` matches the manifest.
    `ast.literal_eval` is used rather than `eval` (no code execution) and rather
    than `json.loads` (the strings are Python reprs: single quotes, True/False).
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "{}"
    if isinstance(s, dict):
        d = s
    else:
        txt = str(s).strip()
        if not txt or txt.lower() in ("nan", "none"):
            return "{}"
        d = ast.literal_eval(txt)
    if not isinstance(d, dict):
        raise ValueError(f"expected a dict literal, got {type(d).__name__}: {s!r}")
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def config_hash(row: Dict[str, Any]) -> str:
    payload = "|".join(
        [
            str(row["cross_validation"]),
            str(row["dataset"]),
            str(row["featurizer"]),
            canonical_args(row["featurizer_args"]),
            str(row["model"]),
            canonical_args(row["model_args"]),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def add_hash(df: pd.DataFrame, colname: str = "config_hash_recomputed") -> pd.DataFrame:
    df = df.copy()
    df[colname] = [config_hash(r) for _, r in df[HASH_COLUMNS].iterrows()]
    return df


# ------------------------------------------------------------------ collect
def collect(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_parquet(p))
        except Exception as exc:
            print(f"[warn] unreadable, skipped: {p}: {type(exc).__name__}: {exc}")
    if not frames:
        raise SystemExit("no readable per-chunk parquets")
    df = pd.concat(frames, ignore_index=True)

    # Idempotence: a resumed or retried chunk can publish the same row twice.
    # Keying on (config_hash, fold, arm) and keeping the last occurrence makes
    # collection safe to re-run over a partially-rewritten prefix.
    before = len(df)
    if {"config_hash", "train_fold", "arm"} <= set(df.columns):
        df = df.drop_duplicates(subset=["arm", "config_hash", "train_fold"], keep="last")
    return df.assign(_dropped_duplicate_rows=before - len(df))


def verify_join(
    repro: pd.DataFrame, published_csv: str, arm: str = "primary"
) -> Dict[str, Any]:
    """Join reproduced rows to the published CSV on a recomputed config_hash."""
    pub = pd.read_csv(published_csv)
    pub_h = add_hash(pub)
    rep = repro[repro.arm == arm] if "arm" in repro.columns else repro
    rep_ok = rep[rep.status == "ok"] if "status" in rep.columns else rep
    rep_h = add_hash(rep_ok)

    pub_cfgs = set(pub_h.config_hash_recomputed)
    rep_cfgs = set(rep_h.config_hash_recomputed)

    out: Dict[str, Any] = {
        "arm": arm,
        "published_rows": int(len(pub)),
        "published_configs": int(len(pub_cfgs)),
        "reproduced_rows": int(len(rep_ok)),
        "reproduced_configs": int(len(rep_cfgs)),
        "configs_matched": int(len(rep_cfgs & pub_cfgs)),
        "configs_unmatched": int(len(rep_cfgs - pub_cfgs)),
        "join_rate_pct": round(
            100.0 * len(rep_cfgs & pub_cfgs) / max(len(rep_cfgs), 1), 4
        ),
        "unmatched_examples": sorted(rep_cfgs - pub_cfgs)[:10],
    }

    # Internal consistency: the hash the worker wrote must equal the hash
    # recomputed from the published-schema columns.  If these disagree, the
    # manifest's canonicalization and this file's have drifted.
    if "config_hash" in rep_h.columns:
        agree = (rep_h.config_hash == rep_h.config_hash_recomputed)
        out["worker_hash_agrees_pct"] = round(100.0 * agree.mean(), 4)
        out["worker_hash_disagreements"] = int((~agree).sum())

    # Value-level delta on the joined configs, per-config mean over folds.  The
    # harness is unseeded by design (published methodology), so per-fold values
    # are NOT expected to match; distribution-level agreement is the test.
    if out["configs_matched"]:
        keys = ["config_hash_recomputed"]
        pm = pub_h.groupby(keys).test_pearson_correlation.agg(["mean", "std", "count"])
        rm = rep_h.groupby(keys).test_pearson_correlation.agg(["mean", "std", "count"])
        j = pm.join(rm, lsuffix="_pub", rsuffix="_rep", how="inner")
        out["value_comparison"] = {
            "configs": int(len(j)),
            "mean_abs_delta_test_pearson": round(
                float((j["mean_pub"] - j["mean_rep"]).abs().mean()), 5
            ),
            "max_abs_delta_test_pearson": round(
                float((j["mean_pub"] - j["mean_rep"]).abs().max()), 5
            ),
            "folds_per_config_pub": sorted(j["count_pub"].unique().tolist()),
            "folds_per_config_rep": sorted(j["count_rep"].unique().tolist()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect OligoGym chunk results")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="per-chunk parquet paths or globs")
    ap.add_argument("--summaries", nargs="*", default=[],
                    help="per-chunk summary JSON paths or globs")
    ap.add_argument("--out", default="results.parquet")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--published-shape", default=None,
                    help="also write the 18-column published-schema view here")
    ap.add_argument("--published-csv", default=None,
                    help="oligogym_benchmarks.csv, to verify the join")
    ap.add_argument("--arm", default="primary")
    ap.add_argument("--report", default="collection_report.json")
    args = ap.parse_args()

    paths: List[str] = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    paths = [p for p in paths if os.path.exists(p)]
    df = collect(paths)

    df.to_parquet(args.out, index=False)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
    if args.published_shape:
        missing = [c for c in PUBLISHED_COLUMNS if c not in df.columns]
        if missing:
            raise SystemExit(f"cannot write published shape, missing: {missing}")
        df[PUBLISHED_COLUMNS].to_csv(args.published_shape, index=False)

    report: Dict[str, Any] = {
        "n_chunk_files": len(paths),
        "n_rows": int(len(df)),
        "n_configs": int(df.config_hash.nunique()) if "config_hash" in df else None,
        "dropped_duplicate_rows": int(df._dropped_duplicate_rows.iloc[0])
        if "_dropped_duplicate_rows" in df.columns and len(df)
        else 0,
        "schema_matches_published_prefix": list(df.columns[: len(PUBLISHED_COLUMNS)])
        == PUBLISHED_COLUMNS,
    }
    if "status" in df.columns:
        report["n_ok"] = int((df.status == "ok").sum())
        report["n_error"] = int((df.status != "ok").sum())
        errs = df[df.status != "ok"]
        if len(errs):
            report["error_examples"] = (
                errs.groupby("error").size().sort_values(ascending=False).head(10).to_dict()
            )
    for col in ("arm", "seed_used", "device_name", "instance", "correction_level"):
        if col in df.columns:
            report[f"{col}_values"] = {
                str(k): int(v) for k, v in df[col].value_counts().head(12).items()
            }

    # Cache evidence, aggregated from the per-chunk summaries.
    spaths: List[str] = []
    for pat in args.summaries:
        spaths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    if spaths:
        hits = misses = 0
        feat_s = 0.0
        per_chunk = []
        for sp in spaths:
            try:
                s = json.load(open(sp))
            except Exception:
                continue
            hits += int(s.get("cache_hits", 0))
            misses += int(s.get("cache_misses", 0))
            feat_s += float(s.get("featurize_s_total", 0.0))
            per_chunk.append(
                {
                    "chunk_id": s.get("chunk_id"),
                    "queue": s.get("queue"),
                    "n_configs": s.get("n_configs"),
                    "cache_hits": s.get("cache_hits"),
                    "cache_misses": s.get("cache_misses"),
                    "featurize_s_total": s.get("featurize_s_total"),
                    "device": (s.get("device") or {}).get("device_name"),
                    "instance": s.get("instance"),
                    "task_peak_rss_mb": s.get("task_peak_rss_mb"),
                }
            )
        report["cache"] = {
            "hits": hits,
            "misses": misses,
            "hit_rate_pct": round(100.0 * hits / max(hits + misses, 1), 2),
            "featurize_s_total": round(feat_s, 2),
            "per_chunk": per_chunk,
        }

    if args.published_csv:
        report["join"] = verify_join(df, args.published_csv, arm=args.arm)

    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
