#!/usr/bin/env python
"""Explode the chunk manifest into one descriptor JSON + one config CSV per chunk.

Run as a single task rather than one per chunk: 175 extra Batch tasks to write
175 small CSVs would cost more in scheduling than the split itself.  The emitted
files are content-addressed by Nextflow, so a downstream chunk task whose two
input files are byte-identical is still skipped by `-resume`.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--compute-class", default="all", choices=["all", "cpu", "gpu"])
    ap.add_argument("--only-chunks", default=None,
                    help="comma-separated chunk_id filter (smoke runs)")
    ap.add_argument("--max-chunks", type=int, default=None)
    args = ap.parse_args()

    chunks = pd.read_csv(args.chunks)
    assign = pd.read_csv(args.assignments)
    if args.compute_class != "all":
        chunks = chunks[chunks.queue == args.compute_class]
    if args.only_chunks:
        keep = {t.strip() for t in args.only_chunks.split(",") if t.strip()}
        missing = keep - set(chunks.chunk_id)
        assert not missing, f"--only-chunks names unknown chunk_ids: {sorted(missing)}"
        chunks = chunks[chunks.chunk_id.isin(keep)]
    if args.max_chunks:
        chunks = chunks.head(args.max_chunks)
    assert len(chunks), "no chunks left after filtering"

    os.makedirs(args.outdir, exist_ok=True)
    by_chunk = dict(tuple(assign.groupby("chunk_id")))

    # A slim scheduling table with NO free-text or JSON columns.  Nextflow's
    # splitCsv cannot parse RFC-4180 doubled quotes, which the featurizer_args
    # JSON column contains ("{""flatten"":false}"), so the driver reads only
    # these scalar scheduling fields; the full descriptor travels to the task as
    # the chunk_<id>.json file, where a real JSON parser reads it.
    sched_cols = [
        "chunk_id", "arm", "queue", "mem_class", "mem_gb_start", "procs",
        "n_configs", "n_fold_fits", "dataset_size_tier", "est_minutes",
    ]
    sched = chunks[sched_cols].copy()
    for c in ("mem_gb_start", "procs", "n_configs", "n_fold_fits"):
        sched[c] = sched[c].astype(int)
    sched.to_csv(os.path.join(args.outdir, "schedule.csv"), index=False)

    written = 0
    for _, row in chunks.iterrows():
        cid = row["chunk_id"]
        sub = by_chunk.get(cid)
        # A chunk with no configs is a partitioner bug, not a runtime condition:
        # fail here rather than launch an empty task.
        assert sub is not None and len(sub), f"chunk {cid} has no assigned configs"
        assert len(sub) == int(row["n_configs"]), (
            f"chunk {cid}: manifest says {row['n_configs']} configs, "
            f"assignment table has {len(sub)}"
        )
        sub.to_csv(os.path.join(args.outdir, f"configs_{cid}.csv"), index=False)
        with open(os.path.join(args.outdir, f"chunk_{cid}.json"), "w") as fh:
            json.dump({k: (None if pd.isna(v) else v) for k, v in row.items()}, fh,
                      indent=2, default=str)
        written += 1

    print(json.dumps({
        "chunks_written": written,
        "configs_written": int(sum(len(by_chunk[c]) for c in chunks.chunk_id)),
        "compute_class": args.compute_class,
    }))


if __name__ == "__main__":
    main()
