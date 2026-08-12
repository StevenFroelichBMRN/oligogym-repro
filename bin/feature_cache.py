#!/usr/bin/env python
"""Disk-backed feature cache for the OligoGym benchmark sweep.

Why a disk cache and not the in-process dict
--------------------------------------------
`calibrate.py --cache` keeps featurized matrices in a dict for the lifetime of
one process, so it only amortises across the 5 folds of a *single* config. The
real redundancy in this sweep is *across* configs: 9,190 target configs share
only 239 distinct (dataset, featurizer, featurizer_args, split) groups, because
every Transformer/CNN/GRU hyperparameter variant re-featurizes the identical
one-hot matrix. Capturing that requires a cache that outlives the process.

Key
---
    (dataset_downloader_key, dataset_config_key, featurizer,
     canonical(featurizer_args), cross_validation, n_folds, seed, fold)

The fold index and seed are part of the key because the train/test split -- and
therefore the matrix -- differs per fold. `canonical()` sorts dict keys so
`{"k": [1], "modification_abundance": false}` and the same dict in another
order hash identically.

Storage
-------
One `.npz` per key under `cache_dir`, named by a short blake2b digest of the
key, plus a sidecar `.json` recording the human-readable key (so a cache
directory is auditable and a stale entry is identifiable without loading it).
Graph featurizers (HELMGraph/SMILESGraph) produce lists of PyG `Data` objects
rather than arrays; those are stored with `torch.save` instead, and the loader
picks the right backend from the sidecar's `payload` field.

Concurrency
-----------
Writes go to a unique temp name in the same directory and are moved into place
with `os.replace`, which is atomic on POSIX. Two tasks featurizing the same key
concurrently therefore both succeed and the last writer wins with a complete
file -- a reader never observes a partial one. No lock is taken: recomputing a
matrix is cheap relative to the cost of a lock held across a 300 s
featurization of sherwood.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from typing import Any, Optional, Tuple

import numpy as np


def canonical(obj: Any) -> str:
    """Order-insensitive, stable JSON for use inside a cache key."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(
    config: dict,
    fold: int,
    n_folds: int,
    seed: int,
) -> dict:
    """Human-readable cache key. Hashed by `key_digest` for the filename."""
    return {
        "dataset": config.get("dataset"),
        "dataset_downloader_key": config.get("dataset_downloader_key"),
        "featurizer": config.get("featurizer"),
        "featurizer_args": canonical(config.get("featurizer_args") or {}),
        "cross_validation": config.get("cross_validation"),
        "n_folds": n_folds,
        "seed": seed,
        "fold": fold,
    }


def key_digest(key: dict) -> str:
    return hashlib.blake2b(canonical(key).encode(), digest_size=12).hexdigest()


class FeatureCache:
    """Disk-backed cache of featurized (X_train, X_test) pairs.

    Parameters
    ----------
    cache_dir : str or None
        Directory to hold cache entries. ``None`` disables the cache entirely
        (every ``get`` misses and every ``put`` is a no-op), which is how the
        A/B "no cache" arm is run through identical code.
    """

    def __init__(self, cache_dir: Optional[str]):
        self.dir = cache_dir
        self.enabled = cache_dir is not None
        self.hits = 0
        self.misses = 0
        self.bytes_written = 0
        self.load_seconds = 0.0
        self.store_seconds = 0.0
        self.write_errors = 0
        self.last_write_error = None
        if self.enabled:
            os.makedirs(cache_dir, exist_ok=True)

    # -- paths ------------------------------------------------------------
    def _paths(self, digest: str) -> Tuple[str, str, str]:
        base = os.path.join(self.dir, digest)
        return base + ".npz", base + ".pt", base + ".json"

    # -- read -------------------------------------------------------------
    def get(self, key: dict):
        """Return (X_train, X_test, extra) or None on miss."""
        if not self.enabled:
            self.misses += 1
            return None
        digest = key_digest(key)
        npz, pt, meta = self._paths(digest)
        if not os.path.exists(meta):
            self.misses += 1
            return None
        t0 = time.perf_counter()
        try:
            with open(meta) as fh:
                side = json.load(fh)
            if side.get("payload") == "torch":
                import torch

                obj = torch.load(pt, weights_only=False)
                out = (obj["train"], obj["test"], obj.get("extra"))
            else:
                with np.load(npz, allow_pickle=False) as z:
                    out = (z["train"], z["test"], None)
        except (OSError, ValueError, KeyError, EOFError):
            # A corrupt or half-written entry must degrade to a miss, never
            # crash the run: featurization is always reproducible.
            self.misses += 1
            return None
        self.hits += 1
        self.load_seconds += time.perf_counter() - t0
        return out

    # -- write ------------------------------------------------------------
    def put(self, key: dict, X_train, X_test, extra=None) -> None:
        if not self.enabled:
            return
        digest = key_digest(key)
        npz, pt, meta = self._paths(digest)
        t0 = time.perf_counter()
        is_array = isinstance(X_train, np.ndarray) and isinstance(X_test, np.ndarray)
        try:
            if is_array:
                # VERIFIED BEHAVIOUR, not assumed: np.savez APPENDS ".npz" when
                # the target name does not already end in it, so writing to
                # "<tmp>.npz.tmp" silently produces "<tmp>.npz.tmp.npz" and
                # renaming <tmp> moves the empty mkstemp placeholder instead of
                # the data. Pass a file OBJECT so numpy writes exactly there.
                fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".npz.tmp")
                with os.fdopen(fd, "wb") as fh:
                    np.savez(fh, train=X_train, test=X_test)
                os.replace(tmp, npz)
                payload = "npz"
                size = os.path.getsize(npz)
            else:
                import torch

                fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".pt.tmp")
                os.close(fd)
                torch.save({"train": X_train, "test": X_test, "extra": extra}, tmp)
                os.replace(tmp, pt)
                payload = "torch"
                size = os.path.getsize(pt)
            fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".json.tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump({"key": key, "payload": payload, "bytes": size}, fh)
            os.replace(tmp, meta)
            self.bytes_written += size
        except (OSError, ValueError, ImportError) as exc:
            # A cache that cannot write is a cache that misses -- never fatal to
            # the run. But it MUST be visible: an earlier version of this method
            # swallowed the error silently and the A/B measured a cache that was
            # never populated (bytes_written stayed 0 while hits looked
            # plausible). Record the failure so a caller can assert on it.
            self.write_errors += 1
            self.last_write_error = f"{type(exc).__name__}: {exc}"
            return
        finally:
            self.store_seconds += time.perf_counter() - t0

    # -- reporting --------------------------------------------------------
    def stats(self) -> dict:
        return {
            "cache_enabled": self.enabled,
            "cache_dir": self.dir,
            "hits": self.hits,
            "misses": self.misses,
            "bytes_written": self.bytes_written,
            "load_seconds": round(self.load_seconds, 4),
            "store_seconds": round(self.store_seconds, 4),
            "write_errors": self.write_errors,
            "last_write_error": self.last_write_error,
        }

    def assert_healthy(self) -> None:
        """Raise if the cache was enabled but never actually stored anything.

        Guards the failure mode this module already hit once: a silent write
        error makes the A/B compare 'no cache' against 'no cache'.
        """
        if not self.enabled:
            return
        if self.write_errors:
            raise RuntimeError(
                f"feature cache had {self.write_errors} write errors; "
                f"last: {self.last_write_error}"
            )
        if self.misses and self.bytes_written == 0:
            raise RuntimeError(
                f"feature cache stored 0 bytes after {self.misses} misses -- "
                "writes are failing silently, so any A/B result is invalid."
            )
