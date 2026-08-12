"""
oligogym_patch -- support module for the patched OligoGym benchmark harness.

This module contains everything the patched harness needs that is *missing or
broken* in oligogym @ 97f5b9f, kept separate from ``train_model_patched.py`` so
that the repairs are auditable:

1. ``DATASET_KEY_MAP`` / ``resolve_dataset_key`` -- the benchmark config YAMLs use
   dataset labels that ``DatasetDownloader.download`` rejects.  The map below was
   derived by numerically matching the published result table
   (benchmarks/oligogym_benchmarks.csv) against Table 2 of the paper; see
   harness_repair_notes.md for the evidence.
2. ``RNAFMEmbeddingsFixed`` -- ``oligogym.features.RNAFMEmbeddings.transform``
   falls off the end of the function (its ``else:`` branch is a copy-paste of
   ``TargetDescriptors`` code referring to undefined names) and therefore always
   returns ``None``; ``fit_transform`` is additionally shadowed by a
   ``TargetDescriptors``-style signature requiring a ``targets`` argument.  This
   subclass restores the documented behaviour (padded ``(n, max_seq_len, dim)``
   array, optional mean/max/cls pooling when ``flatten=True``).
3. Graph-feature helpers -- ``HELMGraph.transform`` returns
   ``List[Dict[str, np.ndarray]]``, so the harness' ndarray-shaped code paths
   (``len(X.shape) == 3`` reshape, ``X.shape[-1]`` input_dim) do not apply.
4. ``seed_everything`` -- optional, flag-gated determinism (see notes).
"""

from __future__ import annotations

import os
import random
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from oligogym.features import RNAFMEmbeddings


# ---------------------------------------------------------------------------
# 1. dataset key mapping
# ---------------------------------------------------------------------------

#: config-YAML dataset label -> key accepted by ``DatasetDownloader.download``.
#: Verified: with this map, the best-configuration test-PCC computed from
#: benchmarks/oligogym_benchmarks.csv reproduces all 96 cells of paper Table 2
#: exactly at 2 decimal places.
DATASET_KEY_MAP: Dict[str, str] = {
    "openASO": "openaso",
    "asoptimizer_cleaned": "asoptimizer",
    "immune_modulation_TLR7": "tlr7",
    "immune_modulation_TLR8": "tlr8",
    "cytotox_lna": "cytotox lna",
    "acute_neurotox_lna": "neurotox lna",
    "acute_neurotox_moe_cleaned": "neurotox moe",
    "siRNAmod": "sirnamod",
    "sherwood": "sherwood",
    "siRNA1": "huesken",
    "siRNA2": "ichihara",
    "siRNA3": "shmushkovich",
}

#: config-YAML labels with no shipped counterpart and no published results.
UNRESOLVABLE_KEYS: Dict[str, str] = {
    "hepatotox_lna": (
        "Appears in config.yaml and config_rnafm.yaml but not in "
        "benchmarks/oligogym_benchmarks.csv, not in all_datasets_info.csv.gz, and "
        "not in paper Table 1. The only hepatotoxicity item in the paper is "
        "reference [11] (Hagedorn et al. 2013, hepatotoxic potential of "
        "oligonucleotides), which is cited but contributes no dataset. Treated as "
        "a leftover config entry for a dataset that was not released."
    ),
    "acute_neurotox_moe": (
        "Uncleaned variant. Only the '_cleaned' label appears in the published "
        "results; the shipped MOE_Neurotox_1 file is the cleaned release "
        "(key 'Neurotox MOE'). No uncleaned source ships with the package."
    ),
    "asoptimizer": (
        "Uncleaned variant. Only 'asoptimizer_cleaned' appears in the published "
        "results; the shipped Hwang_2024_1 file (key 'ASOptimizer') is the cleaned "
        "release. Note the *downloader* key is coincidentally spelled "
        "'asoptimizer', so this config label resolves by accident -- it would "
        "silently load the cleaned data. The harness therefore rejects it "
        "explicitly rather than resolving it."
    ),
}


class DatasetKeyError(KeyError):
    """Raised when a config dataset label cannot be resolved."""


def resolve_dataset_key(config_key: str) -> str:
    """Translate a benchmark-config dataset label into a downloader key.

    Raises:
        DatasetKeyError: for labels that are known-unresolvable or unknown.
    """
    if config_key in DATASET_KEY_MAP:
        return DATASET_KEY_MAP[config_key]
    if config_key in UNRESOLVABLE_KEYS:
        raise DatasetKeyError(
            f"dataset label {config_key!r} has no shipped counterpart: "
            f"{UNRESOLVABLE_KEYS[config_key]}"
        )
    # allow a raw downloader key to pass through unchanged
    if config_key.lower() in set(DATASET_KEY_MAP.values()):
        return config_key.lower()
    raise DatasetKeyError(
        f"unknown dataset label {config_key!r}; expected one of "
        f"{sorted(DATASET_KEY_MAP)} or a downloader key "
        f"{sorted(set(DATASET_KEY_MAP.values()))}"
    )


# ---------------------------------------------------------------------------
# 2. repaired RNA-FM featurizer
# ---------------------------------------------------------------------------


class RNAFMEmbeddingsFixed(RNAFMEmbeddings):
    """``RNAFMEmbeddings`` with a working ``transform`` / ``fit_transform``.

    Upstream ``transform`` computes per-sequence embeddings and then falls into a
    dead ``else`` branch (copied from ``TargetDescriptors``) that references
    undefined names, so the function returns ``None`` for every input.  This
    subclass keeps the upstream embedding computation (``_get_embeddings_batch``,
    layer-12 representations, BOS/EOS stripped) and adds the padding / pooling /
    return that the docstring specifies.
    """

    def transform(self, oligo_list: Sequence[str]):
        if isinstance(oligo_list, np.ndarray):
            oligo_list = oligo_list.tolist()
        oligo_list = list(oligo_list)
        n = len(oligo_list)

        fasta_sequences = [self._helm_to_fasta(o) for o in oligo_list]
        valid_sequences, valid_indices = [], []
        for i, seq in enumerate(fasta_sequences):
            if seq:
                valid_sequences.append(seq)
                valid_indices.append(i)

        valid_embeddings: List[np.ndarray] = []
        for i in range(0, len(valid_sequences), self.batch_size):
            valid_embeddings.extend(
                self._get_embeddings_batch(valid_sequences[i : i + self.batch_size])
            )

        if not valid_embeddings:
            raise ValueError(
                "RNAFMEmbeddings: no HELM string could be converted to a FASTA "
                "sequence, so no embeddings could be computed."
            )

        embedding_dim = int(valid_embeddings[0].shape[1])
        max_seq_len = int(
            self.max_length
            if self.max_length is not None
            else max(e.shape[0] for e in valid_embeddings)
        )

        out = np.zeros((n, max_seq_len, embedding_dim), dtype=np.float32)
        for idx, emb in zip(valid_indices, valid_embeddings):
            length = min(int(emb.shape[0]), max_seq_len)
            out[idx, :length, :] = emb[:length]

        if not self.flatten:
            return out

        # flatten=True -> pool over the sequence axis, as documented
        if self.pooling_strategy == "mean":
            pooled = out.mean(axis=1)
        elif self.pooling_strategy == "max":
            pooled = out.max(axis=1)
        else:  # "cls" -> first token position
            pooled = out[:, 0, :]
        import pandas as pd

        return pd.DataFrame(
            pooled, columns=[f"rnafm_{i}" for i in range(embedding_dim)]
        )

    def fit_transform(self, oligo_list: Sequence[str], *args, **kwargs):
        """Stateless featurizer: ``fit_transform`` == ``transform``.

        Upstream defines a second ``fit_transform`` on this class with a
        ``TargetDescriptors`` signature (requiring ``targets``), which shadows the
        intended one; extra positional/keyword arguments are accepted and ignored
        here so either call style works.
        """
        return self.transform(oligo_list)


# ---------------------------------------------------------------------------
# 3. graph-feature helpers
# ---------------------------------------------------------------------------

GRAPH_FEATURIZERS = frozenset({"HELMGraph", "SMILESGraph"})


def is_graph_features(X: Any) -> bool:
    """True if ``X`` is a list of ``{'node_features', 'edge_index'}`` dicts."""
    return (
        isinstance(X, (list, tuple))
        and len(X) > 0
        and isinstance(X[0], dict)
        and "node_features" in X[0]
        and "edge_index" in X[0]
    )


def graph_node_feature_dim(X: Sequence[Dict[str, np.ndarray]]) -> int:
    """Node-feature dimensionality (= GNN ``input_dim``) of a graph list."""
    return int(np.asarray(X[0]["node_features"]).shape[1])


def graphs_to_pyg(
    X: Sequence[Dict[str, np.ndarray]], y: Optional[Sequence[float]] = None
):
    """Build ``torch_geometric.data.Data`` objects from HELMGraph output.

    ``oligogym.models.GNN.fit`` / ``.predict`` already perform this conversion
    internally, so the harness passes the raw dict list through.  This helper
    exists for the assertion in the harness (that every graph is convertible and
    edge indices are in range) and for callers that need the PyG objects.
    """
    import torch
    from torch_geometric.data import Data

    out = []
    for i, g in enumerate(X):
        node_features = np.asarray(g["node_features"], dtype=np.float32)
        edge_index = np.asarray(g["edge_index"], dtype=np.int64)
        if edge_index.size and edge_index.max() >= node_features.shape[0]:
            raise ValueError(
                f"graph {i}: edge_index references node "
                f"{int(edge_index.max())} but only {node_features.shape[0]} nodes"
            )
        data = Data(
            x=torch.tensor(node_features),
            edge_index=torch.tensor(edge_index),
        )
        if y is not None:
            data.y = torch.tensor([y[i]], dtype=torch.float32)
        out.append(data)
    return out


# ---------------------------------------------------------------------------
# 4. optional determinism
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch RNGs.

    OFF BY DEFAULT in the harness.  See harness_repair_notes.md: the published
    methodology seeds nothing, so enabling this changes which folds and which
    nucleobase clusters are produced, and results will not be bitwise comparable
    to unseeded runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        warnings.warn("torch not importable; torch RNG left unseeded")
