#!/usr/bin/env python
"""
Patched fork of oligogym/benchmarks/train_model.py (upstream commit 97f5b9f).

Drop-in replacement: same CLI (``--config path/to/config.yaml``), same output
files written next to the config (regression_metrics_train.csv,
regression_metrics_test.csv, train_indices.csv, test_indices.csv).

What this fork adds over upstream -- see harness_repair_notes.md for the full
rationale of each item:

  * dataset labels from the benchmark config YAMLs are translated to
    ``DatasetDownloader`` keys (upstream raises ValueError for 14 of 15 labels);
  * ``prepare_model`` gains Transformer and GNN branches, and Transformer's
    required ``seq_len`` is derived from the featurized data;
  * ``prepare_featurizer`` gains HELMGraph and RNAFMEmbeddings branches (the
    latter via the repaired ``RNAFMEmbeddingsFixed`` subclass);
  * graph-valued features (list of dicts) bypass the ndarray reshape and set
    ``input_dim`` from the node-feature dimensionality;
  * DataFrame-valued features (KMersCounts, Thermodynamics) are converted to
    ndarray before being handed to torch models;
  * the dataset is downloaded once per run instead of once per fold;
  * ``check_config`` additionally rejects KMersCounts+Transformer (2-D features
    have no sequence axis for attention; absent from the published results);
  * optional, flag-gated determinism (``--seed`` / ``--proper-kfold``), OFF by
    default so that the published methodology is reproduced unchanged.

Usage:
    python train_model_patched.py --config run_dir/config.yaml
    python train_model_patched.py --config run_dir/config.yaml --seed 0
"""

import argparse
import inspect
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

from oligogym.data import *  # noqa: F401,F403  (DatasetDownloader, Dataset)
from oligogym.features import *  # noqa: F401,F403
from oligogym.metrics import *  # noqa: F401,F403
from oligogym.models import *  # noqa: F401,F403

from oligogym_patch import (
    DATASET_KEY_MAP,
    DatasetKeyError,
    GRAPH_FEATURIZERS,
    RNAFMEmbeddingsFixed,
    graph_node_feature_dim,
    graphs_to_pyg,
    is_graph_features,
    resolve_dataset_key,
    seed_everything,
)

N_FOLDS = 5

#: models that consume flat (n_samples, n_features) input
FLAT_INPUT_MODELS = [
    "LinearModel",
    "RandomForestModel",
    "GaussianProcessModel",
    "XGBoostModel",
    "CatBoostModel",
    "NearestNeighborsModel",
    "TabPFNModel",
    "MLP",
]


class ConfigurationError(Exception):
    pass


class DatasetDownloaderError(Exception):
    pass


def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def check_config(config):
    """Upstream compatibility checks, plus two documented additions."""
    # --- upstream checks, unchanged -----------------------------------------
    if (config["featurizer"] == "KMersCounts") and (
        config["model"] in ["CNN", "GRU", "CausalCNN"]
    ):
        raise ConfigurationError(
            "KMersCounts featurizer is not compatible with CNN, GRU, or CausalCNN models"
        )
    if config["model"] == "CNN":
        if (config["model_args"].get("depth") == 2) and (
            config["model_args"].get("kernel_size", 0) > 5
        ):
            raise ConfigurationError("CNN with depth 2 requires kernel_size <= 5")

    # --- additions ----------------------------------------------------------
    # KMersCounts produces (n_samples, n_kmers): there is no sequence axis for
    # self-attention to operate over, and Transformer requires seq_len.  No
    # KMersCounts+Transformer row exists in benchmarks/oligogym_benchmarks.csv,
    # so the published grid excluded this pair de facto.
    if config["featurizer"] == "KMersCounts" and config["model"] == "Transformer":
        raise ConfigurationError(
            "KMersCounts featurizer is not compatible with the Transformer model "
            "(2-D features have no sequence axis); this pair is absent from the "
            "published results"
        )
    # Graph features only feed the GNN, and the GNN only accepts graph features.
    if (config["featurizer"] in GRAPH_FEATURIZERS) != (config["model"] == "GNN"):
        raise ConfigurationError(
            f"featurizer {config['featurizer']!r} and model {config['model']!r} are "
            "incompatible: graph featurizers pair only with GNN and vice versa"
        )


def download_data_with_retries(config, max_retries=3):
    """Load the dataset named in the config.

    Upstream passes ``config['dataset']`` straight to ``DatasetDownloader`` and
    retries 100x on failure; because the config labels are not downloader keys,
    that loop spins 100 times on a deterministic ValueError and then raises.
    Here the label is translated first, and an unresolvable label fails fast.
    Nothing is fetched over the network -- the datasets ship inside the package.
    """
    dataset_key = resolve_dataset_key(config["dataset"])
    last_error = None
    for attempt in range(max_retries):
        try:
            return DatasetDownloader().download(dataset_key)
        except Exception as exc:  # pragma: no cover - defensive, as upstream
            last_error = exc
            logging.error("Attempt %d failed with error: %s", attempt, exc)
    raise DatasetDownloaderError(
        f"Failed to load dataset {config['dataset']!r} (key {dataset_key!r}) "
        f"after {max_retries} attempts: {last_error}"
    )


def prepare_data(data, split_strategy="random", random_state=None):
    return data.split(
        split_strategy, return_index=True, random_state=random_state
    )


def prepare_data_fold(data, k, rng=None, proper_kfold=False, seed=None):
    """Upstream random-CV fold construction.

    Upstream calls ``np.random.shuffle`` on the global RNG with no seed and
    re-shuffles independently for every fold k, so the five test sets are not a
    partition (they overlap, and some samples never appear in any test set).
    That behaviour is preserved by default.  ``rng`` merely makes it
    reproducible; ``proper_kfold=True`` switches to a single permutation split
    into five disjoint folds and is a documented divergence from the published
    methodology.
    """
    X, y = data.x, data.y
    n = len(X)
    fold_size = n // N_FOLDS

    if proper_kfold:
        if seed is None:
            raise ValueError("--proper-kfold requires --seed")
        # One permutation for all five folds (independent of k), so the five test
        # sets form a true partition.  Derived from `seed`, not from `rng`, because
        # `rng` advances between folds.
        indices = np.random.default_rng(seed).permutation(n)
    else:
        indices = np.arange(n)
        if rng is None:
            np.random.shuffle(indices)  # upstream behaviour: unseeded
        else:
            rng.shuffle(indices)

    test_indices = indices[k * fold_size : (k + 1) * fold_size]
    train_indices = np.concatenate(
        [indices[: k * fold_size], indices[(k + 1) * fold_size :]]
    )
    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
        train_indices,
        test_indices,
    )


def prepare_featurizer(config):
    name = config["featurizer"]
    args = config.get("featurizer_args") or {}

    if name == "OneHotEncoder":
        return OneHotEncoder(**args)
    if name == "KMersCounts":
        return KMersCounts(**args)
    if name == "Thermodynamics":
        # Upstream passes the args dict positionally, but Thermodynamics.__init__
        # takes no arguments -> TypeError.  Ignore args (with a warning).
        if args:
            logging.warning("Thermodynamics takes no arguments; ignoring %s", args)
        return Thermodynamics()
    if name == "TargetDescriptors":
        return TargetDescriptors(**args)
    if name == "HELMGraph":
        return HELMGraph(**args)
    if name == "SMILESGraph":
        return SMILESGraph(**args)
    if name == "RNAFMEmbeddings":
        valid = inspect.signature(RNAFMEmbeddings).parameters.keys()
        return RNAFMEmbeddingsFixed(**{k: v for k, v in args.items() if k in valid})
    raise ConfigurationError(f"unknown featurizer {name!r}")


def prepare_model(config):
    """Instantiate the model, filtering model_args by the constructor signature.

    Upstream falls through to an implicit ``return None`` for any model without
    an explicit branch (Transformer, GNN, CatBoostModel), which then fails with
    ``AttributeError: 'NoneType' object has no attribute 'fit'``.
    """
    name = config["model"]
    registry = {
        "NearestNeighborsModel": NearestNeighborsModel,
        "LinearModel": LinearModel,
        "RandomForestModel": RandomForestModel,
        "GaussianProcessModel": GaussianProcessModel,
        "XGBoostModel": XGBoostModel,
        "CatBoostModel": CatBoostModel,
        "TabPFNModel": TabPFNModel,
        "CNN": CNN,
        "MLP": MLP,
        "GRU": GRU,
        "CausalCNN": CausalCNN,
        "Transformer": Transformer,
        "GNN": GNN,
    }
    if name not in registry:
        raise ConfigurationError(
            f"unknown model {name!r}; known models: {sorted(registry)}"
        )
    cls = registry[name]
    valid_args = inspect.signature(cls).parameters.keys()
    model_args = {k: v for k, v in config["model_args"].items() if k in valid_args}
    return cls(**model_args)


def _as_array(X):
    """DataFrame/list -> ndarray, leaving graph feature lists untouched."""
    if isinstance(X, pd.DataFrame):
        return X.values
    if is_graph_features(X):
        return X
    return np.asarray(X)


def featurize(X_train, X_test, featurizer, config):
    """Featurize and set the model's input shape arguments.

    Upstream does ``config['model_args']['input_dim'] = X_train.shape[-1]`` in
    ``main`` unconditionally, which breaks for graph features (a list of dicts
    has no ``.shape``) and never supplies Transformer's required ``seq_len``.
    Both are handled here, next to the featurization that determines them.
    """
    X_train = featurizer.fit_transform(X_train)
    X_test = featurizer.transform(X_test)

    if config["featurizer"] in GRAPH_FEATURIZERS:
        if not is_graph_features(X_train):
            raise ConfigurationError(
                f"{config['featurizer']} did not return graph dictionaries"
            )
        # validate convertibility; GNN.fit builds its own PyG objects
        graphs_to_pyg(X_train[:1])
        config["model_args"]["input_dim"] = graph_node_feature_dim(X_train)
        return X_train, X_test

    X_train, X_test = _as_array(X_train), _as_array(X_test)

    if config["model"] in FLAT_INPUT_MODELS and len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    config["model_args"]["input_dim"] = X_train.shape[-1]
    if config["model"] == "Transformer":
        if len(X_train.shape) != 3:
            raise ConfigurationError(
                f"Transformer requires 3-D (n, seq_len, features) input; got "
                f"shape {X_train.shape} from featurizer {config['featurizer']!r}"
            )
        config["model_args"]["seq_len"] = int(X_train.shape[1])
    return X_train, X_test


def predict(model, X_train, X_test, y_train, fit_kwargs=None):
    """Fit and predict, filtering ``fit_kwargs`` by the model's fit signature.

    ``SKLearnModel.fit`` and ``LightningModel.fit`` take different arguments
    (e.g. only the latter accepts ``verbose``/``max_epochs``), so callers can
    pass a single dict of training options for any model class.
    """
    kwargs = dict(fit_kwargs or {})
    if kwargs:
        params = inspect.signature(model.fit).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in params}
        else:
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in params or _accepts_trainer_kwarg(model, k)
            }
    model.fit(X_train, y_train, **kwargs)
    return model.predict(X_train), model.predict(X_test)


def _accepts_trainer_kwarg(model, name):
    """LightningModel.fit forwards **trainer_kwargs to pl.Trainer."""
    if not isinstance(model, LightningModel):
        return False
    import pytorch_lightning as pl

    return name in inspect.signature(pl.Trainer).parameters


def _metrics_frame(y_true, y_pred, fold):
    return pd.DataFrame(
        regression_metrics(np.asarray(y_true).squeeze(), np.asarray(y_pred).squeeze()),
        index=[0],
    ).assign(fold=fold)


def run_fold(config, data, k, rng=None, proper_kfold=False, fit_kwargs=None, seed=None):
    """One fold / repeat.  Returns (train metrics, test metrics, idx_tr, idx_te)."""
    if config["cross_validation"] == "random":
        X_train, X_test, y_train, y_test, idx_tr, idx_te = prepare_data_fold(
            data, k, rng=rng, proper_kfold=proper_kfold, seed=seed
        )
    else:
        random_state = None if rng is None else int(rng.integers(0, 2**31 - 1))
        X_train, X_test, y_train, y_test, idx_tr, idx_te = prepare_data(
            data,
            split_strategy=config["cross_validation"],
            random_state=random_state,
        )

    featurizer = prepare_featurizer(config)
    X_train, X_test = featurize(X_train, X_test, featurizer, config)
    model = prepare_model(config)
    y_pred_train, y_pred_test = predict(
        model, X_train, X_test, y_train, fit_kwargs=fit_kwargs
    )
    return (
        _metrics_frame(y_train, y_pred_train, k),
        _metrics_frame(y_test, y_pred_test, k),
        idx_tr,
        idx_te,
    )


def main(config_path, seed=None, proper_kfold=False, fit_kwargs=None):
    results_dir = os.path.abspath(os.path.dirname(config_path) or ".")
    config = load_yaml(config_path)
    check_config(config)

    rng = None
    if seed is not None:
        seed_everything(seed)
        rng = np.random.default_rng(seed)
    if proper_kfold and rng is None:
        raise ConfigurationError("--proper-kfold requires --seed")

    # Upstream downloads the dataset inside the fold loop (5x the identical
    # package-resource read); once is enough and keeps folds comparable.
    data = download_data_with_retries(config)

    cv = config["cross_validation"]
    if cv not in ("random", "nucleobase", "none"):
        raise ConfigurationError(f"unknown cross_validation {cv!r}")

    if cv == "none":
        featurizer = prepare_featurizer(config)
        X_train, X_test, y_train, y_test, idx_tr, idx_te = prepare_data(
            data, random_state=None if rng is None else seed
        )
        X_train, X_test = featurize(X_train, X_test, featurizer, config)
        model = prepare_model(config)
        y_pred_train, y_pred_test = predict(
            model, X_train, X_test, y_train, fit_kwargs=fit_kwargs
        )
        _metrics_frame(y_train, y_pred_train, 0).drop(columns="fold").to_csv(
            os.path.join(results_dir, "regression_metrics_train.csv"), index=False
        )
        _metrics_frame(y_test, y_pred_test, 0).drop(columns="fold").to_csv(
            os.path.join(results_dir, "regression_metrics_test.csv"), index=False
        )
        pd.DataFrame({"train_indices": idx_tr}).to_csv(
            os.path.join(results_dir, "train_indices.csv"), index=False
        )
        pd.DataFrame({"test_indices": idx_te}).to_csv(
            os.path.join(results_dir, "test_indices.csv"), index=False
        )
        return None

    train_metrics, test_metrics, train_idx, test_idx = [], [], [], []
    for k in range(N_FOLDS):
        m_tr, m_te, idx_tr, idx_te = run_fold(
            config,
            data,
            k,
            rng=rng,
            proper_kfold=proper_kfold,
            fit_kwargs=fit_kwargs,
            seed=seed,
        )
        train_metrics.append(m_tr)
        test_metrics.append(m_te)
        train_idx.append(pd.DataFrame({f"fold_{k}": idx_tr}))
        test_idx.append(pd.DataFrame({f"fold_{k}": idx_te}))

    pd.concat(train_idx, axis=1).to_csv(
        os.path.join(results_dir, "train_indices.csv"), index=False
    )
    pd.concat(test_idx, axis=1).to_csv(
        os.path.join(results_dir, "test_indices.csv"), index=False
    )
    pd.concat(train_metrics).to_csv(
        os.path.join(results_dir, "regression_metrics_train.csv"), index=False
    )
    pd.concat(test_metrics).to_csv(
        os.path.join(results_dir, "regression_metrics_test.csv"), index=False
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one OligoGym benchmark configuration"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to a run config.yaml"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed. OFF by default: the published methodology seeds "
        "nothing, so seeding changes which folds/clusters are produced.",
    )
    parser.add_argument(
        "--proper-kfold",
        action="store_true",
        help="Use disjoint 5-fold test sets instead of upstream's independent "
        "per-fold reshuffle. Requires --seed. Diverges from the published method.",
    )
    args = parser.parse_args()
    main(args.config, seed=args.seed, proper_kfold=args.proper_kfold)
