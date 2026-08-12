"""Frozen linear-probe training for canonical embedding caches."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import sha256_path
from .feature_projection import (
    HEAD_INPUT_DIM, PROJECTION_PROTOCOL_ID, project_numpy,
)
from .metrics import evaluate_predictions


CLASSIFICATION_C = [0.01, 0.1, 1.0, 10.0]
ARTICLE_CLASSIFICATION_C = [0.01, 0.1, 1.0, 10.0, 100.0]
RIDGE_ALPHA = [0.1, 1.0, 10.0, 100.0]
PROBE_PROTOCOL_ID = "paired_seeded_train_bootstrap_hash256_linear_probe_v4"
ARTICLE_PROBE_PROTOCOL_ID = "paired_seeded_train_bootstrap_hash256_linear_probe_v5_article_few_shot"


def _normalise_splits(values):
    mapping = {"val": "validation", "dev": "validation", "valid": "validation"}
    return np.asarray([mapping.get(str(value), str(value)) for value in values])


def _model(task_kind, parameter, seed):
    if task_kind in {"binary_classification", "multiclass_classification", "candidate_ranking"}:
        return Pipeline([
            ("scale", StandardScaler()),
            ("head", LogisticRegression(
                C=float(parameter), class_weight="balanced", max_iter=3000,
                random_state=int(seed), solver="saga", tol=1e-4,
            )),
        ])
    if task_kind == "multilabel_classification":
        return Pipeline([
            ("scale", StandardScaler()),
            ("head", OneVsRestClassifier(LogisticRegression(
                C=float(parameter), class_weight="balanced", max_iter=3000,
                random_state=int(seed), solver="saga", tol=1e-4,
            ))),
        ])
    if task_kind in {"regression", "multioutput_regression"}:
        return Pipeline([
            ("scale", StandardScaler()),
            ("head", Ridge(
                alpha=float(parameter), solver="sag", max_iter=5000,
                tol=1e-4, random_state=int(seed),
            )),
        ])
    raise ValueError(f"unsupported probe task kind: {task_kind}")


def _predict(model, task_kind, features):
    if task_kind == "binary_classification" or task_kind == "candidate_ranking":
        probabilities = model.predict_proba(features)
        return probabilities[:, 1], (probabilities[:, 1] >= 0.5).astype(int)
    if task_kind == "multiclass_classification":
        probabilities = model.predict_proba(features)
        return probabilities, probabilities.argmax(axis=1)
    if task_kind == "multilabel_classification":
        probabilities = np.asarray(model.predict_proba(features))
        return probabilities, (probabilities >= 0.5).astype(int)
    predictions = np.asarray(model.predict(features))
    return predictions, predictions


def _metrics(task_kind, truth, scores, predictions, groups=None):
    return evaluate_predictions(
        task_kind, truth, scores=scores, predictions=predictions, group_ids=groups,
    )


def _predictions_from_scores(task_kind, scores):
    if task_kind in {"binary_classification", "candidate_ranking"}:
        return (scores >= 0.5).astype(int)
    if task_kind == "multiclass_classification":
        return scores.argmax(axis=1)
    if task_kind == "multilabel_classification":
        return (scores >= 0.5).astype(int)
    return scores


def _selection_value(task_kind, metrics):
    key = {
        "binary_classification": "macro_f1",
        "multiclass_classification": "macro_f1",
        "candidate_ranking": "mrr",
        "multilabel_classification": "macro_auprc",
        "regression": "pearson",
        "multioutput_regression": "macro_pearson",
    }[task_kind]
    return float(metrics[key])


def _parameter_grid(task_kind, article_mode=False):
    if "regression" in task_kind:
        return RIDGE_ALPHA
    return ARTICLE_CLASSIFICATION_C if article_mode else CLASSIFICATION_C


def _bootstrap_indices(indices, labels, groups, task_kind, seed):
    """Return deterministic paired train resamples without touching validation/test."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("cannot bootstrap an empty training set")
    rng = np.random.default_rng(int(seed))
    if groups is not None:
        values = np.asarray(groups)[indices].astype(str)
        unique_groups = sorted(set(values) - {""})
        if len(unique_groups) >= 2:
            reference_labels = np.asarray(labels)[indices]
            for _attempt in range(100):
                selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                result = np.concatenate([indices[values == group] for group in selected_groups])
                sampled_labels = np.asarray(labels)[result]
                if "regression" in task_kind:
                    return result
                if reference_labels.ndim == 1:
                    if np.array_equal(np.unique(sampled_labels), np.unique(reference_labels)):
                        return result
                elif all(
                    np.array_equal(np.unique(sampled_labels[:, column]), np.unique(reference_labels[:, column]))
                    for column in range(reference_labels.shape[1])
                ):
                    return result
            raise ValueError("group bootstrap could not preserve label support after 100 attempts")
    values = np.asarray(labels)[indices]
    if (
        task_kind in {"binary_classification", "multiclass_classification", "candidate_ranking"}
        and values.ndim == 1
    ):
        sampled = []
        for label in np.unique(values):
            members = indices[values == label]
            sampled.append(rng.choice(members, size=len(members), replace=True))
        result = np.concatenate(sampled)
        return result[rng.permutation(len(result))]
    return rng.choice(indices, size=len(indices), replace=True)


def _bootstrap_audit(indices):
    indices = np.asarray(indices, dtype=np.int64)
    return {
        "rows": int(len(indices)),
        "unique_rows": int(len(np.unique(indices))),
        "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }


def _select_parameter(task_kind, x_train, y_train, x_validation, y_validation,
                      validation_groups, seed, article_mode=False):
    trials = []
    best = None
    for parameter in _parameter_grid(task_kind, article_mode=article_mode):
        model = _model(task_kind, parameter, seed)
        model.fit(x_train, y_train)
        scores, predictions = _predict(model, task_kind, x_validation)
        metrics = _metrics(task_kind, y_validation, scores, predictions, validation_groups)
        value = _selection_value(task_kind, metrics)
        trials.append({"parameter": parameter, "selection_value": value, "metrics": metrics})
        candidate = (value, -float(parameter))
        if best is None or candidate > best[0]:
            best = (candidate, parameter)
    return best[1], trials


def _capped_train_indices(indices, labels, task_kind, cap, seed=2025):
    indices = np.asarray(indices, dtype=np.int64)
    if cap is None or len(indices) <= int(cap):
        return indices
    cap = int(cap)
    if cap <= 0:
        raise ValueError("training_row_cap must be positive")
    rng = np.random.default_rng(int(seed))
    values = np.asarray(labels)[indices]
    if task_kind in {"binary_classification", "multiclass_classification", "candidate_ranking"} and values.ndim == 1:
        classes, counts = np.unique(values, return_counts=True)
        if cap < len(classes):
            raise ValueError("training_row_cap is below the number of classes")
        raw = cap * counts.astype(float) / float(counts.sum())
        quotas = np.floor(raw).astype(int)
        quotas = np.maximum(quotas, 1)
        quotas = np.minimum(quotas, counts)
        while quotas.sum() < cap:
            candidates = [
                index for index in range(len(classes)) if quotas[index] < counts[index]
            ]
            selected = max(candidates, key=lambda index: (raw[index] - quotas[index], counts[index], -index))
            quotas[selected] += 1
        while quotas.sum() > cap:
            candidates = [index for index in range(len(classes)) if quotas[index] > 1]
            selected = min(candidates, key=lambda index: (raw[index] - quotas[index], -counts[index], index))
            quotas[selected] -= 1
        sampled = [
            rng.choice(indices[values == label], size=int(quota), replace=False)
            for label, quota in zip(classes, quotas)
        ]
        result = np.concatenate(sampled).astype(np.int64, copy=False)
        return result[rng.permutation(len(result))]
    return rng.choice(indices, size=cap, replace=False).astype(np.int64, copy=False)


def _standard_evaluation(features, labels, splits, groups, task_kind, seeds,
                         training_row_cap=None, article_mode=False):
    masks = {name: splits == name for name in ("train", "validation", "test")}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("embedding cache must contain train, validation and test")
    selected = []
    fitted = []
    raw_train_indices = np.flatnonzero(masks["train"])
    train_indices = _capped_train_indices(
        raw_train_indices, labels, task_kind, training_row_cap,
    )
    validation_indices = np.flatnonzero(masks["validation"])
    train_val_indices = np.concatenate([train_indices, validation_indices])
    training_cap_audit = {
        "requested_cap": int(training_row_cap) if training_row_cap is not None else None,
        "applied": bool(len(train_indices) < len(raw_train_indices)),
        "input_train_rows": int(len(raw_train_indices)),
        "effective_train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "test_rows": int(masks["test"].sum()),
        "effective_train_indices_sha256": hashlib.sha256(
            np.asarray(train_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    bootstrap_groups = groups if task_kind == "candidate_ranking" else None
    for seed in seeds:
        selection_indices = _bootstrap_indices(
            train_indices, labels, bootstrap_groups, task_kind, seed,
        )
        parameter, trials = _select_parameter(
            task_kind, features[selection_indices], labels[selection_indices],
            features[masks["validation"]], labels[masks["validation"]],
            groups[masks["validation"]] if groups is not None else None, seed,
            article_mode=article_mode,
        )
        model = _model(task_kind, parameter, seed)
        final_indices = _bootstrap_indices(
            train_val_indices, labels, bootstrap_groups, task_kind, int(seed) + 1_000_003,
        )
        model.fit(features[final_indices], labels[final_indices])
        fitted.append(model)
        selected.append({
            "seed": int(seed), "selected_parameter": parameter, "trials": trials,
            "selection_train_bootstrap": _bootstrap_audit(selection_indices),
            "final_train_bootstrap": _bootstrap_audit(final_indices),
        })
    x_test = features[masks["test"]]
    score_list, prediction_list = [], []
    for model in fitted:
        scores, predictions = _predict(model, task_kind, x_test)
        score_list.append(scores); prediction_list.append(predictions)
    mean_scores = np.mean(np.stack(score_list), axis=0)
    ensemble_predictions = _predictions_from_scores(task_kind, mean_scores)
    test_groups = groups[masks["test"]] if groups is not None else None
    metrics = _metrics(task_kind, labels[masks["test"]], mean_scores, ensemble_predictions, test_groups)
    seed_metrics = [
        {
            "seed": int(seed),
            "metrics": _metrics(
                task_kind, labels[masks["test"]], scores, predictions, test_groups,
            ),
        }
        for seed, scores, predictions in zip(seeds, score_list, prediction_list)
    ]
    test_indices = np.flatnonzero(masks["test"])
    return metrics, selected, 1, None, {
        "indices": test_indices, "truth": labels[masks["test"]],
        "scores": mean_scores, "predictions": ensemble_predictions,
        "groups": test_groups, "seed_metrics": seed_metrics,
        "seed_scores": np.stack(score_list),
        "seed_predictions": np.stack(prediction_list),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "training_row_cap": training_cap_audit,
    }


def _logo_evaluation(features, labels, groups, task_kind, seeds,
                     article_mode=False):
    unique_groups = sorted(set(map(str, groups)))
    if len(unique_groups) < 3:
        raise ValueError("leave-one-group-out requires at least three groups")
    all_truth, all_scores, all_predictions, all_groups, all_indices = [], [], [], [], []
    fold_receipts = []
    per_seed_scores = [[] for _ in seeds]
    for held_out in unique_groups:
        test_mask = groups.astype(str) == held_out
        remaining = [group for group in unique_groups if group != held_out]
        validation_group = remaining[-1]
        validation_mask = groups.astype(str) == validation_group
        train_mask = ~(test_mask | validation_mask)
        fold_scores = []
        fold_parameters = []
        fold_offset = int(hashlib.sha256(held_out.encode("utf-8")).hexdigest()[:8], 16)
        train_indices = np.flatnonzero(train_mask)
        final_indices_base = np.flatnonzero(~test_mask)
        for seed in seeds:
            selection_indices = _bootstrap_indices(
                train_indices, labels, groups, task_kind, int(seed) + fold_offset,
            )
            parameter, trials = _select_parameter(
                task_kind, features[selection_indices], labels[selection_indices],
                features[validation_mask], labels[validation_mask], groups[validation_mask], seed,
                article_mode=article_mode,
            )
            model = _model(task_kind, parameter, seed)
            final_indices = _bootstrap_indices(
                final_indices_base, labels, groups, task_kind,
                int(seed) + fold_offset + 1_000_003,
            )
            model.fit(features[final_indices], labels[final_indices])
            scores, _ = _predict(model, task_kind, features[test_mask])
            fold_scores.append(scores)
            fold_parameters.append({
                "seed": int(seed), "selected_parameter": parameter, "trials": trials,
                "selection_train_bootstrap": _bootstrap_audit(selection_indices),
                "final_train_bootstrap": _bootstrap_audit(final_indices),
            })
        for seed_index, scores in enumerate(fold_scores):
            per_seed_scores[seed_index].append(scores)
        mean_scores = np.mean(np.stack(fold_scores), axis=0)
        predictions = _predictions_from_scores(task_kind, mean_scores)
        all_truth.append(labels[test_mask]); all_scores.append(mean_scores)
        all_predictions.append(predictions); all_groups.append(groups[test_mask])
        all_indices.append(np.flatnonzero(test_mask))
        fold_receipts.append({"held_out_group": held_out, "inner_validation_group": validation_group, "parameters": fold_parameters})
    truth = np.concatenate(all_truth, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    ordered_groups = np.concatenate(all_groups, axis=0)
    metrics = _metrics(task_kind, truth, scores, predictions, ordered_groups)
    seed_metrics = []
    for seed, fold_scores in zip(seeds, per_seed_scores):
        seed_scores = np.concatenate(fold_scores, axis=0)
        seed_metrics.append({
            "seed": int(seed),
            "metrics": _metrics(
                task_kind, truth, seed_scores,
                _predictions_from_scores(task_kind, seed_scores), ordered_groups,
            ),
        })
    seed_score_arrays = [np.concatenate(values, axis=0) for values in per_seed_scores]
    return metrics, fold_receipts, len(unique_groups), len(unique_groups), {
        "indices": np.concatenate(all_indices, axis=0), "truth": truth,
        "scores": scores, "predictions": predictions, "groups": ordered_groups,
        "seed_metrics": seed_metrics,
        "seed_scores": np.stack(seed_score_arrays),
        "seed_predictions": np.stack([
            _predictions_from_scores(task_kind, values) for values in seed_score_arrays
        ]),
        "seeds": np.asarray(seeds, dtype=np.int64),
    }


def _few_shot_train_indices(labels, train_indices, shots_per_class, seed):
    labels = np.asarray(labels)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("few-shot probing requires one-dimensional class labels")
    shots = int(shots_per_class)
    if shots <= 0:
        raise ValueError("shots_per_class must be positive")
    rng = np.random.default_rng(int(seed))
    selected = []
    for label in sorted(np.unique(labels[train_indices]).tolist()):
        members = train_indices[labels[train_indices] == label]
        if len(members) < shots:
            raise ValueError(
                f"class {label} has {len(members)} training rows, below {shots}-shot"
            )
        selected.append(rng.choice(members, size=shots, replace=False))
    result = np.concatenate(selected).astype(np.int64, copy=False)
    return result[rng.permutation(len(result))]


def _mean_scalar_metrics(seed_metrics):
    keys = sorted(set.intersection(*[
        {key for key, value in row["metrics"].items() if isinstance(value, (int, float))}
        for row in seed_metrics
    ]))
    return {
        key: float(np.mean([row["metrics"][key] for row in seed_metrics]))
        for key in keys
    }


def _few_shot_evaluation(features, labels, splits, task_kind, seeds, regimes):
    if task_kind not in {"binary_classification", "multiclass_classification"}:
        raise ValueError("few-shot probing supports binary or multiclass classification")
    train_indices = np.flatnonzero(splits == "train")
    test_mask = splits == "test"
    if not len(train_indices) or not np.any(test_mask):
        raise ValueError("few-shot probing requires immutable train and test splits")
    rows = []
    for shots in sorted(set(map(int, regimes or ()))):
        seed_metrics = []
        for seed in seeds:
            selected = _few_shot_train_indices(
                labels, train_indices, shots_per_class=shots, seed=seed,
            )
            model = _model(task_kind, 1.0, seed)
            model.fit(features[selected], labels[selected])
            scores, predictions = _predict(model, task_kind, features[test_mask])
            seed_metrics.append({
                "seed": int(seed),
                "selected_train_rows": int(len(selected)),
                "selected_train_indices_sha256": hashlib.sha256(
                    selected.tobytes()
                ).hexdigest(),
                "metrics": _metrics(
                    task_kind, labels[test_mask], scores, predictions,
                ),
            })
        rows.append({
            "shots_per_class": int(shots),
            "regularization_C": 1.0,
            "selection_source": "training_split_only",
            "validation_and_test_subsampled": False,
            "seed_metrics": seed_metrics,
            "mean_metrics": _mean_scalar_metrics(seed_metrics),
        })
    return rows


def evaluate_embedding_cache(cache_path, task_kind, output_dir, seeds=(13, 29, 43, 71, 97),
                             cv_policy="fixed_train_validation_test",
                             few_shot_regimes=(), training_row_cap=None):
    cache_path = Path(cache_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {"embeddings", "labels", "splits", "sample_ids"}
        missing = required - set(cache.files)
        if missing:
            raise ValueError(f"embedding cache lacks keys: {sorted(missing)}")
        features = np.asarray(cache["embeddings"], dtype=np.float32)
        sample_ids = np.asarray(cache["sample_ids"]).astype(str)
        labels = np.asarray(cache["labels"])
        splits = _normalise_splits(cache["splits"])
        groups = None
        if "group_ids" in cache.files:
            groups = np.asarray(cache["group_ids"]).astype(str)
        elif "assemblies" in cache.files:
            groups = np.asarray(cache["assemblies"]).astype(str)
    if features.ndim != 2:
        raise ValueError("pooled embedding cache must have shape [sample, feature]")
    source_embedding_dim = int(features.shape[1])
    features = project_numpy(features)
    if len(features) != len(labels) or len(features) != len(splits):
        raise ValueError("embedding cache row count mismatch")
    article_mode = bool(
        few_shot_regimes or training_row_cap is not None
        or tuple(seeds) == (13, 17, 42, 123, 997)
    )
    if cv_policy == "leave_one_group_out":
        if groups is None:
            raise ValueError("leave-one-group-out requires group_ids or assemblies")
        metrics, selected, test_access_count, folds, prediction_payload = _logo_evaluation(
            features, labels, groups, task_kind, tuple(seeds),
            article_mode=article_mode,
        )
    elif cv_policy == "fixed_train_validation_test":
        metrics, selected, test_access_count, folds, prediction_payload = _standard_evaluation(
            features, labels, splits, groups, task_kind, tuple(seeds),
            training_row_cap=training_row_cap,
            article_mode=article_mode,
        )
    else:
        raise ValueError(f"unsupported cv_policy: {cv_policy}")
    few_shot_results = []
    if few_shot_regimes:
        if cv_policy != "fixed_train_validation_test":
            raise ValueError("few-shot probing currently requires a fixed split")
        few_shot_results = _few_shot_evaluation(
            features, labels, splits, task_kind, tuple(seeds), few_shot_regimes,
        )
    prediction_path = output_dir / "test_predictions.npz"
    prediction_groups = prediction_payload["groups"]
    if prediction_groups is None:
        prediction_groups = np.asarray([""] * len(prediction_payload["indices"]))
    np.savez_compressed(
        prediction_path,
        sample_ids=sample_ids[prediction_payload["indices"]],
        labels=prediction_payload["truth"], scores=prediction_payload["scores"],
        predictions=prediction_payload["predictions"],
        group_ids=np.asarray(prediction_groups).astype(str),
        seeds=prediction_payload["seeds"],
        seed_scores=prediction_payload["seed_scores"],
        seed_predictions=prediction_payload["seed_predictions"],
    )
    active_protocol_id = (
        ARTICLE_PROBE_PROTOCOL_ID
        if article_mode
        else PROBE_PROTOCOL_ID
    )
    receipt = {
        "status": "ok",
        "probe_protocol_id": active_protocol_id,
        "head_training": {
            "classification_solver": "saga",
            "regression_solver": "sag",
            "stochastic_seeded_optimisation": True,
            "paired_train_bootstrap": True,
            "feature_projection": PROJECTION_PROTOCOL_ID,
            "head_input_dim": HEAD_INPUT_DIM,
            "bootstrap_unit": "candidate_group_for_ranking_and_held_in_group_for_logo_else_stratified_row_for_single_label_classification_else_row",
            "validation_and_test_resampled": False,
        },
        "task_kind": task_kind,
        "cv_policy": cv_policy,
        "seeds": [int(seed) for seed in seeds],
        "embedding_cache": str(cache_path),
        "embedding_cache_sha256": sha256_path(cache_path),
        "rows": int(len(features)),
        "source_embedding_dim": source_embedding_dim,
        "embedding_dim": int(features.shape[1]),
        "test_access_count": int(test_access_count),
        "folds": folds,
        "selection": selected,
        "test_metrics": metrics,
        "seed_test_metrics": prediction_payload["seed_metrics"],
        "few_shot_results": few_shot_results,
        "few_shot_test_access_count": int(
            len(few_shot_results) * len(tuple(seeds))
        ),
        "training_row_cap": prediction_payload.get("training_row_cap", {
            "requested_cap": None,
            "applied": False,
        }),
        "test_predictions": str(prediction_path),
        "test_predictions_sha256": sha256_path(prediction_path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    canonical = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (output_dir / "FINAL_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return receipt
