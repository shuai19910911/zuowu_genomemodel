"""Unified publication metrics for downstream v4 task types."""

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, mean_absolute_error, mean_squared_error,
    r2_score, roc_auc_score,
)


def _finite_correlation(function, truth, prediction):
    value = float(function(truth, prediction).statistic)
    return value if math.isfinite(value) else 0.0


def _binary(truth, scores, predictions=None):
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.shape != scores.shape:
        raise ValueError("binary truth/score shape mismatch")
    predictions = (scores >= 0.5).astype(np.int64) if predictions is None else np.asarray(predictions, dtype=np.int64).reshape(-1)
    if predictions.shape != truth.shape:
        raise ValueError("binary truth/prediction shape mismatch")
    result = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, predictions)),
    }
    if len(np.unique(truth)) == 2:
        result["auroc"] = float(roc_auc_score(truth, scores))
        result["auprc"] = float(average_precision_score(truth, scores))
    else:
        result["auroc"] = 0.0
        result["auprc"] = float(truth.mean())
    top_count = max(1, int(math.ceil(0.01 * len(truth))))
    order = np.argsort(-scores, kind="mergesort")
    top = truth[order[:top_count]]
    remainder = truth[order[top_count:]]
    top_odds = (float(top.sum()) + 0.5) / (float(len(top) - top.sum()) + 0.5)
    remainder_odds = (
        (float(remainder.sum()) + 0.5)
        / (float(len(remainder) - remainder.sum()) + 0.5)
        if len(remainder) else 1.0
    )
    result["top_1pct_odds_ratio"] = top_odds / remainder_odds
    return result


def _multiclass(truth, scores=None, predictions=None):
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    if predictions is None:
        if scores is None:
            raise ValueError("multiclass metrics require scores or predictions")
        predictions = np.asarray(scores).argmax(axis=1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if predictions.shape != truth.shape:
        raise ValueError("multiclass truth/prediction shape mismatch")
    labels = np.unique(np.concatenate([truth, predictions]))
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, predictions)),
        "per_class_f1": f1_score(truth, predictions, labels=labels, average=None, zero_division=0).astype(float).tolist(),
    }


def _regression(truth, predictions):
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if truth.shape != predictions.shape:
        raise ValueError("regression shape mismatch")
    return {
        "pearson": _finite_correlation(pearsonr, truth, predictions),
        "spearman": _finite_correlation(spearmanr, truth, predictions),
        "r2": float(r2_score(truth, predictions)),
        "mae": float(mean_absolute_error(truth, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predictions))),
    }


def _multioutput_regression(truth, predictions):
    truth = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if truth.ndim != 2 or truth.shape != predictions.shape:
        raise ValueError("multioutput regression shape mismatch")
    per_pearson = [_finite_correlation(pearsonr, truth[:, i], predictions[:, i]) for i in range(truth.shape[1])]
    per_spearman = [_finite_correlation(spearmanr, truth[:, i], predictions[:, i]) for i in range(truth.shape[1])]
    per_r2 = [float(r2_score(truth[:, i], predictions[:, i])) for i in range(truth.shape[1])]
    return {
        "macro_pearson": float(np.mean(per_pearson)),
        "macro_spearman": float(np.mean(per_spearman)),
        "macro_r2": float(np.mean(per_r2)),
        "macro_mae": float(mean_absolute_error(truth, predictions)),
        "macro_rmse": float(np.sqrt(mean_squared_error(truth, predictions))),
        "per_output_pearson": per_pearson,
        "per_output_spearman": per_spearman,
    }


def _multilabel(truth, scores, predictions=None):
    truth = np.asarray(truth, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if truth.ndim != 2 or truth.shape != scores.shape:
        raise ValueError("multilabel truth/score shape mismatch")
    predictions = (scores >= 0.5).astype(np.int64) if predictions is None else np.asarray(predictions, dtype=np.int64)
    if predictions.shape != truth.shape:
        raise ValueError("multilabel truth/prediction shape mismatch")
    per_auprc = []
    for index in range(truth.shape[1]):
        per_auprc.append(float(average_precision_score(truth[:, index], scores[:, index])) if truth[:, index].sum() else 0.0)
    return {
        "macro_auprc": float(np.mean(per_auprc)),
        "micro_auprc": float(average_precision_score(truth.reshape(-1), scores.reshape(-1))),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predictions, average="micro", zero_division=0)),
        "per_label_auprc": per_auprc,
    }


def _ranking(truth, scores, group_ids):
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    if not (truth.shape == scores.shape == groups.shape):
        raise ValueError("ranking arrays have different shapes")
    ranks = []
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        if int((truth[indices] == 1).sum()) != 1:
            raise ValueError(f"ranking group must contain exactly one positive: {group}")
        order = np.argsort(-scores[indices], kind="mergesort")
        ranks.append(int(np.flatnonzero(truth[indices][order] == 1)[0]) + 1)
    ranks = np.asarray(ranks, dtype=np.int64)
    return {
        "groups": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(ranks.mean()),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "auprc": float(average_precision_score(truth, scores)),
    }


def _token_metrics(truth, predictions, boundary_tolerance):
    truth = np.asarray(truth, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if truth.ndim != 2 or truth.shape != predictions.shape:
        raise ValueError("token arrays must have matching [sample, position] shape")
    valid = truth != 255
    if not np.any(valid):
        raise ValueError("token arrays contain no evaluated positions")
    result = _multiclass(truth[valid], predictions=predictions[valid])
    true_total = predicted_total = matched_total = 0
    tolerance = int(boundary_tolerance)
    for true_row, predicted_row in zip(truth, predictions):
        row_valid = true_row != 255
        true_row = true_row[row_valid]
        predicted_row = predicted_row[row_valid]
        if not len(true_row):
            continue
        true_boundaries = (np.flatnonzero(true_row[1:] != true_row[:-1]) + 1).tolist()
        predicted_boundaries = (np.flatnonzero(predicted_row[1:] != predicted_row[:-1]) + 1).tolist()
        unmatched = set(range(len(predicted_boundaries)))
        matched = 0
        for position in true_boundaries:
            candidates = [i for i in unmatched if abs(predicted_boundaries[i] - position) <= tolerance]
            if candidates:
                selected = min(candidates, key=lambda i: (abs(predicted_boundaries[i] - position), i))
                unmatched.remove(selected)
                matched += 1
        true_total += len(true_boundaries)
        predicted_total += len(predicted_boundaries)
        matched_total += matched
    precision = matched_total / predicted_total if predicted_total else 0.0
    recall = matched_total / true_total if true_total else 0.0
    result.update({
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "boundary_tolerance_bp": tolerance,
    })
    return result


def evaluate_predictions(task_kind, truth, *, scores=None, predictions=None, group_ids=None, boundary_tolerance=20):
    if task_kind in {"binary_classification", "zero_shot_binary", "zero_shot_variant"}:
        if scores is None:
            scores = predictions
        return _binary(truth, scores, predictions)
    if task_kind == "multiclass_classification":
        return _multiclass(truth, scores, predictions)
    if task_kind == "regression":
        return _regression(truth, predictions)
    if task_kind == "multioutput_regression":
        return _multioutput_regression(truth, predictions)
    if task_kind == "multilabel_classification":
        return _multilabel(truth, scores, predictions)
    if task_kind == "candidate_ranking":
        return _ranking(truth, scores, group_ids)
    if task_kind in {"token_binary", "token_multiclass"}:
        return _token_metrics(truth, predictions, boundary_tolerance)
    raise ValueError(f"unsupported task_kind: {task_kind}")
