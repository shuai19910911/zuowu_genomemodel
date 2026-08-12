import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.probe import (
    ARTICLE_PROBE_PROTOCOL_ID, PROBE_PROTOCOL_ID, _bootstrap_indices,
    _model, _parameter_grid, evaluate_embedding_cache,
)


def _write_cache(path, labels, splits, groups=None):
    rng = np.random.default_rng(7)
    labels_array = np.asarray(labels)
    feature_target = labels_array[:, 0] if labels_array.ndim == 2 else labels_array
    embeddings = rng.normal(size=(len(labels), 8)).astype(np.float32)
    embeddings[:, 0] += np.asarray(feature_target, dtype=float) * 12.0
    payload = {
        "embeddings": embeddings,
        "sample_ids": np.asarray([f"s{i}" for i in range(len(labels))]),
        "labels": labels_array,
        "splits": np.asarray(splits),
        "species": np.asarray(["plant"] * len(labels)),
        "assemblies": np.asarray(groups or ["a"] * len(labels)),
    }
    np.savez_compressed(path, **payload)


def test_binary_and_regression_embedding_probes_close_receipt(tmp_path):
    splits = ["train"] * 40 + ["validation"] * 20 + ["test"] * 20
    labels = [i % 2 for i in range(80)]
    binary_cache = tmp_path / "binary.npz"
    _write_cache(binary_cache, labels, splits)
    result = evaluate_embedding_cache(binary_cache, "binary_classification", tmp_path / "binary_out", seeds=[13, 29])
    assert result["status"] == "ok"
    assert result["test_metrics"]["auroc"] > 0.75
    assert result["test_access_count"] == 1
    assert (tmp_path / "binary_out/FINAL_RECEIPT.json").is_file()
    prediction_path = tmp_path / "binary_out/test_predictions.npz"
    assert prediction_path.is_file()
    predictions = np.load(prediction_path)
    assert predictions["sample_ids"].tolist() == [f"s{i}" for i in range(60, 80)]
    assert result["test_predictions_sha256"]
    assert len(result["seed_test_metrics"]) == 2
    assert result["probe_protocol_id"] == PROBE_PROTOCOL_ID

    y = np.linspace(-1, 1, 80)
    regression_cache = tmp_path / "regression.npz"
    _write_cache(regression_cache, y, splits)
    result = evaluate_embedding_cache(regression_cache, "regression", tmp_path / "reg_out", seeds=[13])
    assert result["test_metrics"]["pearson"] > 0.7


def test_probe_seed_controls_stochastic_optimiser():
    classifier = _model("binary_classification", 1.0, 29).named_steps["head"]
    regressor = _model("regression", 1.0, 43).named_steps["head"]
    assert classifier.solver == "saga" and classifier.random_state == 29
    assert regressor.solver == "sag" and regressor.random_state == 43
    indices = np.arange(20); labels = np.asarray([0] * 10 + [1] * 10)
    first = _bootstrap_indices(indices, labels, None, "binary_classification", 13)
    repeated = _bootstrap_indices(indices, labels, None, "binary_classification", 13)
    second = _bootstrap_indices(indices, labels, None, "binary_classification", 29)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, second)
    assert np.bincount(labels[first]).tolist() == [10, 10]
    assert _parameter_grid("binary_classification") == [0.01, 0.1, 1.0, 10.0]
    assert _parameter_grid("binary_classification", article_mode=True) == [
        0.01, 0.1, 1.0, 10.0, 100.0,
    ]


def test_multilabel_and_leave_one_group_out_probes(tmp_path):
    groups = [str((i % 5) + 1) for i in range(100)]
    labels = np.asarray([[i % 2, (i // 2) % 2] for i in range(100)])
    cache = tmp_path / "multi.npz"
    _write_cache(cache, labels, ["test"] * 100, groups=groups)
    result = evaluate_embedding_cache(
        cache, "multilabel_classification", tmp_path / "logo", seeds=[13],
        cv_policy="leave_one_group_out",
    )
    assert result["status"] == "ok"
    assert result["folds"] == 5
    assert "macro_auprc" in result["test_metrics"]
    predictions = np.load(tmp_path / "logo/test_predictions.npz")
    assert len(predictions["sample_ids"]) == 100
    assert len(result["seed_test_metrics"]) == 1


def test_candidate_ranking_uses_explicit_group_ids(tmp_path):
    splits = ["train"] * 12 + ["validation"] * 8 + ["test"] * 8
    groups = [f"g{i // 4}" for i in range(28)]
    labels = [1 if i % 4 == 0 else 0 for i in range(28)]
    cache = tmp_path / "ranking.npz"
    _write_cache(cache, labels, splits, groups=groups)
    result = evaluate_embedding_cache(cache, "candidate_ranking", tmp_path / "rank", seeds=[13])
    assert 0.0 <= result["test_metrics"]["mrr"] <= 1.0
    assert result["test_metrics"]["groups"] == 2


def test_few_shot_probe_uses_fixed_test_and_five_declared_seeds(tmp_path):
    splits = ["train"] * 40 + ["validation"] * 20 + ["test"] * 20
    labels = [i % 2 for i in range(80)]
    cache = tmp_path / "few_shot.npz"
    _write_cache(cache, labels, splits)
    seeds = [13, 17, 42, 123, 997]
    result = evaluate_embedding_cache(
        cache, "binary_classification", tmp_path / "few_shot_out",
        seeds=seeds, few_shot_regimes=[1, 10],
    )
    assert [row["shots_per_class"] for row in result["few_shot_results"]] == [1, 10]
    assert result["few_shot_test_access_count"] == 10
    assert result["probe_protocol_id"] == ARTICLE_PROBE_PROTOCOL_ID
    for row in result["few_shot_results"]:
        assert [entry["seed"] for entry in row["seed_metrics"]] == seeds
        assert all(
            entry["selected_train_rows"] == 2 * row["shots_per_class"]
            for entry in row["seed_metrics"]
        )
        assert "mcc" in row["mean_metrics"]
    predictions = np.load(tmp_path / "few_shot_out/test_predictions.npz")
    assert predictions["sample_ids"].tolist() == [f"s{i}" for i in range(60, 80)]


def test_training_row_cap_never_subsamples_validation_or_test(tmp_path):
    splits = ["train"] * 40 + ["validation"] * 20 + ["test"] * 20
    labels = [i % 2 for i in range(80)]
    cache = tmp_path / "capped.npz"
    _write_cache(cache, labels, splits)
    result = evaluate_embedding_cache(
        cache, "binary_classification", tmp_path / "capped_out",
        seeds=[13], training_row_cap=20,
    )
    audit = result["training_row_cap"]
    assert audit["applied"] is True
    assert audit["input_train_rows"] == 40
    assert audit["effective_train_rows"] == 20
    assert audit["validation_rows"] == 20
    assert audit["test_rows"] == 20
    predictions = np.load(tmp_path / "capped_out/test_predictions.npz")
    assert len(predictions["sample_ids"]) == 20
