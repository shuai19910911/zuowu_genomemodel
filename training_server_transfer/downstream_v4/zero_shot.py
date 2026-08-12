"""PlantCAD2/GPN-compatible zero-shot scoring for CropGenome-FM and HF MLMs."""

import csv
import hashlib
import importlib.util
import inspect
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .adapters import PLANTCAD2_ZERO_SHOT_SPECS
from .data import read_canonical_rows, sha256_path
from .metrics import evaluate_predictions


BASES = np.asarray(list("ACGT"))
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}
ZERO_SHOT_PROTOCOL_ID = "deterministic_zero_shot_stratified_bootstrap_v2"
ZERO_SHOT_BOOTSTRAP_REPLICATES = 1000
ZERO_SHOT_BOOTSTRAP_SEED = 20260801


def _sealed_test_rows(rows):
    selected = [row for row in rows if row.get("split") == "test"]
    if not selected:
        raise ValueError("zero-shot dataset has no sealed test rows")
    return selected


def adjust_center_positions(source_length, target_length, positions):
    source_length, target_length = int(source_length), int(target_length)
    if target_length > source_length or (source_length - target_length) % 2:
        raise ValueError("center crop requires target <= source and an even length difference")
    offset = (source_length - target_length) // 2
    adjusted = [int(position) - offset for position in positions]
    if any(position < 0 or position >= target_length for position in adjusted):
        raise ValueError("masked position lies outside cropped context")
    return adjusted


def center_crop(sequence, target_length):
    if len(sequence) < target_length:
        raise ValueError(f"sequence length {len(sequence)} is shorter than context {target_length}")
    if len(sequence) == target_length:
        return sequence
    start = (len(sequence) - target_length) // 2
    return sequence[start:start + target_length]


def motif_metrics(probabilities, sequences, positions):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(sequences), len(positions), 4):
        raise ValueError("motif probability shape mismatch")
    predicted = BASES[probabilities.argmax(axis=2)]
    truth = np.asarray([[sequence[position].upper() for position in positions] for sequence in sequences])
    valid = np.isin(truth, BASES)
    token_accuracy = float((predicted[valid] == truth[valid]).mean()) if valid.any() else 0.0
    valid_motifs = valid.all(axis=1)
    motif_accuracy = float((predicted[valid_motifs] == truth[valid_motifs]).all(axis=1).mean()) if valid_motifs.any() else 0.0
    return {
        "token_accuracy": token_accuracy, "top1_accuracy": token_accuracy,
        "motif_accuracy": motif_accuracy, "valid_motifs": int(valid_motifs.sum()),
    }


def reference_base_scores(probabilities, sequences, positions, aggregation="product"):
    probabilities = np.asarray(probabilities, dtype=float)
    values = np.zeros((len(sequences), len(positions)), dtype=float)
    for row_index, sequence in enumerate(sequences):
        for column_index, position in enumerate(positions):
            base_index = BASE_TO_INDEX.get(sequence[position].upper())
            if base_index is not None:
                values[row_index, column_index] = probabilities[row_index, column_index, base_index]
    if aggregation == "product":
        return values.prod(axis=1)
    if aggregation == "mean":
        return values.mean(axis=1)
    raise ValueError(f"unsupported aggregation: {aggregation}")


def snv_effect_scores(probabilities, reference_bases, alternate_bases):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(reference_bases), 4):
        raise ValueError("SNV probability shape mismatch")
    result = np.zeros(len(reference_bases), dtype=float)
    for index, (reference, alternate) in enumerate(zip(reference_bases, alternate_bases)):
        ref_index = BASE_TO_INDEX.get(str(reference).upper())
        alt_index = BASE_TO_INDEX.get(str(alternate).upper())
        if ref_index is None or alt_index is None:
            continue
        result[index] = math.log(max(probabilities[index, ref_index], 1e-12)) - math.log(max(probabilities[index, alt_index], 1e-12))
    return result


def structural_variant_boundary_scores(ref_sequences, mut_sequences, left_values, right_values,
                                       ref_probs, mut_probs, flanking=5):
    """Apache-2.0-compatible reimplementation of PlantCAD2 boundary scoring."""
    ref_probs = np.asarray(ref_probs, dtype=float)
    mut_probs = np.asarray(mut_probs, dtype=float)
    if ref_probs.shape != mut_probs.shape or ref_probs.ndim != 3 or ref_probs.shape[2] != 4:
        raise ValueError("SV probability arrays must have identical (n,length,4) shape")
    length = ref_probs.shape[1]
    center0 = length // 2
    mut_left0 = list(range(center0 - flanking, center0))
    mut_right0 = list(range(center0, center0 + flanking))
    scores = np.zeros(len(ref_sequences), dtype=float)
    for index in range(len(ref_sequences)):
        left_end = int(left_values[index]) - 1
        left_ref = list(range(left_end - (flanking - 1), left_end + 1))
        right_start = int(right_values[index]) + 1
        right_ref = list(range(right_start, right_start + flanking))
        mut_start = mut_left0[0]
        center_sequence = mut_sequences[index][mut_start:mut_start + 2 * flanking]
        values = []
        for offset in range(flanking):
            for ref_position, mut_position, base in (
                (left_ref[offset] - 1, mut_left0[offset], center_sequence[offset]),
                (right_ref[offset] - 1, mut_right0[offset], center_sequence[flanking + offset]),
            ):
                base_index = BASE_TO_INDEX.get(base.upper())
                if base_index is None:
                    values.append(0.0)
                    continue
                if not (0 <= ref_position < length and 0 <= mut_position < length):
                    raise ValueError("SV boundary position outside model context")
                ref_value = ref_probs[index, ref_position, base_index]
                mut_value = mut_probs[index, mut_position, base_index]
                values.append(math.log(max(mut_value, 1e-12) / max(ref_value, 1e-12)))
        scores[index] = -float(np.mean(values))
    return scores


class CropGenomeMLMScorer:
    def __init__(self, project_root, checkpoint, model_config, context_length, device="cuda"):
        project_root = Path(project_root).resolve()
        extractor_path = project_root / "scripts/extract_cropgenome_bench_v1_embeddings.py"
        spec = importlib.util.spec_from_file_location("cropgenome_v4_embedding_loader", extractor_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.train_module = module.load_train_module(project_root)
        configuration = json.loads(Path(model_config).read_text(encoding="utf-8"))
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = requested
        self.model, self.checkpoint_step, self.load_report, _ = module.build_model(
            self.train_module, configuration, Path(checkpoint), self.device,
        )
        self.context_length = int(context_length)
        self.model_id = f"CropGenomeFM_step{self.checkpoint_step}"

    def _input(self, sequences):
        if any(len(sequence) != len(sequences[0]) for sequence in sequences):
            raise ValueError("zero-shot batches require equal sequence lengths")
        values = np.asarray([[BASE_TO_INDEX.get(base.upper(), 4) for base in sequence] for sequence in sequences], dtype=np.int64)
        input_ids = torch.as_tensor(values, dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        return input_ids, attention_mask

    def masked_probabilities(self, sequences, positions):
        input_ids, attention_mask = self._input(sequences)
        masked = input_ids.clone(); masked[:, positions] = 5
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda",
        ):
            output = self.model(masked, attention_mask, return_aux=False)
            logits = output[0] if isinstance(output, (tuple, list)) else output
        selected = logits[:, positions, :4].float()
        return torch.softmax(selected, dim=-1).cpu().numpy()

    def unmasked_probabilities(self, sequences):
        input_ids, attention_mask = self._input(sequences)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda",
        ):
            output = self.model(input_ids, attention_mask, return_aux=False)
            logits = output[0] if isinstance(output, (tuple, list)) else output
        return torch.softmax(logits[..., :4].float(), dim=-1).cpu().numpy()


class HuggingFaceMLMScorer:
    def __init__(self, model_path, context_length, device="cuda",
                 inference_dtype="float16", load_dtype=None):
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
        from .model_adapters import register_model_plugin
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        dtype_by_name = {
            "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
        }
        dtype = dtype_by_name[inference_dtype] if self.device.type == "cuda" else torch.float32
        construction_dtype = (
            dtype_by_name[load_dtype] if load_dtype and self.device.type == "cuda" else dtype
        )
        plugin = register_model_plugin(model_path)
        trust_remote_code = (
            plugin.force_trust_remote_code
            if plugin.force_trust_remote_code is not None else True
        )
        config = None
        if plugin.config_overrides:
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, local_files_only=True,
            )
            for key, value in plugin.config_overrides.items():
                setattr(config, key, value)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, torch_dtype=construction_dtype,
            local_files_only=True, config=config,
        ).to(device=self.device, dtype=dtype).eval()
        self.tokenizer = plugin.tokenizer or AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, local_files_only=True,
        )
        vocabulary = self.tokenizer.get_vocab()
        self.base_ids = []
        for base in "ACGT":
            if base in vocabulary:
                self.base_ids.append(vocabulary[base])
            elif base.lower() in vocabulary:
                self.base_ids.append(vocabulary[base.lower()])
            else:
                raise ValueError(f"single-base token {base} absent from tokenizer")
        if self.tokenizer.mask_token_id is None:
            raise ValueError("model tokenizer has no mask token")
        self.context_length = int(context_length)
        self.model_id = Path(model_path).name

    def _tokenize(self, sequences):
        encoded = self.tokenizer(
            list(sequences), add_special_tokens=False, padding=False, truncation=False,
            return_attention_mask=True, return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] != len(sequences[0]):
            raise ValueError("zero-shot PlantCAD tasks require a one-token-per-base tokenizer")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _forward_inputs(self, encoded):
        if not hasattr(self, "_accepted_forward_inputs"):
            parameters = inspect.signature(self.model.forward).parameters
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            self._accepted_forward_inputs = None if accepts_kwargs else set(parameters)
        if self._accepted_forward_inputs is None:
            return encoded
        return {
            key: value for key, value in encoded.items()
            if key in self._accepted_forward_inputs
        }

    def masked_probabilities(self, sequences, positions):
        encoded = self._tokenize(sequences)
        encoded["input_ids"][:, positions] = self.tokenizer.mask_token_id
        with torch.inference_mode():
            logits = self.model(**self._forward_inputs(encoded)).logits[:, positions][:, :, self.base_ids]
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    def unmasked_probabilities(self, sequences):
        encoded = self._tokenize(sequences)
        with torch.inference_mode():
            logits = self.model(**self._forward_inputs(encoded)).logits[:, :, self.base_ids]
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()


def _batch_ranges(length, batch_size):
    for start in range(0, length, batch_size):
        yield start, min(length, start + batch_size)


def _bootstrap_indices(labels, rng, stratified):
    labels = np.asarray(labels)
    if stratified and labels.ndim == 1 and len(np.unique(labels)) > 1:
        selected = []
        for value in np.unique(labels):
            cohort = np.flatnonzero(labels == value)
            selected.extend(rng.choice(cohort, size=len(cohort), replace=True).tolist())
        return np.asarray(selected, dtype=np.int64)
    return rng.integers(0, len(labels), size=len(labels), dtype=np.int64)


def _bootstrap_primary_metric(labels, primary_metric, metric_function, output_path,
                              replicates=ZERO_SHOT_BOOTSTRAP_REPLICATES,
                              seed=ZERO_SHOT_BOOTSTRAP_SEED, stratified=True):
    rng = np.random.default_rng(int(seed)); values = []; digest = hashlib.sha256()
    for replicate in range(int(replicates)):
        indices = _bootstrap_indices(labels, rng, stratified)
        digest.update(indices.tobytes(order="C"))
        metrics = metric_function(indices)
        value = float(metrics[primary_metric])
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite zero-shot bootstrap metric: {primary_metric}")
        values.append(value)
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["replicate", "metric", "value", "source_id"])
        writer.writeheader()
        for replicate, value in enumerate(values):
            writer.writerow({
                "replicate": replicate, "metric": primary_metric,
                "value": format(value, ".17g"), "source_id": f"bootstrap_{replicate:04d}",
            })
    return {
        "method": "sealed_test_stratified_row_bootstrap_percentile_v1" if stratified else "sealed_test_row_bootstrap_percentile_v1",
        "replicates": int(replicates), "seed": int(seed),
        "primary_metric": primary_metric, "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "resample_indices_sha256": digest.hexdigest(),
        "source_data": str(output_path), "source_data_sha256": sha256_path(output_path),
    }


def run_zero_shot_task(task_id, dataset_root, scorer, output_dir, batch_size=4,
                       primary_metric=None):
    if task_id not in set(PLANTCAD2_ZERO_SHOT_SPECS) | {"D02", "D03"}:
        raise ValueError(f"unsupported zero-shot task: {task_id}")
    dataset_root = Path(dataset_root).resolve(); output_dir = Path(output_dir).resolve()
    rows = _sealed_test_rows(read_canonical_rows(dataset_root / "samples.tsv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([row["label"] for row in rows])
    all_scores = []
    motif_accumulators = []
    if task_id in {"D02", "D03"}:
        position = scorer.context_length // 2
        for start, end in _batch_ranges(len(rows), batch_size):
            sequences = [center_crop(row["sequence"], scorer.context_length) for row in rows[start:end]]
            probabilities = scorer.masked_probabilities(sequences, [position])[:, 0, :]
            references = [row["metadata"]["ref"] for row in rows[start:end]]
            alternates = [row["metadata"]["alt"] for row in rows[start:end]]
            all_scores.append(snv_effect_scores(probabilities, references, alternates))
        scores = np.concatenate(all_scores)
        metrics = evaluate_predictions("zero_shot_variant", labels, scores=scores)
    else:
        spec = PLANTCAD2_ZERO_SHOT_SPECS[task_id]
        if task_id == "C28":
            if any(len(row["sequence"]) != scorer.context_length for row in rows):
                raise ValueError("C28 official SV scoring requires full 8192-bp context")
            for start, end in _batch_ranges(len(rows), batch_size):
                batch = rows[start:end]
                ref_sequences = [row["sequence"] for row in batch]
                mut_sequences = [row["sequence_b"] for row in batch]
                ref_probs = scorer.unmasked_probabilities(ref_sequences)
                mut_probs = scorer.unmasked_probabilities(mut_sequences)
                all_scores.append(structural_variant_boundary_scores(
                    ref_sequences, mut_sequences,
                    [row["metadata"]["left"] for row in batch],
                    [row["metadata"]["right"] for row in batch],
                    ref_probs, mut_probs, flanking=spec["flanking"],
                ))
            scores = np.concatenate(all_scores)
            metrics = evaluate_predictions("zero_shot_variant", labels, scores=scores)
        else:
            original_positions = spec["mask_indexes_8192"]
            positions = adjust_center_positions(8192, scorer.context_length, original_positions)
            all_sequences = []
            all_probabilities = []
            for start, end in _batch_ranges(len(rows), batch_size):
                sequences = [center_crop(row["sequence"], scorer.context_length) for row in rows[start:end]]
                all_sequences.extend(sequences)
                all_probabilities.append(scorer.masked_probabilities(sequences, positions))
            probabilities = np.concatenate(all_probabilities, axis=0)
            if task_id in {"C17", "C18", "C19", "C20"}:
                metrics = motif_metrics(probabilities, all_sequences, positions)
                scores = reference_base_scores(probabilities, all_sequences, positions, aggregation="product")
            else:
                scores = reference_base_scores(probabilities, all_sequences, positions, aggregation="product")
                metrics = evaluate_predictions("zero_shot_binary", labels, scores=scores)
    if not np.isfinite(np.asarray(scores, dtype=float)).all():
        raise RuntimeError("non-finite zero-shot scores detected")
    if task_id in {"C17", "C18", "C19", "C20"} and not np.isfinite(probabilities).all():
        raise RuntimeError("non-finite zero-shot motif probabilities detected")
    if not primary_metric:
        raise ValueError("zero-shot evaluation requires the registry primary_metric")
    if primary_metric not in metrics:
        raise KeyError(f"primary metric {primary_metric} absent from zero-shot metrics")
    if task_id in {"C17", "C18", "C19", "C20"}:
        metric_function = lambda indices: motif_metrics(
            probabilities[indices], [all_sequences[index] for index in indices], positions,
        )
        stratified = False
    else:
        metric_kind = "zero_shot_variant" if task_id in {"C28", "D02", "D03"} else "zero_shot_binary"
        metric_function = lambda indices: evaluate_predictions(
            metric_kind, labels[indices], scores=scores[indices],
        )
        stratified = True
    bootstrap_path = output_dir / "bootstrap_primary_metric.tsv"
    bootstrap = _bootstrap_primary_metric(
        labels, primary_metric, metric_function, bootstrap_path, stratified=stratified,
    )
    predictions_path = output_dir / "predictions.npz"
    prediction_payload = {
        "sample_ids": np.asarray([row["sample_id"] for row in rows]),
        "labels": labels, "scores": scores,
    }
    if task_id in {"C17", "C18", "C19", "C20"}:
        prediction_payload["masked_base_probabilities"] = probabilities
    np.savez_compressed(predictions_path, **prediction_payload)
    receipt = {
        "status": "ok", "task_id": task_id, "model_id": scorer.model_id,
        "evaluation_protocol_id": ZERO_SHOT_PROTOCOL_ID,
        "dataset_root": str(dataset_root), "dataset_manifest_sha256": sha256_path(dataset_root / "dataset_manifest.json"),
        "rows": len(rows), "context_length": scorer.context_length, "batch_size": int(batch_size),
        "metrics": metrics, "primary_metric": primary_metric,
        "bootstrap": bootstrap, "predictions": str(predictions_path),
        "split": "test", "test_access_count": 1,
        "predictions_sha256": sha256_path(predictions_path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    canonical = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (output_dir / "FINAL_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt
