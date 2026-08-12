"""Canonical sequence dataset storage and integrity gates."""

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASE_TO_TOKEN = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
TOKEN_TO_BASE = np.asarray(list("ACGTN"))
VALID_SPLITS = {"train", "validation", "test"}
IUPAC_AMBIGUITY = set("RYSWKMBDHV")
IUPAC_TO_N = str.maketrans({base: "N" for base in IUPAC_AMBIGUITY})


class DatasetContractError(ValueError):
    pass


def sha256_path(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def public_model_snapshot_sha256(root, weight_file):
    """Hash every non-weight file in a symlink-free public-model snapshot."""
    root = Path(root).resolve()
    weight_file = Path(weight_file)
    if weight_file.name != str(weight_file) or not str(weight_file):
        raise ValueError("public model weight_file must be a basename")
    if not root.is_dir():
        raise ValueError(f"public model root is missing: {root}")
    weight_path = root / str(weight_file)
    digest = hashlib.sha256()
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"public model snapshot contains a symlink: {path}")
        if path.is_file() and path != weight_path:
            files.append(path)
    if not files:
        raise ValueError(f"public model snapshot has no non-weight files: {root}")
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(
            relative.encode("utf-8") + b"\0"
            + file_digest.hexdigest().encode("ascii") + b"\n"
        )
    return digest.hexdigest()


def _normalise_split(value):
    value = str(value).strip().lower()
    return "validation" if value in {"val", "dev", "valid"} else value


def _normalise_sequence(value, sample_id):
    sequence = str(value).upper().replace("U", "T")
    invalid = sorted(set(sequence) - set(BASE_TO_TOKEN) - IUPAC_AMBIGUITY)
    if invalid:
        raise DatasetContractError(f"unsupported DNA symbol for {sample_id}: {invalid}")
    sequence = sequence.translate(IUPAC_TO_N)
    if not sequence:
        raise DatasetContractError(f"empty DNA sequence for {sample_id}")
    return sequence


def _normalise_label(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [float(item) if isinstance(item, (float, np.floating)) else int(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            raise DatasetContractError("non-finite scalar label")
        return value
    if isinstance(value, bool):
        return int(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetContractError(f"unsupported label value: {value!r}") from error
    if not np.isfinite(parsed):
        raise DatasetContractError("non-finite scalar label")
    return int(parsed) if parsed.is_integer() else parsed


def _validate_rows(rows, allow_cross_split_exact_duplicates=False, required_splits=None,
                   allow_cross_split_groups=False):
    if not rows:
        raise DatasetContractError("dataset has no rows")
    seen_ids = set()
    sequence_split_masks = {}
    group_split_masks = {}
    split_bits = {"train": 1, "validation": 2, "test": 4}
    observed_splits = set()
    vector_width = None
    in_place = isinstance(rows, list)
    normalised = rows if in_place else []
    for index, raw in enumerate(rows):
        row = dict(raw)
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise DatasetContractError(f"row {index} has no sample_id")
        if sample_id in seen_ids:
            raise DatasetContractError(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        split = _normalise_split(row.get("split", ""))
        if split not in VALID_SPLITS:
            raise DatasetContractError(f"invalid split for {sample_id}: {split!r}")
        raw_sequence = str(row.get("sequence", "")).upper()
        sequence_iupac = Counter(base for base in raw_sequence if base in IUPAC_AMBIGUITY)
        sequence = _normalise_sequence(raw_sequence, sample_id)
        sequence_b = None
        sequence_b_iupac = Counter()
        sequence_b_u_count = 0
        if row.get("sequence_b") not in (None, ""):
            raw_sequence_b = str(row["sequence_b"]).upper()
            sequence_b_iupac.update(base for base in raw_sequence_b if base in IUPAC_AMBIGUITY)
            sequence_b_u_count = raw_sequence_b.count("U")
            sequence_b = _normalise_sequence(raw_sequence_b, sample_id + ":sequence_b")
        label = _normalise_label(row.get("label"))
        if isinstance(label, list):
            if vector_width is None:
                vector_width = len(label)
            if not label or len(label) != vector_width:
                raise DatasetContractError("inconsistent vector-label width")
            if not np.isfinite(np.asarray(label, dtype=np.float64)).all():
                raise DatasetContractError("non-finite vector label")
        sequence_digest = hashlib.sha256(sequence.encode("ascii")).digest()
        sequence_hash = sequence_digest.hex()
        sequence_split_masks[sequence_digest] = (
            sequence_split_masks.get(sequence_digest, 0) | split_bits[split]
        )
        observed_splits.add(split)
        group_id = str(row.get("group_id", "")).strip()
        if group_id:
            group_split_masks[group_id] = group_split_masks.get(group_id, 0) | split_bits[split]
        normalised_row = {
            **row,
            "sample_id": sample_id,
            "split": split,
            "sequence": sequence,
            "sequence_b": sequence_b,
            "label": label,
            "species": str(row.get("species", "unknown_plant")),
            "assembly_id": str(row.get("assembly_id", row.get("species", "unknown_plant"))),
            "group_id": group_id,
            "sequence_sha256": sequence_hash,
            "_normalization_iupac_counts": dict(sequence_iupac + sequence_b_iupac) or None,
            "_normalization_u_count": raw_sequence.count("U") + sequence_b_u_count,
        }
        if in_place:
            rows[index] = normalised_row
        else:
            normalised.append(normalised_row)
    if not allow_cross_split_exact_duplicates:
        leaked_count = sum(mask & (mask - 1) != 0 for mask in sequence_split_masks.values())
        if leaked_count:
            raise DatasetContractError(f"cross-split exact-sequence leakage: {leaked_count} sequence hashes")
    leaked_groups = [
        group for group, mask in group_split_masks.items() if mask & (mask - 1) != 0
    ]
    if leaked_groups and not allow_cross_split_groups:
        raise DatasetContractError(f"group appears in multiple splits: {leaked_groups[:5]}")
    required_splits = set(required_splits or VALID_SPLITS)
    if not required_splits <= VALID_SPLITS:
        raise DatasetContractError(f"unsupported required splits: {sorted(required_splits - VALID_SPLITS)}")
    missing_splits = required_splits - observed_splits
    if missing_splits:
        raise DatasetContractError(f"dataset lacks required splits: {sorted(missing_splits)}")
    return normalised


def materialize_rows(rows, output_dir, task_id, task_kind, source_receipt,
                     allow_cross_split_exact_duplicates=False, required_splits=None,
                     allow_cross_split_groups=False):
    output_dir = Path(output_dir).resolve()
    normalised = _validate_rows(
        rows, allow_cross_split_exact_duplicates, required_splits,
        allow_cross_split_groups,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=str(output_dir.parent)))
    try:
        sequence_path = staging / "sequences.u8"
        has_sequence_b = any(row["sequence_b"] is not None for row in normalised)
        if has_sequence_b and not all(row["sequence_b"] is not None for row in normalised):
            raise DatasetContractError("paired-sequence dataset mixes paired and unpaired rows")
        sequence_b_path = staging / "sequences_b.u8"
        samples_path = staging / "samples.tsv"
        offset = 0
        offset_b = 0
        row_count = len(normalised)
        vector_label_width = (
            len(normalised[0]["label"]) if isinstance(normalised[0]["label"], list) else None
        )
        split_counts = Counter()
        iupac_counts = Counter()
        rows_with_iupac = 0
        u_to_t_count = 0
        group_splits = defaultdict(set)
        fields = [
            "sample_id", "split", "input_path", "offset", "length",
            "input_path_b", "offset_b", "length_b", "label", "label_json",
            "species", "assembly_id", "group_id", "sequence_sha256", "metadata_json",
        ]
        with sequence_path.open("wb") as sequence_handle, (
            sequence_b_path.open("wb") if has_sequence_b else open(os.devnull, "wb")
        ) as sequence_b_handle, samples_path.open("w", encoding="utf-8", newline="") as samples_handle:
            writer = csv.DictWriter(
                samples_handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            for index, row in enumerate(normalised):
                encoded = bytes(BASE_TO_TOKEN[base] for base in row["sequence"])
                sequence_handle.write(encoded)
                encoded_b = b""
                if has_sequence_b:
                    encoded_b = bytes(BASE_TO_TOKEN[base] for base in row["sequence_b"])
                    sequence_b_handle.write(encoded_b)
                label = row["label"]
                standard_keys = {
                    "sample_id", "split", "sequence", "sequence_b", "label", "species",
                    "assembly_id", "group_id", "sequence_sha256",
                    "_normalization_iupac_counts", "_normalization_u_count",
                }
                metadata = {key: value for key, value in row.items() if key not in standard_keys}
                writer.writerow({
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "input_path": "sequences.u8",
                    "offset": offset,
                    "length": len(encoded),
                    "input_path_b": "sequences_b.u8" if has_sequence_b else "",
                    "offset_b": offset_b if has_sequence_b else "",
                    "length_b": len(encoded_b) if has_sequence_b else "",
                    "label": label if not isinstance(label, list) else "",
                    "label_json": json.dumps(label, ensure_ascii=False, separators=(",", ":")) if isinstance(label, list) else "",
                    "species": row["species"],
                    "assembly_id": row["assembly_id"],
                    "group_id": row["group_id"],
                    "sequence_sha256": row["sequence_sha256"],
                    "metadata_json": json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                })
                split_counts[row["split"]] += 1
                row_iupac = row["_normalization_iupac_counts"] or {}
                iupac_counts.update(row_iupac)
                rows_with_iupac += bool(row_iupac)
                u_to_t_count += int(row["_normalization_u_count"])
                if row["group_id"]:
                    group_splits[row["group_id"]].add(row["split"])
                offset += len(encoded)
                offset_b += len(encoded_b)
                if isinstance(normalised, list):
                    normalised[index] = None
        cross_split_groups = sum(len(splits) > 1 for splits in group_splits.values())
        ordered_splits = [key for key in ("train", "validation", "test") if split_counts[key]]
        manifest = {
            "status": "ok",
            "schema_version": "canonical-sequence-v4.1",
            "implementation_sha256": sha256_path(Path(__file__).resolve()),
            "task_id": task_id,
            "task_kind": task_kind,
            "rows": row_count,
            "total_bases": offset,
            "split_counts": {key: split_counts[key] for key in ordered_splits},
            "vector_label_width": vector_label_width,
            "source_receipt": dict(source_receipt or {}),
            "cross_split_exact_sequence_policy": "allowed_and_disclosed" if allow_cross_split_exact_duplicates else "forbidden",
            "group_split_audit": {
                "policy": "official_split_informational" if allow_cross_split_groups else "forbidden",
                "cross_split_groups": int(cross_split_groups),
            },
            "paired_sequences": has_sequence_b,
            "sequence_normalization": {
                "protocol_id": "iupac_ambiguity_to_n_and_u_to_t_v1",
                "iupac_symbols_to_n": dict(sorted(iupac_counts.items())),
                "iupac_bases_to_n": int(sum(iupac_counts.values())),
                "rows_with_iupac": int(rows_with_iupac),
                "u_bases_to_t": int(u_to_t_count),
            },
            "artifacts": {
                "samples.tsv": {"sha256": sha256_path(samples_path), "size_bytes": samples_path.stat().st_size},
                "sequences.u8": {"sha256": sha256_path(sequence_path), "size_bytes": sequence_path.stat().st_size},
            },
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if has_sequence_b:
            manifest["artifacts"]["sequences_b.u8"] = {
                "sha256": sha256_path(sequence_b_path), "size_bytes": sequence_b_path.stat().st_size,
            }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            if any(output_dir.iterdir()):
                raise DatasetContractError(f"output directory is not empty: {output_dir}")
            output_dir.rmdir()
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_canonical_rows(samples_path):
    samples_path = Path(samples_path).resolve()
    mmap_cache = {}
    rows = []
    with samples_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            input_path = Path(raw["input_path"])
            if not input_path.is_absolute():
                input_path = samples_path.parent / input_path
            key = str(input_path.resolve())
            if key not in mmap_cache:
                mmap_cache[key] = np.memmap(key, dtype=np.uint8, mode="r")
            offset = int(raw["offset"])
            length = int(raw["length"])
            values = np.asarray(mmap_cache[key][offset:offset + length], dtype=np.uint8)
            if np.any(values > 4):
                raise DatasetContractError(f"invalid sequence token in {key}")
            sequence = "".join(TOKEN_TO_BASE[values].tolist())
            sequence_b = None
            if raw.get("input_path_b"):
                input_path_b = Path(raw["input_path_b"])
                if not input_path_b.is_absolute():
                    input_path_b = samples_path.parent / input_path_b
                key_b = str(input_path_b.resolve())
                if key_b not in mmap_cache:
                    mmap_cache[key_b] = np.memmap(key_b, dtype=np.uint8, mode="r")
                offset_b = int(raw["offset_b"])
                length_b = int(raw["length_b"])
                values_b = np.asarray(mmap_cache[key_b][offset_b:offset_b + length_b], dtype=np.uint8)
                if np.any(values_b > 4):
                    raise DatasetContractError(f"invalid paired sequence token in {key_b}")
                sequence_b = "".join(TOKEN_TO_BASE[values_b].tolist())
            if raw.get("label_json"):
                label = json.loads(raw["label_json"])
            else:
                value = float(raw["label"])
                label = int(value) if value.is_integer() else value
            metadata = json.loads(raw.get("metadata_json") or "{}")
            rows.append({
                **raw, "offset": offset, "length": length, "sequence": sequence,
                "sequence_b": sequence_b, "label": label, "metadata": metadata,
            })
    return rows
