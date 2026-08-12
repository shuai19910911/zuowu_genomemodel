"""Revision-pinned public-source patterns and format adapters."""

import csv
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SOURCE_PATTERNS_BY_TASK = {
    "A04": ["pro_seq/*.fa"],
    "A05": ["lncrna/*.fa"],
    "A06": ["gene_exp/arabidopsis_thaliana_*.fa"],
    "A07": ["gene_exp/glycine_max_*.fa"],
    "A08": ["gene_exp/oryza_sativa_*.fa"],
    "A09": ["gene_exp/solanum_lycopersicum_*.fa"],
    "A10": ["gene_exp/zea_mays_*.fa"],
    "C01": ["poly_a/*.fa"],
    "C02": ["splicing/*.fa"],
    "C03": ["promoter_strength/*.fa"],
    "C04": ["terminator_strength/*.fa"],
    "C05": ["chromatin_access/*.fa"],
    "C06": ["train.csv", "dev.csv", "test.csv"],
    "C07": ["train.csv", "dev.csv", "test.csv"],
    "C08": ["H3K27ac/*.csv"],
    "C09": ["H3K27me3/*.csv"],
    "C10": ["H3K4me3/*.csv"],
    "C11": ["full_datasets/*.csv"],
    "C12": ["TIS/*.tsv"],
    "C13": ["TTS/*.tsv"],
    "C14": ["Donor/*.tsv"],
    "C15": ["Acceptor/*.tsv"],
    "C16": ["Evolutionary_constraint/*.tsv"],
    "C17": ["tis_recovery/*.parquet"],
    "C18": ["tts_recovery/*.parquet"],
    "C19": ["donor_recovery/*.parquet"],
    "C20": ["acceptor_recovery/*.parquet"],
    "C21": ["tis_core_noncore_classification/*.parquet"],
    "C22": ["tts_core_noncore_classification/*.parquet"],
    "C23": ["donor_core_noncore_classification/*.parquet"],
    "C24": ["acceptor_core_noncore_classification/*.parquet"],
    "C25": ["conservation_within_poaceae_non_tis/*.parquet"],
    "C26": ["conservation_within_andropogoneae/*.parquet"],
    "C27": ["conservation_within_poaceae_tis/*.parquet"],
    "C28": ["structural_variant_effect_prediction/*.parquet"],
    "C29": ["cross_species_acr_train_on_arabidopsis/*.parquet"],
    "C30": ["cross_species_acr_train_on_nine_species/*.parquet"],
    "C31": ["cell_type_specific_acr/*.parquet"],
    "C32": ["cross_species_leaf_on_off_expression/*.parquet"],
    "C33": ["cross_species_leaf_absolute_expression/*.parquet"],
    "C34": ["cross_species_leaf_on_off_translation/*.parquet"],
    "C35": ["cross_species_leaf_absolute_translation/*.parquet"],
    "C36": ["train.parquet", "validation.parquet", "test.parquet", "labels.txt"],
    "D01": ["embedding/windows.parquet"],
    "D02": ["variants/filt.processed.parquet"],
    "D03": ["aragwas/coordinates.parquet", "aragwas/api_chr*.csv.gz", "aragwas/LD_100000_0.1.npz"],
}


PLANTCAD2_ZERO_SHOT_SPECS = {
    "C17": {"dataset_config": "tis_recovery", "mask_indexes_8192": [4094, 4095, 4096], "motif_len": 3},
    "C18": {"dataset_config": "tts_recovery", "mask_indexes_8192": [4094, 4095, 4096], "motif_len": 3},
    "C19": {"dataset_config": "donor_recovery", "mask_indexes_8192": [4095, 4096], "motif_len": 2},
    "C20": {"dataset_config": "acceptor_recovery", "mask_indexes_8192": [4095, 4096], "motif_len": 2},
    "C21": {"dataset_config": "tis_core_noncore_classification", "mask_indexes_8192": [4094, 4095, 4096], "motif_len": 3},
    "C22": {"dataset_config": "tts_core_noncore_classification", "mask_indexes_8192": [4094, 4095, 4096], "motif_len": 3},
    "C23": {"dataset_config": "donor_core_noncore_classification", "mask_indexes_8192": [4095, 4096], "motif_len": 2},
    "C24": {"dataset_config": "acceptor_core_noncore_classification", "mask_indexes_8192": [4095, 4096], "motif_len": 2},
    "C25": {"dataset_config": "conservation_within_poaceae_non_tis", "mask_indexes_8192": [4095], "motif_len": 1},
    "C26": {"dataset_config": "conservation_within_andropogoneae", "mask_indexes_8192": [4095], "motif_len": 1},
    "C27": {"dataset_config": "conservation_within_poaceae_tis", "mask_indexes_8192": [4095], "motif_len": 1},
    "C28": {"dataset_config": "structural_variant_effect_prediction", "flanking": 5},
}


def source_patterns(task_id, profile="formal"):
    if profile not in {"smoke", "formal"}:
        raise ValueError(f"unsupported source profile: {profile}")
    patterns = list(SOURCE_PATTERNS_BY_TASK.get(task_id, []))
    if profile == "smoke" and task_id == "C11":
        return ["mini_datasets/dev.csv", "mini_datasets/test.csv", "mini_datasets/train.csv"]
    if profile == "smoke" and task_id in {"C12", "C13", "C14", "C15", "C16"}:
        directory = patterns[0].split("/", 1)[0]
        return [f"{directory}/valid.tsv"]
    if profile == "smoke" and task_id in {"C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24"}:
        directory = patterns[0].split("/", 1)[0]
        return [f"{directory}/test_tomato-00000-of-*.parquet"]
    if profile == "smoke" and task_id in {"C25", "C26", "C27", "C28"}:
        directory = patterns[0].split("/", 1)[0]
        return [f"{directory}/test-00000-of-*.parquet"]
    if profile == "smoke" and task_id in {"C29", "C30", "C31", "C32", "C33", "C34", "C35"}:
        directory = patterns[0].split("/", 1)[0]
        return [f"{directory}/test-00000-of-00001.parquet"]
    if profile == "smoke" and task_id == "C36":
        return ["test.parquet", "labels.txt"]
    return patterns


def plantcad2_zero_shot_spec(task_id):
    try:
        return dict(PLANTCAD2_ZERO_SHOT_SPECS[task_id])
    except KeyError as error:
        raise ValueError(f"not a PlantCAD2 zero-shot task: {task_id}") from error


def _infer_split(path):
    name = Path(path).name.lower()
    if any(token in name for token in ("validation", "valid", "dev")):
        return "validation"
    if "test" in name:
        return "test"
    if "train" in name:
        return "train"
    raise ValueError(f"cannot infer split from filename: {path}")


def _sample_id(task_id, path, index, source_name=""):
    identity = f"{task_id}|{Path(path).as_posix()}|{index}|{source_name}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _parse_label(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (np.integer, int, bool)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    text = str(value).strip()
    if text.startswith("["):
        return json.loads(text)
    if len(text) > 1 and set(text) <= {"0", "1"}:
        return [int(base) for base in text]
    number = float(text)
    return int(number) if number.is_integer() else number


def _parse_fasta(path):
    header = None
    chunks = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            elif header is not None:
                chunks.append(line)
            elif line:
                raise ValueError(f"sequence before FASTA header: {path}")
    if header is not None:
        yield header, "".join(chunks)


def _fasta_source_partition(path):
    stem = Path(path).stem
    for suffix in ("_validation", "_validate", "_valid", "_train", "_test", "_dev"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def load_fasta_rows(paths, task_id, species="unknown_plant"):
    rows = []
    for path in sorted(map(Path, paths)):
        split = _infer_split(path)
        source_species = _fasta_source_partition(path) if species == "unknown_plant" else species
        for index, (header, sequence) in enumerate(_parse_fasta(path)):
            fields = header.split("|")
            if len(fields) < 2:
                raise ValueError(f"FASTA header has no label: {header[:160]!r}")
            labels = [_parse_label(value) for value in fields[1:]]
            label = labels[0] if len(labels) == 1 else labels
            rows.append({
                "sample_id": _sample_id(task_id, path, index, fields[0]),
                "split": split,
                "sequence": sequence,
                "label": label,
                "species": source_species,
                "source_name": fields[0],
            })
    return rows


def _sequence_column(fieldnames):
    for name in ("sequence", "sequences", "seq"):
        if name in fieldnames:
            return name
    raise ValueError(f"no DNA sequence column in {fieldnames}")


def _label_column(fieldnames):
    for name in ("label", "labels"):
        if name in fieldnames:
            return name
    raise ValueError(f"no label column in {fieldnames}")


def load_csv_rows(paths, task_id, default_species="unknown_plant"):
    rows = []
    for path in sorted(map(Path, paths)):
        split = _infer_split(path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            sequence_col = _sequence_column(reader.fieldnames or [])
            label_col = _label_column(reader.fieldnames or [])
            for index, record in enumerate(reader):
                rows.append({
                    "sample_id": _sample_id(task_id, path, index),
                    "split": split,
                    "sequence": record[sequence_col],
                    "label": _parse_label(record[label_col]),
                    "species": record.get("species") or record.get("genome") or default_species,
                    "group_id": record.get("group_id") or record.get("orthogroup") or "",
                })
    return rows


def load_tsv_rows(paths, task_id, default_species="unknown_plant"):
    rows = []
    for path in sorted(map(Path, paths)):
        split = _infer_split(path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            sequence_col = _sequence_column(reader.fieldnames or [])
            label_col = _label_column(reader.fieldnames or [])
            for index, record in enumerate(reader):
                rows.append({
                    "sample_id": _sample_id(task_id, path, index),
                    "split": split,
                    "sequence": record[sequence_col],
                    "label": _parse_label(record[label_col]),
                    "species": record.get("species") or default_species,
                    "group_id": record.get("group_id", ""),
                })
    return rows


def load_parquet_rows(paths, task_id, default_species="unknown_plant", batch_size=8192):
    rows = []
    for path in sorted(map(Path, paths)):
        split = _infer_split(path)
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        sequence_col = _sequence_column(names)
        label_col = _label_column(names)
        row_index = 0
        selected = [sequence_col, label_col]
        for optional in ("species", "genome", "orthogroup", "group_id", "id", "gene"):
            if optional in names and optional not in selected:
                selected.append(optional)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
            payload = batch.to_pydict()
            for local_index in range(batch.num_rows):
                record = {key: payload[key][local_index] for key in selected}
                rows.append({
                    "sample_id": _sample_id(task_id, path, row_index, str(record.get("id") or record.get("gene") or "")),
                    "split": split,
                    "sequence": record[sequence_col],
                    "label": _parse_label(record[label_col]),
                    "species": record.get("species") or record.get("genome") or default_species,
                    "group_id": record.get("group_id") or record.get("orthogroup") or "",
                })
                row_index += 1
    return rows


def deterministic_hash_sample(rows, cap_per_split, seed):
    """Keep the lowest stable hashes per split without depending on source order."""
    if cap_per_split is None:
        return list(rows)
    cap = int(cap_per_split)
    if cap <= 0:
        raise ValueError("cap_per_split must be positive")
    heaps = {split: [] for split in ("train", "validation", "test")}
    for row in rows:
        split = row["split"]
        digest = hashlib.sha256(f"{seed}|{row['sample_id']}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        item = (-value, row["sample_id"], row)
        heap = heaps[split]
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = []
    for split in ("train", "validation", "test"):
        selected.extend(item[2] for item in sorted(heaps[split], key=lambda value: (-value[0], value[1])))
    return selected
