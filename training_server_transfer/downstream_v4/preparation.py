"""Task-specific dataset preparation into the canonical v4 contract."""

import csv
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .adapters import (
    PLANTCAD2_ZERO_SHOT_SPECS, deterministic_hash_sample, load_csv_rows,
    load_fasta_rows, load_parquet_rows, load_tsv_rows, source_patterns,
)
from .data import DatasetContractError, materialize_rows, sha256_path
from .download import DEFAULT_DATA_ROOT


EXISTING_DATASETS = {
    "A01": "training_server_transfer/runs/cropgenome_bench_v1_8k_complete/datasets/context_8192/splice_donor_acceptor",
    "A02": "training_server_transfer/runs/cropgenome_bench_v1_8k_complete/datasets/context_8192/promoter_TSS",
    "A03": "training_server_transfer/runs/cropgenome_bench_v1_8k_complete/datasets/context_8192/TES_polyA",
    "A04": "training_server_transfer/runs/plant_genomic_benchmark_publication_v2/datasets/lncrna_multispecies",
    "A05": "training_server_transfer/runs/plant_genomic_benchmark_publication_v2/datasets/enhancer_cassava_proseq",
    "A06": "training_server_transfer/runs/plant_genomic_benchmark_publication_v2/datasets/gene_expression_glycine_max",
    "A07": "training_server_transfer/runs/plant_genomic_benchmark_publication_v2/datasets/gene_expression_oryza_sativa",
    "A08": "training_server_transfer/runs/plant_genomic_benchmark_publication_v2/datasets/gene_expression_zea_mays",
    "A09": "training_server_transfer/runs/plant_genomic_benchmark_publication_v2/datasets/gene_expression_solanum_lycopersicum",
    "A10": "training_server_transfer/runs/plant_genomic_benchmark_publication_v2/datasets/gene_expression_arabidopsis_thaliana",
    "B13": "training_server_transfer/runs/cropgenome_specialty_non_edta_v1/datasets/crop_gene_architecture_7state",
    "B14": "training_server_transfer/runs/cropgenome_specialty_non_edta_v1/datasets/crop_long_intron_splice_pair_ranking",
    "B15": "training_server_transfer/runs/cropgenome_specialty_non_edta_v1/datasets/crop_complete_gene_boundary_pair_ranking",
    "B16": "training_server_transfer/runs/cropgenome_specialty_non_edta_v1/datasets/crop_exon_order_same_gene_membership",
}

BUDGET_CAPS = {
    "C05": 200000,
    "C11": 200000,
    "C12": 200000, "C13": 200000, "C14": 200000, "C15": 200000, "C16": 200000,
    "C17": 10000, "C18": 10000, "C19": 10000, "C20": 10000,
    "C21": 10000, "C22": 10000, "C23": 10000, "C24": 10000,
    "C25": 10000, "C26": 10000, "C27": 10000, "C28": 10000,
    "C29": 200000, "C30": 200000, "C31": 200000,
    "C32": 200000, "C33": 200000,
    "D01": 200000, "D02": 200000, "D03": 200000,
}

REGION_LABELS = {
    "intergenic": 0, "CDS": 1, "intron": 2, "three_prime_UTR": 3,
    "five_prime_UTR": 4, "ncRNA_gene": 5, "Repeat": 6,
}
BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def _task_by_id(registry, task_id):
    matches = [row for row in registry["tasks"] if row["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate task_id: {task_id}")
    return matches[0]


def _source_by_id(registry, source_id):
    return next(row for row in registry["sources"] if row["source_id"] == source_id)


def _frozen_manifest_path(dataset_root):
    dataset_root = Path(dataset_root)
    candidates = (
        dataset_root / "dataset_manifest.json",
        dataset_root / "task_manifest.json",
        dataset_root.parent / "dataset_manifest.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def _resolve_patterns(root, patterns):
    root = Path(root)
    files = []
    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    files = sorted(set(path.resolve() for path in files))
    if not files:
        raise FileNotFoundError(f"no source files under {root} for patterns {patterns}")
    return files


def _source_receipt(source_root, source, task_id, profile):
    receipt_path = Path(source_root) / "download_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"source download receipt missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "ok"
        or receipt.get("source_id") != source["source_id"]
        or receipt.get("revision") != source.get("revision")
    ):
        raise DatasetContractError(f"source receipt revision mismatch for {source['source_id']}")
    if task_id not in set(receipt.get("selected_tasks") or []):
        raise DatasetContractError(f"source receipt does not cover task {task_id}")
    task_profile = (receipt.get("task_profiles") or {}).get(task_id, receipt.get("profile"))
    accepted_profiles = {"formal", "smoke"} if profile == "smoke" else {"formal"}
    if task_profile not in accepted_profiles:
        raise DatasetContractError(
            f"source receipt profile {task_profile!r} cannot serve {profile!r} preparation"
        )
    for record in receipt.get("artifacts") or []:
        artifact = Path(source_root) / record["path"]
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(record["size_bytes"])
            or sha256_path(artifact) != record["sha256"]
        ):
            raise DatasetContractError(f"source artifact hash mismatch: {artifact}")
    return receipt


def _derive_validation(rows, fraction=0.1, seed=42):
    if any(row["split"] == "validation" for row in rows):
        return rows
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        return rows
    ranked = sorted(
        train,
        key=lambda row: hashlib.sha256(f"{seed}|{row['sample_id']}".encode("utf-8")).digest(),
    )
    n_validation = max(1, int(round(len(train) * fraction)))
    selected = {row["sample_id"] for row in ranked[:n_validation]}
    result = []
    for row in rows:
        copy = dict(row)
        if copy["sample_id"] in selected:
            copy["split"] = "validation"
        result.append(copy)
    return result


def _smoke_repartition(rows, seed=42):
    """Parser-only smoke split; never used by budgeted/full scientific runs."""
    result = []
    for row in rows:
        copy = dict(row)
        bucket = int(hashlib.sha256(f"{seed}|{row['sample_id']}".encode()).hexdigest()[:8], 16) % 10
        copy["split"] = "test" if bucket < 2 else "validation" if bucket < 4 else "train"
        result.append(copy)
    if {row["split"] for row in result} != {"train", "validation", "test"}:
        raise DatasetContractError("smoke repartition did not produce all three splits")
    return result


def _decontaminate_exact(rows):
    priority = {"train": 0, "validation": 1, "test": 2}
    by_hash = defaultdict(list)
    for row in rows:
        digest = hashlib.sha256(row["sequence"].upper().encode("ascii")).hexdigest()
        by_hash[digest].append(row)
    keep = []
    removed = Counter()
    for records in by_hash.values():
        best = max(priority[row["split"]] for row in records)
        for row in records:
            if priority[row["split"]] == best:
                keep.append(row)
            else:
                removed[row["split"]] += 1
    keep.sort(key=lambda row: ("train validation test".split().index(row["split"]), row["sample_id"]))
    return keep, dict(removed)


def _profile_cap(task_id, profile):
    if profile == "smoke":
        return 32
    if profile == "budgeted":
        return BUDGET_CAPS.get(task_id)
    if profile == "full":
        return None
    raise ValueError(f"unsupported preparation profile: {profile}")


def _load_reference(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    sequences = {}
    name = None
    chunks = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(chunks).upper()
                name = raw[1:].split()[0]
                chunks = []
            else:
                chunks.append(raw.strip())
        if name is not None:
            sequences[name] = "".join(chunks).upper()
    if not sequences:
        raise DatasetContractError(f"no sequences in reference: {path}")
    return sequences


def _find_reference_file(aux_root):
    candidates = sorted(Path(aux_root).glob("*.fa*"))
    if len(candidates) != 1:
        raise DatasetContractError(f"expected exactly one TAIR10 reference file under {aux_root}, found {len(candidates)}")
    return candidates[0]


def _window_pair(reference, chrom, pos1, ref, alt, length=512):
    chrom = str(chrom).removeprefix("Chr").removeprefix("chr")
    if chrom not in reference:
        raise DatasetContractError(f"reference chromosome absent: {chrom}")
    pos0 = int(pos1) - 1
    start = pos0 - length // 2
    end = start + length
    if start < 0 or end > len(reference[chrom]):
        return None
    sequence = reference[chrom][start:end]
    ref = str(ref).upper(); alt = str(alt).upper()
    if len(ref) != 1 or len(alt) != 1 or ref not in BASE_INDEX or alt not in BASE_INDEX:
        return None
    if sequence[length // 2] != ref:
        raise DatasetContractError(
            f"TAIR10 reference mismatch at chr{chrom}:{pos1}: {sequence[length // 2]} != {ref}"
        )
    mutant = sequence[: length // 2] + alt + sequence[length // 2 + 1 :]
    return sequence, mutant, chrom


def _load_plantcad2_zero_rows(paths, task_id):
    spec = PLANTCAD2_ZERO_SHOT_SPECS[task_id]
    rows = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        for batch in parquet.iter_batches(batch_size=4096):
            data = batch.to_pydict()
            for index in range(batch.num_rows):
                identity = f"{task_id}|{path}|{len(rows)}"
                sample_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                if task_id == "C28":
                    rows.append({
                        "sample_id": sample_id, "split": "test", "sequence": data["RefSeq"][index],
                        "sequence_b": data["MutSeq"][index], "label": int(data["label"][index]),
                        "species": "plant_mixed", "left": int(data["left"][index]),
                        "right": int(data["right"][index]),
                    })
                    continue
                sequence = data["sequence"][index]
                if "label" in names:
                    label = int(data["label"][index])
                else:
                    label = [BASE_INDEX.get(sequence[position].upper(), 4) for position in spec["mask_indexes_8192"]]
                rows.append({
                    "sample_id": sample_id, "split": "test", "sequence": sequence,
                    "label": label, "species": "Zea mays_or_Solanum lycopersicum",
                    "mask_indexes_8192": spec.get("mask_indexes_8192", []),
                })
    return rows


def _load_gpn_rows(task_id, source_root, reference_path, cap, seed=42):
    reference = _load_reference(reference_path)
    source_root = Path(source_root)
    rows = []
    if task_id == "D01":
        table = pq.read_table(source_root / "embedding/windows.parquet").to_pydict()
        for index, chrom in enumerate(table["chrom"]):
            chrom_key = str(chrom).removeprefix("Chr").removeprefix("chr")
            start, end = int(table["center_start"][index]), int(table["center_end"][index])
            if chrom_key not in reference or start < 0 or end > len(reference[chrom_key]):
                continue
            region = table["Region"][index]
            rows.append({
                "sample_id": hashlib.sha256(f"D01|{chrom_key}|{start}|{end}".encode()).hexdigest()[:24],
                "split": "test", "sequence": reference[chrom_key][start:end],
                "label": REGION_LABELS[region], "class_name": region,
                "species": "Arabidopsis thaliana", "group_id": chrom_key,
            })
        return deterministic_hash_sample(rows, cap, seed) if cap else rows
    if task_id == "D02":
        parquet = pq.ParquetFile(source_root / "variants/filt.processed.parquet")
        required_columns = ["chrom", "pos", "ref", "alt", "AC", "AN"]
        missing = sorted(set(required_columns) - set(parquet.schema_arrow.names))
        if missing:
            raise DatasetContractError(f"D02 parquet lacks columns: {missing}")
        selected_columns = required_columns + (
            ["consequence"] if "consequence" in parquet.schema_arrow.names else []
        )
        candidates = []
        for batch in parquet.iter_batches(batch_size=65536, columns=selected_columns):
            data = batch.to_pydict()
            for index in range(batch.num_rows):
                ac, an = int(data["AC"][index]), int(data["AN"][index])
                if an <= 0:
                    continue
                maf = min(ac / an, 1.0 - ac / an)
                if ac == 1:
                    label = 1
                elif maf >= 0.05:
                    label = 0
                else:
                    continue
                item = (
                    str(data["chrom"][index]), int(data["pos"][index]),
                    data["ref"][index], data["alt"][index], label,
                    data["consequence"][index] if "consequence" in data else None,
                )
                candidates.append(item)
        if cap and len(candidates) > cap:
            candidates.sort(key=lambda row: hashlib.sha256(f"{seed}|{row[:5]}".encode()).digest())
            half = cap // 2
            selected = [row for row in candidates if row[4] == 1][:half] + [row for row in candidates if row[4] == 0][:cap-half]
        else:
            selected = candidates
        for chrom, pos, ref, alt, label, consequence in selected:
            pair = _window_pair(reference, chrom, pos, ref, alt)
            if pair is None:
                continue
            sequence, mutant, chrom_key = pair
            rows.append({
                "sample_id": hashlib.sha256(f"D02|{chrom_key}|{pos}|{ref}|{alt}".encode()).hexdigest()[:24],
                "split": "test", "sequence": sequence, "sequence_b": mutant, "label": label,
                "species": "Arabidopsis thaliana", "chrom": chrom_key, "pos": pos,
                "ref": ref, "alt": alt, "consequence": consequence,
            })
        return rows
    if task_id == "D03":
        aggregated = {}
        for path in sorted((source_root / "aragwas").glob("api_chr*.csv.gz")):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                names = set(reader.fieldnames or [])
                official = {
                    "chrom": "snp.chr", "pos": "snp.position",
                    "ref": "snp.ref", "alt": "snp.alt", "label": "overPermutation",
                }
                legacy = {
                    "chrom": "SNPChr", "pos": "SNPPos",
                    "ref": "SNPRef", "alt": "SNPAlt", "label": "overPermutation",
                }
                columns = official if set(official.values()) <= names else legacy
                missing = sorted(set(columns.values()) - names)
                if missing:
                    raise DatasetContractError(f"D03 AraGWAS CSV lacks columns in {path.name}: {missing}")
                for record in reader:
                    key = (
                        record[columns["chrom"]], int(record[columns["pos"]]),
                        record[columns["ref"]], record[columns["alt"]],
                    )
                    label = int(str(record["overPermutation"]).lower() == "true")
                    aggregated[key] = max(label, aggregated.get(key, 0))
        candidates = list(aggregated.items())
        if cap and len(candidates) > cap:
            candidates.sort(key=lambda row: hashlib.sha256(f"{seed}|{row[0]}".encode()).digest())
            positives = [row for row in candidates if row[1] == 1]
            negatives = [row for row in candidates if row[1] == 0]
            n_pos = min(len(positives), max(1, cap // 2))
            candidates = positives[:n_pos] + negatives[: cap - n_pos]
        for (chrom, pos, ref, alt), label in candidates:
            pair = _window_pair(reference, chrom, pos, ref, alt)
            if pair is None:
                continue
            sequence, mutant, chrom_key = pair
            rows.append({
                "sample_id": hashlib.sha256(f"D03|{chrom_key}|{pos}|{ref}|{alt}".encode()).hexdigest()[:24],
                "split": "test", "sequence": sequence, "sequence_b": mutant, "label": label,
                "species": "Arabidopsis thaliana", "chrom": chrom_key, "pos": pos,
                "ref": ref, "alt": alt,
            })
        return rows
    raise ValueError(task_id)


def _crop_window(sequence, center, length):
    start = center - length // 2
    end = start + length
    if start < 0 or end > len(sequence):
        return None
    return sequence[start:end]


def _parse_attributes(text):
    result = {}
    for part in text.split(";"):
        if "=" in part:
            key, value = part.split("=", 1); result[key] = value
    return result


def load_edta_rows(manifest_path, task_id, context=8192, seed=42):
    """Build B11/B12 rows from a frozen TSV: assembly_id, fasta_path, edta_gff3."""
    entries = []
    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"assembly_id", "fasta_path", "edta_gff3"}
        if not required <= set(reader.fieldnames or []):
            raise DatasetContractError(f"EDTA manifest lacks columns: {sorted(required - set(reader.fieldnames or []))}")
        entries = list(reader)
    rows = []
    classes = {}
    for entry in entries:
        assembly = entry["assembly_id"]
        digest = int(hashlib.sha256(f"{seed}|{assembly}".encode()).hexdigest()[:8], 16) % 10
        split = "test" if digest == 0 else "validation" if digest == 1 else "train"
        genome = _load_reference(entry["fasta_path"])
        elements = []
        with Path(entry["edta_gff3"]).open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw or raw.startswith("#"):
                    continue
                fields = raw.rstrip("\n").split("\t")
                if len(fields) != 9:
                    continue
                chrom, feature = fields[0], fields[2]
                if chrom not in genome or "repeat" not in feature.lower() and "transpos" not in feature.lower():
                    continue
                start1, end1 = int(fields[3]), int(fields[4])
                attrs = _parse_attributes(fields[8])
                classification = attrs.get("Classification") or attrs.get("classification") or attrs.get("Name", "unknown")
                superfamily = classification.split("/")[-1].split("_")[0]
                elements.append((chrom, start1, end1, superfamily))
        if task_id == "B11":
            boundaries = set()
            for chrom, start1, end1, _ in elements:
                boundaries.update({(chrom, start1), (chrom, end1)})
            for element_index, (chrom, start1, end1, _) in enumerate(elements):
                for side, center1 in (("left", start1), ("right", end1)):
                    sequence = _crop_window(genome[chrom], center1 - 1, context)
                    if sequence:
                        rows.append({"sample_id": f"{assembly}|{element_index}|{side}|pos", "split": split, "sequence": sequence, "label": 1, "species": assembly, "assembly_id": assembly, "group_id": assembly})
                    shift = max(context, (end1 - start1 + 1) // 2)
                    neg1 = center1 + shift if center1 + shift < len(genome[chrom]) else center1 - shift
                    while (chrom, neg1) in boundaries:
                        neg1 += context
                    negative = _crop_window(genome[chrom], neg1 - 1, context)
                    if negative:
                        rows.append({"sample_id": f"{assembly}|{element_index}|{side}|neg", "split": split, "sequence": negative, "label": 0, "species": assembly, "assembly_id": assembly, "group_id": assembly})
        elif task_id == "B12":
            for _, _, _, superfamily in elements:
                classes.setdefault(superfamily, len(classes))
            for element_index, (chrom, start1, end1, superfamily) in enumerate(elements):
                center = (start1 + end1) // 2 - 1
                sequence = _crop_window(genome[chrom], center, context)
                if sequence:
                    rows.append({"sample_id": f"{assembly}|{element_index}", "split": split, "sequence": sequence, "label": classes[superfamily], "class_name": superfamily, "species": assembly, "assembly_id": assembly, "group_id": assembly})
        else:
            raise ValueError(task_id)
    return rows


def prepare_task(registry, task_id, project_root, profile="budgeted", source_root=None,
                 output_root=None, auxiliary_root=None, edta_manifest=None, seed=42):
    project_root = Path(project_root).resolve()
    task = _task_by_id(registry, task_id)
    source = _source_by_id(registry, task["source_id"])
    if task_id in EXISTING_DATASETS:
        root = project_root / EXISTING_DATASETS[task_id]
        samples_path = root / "samples.tsv"
        if not samples_path.is_file():
            raise FileNotFoundError(samples_path)
        manifest_path = _frozen_manifest_path(root)
        if manifest_path is None:
            raise FileNotFoundError(f"frozen dataset manifest missing under {root}")
        return {
            "status": "adopted", "task_id": task_id, "dataset_root": str(root),
            "profile": "existing_frozen", "samples_sha256": sha256_path(samples_path),
            "manifest_path": str(manifest_path), "manifest_sha256": sha256_path(manifest_path),
        }
    if task_id == "B17":
        return {"status": "view_only", "task_id": task_id, "depends_on": ["B13", "B14", "B15", "B16"]}
    if output_root is None:
        output_root = project_root / "training_server_transfer/runs/downstream_v4/datasets" / profile / task_id
    else:
        output_root = Path(output_root)
    if task_id in {"B11", "B12"}:
        if not edta_manifest:
            raise DatasetContractError("B11/B12 require --edta-manifest after EDTA finalization")
        rows = load_edta_rows(edta_manifest, task_id, seed=seed)
        receipt = {"source_id": source["source_id"], "revision": source["revision"], "edta_manifest_sha256": sha256_path(edta_manifest)}
    else:
        source_root = Path(source_root or (project_root / DEFAULT_DATA_ROOT / source["source_id"]))
        receipt = _source_receipt(source_root, source, task_id, profile)
        files = _resolve_patterns(source_root, source_patterns(task_id, "smoke" if profile == "smoke" else "formal"))
        if task["adapter"] == "hf_fasta":
            rows = load_fasta_rows(files, task_id)
            if task_id == "C05":
                widths = set()
                for row in rows:
                    labels = row.get("label")
                    if not isinstance(labels, list) or not labels:
                        raise DatasetContractError("C05 requires non-empty source multilabel vectors")
                    if any(label not in (0, 1) for label in labels):
                        raise DatasetContractError("C05 source multilabel vectors must be binary")
                    widths.add(len(labels))
                    row["label"] = int(any(labels))
                receipt["label_transform"] = (
                    "any_accessible_binary_from_species_specific_multilabel"
                )
                receipt["source_label_widths"] = sorted(widths)
        elif task["adapter"] == "hf_csv":
            rows = load_csv_rows(files, task_id)
        elif task["adapter"] == "hf_tsv":
            rows = load_tsv_rows(files, task_id)
        elif task["adapter"] == "hf_parquet":
            rows = load_parquet_rows([path for path in files if path.suffix == ".parquet"], task_id)
        elif task["adapter"] == "hf_parquet_zero_shot":
            rows = _load_plantcad2_zero_rows(files, task_id)
        elif task["adapter"] == "gpn_parquet":
            if auxiliary_root is None:
                auxiliary_root = project_root / DEFAULT_DATA_ROOT / task["auxiliary_source_ids"][0]
            reference_path = _find_reference_file(auxiliary_root)
            rows = _load_gpn_rows(task_id, source_root, reference_path, _profile_cap(task_id, profile), seed)
        else:
            raise ValueError(f"adapter is not preparable here: {task['adapter']}")
    zero_shot = task["task_kind"].startswith("zero_shot") or task_id == "D01"
    if not zero_shot:
        if profile == "smoke" and {row["split"] for row in rows} != {"train", "validation", "test"}:
            rows = _smoke_repartition(rows, seed=seed)
        else:
            rows = _derive_validation(rows, seed=seed)
    cap = _profile_cap(task_id, profile)
    if cap and task["adapter"] != "gpn_parquet":
        rows = deterministic_hash_sample(rows, cap, seed)
    rows, removed = _decontaminate_exact(rows)
    receipt = {
        **receipt,
        "task_id": task_id,
        "profile": profile,
        "decontaminated_rows": removed,
        "seed": seed,
        "preparation_implementation_sha256": sha256_path(Path(__file__).resolve()),
        "adapter_implementation_sha256": sha256_path(Path(__file__).with_name("adapters.py")),
        "cross_split_group_policy": (
            "official_split_informational" if task_id in {"C32", "C33"} else "forbidden"
        ),
    }
    required_splits = {"test"} if zero_shot else {"train", "validation", "test"}
    manifest = materialize_rows(
        rows, output_root, task_id, task["task_kind"], receipt, required_splits=required_splits,
        allow_cross_split_groups=task_id in {"C32", "C33"},
    )
    manifest_path = Path(output_root).resolve() / "dataset_manifest.json"
    return {
        "status": "ok", "task_id": task_id, "dataset_root": str(Path(output_root).resolve()),
        "profile": profile, "manifest": manifest, "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
    }
