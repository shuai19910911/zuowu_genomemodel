#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "research_guide"
OUT = GUIDE / "source_data"
CORE = ROOT / "training_server_transfer/runs/cropgenome_bench_v1_8k_complete/datasets/context_8192"
EXTERNAL = ROOT / "training_server_transfer/runs/plant_genomic_benchmark_publication_v2/datasets"
ASSEMBLIES = ROOT / "data_manifests/assemblies.canonical.tsv"
AGGREGATE = OUT / "aggregate_primary_metrics.tsv"
FUTURE = OUT / "future_task_registry.tsv"

FAMILY = {
    "Arabidopsis thaliana": "Brassicaceae",
    "Arachis hypogaea": "Fabaceae",
    "Beta vulgaris": "Amaranthaceae",
    "Brassica napus": "Brassicaceae",
    "Brassica oleracea": "Brassicaceae",
    "Brassica rapa": "Brassicaceae",
    "Cicer arietinum": "Fabaceae",
    "Citrullus lanatus": "Cucurbitaceae",
    "Cucumis melo": "Cucurbitaceae",
    "Cucumis sativus": "Cucurbitaceae",
    "Daucus carota": "Apiaceae",
    "Glycine max": "Fabaceae",
    "Gossypium hirsutum": "Malvaceae",
    "Helianthus annuus": "Asteraceae",
    "Hordeum vulgare": "Poaceae",
    "Lactuca sativa": "Asteraceae",
    "Malus domestica": "Rosaceae",
    "Manihot esculenta": "Euphorbiaceae",
    "Musa acuminata": "Musaceae",
    "Oryza sativa": "Poaceae",
    "Phaseolus vulgaris": "Fabaceae",
    "Prunus persica": "Rosaceae",
    "Saccharum spontaneum": "Poaceae",
    "Setaria italica": "Poaceae",
    "Solanum lycopersicum": "Solanaceae",
    "Solanum tuberosum": "Solanaceae",
    "Sorghum bicolor": "Poaceae",
    "Triticum aestivum": "Poaceae",
    "Vigna radiata": "Fabaceae",
    "Vitis vinifera": "Vitaceae",
    "Zea mays": "Poaceae",
}

KEY_PLANT = {"AgroNT_1B", "PlantCAD2_Small", "PlantCaduceus_l32"}
PUBLIC_MODELS = {
    "AgroNT_1B", "PlantCAD2_Small", "PlantCaduceus_l32",
    "NTv2_100M_multi_species", "DNABERT2_117M", "HyenaDNA_medium_160k",
    "Caduceus_PS_131k", "Evo2_1B_base",
}

CAPABILITY = {
    "CROP-LONGGENE-SEG": {
        "crop_signal": "64–128K完整长基因中的启动子、UTR、CDS、长内含子、剪接边界和TES联合结构",
        "ours": "需要连续长输入、token级分割和作物GFF区域先验；当前8K模型不足，下一版64–128K候选才具备完整信息范围",
        "agro": "植物域强基线，但冻结配置约1,024个6-mer token；只能进入common或compute-matched chunked赛道",
        "plantcad": "PlantCAD2/PlantCaduceus是必须正面对比的植物长序列强基线；若不能超过，不得宣称本模型独有",
        "nt500": "规模强但约2,048 token且非植物预训练；原生赛道长度不足，必须给chunked结果",
        "evo2": "通用进化尺度字符模型；当前本地正式runtime只冻结到8K，未来需按真实可运行长度重新审计",
        "classic": "Helixer/GeMoMa/AUGUSTUS类基因结构方法＋CNN/CRF分割器",
    },
    "CROP-DISTAL-CIS-PAIR": {
        "crop_signal": "中心2K局部序列匹配、仅远端8–64K背景不同的成对调控样本与反事实远端置换",
        "ours": "目标是证明全局token保存远端信息；必须出现64K相对2K的配对增益且远端置换后增益消失",
        "agro": "只能读取中心局部或分块证据；不能在原生单次前向中联合完整远端上下文",
        "plantcad": "真正的植物长序列对手；需要同样配对样本、同probe和同读取碱基预算",
        "nt500": "强通用短Transformer；用于检验局部motif是否已足够，但不具备原生64K输入",
        "evo2": "可检验通用长序列先验；当前冻结8K配置无法覆盖设计中的64K因果对照",
        "classic": "motif/GC/距离匹配逻辑回归、局部CNN、分块attention聚合",
    },
    "CROP-ISOFORM-LR": {
        "crop_signal": "长内含子、多候选剪接位点和可变poly(A)的组织/胁迫条件使用率",
        "ours": "需要64–128K基因级表示与junction/poly(A)多标签头；尚无冻结数据或结果",
        "agro": "植物域有利，但短窗口需chunking，可能丢失跨内含子和位点竞争关系",
        "plantcad": "植物长模型主基线；必须在gene-family和study隔离下比较",
        "nt500": "通用剪接motif可能强，但缺作物域和完整长基因原生上下文",
        "evo2": "字符级通用模型可提供强局部/中程表示；需检验对组织条件标签的迁移",
        "classic": "剪接motif模型、转录本图模型、表达/可检测性先验",
    },
    "CROP-POLYPLOID-HOMEOLOG": {
        "crop_signal": "小麦、棉花、油菜同源亚基因组基因的表达优势、抑制和平衡状态",
        "ours": "作物预训练与长基因上下文可能有利，但序列本身不能替代组织表达标签；必须与表达强基线比较",
        "agro": "植物域匹配但native上下文短；仍是重要植物迁移基线",
        "plantcad": "最关键植物长模型对手；不能因其架构不同而排除",
        "nt500": "通用规模对照，用于分离模型规模与作物域预训练贡献",
        "evo2": "进化尺度先验可能捕获保守性；需要同源组整体隔离避免记忆",
        "classic": "表达均值/组织先验、线性混合模型、同源组序列相似度与亚基因组注释",
    },
    "CROP-TE-SV-REG": {
        "crop_signal": "TE插入或结构变异前后成对单倍型的边界、方向和表达变化，窗口32–256K",
        "ours": "需要成对ref/alt编码、长上下文和delta头；当前被EDTA QC与功能标签冻结阻塞",
        "agro": "适合作物局部序列，但不能原生覆盖大SV两侧完整上下文",
        "plantcad": "能进入较长native赛道的植物强基线；真实最大长度必须逐模型Gate",
        "nt500": "短上下文只能分块；可作为规模强但信息范围不足的对照",
        "evo2": "通用字符模型是重要长序列基线，但当前本地8K配置不足以覆盖256K",
        "classic": "SV长度/类型、TE家族、距TSS、GC、mappability与eQTL统计模型",
    },
    "CROP-NLR-CLUSTER": {
        "crop_signal": "65–256K重复富集NLR抗病基因簇的完整基因、伪基因、拷贝数和簇边界",
        "ours": "长程＋作物域可能有利；必须在低同源、近重复去泄漏和人工QC层上证明",
        "agro": "植物域强但短输入只能逐块识别局部NLR motif，难以原生计数整簇",
        "plantcad": "植物长模型是最强公平对手；至少比较边界F1和copy-count MAE",
        "nt500": "通用短模型用于局部motif识别上限，不可把N/E记为0分",
        "evo2": "通用进化序列模型可能识别重复与保守域；需重新冻结长输入runtime",
        "classic": "NLR-Annotator/NLR-Parser、HMM/domain规则、repeat-aware gene caller",
    },
    "CROP-ACR-STRESS": {
        "crop_signal": "跨作物、组织和胁迫条件的开放染色质与远端调控活性",
        "ours": "作物域预训练可能改善迁移；长程优势只在远端匹配子集上判断",
        "agro": "强植物短上下文基线，通用局部峰任务必须至少非劣",
        "plantcad": "当前最关键植物长模型基线；外部表达结果已显示其总体更强",
        "nt500": "大规模通用模型，下一版必须冻结500M而不是仅用100M代表",
        "evo2": "通用进化尺度基线，检验大规模字符预训练是否足以替代植物域数据",
        "classic": "GC/mappability/TSS距离匹配、gkm-SVM、CNN/ResNet、motif模型",
    },
    "CROP-PANGENOME-SV": {
        "crop_signal": "水稻、玉米、大豆、小麦非参考单倍型中的PAV/SV类别及转录影响",
        "ours": "需要成对单倍型长编码和reference-bias审计；目前尚未冻结数据",
        "agro": "植物域有利但原生长度不足；chunked track可检验局部证据汇总",
        "plantcad": "植物长序列强基线；必须在非参考单倍型子集保持效果",
        "nt500": "短通用规模基线；不能完整读取复杂SV上下文",
        "evo2": "字符级模型可能适合变异序列；需严格匹配输入长度和计算量",
        "classic": "SV caller特征、PAV规则、k-mer presence/absence、表达delta线性模型",
    },
    "CROP-QTL-VAR-RANK": {
        "crop_signal": "QTL/GWAS位点内候选基因/变异的locus-wise排序，而不是普通全局分类",
        "ours": "只能作为序列表征组件；必须联合位点、等位delta并与非深度强基线比较",
        "agro": "植物序列表示重要，但短局部窗口可能已足够，不能预设长模型必胜",
        "plantcad": "植物强基线；同一locus候选集、同一ranking head和相同特征范围",
        "nt500": "通用规模基线，用于检验规模效应；500M当前尚无本项目成绩",
        "evo2": "通用变异表征基线；不能把序列分数直接称为因果证据",
        "classic": "距lead SNP、功能注释、保守性、eQTL/精细定位posterior与传统ranker",
    },
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def genus(species: str) -> str:
    return species.split()[0]


def yn(value: bool) -> str:
    return "yes" if value else "no"


def pairwise_disjoint(groups: dict[str, set[str]]) -> bool:
    names = ["train", "val", "test"]
    return all(groups[a].isdisjoint(groups[b]) for i, a in enumerate(names) for b in names[i + 1 :])


def taxonomy_tables() -> None:
    assemblies = read_tsv(ASSEMBLIES)
    by_accession = {r["assembly_accession"]: r for r in assemblies}
    stageb_train = [r for r in assemblies if r["split"] == "train"]
    train_species = {r["species"] for r in stageb_train}
    train_genera = {r["genus"] for r in stageb_train}
    missing_families = sorted({r["species"] for r in assemblies} - FAMILY.keys())
    if missing_families:
        raise RuntimeError(f"missing family mappings: {missing_families}")
    train_families = {FAMILY[r["species"]] for r in stageb_train}

    datasets: list[tuple[str, str, list[dict[str, str]], str]] = []
    for task in ("promoter_TSS", "TES_polyA", "splice_donor_acceptor"):
        datasets.append(("CropGenome-Bench-v1", task, read_tsv(CORE / task / "samples.tsv"), "core"))
    for task_dir in sorted(p for p in EXTERNAL.iterdir() if p.is_dir() and (p / "samples.tsv").exists()):
        datasets.append(("Plant-Genomic-Benchmark", task_dir.name, read_tsv(task_dir / "samples.tsv"), "external"))

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for panel, task, samples, task_scope in datasets:
        counts = Counter((r["split"], r["species"]) for r in samples)
        split_species = {s: {sp for (split, sp), n in counts.items() if split == s and n} for s in ("train", "val", "test")}
        split_genera = {s: {genus(sp) for sp in values} for s, values in split_species.items()}
        split_families = {s: {FAMILY[sp] for sp in values} for s, values in split_species.items()}
        assembly_map: dict[tuple[str, str], set[str]] = defaultdict(set)
        if task_scope == "core":
            for row in samples:
                assembly_map[(row["split"], row["species"])].add(row["assembly_id"])

        for split in ("train", "val", "test"):
            for species in sorted(split_species[split]):
                accs = sorted(assembly_map[(split, species)])
                canonical_splits = sorted({by_accession[a]["split"] if a in by_accession else "not_in_canonical_manifest" for a in accs})
                downstream_species_seen = species in split_species["train"]
                downstream_genus_seen = genus(species) in split_genera["train"]
                downstream_family_seen = FAMILY[species] in split_families["train"]
                if split == "train":
                    interpretation = "downstream training taxon"
                elif downstream_species_seen:
                    interpretation = "same species occurs in downstream train; this is within-species supervised generalization"
                elif downstream_genus_seen:
                    interpretation = "species held out downstream, but genus occurs in downstream train"
                elif downstream_family_seen:
                    interpretation = "species and genus held out downstream, but family occurs in downstream train"
                else:
                    interpretation = "species, genus and family held out downstream"
                detail_rows.append({
                    "panel": panel,
                    "task_id": task,
                    "split": split,
                    "species": species,
                    "genus": genus(species),
                    "family": FAMILY[species],
                    "sample_count": counts[(split, species)],
                    "assembly_accessions": ";".join(accs) if accs else "not_available_in_external_dataset",
                    "canonical_assembly_splits": ";".join(canonical_splits) if canonical_splits else "not_auditable_no_accession_mapping",
                    "species_seen_in_downstream_train": yn(downstream_species_seen),
                    "genus_seen_in_downstream_train": yn(downstream_genus_seen),
                    "family_seen_in_downstream_train": yn(downstream_family_seen),
                    "species_seen_in_stageB_train": yn(species in train_species),
                    "genus_seen_in_stageB_train": yn(genus(species) in train_genera),
                    "family_seen_in_stageB_train": yn(FAMILY[species] in train_families),
                    "interpretation": interpretation,
                })

        test_species = split_species["test"]
        test_genera = {genus(sp) for sp in test_species}
        test_families = {FAMILY[sp] for sp in test_species}
        if task_scope == "core":
            test_accs = {a for (split, _sp), values in assembly_map.items() if split == "test" for a in values}
            accession_isolated = yn(test_accs.isdisjoint({r["assembly_accession"] for r in stageb_train}))
        else:
            accession_isolated = "not_auditable_no_accession_mapping"
        unseen_species = sum(sp not in train_species for sp in test_species)
        unseen_genera = sum(g not in train_genera for g in test_genera)
        unseen_families = sum(f not in train_families for f in test_families)
        summary_rows.append({
            "panel": panel,
            "task_id": task,
            "task_scope": task_scope,
            "train_species_count": len(split_species["train"]),
            "validation_species_count": len(split_species["val"]),
            "test_species_count": len(test_species),
            "downstream_species_disjoint": yn(pairwise_disjoint(split_species)),
            "downstream_genus_disjoint": yn(pairwise_disjoint(split_genera)),
            "downstream_family_disjoint": yn(pairwise_disjoint(split_families)),
            "test_families_not_seen_in_downstream_train": f"{sum(f not in split_families['train'] for f in test_families)}/{len(test_families)}",
            "stageB_test_assembly_accession_isolated": accession_isolated,
            "stageB_test_species_unseen": f"{unseen_species}/{len(test_species)}",
            "stageB_test_genera_unseen": f"{unseen_genera}/{len(test_genera)}",
            "stageB_test_families_unseen": f"{unseen_families}/{len(test_families)}",
            "safe_interpretation": (
                "species/genus-disjoint supervised core benchmark; not globally family-disjoint; test accessions isolated from Stage B"
                if task_scope == "core" else
                "official sample-level supervised split; not a species-held-out benchmark; accession overlap with Stage B cannot be audited from this dataset"
            ),
        })

    write_tsv(
        OUT / "downstream_taxonomy_detail.tsv",
        [
            "panel", "task_id", "split", "species", "genus", "family", "sample_count",
            "assembly_accessions", "canonical_assembly_splits", "species_seen_in_downstream_train",
            "genus_seen_in_downstream_train", "family_seen_in_downstream_train",
            "species_seen_in_stageB_train", "genus_seen_in_stageB_train",
            "family_seen_in_stageB_train", "interpretation",
        ],
        detail_rows,
    )
    write_tsv(
        OUT / "downstream_taxonomy_summary.tsv",
        [
            "panel", "task_id", "task_scope", "train_species_count", "validation_species_count",
            "test_species_count", "downstream_species_disjoint", "downstream_genus_disjoint",
            "downstream_family_disjoint", "test_families_not_seen_in_downstream_train",
            "stageB_test_assembly_accession_isolated", "stageB_test_species_unseen",
            "stageB_test_genera_unseen", "stageB_test_families_unseen", "safe_interpretation",
        ],
        summary_rows,
    )


def baseline_gap_table() -> None:
    rows = read_tsv(AGGREGATE)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], row["context_bp"], row["metric"])].append(row)
    out_rows: list[dict[str, object]] = []
    for (panel, context, metric), group in sorted(grouped.items()):
        ours = next(float(r["value"]) for r in group if r["model_id"] == "CropGenomeFM_step14000")
        plant = max((r for r in group if r["model_id"] in KEY_PLANT), key=lambda r: float(r["value"]))
        public = max((r for r in group if r["model_id"] in PUBLIC_MODELS), key=lambda r: float(r["value"]))
        gap_plant = ours - float(plant["value"])
        gap_public = ours - float(public["value"])
        if gap_plant >= 0:
            call = "leads_current_evaluated_plant_baselines"
        elif gap_plant >= (-0.02 if "AUPRC" in metric else -0.03):
            call = "near_parity_point_estimate; formal noninferiority still requires paired CI"
        else:
            call = "gap_to_close_before_parity_claim"
        out_rows.append({
            "panel": panel,
            "context_bp": context,
            "metric": metric,
            "our_value": f"{ours:.8f}",
            "best_current_plant_baseline": plant["model_id"],
            "best_current_plant_value": plant["value"],
            "gap_vs_best_plant": f"{gap_plant:+.8f}",
            "best_current_public_baseline": public["model_id"],
            "best_current_public_value": public["value"],
            "gap_vs_best_public": f"{gap_public:+.8f}",
            "current_call": call,
            "NTv2_500M_status": "planned_not_evaluated; no current result",
            "next_release_rule": (
                "common task noninferiority: paired/group-bootstrap 95% CI lower bound > -0.02 AUPRC"
                if "AUPRC" in metric else
                "common task noninferiority: paired/group-bootstrap 95% CI lower bound > -0.03 Pearson"
            ),
        })
    write_tsv(
        OUT / "baseline_gap_summary.tsv",
        [
            "panel", "context_bp", "metric", "our_value", "best_current_plant_baseline",
            "best_current_plant_value", "gap_vs_best_plant", "best_current_public_baseline",
            "best_current_public_value", "gap_vs_best_public", "current_call",
            "NTv2_500M_status", "next_release_rule",
        ],
        out_rows,
    )


def capability_matrix() -> None:
    future = read_tsv(FUTURE)
    ids = [r["task_id"] for r in future]
    if set(ids) != set(CAPABILITY):
        raise RuntimeError(f"capability registry mismatch: future={ids}, configured={sorted(CAPABILITY)}")
    out_rows = []
    by_id = {r["task_id"]: r for r in future}
    for task_id in ids:
        row = by_id[task_id]
        cap = CAPABILITY[task_id]
        out_rows.append({
            "task_id": task_id,
            "priority": row["priority"],
            "release_status": row["release_status"],
            "crop_specific_signal": cap["crop_signal"],
            "CropGenomeFM_required_capability": cap["ours"],
            "AgroNT_1B_fair_role_and_expected_limit": cap["agro"],
            "PlantCAD2_PlantCaduceus_fair_role": cap["plantcad"],
            "NTv2_500M_fair_role_and_expected_limit": cap["nt500"],
            "Evo2_1B_fair_role_and_expected_limit": cap["evo2"],
            "required_non_foundation_baseline": cap["classic"],
            "formal_success_rule": row["success_rule"],
            "claim_boundary": row["claim_boundary"],
            "evidence_status": "preregistered_design_only; no task performance yet",
        })
    write_tsv(
        OUT / "future_task_baseline_capability_matrix.tsv",
        [
            "task_id", "priority", "release_status", "crop_specific_signal",
            "CropGenomeFM_required_capability", "AgroNT_1B_fair_role_and_expected_limit",
            "PlantCAD2_PlantCaduceus_fair_role", "NTv2_500M_fair_role_and_expected_limit",
            "Evo2_1B_fair_role_and_expected_limit", "required_non_foundation_baseline",
            "formal_success_rule", "claim_boundary", "evidence_status",
        ],
        out_rows,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    taxonomy_tables()
    baseline_gap_table()
    capability_matrix()
    outputs = [
        "downstream_taxonomy_detail.tsv",
        "downstream_taxonomy_summary.tsv",
        "baseline_gap_summary.tsv",
        "future_task_baseline_capability_matrix.tsv",
    ]
    for name in outputs:
        rows = read_tsv(OUT / name)
        print(f"wrote {OUT / name} ({len(rows)} rows, {len(rows[0]) if rows else 0} columns)")


if __name__ == "__main__":
    main()
