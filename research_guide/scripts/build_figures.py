#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research_guide" / "source_data"
OUT = ROOT / "research_guide" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"
if FONT.exists():
    mpl.font_manager.fontManager.addfont(str(FONT))
    mpl.rcParams["font.family"] = "Noto Sans CJK SC"
mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "navy": "#183153",
    "blue": "#2F6BFF",
    "cyan": "#31B7C2",
    "green": "#2E9E6F",
    "orange": "#E8873A",
    "red": "#D9534F",
    "purple": "#7D5FB2",
    "gray": "#6B7280",
    "light": "#EEF3F8",
}


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, wh, text, fc, ec="none", fontsize=8.5, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.0,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)
    return patch


def arrow(ax, start, end, color="#8090A0"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=color))


def architecture_figure():
    fig, ax = plt.subplots(figsize=(14.0, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "CropGenome-FM v2：字符级、反向互补等变、Hyena–稀疏注意力混合架构",
            fontsize=16, weight="bold", color=COLORS["navy"], va="top")
    ax.text(0.02, 0.915, "369,505,287 trainable parameters | 8,192-bp frozen base | 65,536-bp architecture gate",
            fontsize=9.5, color=COLORS["gray"], va="top")

    box(ax, (0.025, 0.70), (0.105, 0.105), "DNA\nA/C/G/T/N", COLORS["light"], COLORS["navy"], 9, "bold")
    arrow(ax, (0.13, 0.752), (0.165, 0.752))
    box(ax, (0.165, 0.70), (0.115, 0.105), "7-token\nEmbedding\nd=1,024", "#DCE8FF", COLORS["blue"], 8.6, "bold")
    arrow(ax, (0.28, 0.752), (0.315, 0.752))

    # Main backbone as eight groups.
    start_x, group_w, gap = 0.315, 0.061, 0.009
    dilations = [1, 2, 4, 8, 16, 32, 64, 128]
    for i, dilation in enumerate(dilations):
        x = start_x + i * (group_w + gap)
        box(ax, (x, 0.72), (group_w, 0.07), "4× HyenaLite", "#DFF3EA", COLORS["green"], 7.1, "bold")
        box(ax, (x, 0.625), (group_w, 0.07), f"Local Attn\nd={dilation}", "#FFE7D1", COLORS["orange"], 7.0, "bold")
        if i < 7:
            arrow(ax, (x + group_w, 0.752), (x + group_w + gap, 0.752))
    ax.text(0.315, 0.585, "32 HyenaLite blocks + 8 attention insertions (after every four HyenaLite blocks)",
            fontsize=8.3, color=COLORS["gray"])
    ax.text(0.315, 0.555, "Attention: 8 heads, chunk=512, dilation schedule 1→128; HyenaLite kernel=127",
            fontsize=8.3, color=COLORS["gray"])

    # RC branch.
    box(ax, (0.12, 0.37), (0.21, 0.09), "Direct strand forward", "#E9EFF8", COLORS["navy"], 9, "bold")
    box(ax, (0.12, 0.225), (0.21, 0.09), "Reverse-complement forward\nflip + nucleotide complement", "#F4EAFB", COLORS["purple"], 8.3, "bold")
    arrow(ax, (0.33, 0.415), (0.415, 0.415))
    arrow(ax, (0.33, 0.270), (0.415, 0.365))
    box(ax, (0.415, 0.33), (0.20, 0.11), "Align RC coordinates\nand class channels\nthen average", "#FFF4D6", "#C69922", 8.5, "bold")
    arrow(ax, (0.615, 0.385), (0.68, 0.385))
    box(ax, (0.68, 0.35), (0.13, 0.09), "Final hidden\nrepresentation", "#DCE8FF", COLORS["blue"], 8.5, "bold")

    box(ax, (0.845, 0.44), (0.13, 0.09), "MLM head\n5 nucleotide classes", "#DFF3EA", COLORS["green"], 8.2, "bold")
    box(ax, (0.845, 0.31), (0.13, 0.09), "Region head\n7 sequence classes", "#FFE7D1", COLORS["orange"], 8.2, "bold")
    box(ax, (0.845, 0.18), (0.13, 0.09), "Frozen embeddings\n+ linear/ridge probe", "#F4EAFB", COLORS["purple"], 8.2, "bold")
    arrow(ax, (0.81, 0.395), (0.845, 0.485))
    arrow(ax, (0.81, 0.395), (0.845, 0.355))
    arrow(ax, (0.81, 0.395), (0.845, 0.225))

    ax.text(0.025, 0.105, "Current objectives", fontsize=10.5, weight="bold", color=COLORS["navy"])
    ax.text(0.025, 0.065,
            "MLM (weight 1.0) + symmetric RC KL consistency (0.02) + window-level region classification (0.05). "
            "Stage C1 uses gradient checkpointing for 64K sequences.", fontsize=8.8, color=COLORS["gray"])
    save(fig, "figure_01_architecture")


def pretraining_data_figure():
    rows = list(csv.DictReader((SOURCE / "pretraining_stage_summary.tsv").open(), delimiter="\t"))
    contexts = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    stages = [r["stage"] for r in rows]
    matrix = np.zeros((len(stages), len(contexts)), dtype=float)
    for i, row in enumerate(rows):
        values = dict(x.split(":") for x in row["context_windows"].split(";"))
        for j, c in enumerate(contexts):
            matrix[i, j] = float(values.get(str(c), 0))
    token_b = [int(r["total_tokens"]) / 1e9 for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.7, 1]})
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.32 + 0.09 * i) for i in range(len(contexts))]
    left = np.zeros(len(stages))
    y = np.arange(len(stages))
    for j, c in enumerate(contexts):
        ax.barh(y, matrix[:, j], left=left, color=colors[j], edgecolor="white", height=0.68,
                label=f"{c//1024}K" if c < 1048576 else str(c))
        left += matrix[:, j]
    ax.set_yticks(y, stages)
    ax.invert_yaxis()
    ax.set_xlabel("窗口数（横轴使用对数刻度）")
    ax.set_xscale("log")
    ax.set_title("预训练数据包：窗口长度组成", loc="left", weight="bold", color=COLORS["navy"])
    ax.legend(title="Context (bp)", ncol=4, frameon=False, loc="lower right")
    for i, total in enumerate(left):
        ax.text(total * 1.04, i, f"{int(total):,}", va="center", fontsize=8)

    status_color = [COLORS["green"], COLORS["orange"], COLORS["gray"], COLORS["gray"]]
    ax2.barh(y, token_b, color=status_color, height=0.68)
    ax2.set_yticks(y, stages)
    ax2.invert_yaxis()
    ax2.set_xlabel("全部split碱基token（十亿）")
    ax2.set_title("数据量与执行状态", loc="left", weight="bold", color=COLORS["navy"])
    labels = ["completed; selected step 14K", "partial; stopped step 569", "data ready; not trained", "data ready; not trained"]
    for i, (v, lab) in enumerate(zip(token_b, labels)):
        ax2.text(v + max(token_b) * 0.02, i, f"{v:.2f}B\n{lab}", va="center", fontsize=8)
    ax2.set_xlim(0, max(token_b) * 1.38)
    fig.suptitle("CropGenome-FM curriculum: data materialization is not equivalent to completed training",
                 x=0.06, ha="left", fontsize=14, weight="bold", color=COLORS["navy"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "figure_02_pretraining_data")


def core_performance_figure():
    rows = list(csv.DictReader((SOURCE / "aggregate_primary_metrics.tsv").open(), delimiter="\t"))
    rows = [r for r in rows if r["panel"] == "core_3_task_macro"]
    display = {
        "CropGenomeFM_step14000": ("CropGenome-FM", COLORS["blue"], 3.0),
        "CropGenomeFM_random_init": ("Random-init same arch.", COLORS["gray"], 1.7),
        "AgroNT_1B": ("AgroNT 1B", COLORS["green"], 1.8),
        "PlantCAD2_Small": ("PlantCAD2 Small", COLORS["orange"], 1.8),
        "PlantCaduceus_l32": ("PlantCaduceus l32", COLORS["purple"], 1.8),
        "NTv2_100M_multi_species": ("NT-v2 100M", COLORS["cyan"], 1.7),
        "Evo2_1B_base": ("Evo2 1B", COLORS["red"], 1.5),
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    contexts = [512, 2048, 8192]
    for mid, (label, color, lw) in display.items():
        vals = []
        for c in contexts:
            x = [float(r["value"]) for r in rows if r["model_id"] == mid and int(r["context_bp"]) == c]
            vals.append(x[0] if x else np.nan)
        ax.plot(contexts, vals, marker="o", markersize=5.5, linewidth=lw, label=label, color=color,
                zorder=4 if mid == "CropGenomeFM_step14000" else 2)
        if mid == "CropGenomeFM_step14000":
            for c, v in zip(contexts, vals):
                ax.text(c, v + 0.012, f"{v:.3f}", ha="center", color=color, weight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts, ["512", "2,048", "8,192"])
    ax.set_ylim(0.50, 0.92)
    ax.set_xlabel("输入窗口长度（bp）")
    ax.set_ylabel("三任务宏平均AUPRC")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.suptitle("CropGenome-Bench-v1正式测试：冻结表示＋线性probe",
                 x=0.08, y=0.98, ha="left", weight="bold", color=COLORS["navy"], fontsize=14)
    fig.text(0.08, 0.925, "5 probe seeds；512 bp排名第3，2,048和8,192 bp排名第1",
             color=COLORS["gray"], fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save(fig, "figure_03_core_performance")


def external_performance_figure():
    rows = list(csv.DictReader((SOURCE / "external_primary_metrics.tsv").open(), delimiter="\t"))
    tasks = [
        "enhancer_cassava_proseq",
        "lncrna_multispecies",
        "gene_expression_arabidopsis_thaliana",
        "gene_expression_glycine_max",
        "gene_expression_oryza_sativa",
        "gene_expression_solanum_lycopersicum",
        "gene_expression_zea_mays",
    ]
    task_labels = ["Cassava\nenhancer", "Multi-species\nlncRNA", "Arabidopsis\nexpression", "Soybean\nexpression", "Rice\nexpression", "Tomato\nexpression", "Maize\nexpression"]
    models = ["CropGenomeFM_step14000", "CropGenomeFM_random_init", "AgroNT_1B", "PlantCAD2_Small", "PlantCaduceus_l32", "NTv2_100M_multi_species", "Evo2_1B_base"]
    model_labels = ["CropGenome-FM", "Random-init", "AgroNT 1B", "PlantCAD2", "PlantCaduceus", "NT-v2 100M", "Evo2 1B"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.05, 1]})
    for ax, context in zip(axes, ["512", "6000"]):
        mat = np.full((len(models), len(tasks)), np.nan)
        for i, model in enumerate(models):
            for j, task in enumerate(tasks):
                x = [float(r["value"]) for r in rows if r["context_bp"] == context and r["model_id"] == model and r["task_id"] == task]
                if x:
                    mat[i, j] = x[0]
        im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=0.9, aspect="auto")
        ax.set_xticks(range(len(tasks)), task_labels, rotation=38, ha="right")
        ax.set_yticks(range(len(models)), model_labels)
        ax.set_title(f"{int(context):,} bp", weight="bold", color=COLORS["navy"])
        for i in range(len(models)):
            for j in range(len(tasks)):
                val = mat[i, j]
                text = "N/E" if np.isnan(val) else f"{val:.3f}"
                color = "#555555" if np.isnan(val) else ("white" if val > 0.58 else "#16324F")
                ax.text(j, i, text, ha="center", va="center", fontsize=7.4, color=color,
                        weight="bold" if i == 0 else "normal")
    fig.suptitle("Plant Genomic Benchmark正式测试：当前模型在长窗口受益，但植物专属基线仍是主要差距",
                 x=0.04, ha="left", fontsize=14, weight="bold", color=COLORS["navy"])
    fig.text(0.04, 0.025, "N/E = zero-truncation protocol下未评估；表中为primary seed 13；binary另有4个seed且full-data值相同，regression为确定性ridge。",
             fontsize=8.8, color=COLORS["gray"])
    fig.subplots_adjust(left=0.10, right=0.88, top=0.86, bottom=0.22, wspace=0.22)
    cax = fig.add_axes([0.91, 0.24, 0.012, 0.60])
    fig.colorbar(im, cax=cax, label="Primary metric: AUPRC (binary) / Pearson (expression)")
    save(fig, "figure_04_external_performance")


def strategy_figure():
    gaps = list(csv.DictReader((SOURCE / "baseline_gap_summary.tsv").open(), delimiter="\t"))
    future = list(csv.DictReader((SOURCE / "future_task_registry.tsv").open(), delimiter="\t"))

    gap_labels = {
        ("core_3_task_macro", "512", "macro AUPRC"): "Core 512\nAUPRC",
        ("core_3_task_macro", "2048", "macro AUPRC"): "Core 2K\nAUPRC",
        ("core_3_task_macro", "8192", "macro AUPRC"): "Core 8K\nAUPRC",
        ("external_group_macro", "512", "macro AUPRC"): "External binary 512\nAUPRC",
        ("external_group_macro", "512", "macro Pearson"): "External expression 512\nPearson",
        ("external_group_macro", "6000", "macro AUPRC"): "External binary 6K\nAUPRC",
        ("external_group_macro", "6000", "macro Pearson"): "External expression 6K\nPearson",
    }
    ordered_keys = [
        ("core_3_task_macro", "512", "macro AUPRC"),
        ("core_3_task_macro", "2048", "macro AUPRC"),
        ("core_3_task_macro", "8192", "macro AUPRC"),
        ("external_group_macro", "512", "macro AUPRC"),
        ("external_group_macro", "512", "macro Pearson"),
        ("external_group_macro", "6000", "macro AUPRC"),
        ("external_group_macro", "6000", "macro Pearson"),
    ]
    by_key = {(r["panel"], r["context_bp"], r["metric"]): r for r in gaps}
    values = [float(by_key[key]["gap_vs_best_plant"]) for key in ordered_keys]
    labels = [gap_labels[key] for key in ordered_keys]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.8), gridspec_kw={"width_ratios": [1.12, 1]})
    y = np.arange(len(values))
    colors = [COLORS["green"] if value >= 0 else COLORS["orange"] for value in values]
    ax.barh(y, values, color=colors, height=0.62)
    ax.axvline(0, color=COLORS["navy"], linewidth=1.0)
    ax.axvline(-0.02, color=COLORS["gray"], linewidth=0.9, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.072, 0.045)
    ax.set_xlabel("CropGenome-FM − best current plant baseline")
    ax.set_title("Current common-task gap", loc="left", weight="bold", color=COLORS["navy"])
    ax.grid(axis="x", alpha=0.18)
    for yi, value in zip(y, values):
        x = value + (0.002 if value >= 0 else -0.002)
        ax.text(x, yi, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=8.2, weight="bold")

    short = {
        "CROP-LONGGENE-SEG": "Long-gene segmentation",
        "CROP-DISTAL-CIS-PAIR": "Distal cis pairs",
        "CROP-ISOFORM-LR": "Long-read isoforms",
        "CROP-POLYPLOID-HOMEOLOG": "Polyploid homoeologs",
        "CROP-TE-SV-REG": "TE/SV regulation",
        "CROP-NLR-CLUSTER": "NLR clusters",
        "CROP-ACR-STRESS": "Stress ACRs",
        "CROP-PANGENOME-SV": "Pan-genome SV",
        "CROP-QTL-VAR-RANK": "QTL variant ranking",
    }
    priority_colors = {"P0": COLORS["blue"], "P1": COLORS["purple"], "P2": COLORS["gray"]}
    task_y = np.arange(len(future))
    max_lengths = [max(int(x) for x in row["input_lengths_bp"].split(";")) for row in future]
    task_colors = [priority_colors[row["priority"].split("_", 1)[0]] for row in future]
    ax2.scatter(max_lengths, task_y, s=95, c=task_colors, edgecolors="white", linewidths=0.8, zorder=3)
    for x, yi, row in zip(max_lengths, task_y, future):
        status = "source only" if row["release_status"].startswith("current source") else ("blocked" if "blocked" in row["release_status"] else "not frozen")
        ax2.text(x * 1.06, yi, status, va="center", fontsize=7.5, color=COLORS["gray"])
    ax2.set_xscale("log", base=2)
    ticks = [2048, 8192, 32768, 65536, 131072, 262144]
    ax2.set_xticks(ticks, ["2K", "8K", "32K", "64K", "128K", "256K"])
    ax2.set_xlim(1500, 560000)
    ax2.set_yticks(task_y, [short[row["task_id"]] for row in future])
    ax2.invert_yaxis()
    ax2.set_xlabel("Maximum planned input length (bp)")
    ax2.set_title("Crop-specific task pressure tests", loc="left", weight="bold", color=COLORS["navy"])
    ax2.grid(axis="x", alpha=0.18)

    fig.suptitle("Common-task parity first; crop-specific superiority only after frozen fair baselines",
                 x=0.04, ha="left", fontsize=14, weight="bold", color=COLORS["navy"])
    fig.text(0.04, 0.022,
             "Left: each gap uses its own primary metric and is not pooled across AUPRC/Pearson. Dashed line marks −0.02 only as a visual reference. "
             "Right: planned context and data status, not observed performance; blue=P0, purple=P1, grey=P2.", fontsize=8.4, color=COLORS["gray"])
    fig.subplots_adjust(left=0.17, right=0.96, top=0.87, bottom=0.14, wspace=0.44)
    save(fig, "figure_05_strategy_map")


if __name__ == "__main__":
    architecture_figure()
    pretraining_data_figure()
    core_performance_figure()
    external_performance_figure()
    strategy_figure()
    print(f"wrote 10 figure files under {OUT}")
