#!/usr/bin/env python3
"""Generate matplotlib PNG training curves for CropGenome-FM v2 logs.

The training script prints JSON lines. This parser keeps train/eval metrics in
TSV source data and draws publication-readable PNG curves with titles, axes,
ticks, legends, raw train curves, rolling medians, validation checkpoints, and
latest-point annotations. Public GitHub artifacts are PNG-only: no SVG/PDF.
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

METRICS = [
    ("loss", "total loss", "总训练目标，越低越好"),
    ("mlm_loss", "MLM loss", "遮盖碱基预测损失，主学习目标，越低越好"),
    ("rc_loss", "RC consistency loss", "反向互补一致性损失，越低说明双链方向更一致"),
    ("selection_loss", "selection loss", "最佳 checkpoint 选择指标：MLM + 小权重 RC，越低越好"),
    ("region_loss", "region auxiliary loss", "区域辅助任务损失，只作弱监督/健康检查"),
    ("region_acc", "region auxiliary accuracy", "区域辅助任务准确率，只作健康检查，越高越好"),
]

COLORS = {
    "train": (21, 101, 192),
    "val": (198, 40, 40),
    "axis": (50, 50, 50),
    "grid": (225, 225, 225),
    "text": (20, 20, 20),
    "bg": (255, 255, 255),
}


def parse_log(path: Path):
    rows = []
    resume_start_step = None
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "resume" in payload and "start_step" in payload:
                resume_start_step = int(payload["start_step"])
            if "step" not in payload:
                continue
            step = int(payload["step"])
            if "loss" in payload:
                rows.append(normalize_row(step, "train", payload))
            if any(key.startswith("val_") for key in payload):
                val_payload = {key[4:]: value for key, value in payload.items() if key.startswith("val_")}
                rows.append(normalize_row(step, "val", val_payload))
    return rows, resume_start_step


def merge_logs(log_paths):
    rows = []
    for log_path in log_paths:
        if not log_path.exists():
            continue
        new_rows, resume_start_step = parse_log(log_path)
        if resume_start_step is not None:
            rows = [row for row in rows if int(row["step"]) <= resume_start_step]
        rows.extend(new_rows)
        deduped = {}
        for row in rows:
            deduped[(row["split"], int(row["step"]))] = row
        rows = sorted(deduped.values(), key=lambda row: (int(row["step"]), row["split"]))
    return rows


def normalize_row(step, split, payload):
    row = {"step": step, "split": split}
    row["lr"] = payload.get("lr", "")
    for key, _, _ in METRICS:
        value = payload.get(key, "")
        row[key] = "" if value == "" or value is None else float(value)
    row["region_valid_count"] = payload.get("region_valid_count", "")
    return row


def write_source_tsv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "split", "lr"] + [key for key, _, _ in METRICS] + ["region_valid_count"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_points(rows, metric, split):
    points = []
    for row in rows:
        if row["split"] != split:
            continue
        value = row.get(metric, "")
        if value == "" or value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            points.append((int(row["step"]), value))
    return points



def rolling_median(points, window=21):
    """Return a centered rolling median series for noisy train points."""
    if not points:
        return []
    half = max(1, int(window) // 2)
    values = [v for _, v in points]
    smoothed = []
    for i, (step, _) in enumerate(points):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        window_values = sorted(values[lo:hi])
        mid = len(window_values) // 2
        if len(window_values) % 2:
            med = window_values[mid]
        else:
            med = 0.5 * (window_values[mid - 1] + window_values[mid])
        smoothed.append((step, med))
    return smoothed


def write_png(path: Path, metric_key: str, title: str, description: str, train_points, val_points):
    """Write a real matplotlib PNG with visible title, axes, ticks and legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=180)
    if train_points:
        xs = [x for x, _ in train_points]
        ys = [y for _, y in train_points]
        ax.plot(xs, ys, color="#9CA3AF", linewidth=0.9, alpha=0.55, label="train raw")
        smooth = rolling_median(train_points, window=21)
        ax.plot([x for x, _ in smooth], [y for _, y in smooth], color="#1D4ED8", linewidth=2.3, label="train rolling median (21)")
        latest_x, latest_y = train_points[-1]
        ax.scatter([latest_x], [latest_y], marker="*", s=170, color="#7C3AED", edgecolor="white", linewidth=0.8, zorder=5, label=f"latest train step {latest_x}")
        ax.annotate(f"latest train\nstep {latest_x}\n{latest_y:.4f}", xy=(latest_x, latest_y), xytext=(12, 18), textcoords="offset points", fontsize=8, color="#4C1D95", arrowprops={"arrowstyle": "->", "color": "#7C3AED", "lw": 0.8})
    if val_points:
        vx = [x for x, _ in val_points]
        vy = [y for _, y in val_points]
        ax.scatter(vx, vy, marker="D", s=70, color="#DC2626", edgecolor="white", linewidth=0.8, zorder=6, label="validation checkpoints")
        ax.plot(vx, vy, color="#DC2626", linewidth=1.5, alpha=0.65)
        if len(val_points) <= 11:
            label_indices = list(range(len(val_points)))
        else:
            stride = max(1, math.ceil((len(val_points) - 1) / 8))
            label_indices = sorted(set(range(0, len(val_points), stride)) | {len(val_points) - 1, min(range(len(val_points)), key=lambda i: val_points[i][1])})
        for label_rank, point_index in enumerate(label_indices):
            x, y = val_points[point_index]
            if point_index == len(val_points) - 1:
                offset, align = (-20, 48), "right"
            else:
                offset = (8, 18) if label_rank % 2 == 0 else (8, -30)
                align = "left"
            ax.annotate(f"val {x}\n{y:.4f}", xy=(x, y), xytext=offset, textcoords="offset points", ha=align, fontsize=8, color="#7F1D1D", arrowprops={"arrowstyle": "->", "color": "#DC2626", "lw": 0.7})
    ax.set_title(title, fontsize=15, weight="bold", pad=12)
    ax.set_xlabel("Training step", fontsize=12)
    ylabel = "Selection loss (MLM + 0.02 × RC; lower is better)" if metric_key == "selection_loss" else f"{title.split('—')[-1].strip()} (lower is better)"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.18)
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=9)
    all_values = [v for _, v in train_points] + [v for _, v in val_points]
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        margin = max(1e-6, (y_max - y_min) * 0.12)
        ax.set_ylim(y_min - margin, y_max + margin)
    fig.text(0.01, 0.01, "Direct matplotlib PNG from TSV source data. Public artifact policy: PNG only; no SVG/PDF.", fontsize=8, color="#4B5563")
    fig.text(0.99, 0.01, "Train raw + rolling median; validation checkpoints; latest train point annotated.", fontsize=8, color="#4B5563", ha="right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", facecolor="white")
    plt.close(fig)

def write_summary(path: Path, rows, figures_dir, metrics, artifact_prefix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "train_points", "val_points", "latest_train_step", "latest_train_value", "latest_val_step", "latest_val_value", "png"])
        for key, _, _ in metrics:
            tr = metric_points(rows, key, "train")
            va = metric_points(rows, key, "val")
            writer.writerow([
                key,
                len(tr),
                len(va),
                tr[-1][0] if tr else "",
                f"{tr[-1][1]:.8g}" if tr else "",
                va[-1][0] if va else "",
                f"{va[-1][1]:.8g}" if va else "",
                str(figures_dir / f"{artifact_prefix}_{key}.png"),
            ])

def write_readme(path: Path, log_path: Path, rows, source_rel: str, summary_rel: str):
    latest_train = max((r for r in rows if r["split"] == "train"), key=lambda r: r["step"], default=None)
    latest_val = max((r for r in rows if r["split"] == "val"), key=lambda r: r["step"], default=None)
    lines = [
        "# CropGenome-FM v2 Stable Stage_B training curves",
        "",
        f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CST",
        "",
        f"训练日志: `{log_path}`",
        f"源数据: `{source_rel}`",
        f"曲线摘要: `{summary_rel}`",
        "",
        "## 当前状态",
        "",
        f"- 最新 train step（训练步）: `{latest_train['step'] if latest_train else '尚未出现'}`",
        f"- 最新 validation step（验证步）: `{latest_val['step'] if latest_val else '尚未出现'}`",
        "",
        "## 单独曲线说明",
        "",
    ]
    for key, title, description in METRICS:
        lines.extend([
            f"### {title}",
            "",
            f"- 含义: {description}。",
            f"- PNG: `figures/v2_stable_stageB_{key}.png`",
            "",
        ])
    lines.extend([
        "## 解释边界",
        "",
        "- `selection_loss`（选择损失）用于 best checkpoint（最佳模型存档点）和 early stopping（早停），不把 `region_loss`（区域损失）放进主选择指标。",
        "- `region_loss/region_acc`（区域辅助损失/准确率）只作为 weak supervision（弱监督）健康检查，不能单独写成正式下游 benchmark（基准评测）胜利。",
        "- 正式下游任务必须在独立目录中生成各自的 AUROC/AUPRC/F1/loss 曲线和说明，不能只引用预训练 loss。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, action="append", help="Training log path. Repeat to merge resumed runs; later resume logs truncate earlier rows after their start_step.")
    p.add_argument("--out-dir", default="docs/training_progress")
    p.add_argument("--artifact-prefix", default="v2_stable_stageB", help="Filename prefix for TSV and PNG artifacts. Use a generation-specific prefix for continuation runs.")
    p.add_argument("--title-prefix", default="CropGenome-FM v2 Stable Stage B", help="English plot-title prefix.")
    p.add_argument("--metrics", default=",".join(key for key, _, _ in METRICS), help="Comma-separated metric keys to plot. Source TSV always retains all metrics.")
    args = p.parse_args()
    log_paths = [Path(path) for path in args.log]
    out_dir = Path(args.out_dir)
    source_dir = out_dir / "source_data"
    figures_dir = out_dir / "figures"
    rows = merge_logs(log_paths)
    metric_map = {key: (key, title, description) for key, title, description in METRICS}
    requested = [key.strip() for key in args.metrics.split(",") if key.strip()]
    unknown = [key for key in requested if key not in metric_map]
    if unknown:
        p.error(f"unknown metric key(s): {', '.join(unknown)}")
    if not requested:
        p.error("--metrics must select at least one metric")
    selected_metrics = [metric_map[key] for key in requested]
    source_path = source_dir / f"{args.artifact_prefix}_metrics.tsv"
    summary_path = source_dir / f"{args.artifact_prefix}_curve_summary.tsv"
    write_source_tsv(source_path, rows)
    for key, title, description in selected_metrics:
        train_points = metric_points(rows, key, "train")
        val_points = metric_points(rows, key, "val")
        write_png(figures_dir / f"{args.artifact_prefix}_{key}.png", key, f"{args.title_prefix} — {title}", description, train_points, val_points)
    write_summary(summary_path, rows, Path("figures"), selected_metrics, args.artifact_prefix)
    print(f"v2_curves: rows={len(rows)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
