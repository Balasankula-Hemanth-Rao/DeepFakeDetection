"""
Generate paper-ready evaluation figures from a JSON metrics file.

Usage:
  python paper/generate_eval_plots.py \
    --input paper/evaluation_metrics_template.json \
    --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_metric_comparison(runs: list[dict], output_dir: Path) -> None:
    names = [r["name"] for r in runs]
    auc = [r["auc"] for r in runs]
    acc = [r["accuracy"] for r in runs]
    f1 = [r["f1"] for r in runs]

    x = np.arange(len(names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, auc, width, label="AUC")
    ax.bar(x, acc, width, label="Accuracy")
    ax.bar(x + width, f1, width, label="F1")

    ax.set_title("Evaluation Metric Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_dir / "metric_comparison.png", dpi=300)
    plt.close(fig)


def plot_confusion_heatmap(run: dict, output_dir: Path, file_name: str) -> None:
    cm = np.array([[run["tn"], run["fp"]], [run["fn"], run["tp"]]], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel("Count", rotation=90)

    ax.set_title(f"Confusion Matrix - {run['name']}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Real", "Pred: Fake"])
    ax.set_yticklabels(["True: Real", "True: Fake"])

    for i in range(2):
        for j in range(2):
            value = int(cm[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_dir / file_name, dpi=300)
    plt.close(fig)


def plot_per_method_accuracy(per_method: dict[str, float], output_dir: Path) -> None:
    methods = list(per_method.keys())
    scores = [per_method[m] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, scores)
    ax.set_title("Per-Method Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_xticklabels(methods, rotation=20, ha="right")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "per_method_accuracy.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for paper evaluation section.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("paper/evaluation_metrics_template.json"),
        help="Path to metrics JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output folder for generated figure files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_metrics(args.input)
    runs = data.get("runs", [])
    per_method = data.get("per_method_accuracy", {})

    if not runs:
        raise ValueError("No runs found in metrics JSON. Add at least one run entry.")

    plot_metric_comparison(runs, output_dir)
    plot_per_method_accuracy(per_method, output_dir)

    # Generate one confusion matrix per run.
    for run in runs:
        normalized_name = run["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_heatmap(run, output_dir, f"confusion_matrix_{normalized_name}.png")

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
