"""Create presentation figures from the bundled reconstruction experiment.

The figures are derived only from the saved experiment artifacts.  They make
the data-cleaning/synthesis pipeline and same-split ablation results easy to
show in a report or demonstration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_pipeline(output: Path, metadata: dict) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.axis("off")
    steps = [
        ("Bundled breath\nrecordings", "#dceeff"),
        ("Decode & crop\n18 windows/source", "#e9f8e5"),
        ("Source-group\nsplit", "#fff0cc"),
        ("Normal + three\ncontrolled conditions", "#fde2e4"),
        ("Log-Mel features\n32 bands", "#e7ddff"),
        ("Hybrid ablation\nheld-out evaluation", "#d9f0ed"),
    ]
    for i, (label, color) in enumerate(steps):
        x = 0.02 + i * 0.162
        rect = plt.Rectangle((x, 0.28), 0.135, 0.43, facecolor=color, edgecolor="#34495e", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + 0.0675, 0.495, label, ha="center", va="center", fontsize=10, weight="semibold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 0.158, 0.495), xytext=(x + 0.137, 0.495), arrowprops={"arrowstyle": "->", "lw": 1.6})
    split = {
        name: [source for source, assigned in metadata["splits"].items() if assigned == name]
        for name in ("train", "val", "test")
    }
    ax.text(
        0.5,
        0.08,
        "Source groups: train {}  |  validation {}  |  held-out test {}".format(
            ", ".join(split["train"]), ", ".join(split["val"]), ", ".join(split["test"])
        ),
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output / "pipeline_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_comparison(output: Path, results: list[dict]) -> None:
    names = [row["experiment"] for row in results]
    f1 = [row["test_macro_f1"] * 100 for row in results]
    accuracy = [row["test_accuracy"] * 100 for row in results]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, accuracy, width, label="Held-out accuracy", color="#4c78a8")
    ax.bar(x + width / 2, f1, width, label="Held-out macro F1", color="#f58518")
    for index, value in enumerate(f1):
        ax.text(index + width / 2, value + 1.1, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent")
    ax.set_title("Same-split hybrid ablation comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "ablation_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_e2_matrix(output: Path, details: dict) -> None:
    labels = ["normal", "ambient", "movable", "fixed"]
    matrix = np.asarray(details["confusion_matrices"]["E2"])
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    ax.set(xticks=np.arange(4), yticks=np.arange(4), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True", title="E2 held-out confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix_E2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/bundled_reconstruction"))
    args = parser.parse_args()
    root = args.experiment_dir
    metadata = json.loads((root / "dataset_metadata.json").read_text(encoding="utf-8"))
    with (root / "ablation" / "ablation_results.csv").open(newline="", encoding="utf-8") as handle:
        results = [
            {"experiment": row["variant"], "test_accuracy": float(row["accuracy"]), "test_macro_f1": float(row["macro_f1"])}
            for row in csv.DictReader(handle)
        ]
    details = json.loads((root / "ablation" / "detailed_reports.json").read_text(encoding="utf-8"))
    _save_pipeline(root, metadata)
    _save_comparison(root, results)
    _save_e2_matrix(root, details)
    print(f"Wrote presentation figures to {root.resolve()}")


if __name__ == "__main__":
    main()
