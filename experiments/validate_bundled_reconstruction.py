"""Validate auditable invariants for the bundled reconstruction dataset.

This script checks what can be established from the prepared arrays and their
manifest: split/group separation, array-manifest counts, finite feature values,
valid labels, exact feature duplicates, class balance, and train-standardised
feature statistics.  It does not turn generated labels into clinical labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/bundled_reconstruction"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.experiment_dir
    output = args.output or root / "VALIDATION_REPORT.md"
    metadata = json.loads((root / "dataset_metadata.json").read_text(encoding="utf-8"))
    with (root / "window_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    arrays = np.load(root / "prepared_source_split.npz")
    class_order = metadata["class_order"]
    groups = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].add(row["split"])
    overlapping_groups = {group: sorted(values) for group, values in groups.items() if len(values) != 1}
    if overlapping_groups:
        raise ValueError(f"A source group crosses splits: {overlapping_groups}")

    report_rows, hashes = [], defaultdict(list)
    for split in ("train", "val", "test"):
        x, y = arrays[f"X_{split}"], arrays[f"y_{split}"]
        manifest_rows = [row for row in rows if row["split"] == split]
        if len(x) != len(y) or len(x) != len(manifest_rows):
            raise ValueError(f"{split}: arrays and manifest have different sizes")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{split}: non-finite values found")
        if not np.isin(y, np.arange(len(class_order))).all():
            raise ValueError(f"{split}: invalid encoded labels")
        for index, sample in enumerate(x):
            hashes[hashlib.sha256(np.ascontiguousarray(sample).tobytes()).hexdigest()].append((split, index))
        expected = [sum(row["class_name"] == label for row in manifest_rows) for label in class_order]
        observed = np.bincount(y, minlength=len(class_order)).tolist()
        if expected != observed:
            raise ValueError(f"{split}: manifest class counts do not match labels")
        report_rows.append((split, len(x), len({row["group_id"] for row in manifest_rows}), observed))
    cross_split_duplicates = [locations for locations in hashes.values() if len({split for split, _ in locations}) > 1]
    if cross_split_duplicates:
        raise ValueError(f"Found {len(cross_split_duplicates)} exact feature hashes across splits")
    train = arrays["X_train"]
    if abs(float(train.mean())) > 1e-5 or abs(float(train.std()) - 1.0) > 1e-5:
        raise ValueError("Training feature standardisation is inconsistent with saved data")
    lines = [
        "# Bundled reconstruction validation report",
        "",
        "**Status: passed.** This report validates the saved reconstruction arrays and manifest. It does not validate clinical meaning of the generated conditions.",
        "",
        "## Checks performed",
        "",
        "- Each source group occurs in exactly one split.",
        "- Each split has matching feature, label and manifest counts.",
        "- Features and labels are finite; labels are within the four declared classes.",
        "- Manifest class counts match the encoded labels.",
        "- No exact log-Mel feature hash appears across splits.",
        "- The saved training features have mean 0 and standard deviation 1 after normalisation.",
        "",
        "## Results",
        "",
        "| Split | Windows | Source groups | normal | ambient | movable synthetic | fixed synthetic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, count, group_count, class_counts in report_rows:
        lines.append(f"| {split} | {count} | {group_count} | " + " | ".join(map(str, class_counts)) + " |")
    lines += [
        "",
        f"Exact feature hashes crossing splits: **{len(cross_split_duplicates)}**.",
        "",
        "## Scope",
        "",
        "The samples are derived from eight bundled ICBHI breathing sources. Ambient and obstruction examples are controlled digital transformations, so this validation supports reproducibility and split hygiene within the reconstruction; it is not external or clinical validation.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
