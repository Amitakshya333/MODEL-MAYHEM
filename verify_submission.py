#!/usr/bin/env python3
"""
Verification script for MODEL MAYHEM: PCB DEFECT DETECTION submission CSV.

Validation Rules:
1. Header must match exactly:
   image_name,class_name,x_min,y_min,x_max,y_max,confidence
2. Class names must belong to the allowed 6 classes:
   - missing_hole
   - mouse_bite
   - open_circuit
   - short
   - spur
   - spurious_copper
3. Coordinates must be non-negative:
   x_min >= 0, y_min >= 0, x_max > x_min, y_max > y_min
4. Confidence scores must be within [0.0, 1.0].
5. No missing/null/NaN values.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ALLOWED_CLASSES = {
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
}

EXPECTED_COLUMNS = [
    "image_name",
    "class_name",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "confidence",
]


def verify_submission(csv_path: str, image_dir: str = None) -> bool:
    path = Path(csv_path)
    print(f"==================================================")
    print(f"🔍 Validating Submission CSV: {path.resolve()}")
    print(f"==================================================")

    if not path.is_file():
        print(f"❌ ERROR: File '{csv_path}' does not exist.")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ ERROR: Failed to parse CSV file: {e}")
        return False

    errors = []
    warnings = []

    # 1. Check header columns
    actual_cols = list(df.columns)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in actual_cols]
    if missing_cols or actual_cols != EXPECTED_COLUMNS:
        errors.append(
            f"Columns do not match expected format.\n"
            f"  Expected: {EXPECTED_COLUMNS}\n"
            f"  Found:    {actual_cols}\n"
            f"  Missing:  {missing_cols}"
        )
        if missing_cols:
            print("\n❌ VALIDATION FAILED:")
            for err in errors:
                print(f"  • {err}")
            return False

    # 2. Check if empty
    if len(df) == 0:
        warnings.append("CSV is empty (0 detections found). If there are defects in test set, this will score 0.")
        print(f"⚠️  WARNING: 0 detections found in {csv_path}")
        return len(errors) == 0

    # 3. Check for NaN / null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        cols_with_null = null_counts[null_counts > 0].to_dict()
        errors.append(f"Found missing / null / NaN values in columns: {cols_with_null}")

    # 4. Check class names
    invalid_classes = set(df["class_name"].unique()) - ALLOWED_CLASSES
    if invalid_classes:
        errors.append(f"Found invalid class names: {invalid_classes}. Allowed: {sorted(ALLOWED_CLASSES)}")

    # 5. Check coordinates
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Coordinate column '{col}' is not numeric.")

    if not errors:
        # Check coordinate bounds
        neg_xmin = (df["x_min"] < 0).sum()
        neg_ymin = (df["y_min"] < 0).sum()
        if neg_xmin > 0:
            errors.append(f"Found {neg_xmin} bounding boxes with x_min < 0")
        if neg_ymin > 0:
            errors.append(f"Found {neg_ymin} bounding boxes with y_min < 0")

        inv_x = (df["x_max"] <= df["x_min"]).sum()
        inv_y = (df["y_max"] <= df["y_min"]).sum()
        if inv_x > 0:
            errors.append(f"Found {inv_x} bounding boxes where x_max <= x_min")
        if inv_y > 0:
            errors.append(f"Found {inv_y} bounding boxes where y_max <= y_min")

    # 6. Check confidence scores
    if pd.api.types.is_numeric_dtype(df["confidence"]):
        conf_out_of_bounds = ((df["confidence"] < 0.0) | (df["confidence"] > 1.0)).sum()
        if conf_out_of_bounds > 0:
            errors.append(f"Found {conf_out_of_bounds} confidence scores outside [0.0, 1.0]")
    else:
        errors.append("Confidence column is not numeric.")

    # 7. Check image existence if image_dir is provided
    if image_dir:
        img_path = Path(image_dir)
        if img_path.is_dir():
            unique_imgs = df["image_name"].unique()
            missing_imgs = [img for img in unique_imgs if not (img_path / img).exists()]
            if missing_imgs:
                warnings.append(
                    f"{len(missing_imgs)} referenced images not found in '{image_dir}' (sample: {missing_imgs[:3]})"
                )
        else:
            warnings.append(f"Image directory '{image_dir}' not found; skipping image existence check.")

    # Print summary statistics
    print(f"\n📊 Submission Statistics:")
    print(f"  • Total detections:    {len(df)}")
    print(f"  • Unique images:       {df['image_name'].nunique()}")
    print(f"  • Classes represented: {df['class_name'].nunique()} / {len(ALLOWED_CLASSES)}")
    print("\n📈 Class Breakdown:")
    class_counts = df["class_name"].value_counts()
    for cls in sorted(ALLOWED_CLASSES):
        count = class_counts.get(cls, 0)
        conf_mean = df[df["class_name"] == cls]["confidence"].mean() if count > 0 else 0.0
        print(f"  - {cls:<16}: {count:>6} detections | Avg Conf: {conf_mean:.4f}")

    if errors:
        print("\n❌ VALIDATION FAILED:")
        for err in errors:
            print(f"  • {err}")
        return False

    if warnings:
        print("\n⚠️  VALIDATION PASSED WITH WARNINGS:")
        for w in warnings:
            print(f"  • {w}")
    else:
        print("\n✅ VALIDATION PASSED: All submission requirements strictly satisfied!")

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify submission CSV for PCB Defect Detection.")
    parser.add_argument("csv_path", type=str, help="Path to submission CSV file.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Optional path to directory containing images to verify filenames.",
    )
    args = parser.parse_args()

    success = verify_submission(args.csv_path, args.image_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
