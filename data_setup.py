#!/usr/bin/env python3
"""
Data setup and preparation script for MODEL MAYHEM: PCB DEFECT DETECTION.

Features:
- Roboflow API download integration (project: 'pcb-defect-detection-9ewqw-duvuw', version 2)
- Direct zip extraction helper
- Dataset integrity inspection (image counts, label counts, defect distribution)
- Automatic dataset.yaml normalization with absolute or relative paths
"""

import os
import sys
import yaml
import shutil
import zipfile
import argparse
from pathlib import Path
from collections import Counter

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

DEFAULT_WORKSPACE = "uthm-project-visionbased-pcb-defect-detection"
DEFAULT_PROJECT = "pcb-defect-detection-9ewqw-duvuw"
DEFAULT_VERSION = 2


def download_from_roboflow(api_key: str, dest_dir: str = "dataset") -> str:
    """Download dataset using Roboflow Python SDK."""
    from roboflow import Roboflow

    print(f"📥 Connecting to Roboflow Universe...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(DEFAULT_WORKSPACE).project(DEFAULT_PROJECT)
    version = project.version(DEFAULT_VERSION)
    print(f"📦 Downloading version {DEFAULT_VERSION} in yolov8 format...")
    dataset = version.download("yolov8", location=dest_dir)
    print(f"✅ Downloaded to: {dataset.location}")
    return dataset.location


def extract_zip(zip_path: str, dest_dir: str = "dataset") -> str:
    """Extract a downloaded zip archive into dest_dir."""
    print(f"📂 Extracting {zip_path} to {dest_dir}...")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"✅ Extracted files to {dest.resolve()}")
    return str(dest)


def inspect_and_fix_dataset(dataset_dir: str, output_yaml: str = "dataset.yaml") -> dict:
    """
    Inspects images and YOLO label files in dataset_dir, verifies class ids,
    counts defect distribution per split, and writes a normalized dataset.yaml.
    """
    root = Path(dataset_dir).resolve()
    print(f"\n🔍 Inspecting dataset at: {root}")

    # Standard splits
    splits = ["train", "valid", "test"]
    stats = {}

    for split in splits:
        split_dir = root / split
        # Check possible directory layouts: split/images or split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists():
            img_dir = split_dir
        if not lbl_dir.exists():
            lbl_dir = split_dir

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        labels = list(lbl_dir.glob("*.txt"))

        # Count classes in annotations
        class_counts = Counter()
        for lbl in labels:
            if lbl.name == "classes.txt":
                continue
            try:
                with open(lbl, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            class_counts[cls_id] += 1
            except Exception:
                pass

        stats[split] = {
            "image_count": len(images),
            "label_file_count": len(labels),
            "class_counts": dict(class_counts),
            "img_dir": str(img_dir),
            "lbl_dir": str(lbl_dir),
        }

    print("\n📊 Dataset Summary:")
    print("-" * 65)
    print(f"{'Split':<10} | {'Images':<8} | {'Labels':<8} | Class Annotations Breakdown")
    print("-" * 65)
    for split, info in stats.items():
        breakdown = ", ".join(
            f"{CLASS_NAMES[k] if 0 <= k < len(CLASS_NAMES) else k}: {v}"
            for k, v in sorted(info["class_counts"].items())
        )
        print(f"{split:<10} | {info['image_count']:<8} | {info['label_file_count']:<8} | {breakdown or 'None'}")
    print("-" * 65)

    # Determine train and val image directories
    train_dir = stats["train"]["img_dir"] if stats["train"]["image_count"] > 0 else str(root / "train")
    val_dir = stats["valid"]["img_dir"] if stats["valid"]["image_count"] > 0 else str(root / "valid")
    test_dir = stats["test"]["img_dir"] if stats["test"]["image_count"] > 0 else str(root / "test")

    yaml_data = {
        "path": str(root).replace("\\", "/"),
        "train": str(Path(train_dir).relative_to(root)).replace("\\", "/"),
        "val": str(Path(val_dir).relative_to(root)).replace("\\", "/"),
        "test": str(Path(test_dir).relative_to(root)).replace("\\", "/") if stats["test"]["image_count"] > 0 else None,
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }

    # If test is None, omit it
    if yaml_data["test"] is None:
        del yaml_data["test"]

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Generated normalized YAML: {Path(output_yaml).resolve()}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="PCB Defect Dataset Setup Helper")
    parser.add_argument("--roboflow-key", type=str, default=None, help="Roboflow API Key to download dataset.")
    parser.add_argument("--zip-file", type=str, default=None, help="Path to local downloaded dataset ZIP file.")
    parser.add_argument("--dir", type=str, default="dataset", help="Target dataset directory.")
    parser.add_argument("--output-yaml", type=str, default="dataset.yaml", help="Output YAML config file.")
    args = parser.parse_args()

    target_dir = args.dir

    if args.roboflow_key:
        target_dir = download_from_roboflow(args.roboflow_key, args.dir)
    elif args.zip_file:
        target_dir = extract_zip(args.zip_file, args.dir)

    if Path(target_dir).exists():
        inspect_and_fix_dataset(target_dir, args.output_yaml)
    else:
        print(f"Dataset directory '{target_dir}' does not exist yet.")
        print("Provide --roboflow-key <KEY> or --zip-file <PATH> to initialize.")


if __name__ == "__main__":
    main()
