#!/usr/bin/env python3
"""
MODEL MAYHEM: PCB DEFECT DETECTION - Training Pipeline
Trains state-of-the-art YOLO models (YOLO11x / YOLOv8x) with domain-specific
hyperparameters for PCB micro-defect localization.

Metric Goals:
- Primary: mAP@50:90
- Tie-Breaker 1: mAP@50
- Tie-Breaker 2: Macro Recall
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO, RTDETR

# Canonical PCB defect classes
CANONICAL_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def train(
    data_yaml: str = "dataset.yaml",
    model_name: str = "yolo11x.pt",
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = 8,
    device: str = "",
    workers: int = 4,
    project: str = "pcb_runs",
    name: str = "yolo11x_pcb",
    resume: bool = False,
    optimizer: str = "auto",
    lr0: float = 0.01,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    export_onnx: bool = False,
) -> dict:
    """
    Executes training with data augmentations tuned for PCB defect detection.
    """
    print("=" * 65)
    print(" PCB DEFECT DETECTION - MODEL TRAINING")
    print("=" * 65)
    print(f"Model:        {model_name}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Epochs:       {epochs}")
    print(f"Image Size:   {imgsz}")
    print(f"Batch Size:   {batch}")
    print(f"Device:       {device or 'auto'}")
    print(f"Optimizer:    {optimizer}")
    print(f"Base LR:      {lr0}")
    print("=" * 65)

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {yaml_path.resolve()}")

    # Initialize model (pre-trained weights)
    print(f"Loading model checkpoint '{model_name}'...")
    if "rtdetr" in str(model_name).lower():
        model = RTDETR(model_name)
    else:
        model = YOLO(model_name)

    # Domain-specific augmentation & training hyperparameters for PCB defects:
    # Small defect detection requires high resolution (1024+), rotation/flip invariance,
    # and copy-paste / mosaic augmentation to expose defect features.
    train_args = {
        "data": str(yaml_path.resolve()),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device if device else None,
        "workers": workers,
        "project": project,
        "name": name,
        "resume": resume,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "weight_decay": weight_decay,
        "warmup_epochs": 3.0,
        "cos_lr": True,
        # Augmentation settings for micro-defects:
        "mosaic": 1.0,          # Combine 4 images to teach scale context
        "mixup": 0.15,          # Subtle mixup for regularization
        "degrees": 15.0,        # Slight rotation (PCBs may be rotated)
        "flipud": 0.5,          # Vertical flip
        "fliplr": 0.5,          # Horizontal flip
        "scale": 0.5,           # Multi-scale variance (+- 50%)
        "shear": 2.0,           # Slight shear
        "perspective": 0.0001,  # Minor perspective shift
        "hsv_h": 0.015,         # Minor hue adjustment
        "hsv_s": 0.5,           # Saturation adjustment
        "hsv_v": 0.4,           # Value (brightness) adjustment
        "copy_paste": 0.15,     # Copy paste defects for class balance
        "close_mosaic": 10,     # Disable mosaic for last 10 epochs for crisp bounding box convergence
        # Loss gains:
        "box": 7.5,             # High box loss gain for tight localization across IoU 0.50:0.90
        "cls": 0.5,             # Classification loss gain
        "dfl": 1.5,             # Distribution Focal Loss gain
        "save": True,
        "save_period": -1,      # Only save best and last
        "plots": True,
        "verbose": True,
    }

    # Remove None values
    train_args = {k: v for k, v in train_args.items() if v is not None}

    print("\nStarting training loop...")
    results = model.train(**train_args)

    best_weights = Path(project) / name / "weights" / "best.pt"
    print(f"\n✅ Training complete! Best weights: {best_weights.resolve()}")

    # Run post-training validation to capture exact competition metrics
    print("\n" + "=" * 65)
    print("📊 Evaluating Validation Metrics (mAP@50:90, mAP@50, Macro Recall)...")
    print("=" * 65)

    if "rtdetr" in str(model_name).lower():
        best_model = RTDETR(str(best_weights))
    else:
        best_model = YOLO(str(best_weights))
    val_results = best_model.val(data=str(yaml_path.resolve()), imgsz=imgsz, batch=batch)

    # Extract metrics
    box_metrics = val_results.box
    map50_90 = box_metrics.map       # mAP@50:90 (Primary metric)
    map50 = box_metrics.map50        # mAP@50 (Tie-breaker 1)
    macro_recall = box_metrics.mr    # Macro Recall (Tie-breaker 2)
    macro_precision = box_metrics.mp # Macro Precision

    print("\n🎯 Competition Evaluation Summary:")
    print("-" * 50)
    print(f"  • Primary Metric (mAP@50:90): {map50_90:.4f}")
    print(f"  • Tie-Breaker 1  (mAP@50):    {map50:.4f}")
    print(f"  • Tie-Breaker 2  (Recall):    {macro_recall:.4f}")
    print(f"  • Macro Precision:            {macro_precision:.4f}")
    print("-" * 50)

    # Class-level breakdown
    if hasattr(box_metrics, "maps") and box_metrics.maps is not None:
        print("\n📈 Per-Class mAP@50:90 Breakdown:")
        names = best_model.names
        for cls_idx, cls_map in enumerate(box_metrics.maps):
            cls_name = names.get(cls_idx, f"Class {cls_idx}")
            print(f"  - {cls_name:<16}: {cls_map:.4f}")

    # Optional TTA validation evaluation
    try:
        print("\n⚡ Evaluating with Test-Time Augmentation (TTA)...")
        val_tta = best_model.val(data=str(yaml_path.resolve()), imgsz=imgsz, batch=batch, augment=True)
        print(f"  • TTA mAP@50:90: {val_tta.box.map:.4f} (Delta: {val_tta.box.map - map50_90:+.4f})")
        print(f"  • TTA mAP@50:    {val_tta.box.map50:.4f} (Delta: {val_tta.box.map50 - map50:+.4f})")
    except Exception as e:
        print(f"  Note: TTA evaluation skipped ({e})")

    # Optional ONNX export
    if export_onnx:
        print(f"\n📦 Exporting model to ONNX...")
        onnx_file = best_model.export(format="onnx", imgsz=imgsz, dynamic=True)
        print(f"✅ ONNX model exported: {onnx_file}")

    return {
        "best_weights": str(best_weights),
        "map50_90": float(map50_90),
        "map50": float(map50),
        "macro_recall": float(macro_recall),
    }


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for PCB Defect Detection")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Base model (e.g. yolo11x.pt, yolov8x.pt, yolo11l.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image resolution for training (default: 1024)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--device", type=str, default="", help="Device: '0', '0,1', 'cpu'")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads for data loader")
    parser.add_argument("--project", type=str, default="pcb_runs", help="Project directory")
    parser.add_argument("--name", type=str, default="pcb_detector", help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer: auto, AdamW, SGD")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--export-onnx", action="store_true", help="Export best weights to ONNX format")

    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        optimizer=args.optimizer,
        lr0=args.lr0,
        export_onnx=args.export_onnx,
    )


if __name__ == "__main__":
    main()
