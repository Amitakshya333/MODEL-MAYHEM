#!/usr/bin/env python3
"""
MODEL MAYHEM: PCB DEFECT DETECTION - Evaluation Script
Evaluates model checkpoints against competition criteria:
- Primary: mAP@50:90
- Tie-Breaker 1: mAP@50
- Tie-Breaker 2: Macro Recall
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO, RTDETR

CANONICAL_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def find_default_weights() -> str:
    """Locate best trained weights file from standard candidate paths."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path("weights/best.pt"),
        Path("best.pt"),
        Path("submissions/CodeSplinter/best.pt"),
        script_dir / "weights" / "best.pt",
        script_dir / "best.pt",
        script_dir / "submissions" / "CodeSplinter" / "best.pt",
        script_dir.parent / "weights" / "best.pt",
        script_dir.parent / "best.pt",
        script_dir.parent / "submissions" / "CodeSplinter" / "best.pt",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return "weights/best.pt"


def find_default_data_yaml() -> str:
    """Locate dataset configuration file from standard candidate paths."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path("dataset.yaml"),
        Path("data.yaml"),
        Path("data1.yaml"),
        Path("dataset/data.yaml"),
        script_dir / "dataset.yaml",
        script_dir / "data.yaml",
        script_dir / "dataset" / "data.yaml",
        script_dir.parent / "data.yaml",
        script_dir.parent / "dataset.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return "dataset.yaml"


def evaluate_model(
    weights: str = "",
    data_yaml: str = "",
    split: str = "val",
    imgsz: int = 1024,
    batch: int = 16,
    device: str = "",
    use_tta: bool = False,
    conf: float = 0.001,
    iou: float = 0.6,
):
    if not weights:
        weights = find_default_weights()
    if not data_yaml:
        data_yaml = find_default_data_yaml()

    print("=" * 65)
    print("🔍 PCB DEFECT DETECTION - MODEL EVALUATION")
    print("=" * 65)
    print(f"Weights:      {weights}")
    print(f"Data YAML:    {data_yaml}")
    print(f"Split:        {split}")
    print(f"Image Size:   {imgsz}")
    print(f"TTA Enabled:  {use_tta}")
    print("=" * 65)

    if not Path(weights).exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")

    import yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Check if requested split exists in dataset dir, otherwise switch gracefully
    dataset_root = Path(data_cfg.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = Path(data_yaml).parent / dataset_root

    split_dir_name = data_cfg.get(split) or f"{split}/images"
    split_dir = dataset_root / split_dir_name if not Path(split_dir_name).is_absolute() else Path(split_dir_name)

    if not split_dir.exists():
        for alt_split in ["val", "test", "valid", "train"]:
            alt_dir_name = data_cfg.get(alt_split) or f"{alt_split}/images"
            alt_dir = dataset_root / alt_dir_name if not Path(alt_dir_name).is_absolute() else Path(alt_dir_name)
            if alt_dir.exists():
                print(f"ℹ️ Requested split '{split}' not found at {split_dir}. Switching to available split '{alt_split}' at {alt_dir}.")
                split = "val" if alt_split in ("val", "valid") else alt_split
                break

    if str(weights).endswith(".pth"):
        import torch
        import yaml
        from torch.utils.data import DataLoader
        from train_torchvision import (
            load_torchvision_checkpoint,
            build_torchvision_detector,
            evaluate_torchvision_model,
            PCBDefectDataset,
            collate_fn,
        )

        with open(data_yaml, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)

        dataset_root = Path(data_cfg.get("path", "."))
        split_key = "val" if split in ("val", "valid") else ("train" if split == "train" else "test")
        split_rel = data_cfg.get(split_key, data_cfg.get("val", "valid/images"))
        val_img_dir = dataset_root / split_rel if not Path(split_rel).is_absolute() else Path(split_rel)

        dev = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        model_name, state_dict = load_torchvision_checkpoint(weights, device=dev)
        model = build_torchvision_detector(model_name=model_name, pretrained=False)
        model.load_state_dict(state_dict)
        model.to(dev)

        val_dataset = PCBDefectDataset(val_img_dir, img_size=imgsz)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        metrics = evaluate_torchvision_model(model, val_loader, device=dev)
        return metrics

    if "rtdetr" in str(weights).lower():
        model = RTDETR(weights)
    else:
        model = YOLO(weights)

    val_kwargs = {
        "data": data_yaml,
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
        "iou": iou,
        "augment": use_tta,
        "plots": True,
        "verbose": True,
    }
    if device:
        val_kwargs["device"] = device

    metrics = model.val(**val_kwargs)

    box = metrics.box
    map50_90 = box.map
    map50 = box.map50
    map75 = box.map75
    macro_recall = box.mr
    macro_precision = box.mp

    print("\n" + "=" * 65)
    print("🎯 OFFICIAL COMPETITION METRICS SUMMARY:")
    print("=" * 65)
    print(f"  🥇 Primary Metric (mAP@50:90): {map50_90:.4f}")
    print(f"  🥈 Tie-Breaker 1  (mAP@50):    {map50:.4f}")
    print(f"  🥉 Tie-Breaker 2  (Recall):    {macro_recall:.4f}")
    print(f"  •  mAP@75:                     {map75:.4f}")
    print(f"  •  Macro Precision:            {macro_precision:.4f}")
    print("-" * 65)

    if hasattr(box, "maps") and box.maps is not None:
        print("\n📈 Per-Class Breakdown:")
        print(f"{'Class ID':<10} | {'Class Name':<18} | {'mAP@50:90':<12}")
        print("-" * 46)
        names = model.names
        for i, val in enumerate(box.maps):
            cname = names.get(i, f"class_{i}")
            print(f"{i:<10} | {cname:<18} | {val:.4f}")
        print("-" * 46)

    return {
        "map50_90": float(map50_90),
        "map50": float(map50),
        "macro_recall": float(macro_recall),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO PCB Defect Detector")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pt). Default: auto-detects weights/best.pt.")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset.yaml. Default: auto-detects dataset.yaml.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Split to evaluate")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device: '0', 'cpu', etc.")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for validation")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold for validation")

    args = parser.parse_args()

    w_arg = args.weights if args.weights else find_default_weights()
    d_arg = args.data if args.data else find_default_data_yaml()

    evaluate_model(
        weights=w_arg,
        data_yaml=d_arg,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        use_tta=args.tta,
        conf=args.conf,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
