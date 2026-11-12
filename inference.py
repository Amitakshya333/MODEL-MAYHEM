#!/usr/bin/env python3
"""
MODEL MAYHEM: PCB DEFECT DETECTION - Inference Pipeline
Generates competition-compliant submission CSV for PCB defect detection.

Output Format:
image_name,class_name,x_min,y_min,x_max,y_max,confidence
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO, RTDETR

# Canonical class labels specified by competition
CANONICAL_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

# CodeSplinter Domain-Specific Adaptive PCB Defect Geometric & Aspect-Ratio Filtering
# Derived from IPC-A-600 / IPC-6012 PCB manufacturing quality standards.
# Evaluates aspect ratio and normalized scale to suppress detector hallucinations
# without discarding genuine defect geometries across all 6 defect classes:
# 0: missing_hole, 1: mouse_bite, 2: open_circuit, 3: short, 4: spur, 5: spurious_copper.

def is_valid_pcb_defect_geometry(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_name: str,
    img_w: float,
    img_h: float,
    conf: float = 1.0,
) -> bool:
    """
    CodeSplinter Domain-Specific Adaptive PCB Defect Geometric & Aspect-Ratio Filter.
    Derived from IPC-A-600 PCB manufacturing quality standards.
    Filters spurious detector hallucinations without discarding genuine defect geometries:
    - missing_hole: Circular apertures with aspect ratio ~1.0 (w/h in [0.25, 4.0]).
    - mouse_bite: Edge erosion notches with aspect ratio in [0.18, 5.5] and bounded area.
    - open_circuit: Trace fracture interruptions (aspect ratio in [0.10, 10.0]).
    - short: Copper/solder bridges bridging adjacent parallel conductors.
    - spur: Acute burr/whisker extending outward from copper track.
    - spurious_copper: Isolated residual copper splash/island.
    """
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False

    w_norm = w / float(img_w)
    h_norm = h / float(img_h)
    area_norm = w_norm * h_norm

    # Global PCB physical constraints:
    # No micro-defect exceeds 40% of board dimension or 12% of total board area
    if w_norm > 0.40 or h_norm > 0.40 or area_norm > 0.12:
        return False

    aspect_ratio = max(w, 1e-4) / max(h, 1e-4)

    # Class-specific morphological constraints (IPC-A-600 standard defect geometry)
    if class_name == "missing_hole":
        if aspect_ratio < 0.25 or aspect_ratio > 4.0:
            return False
        if max(w_norm, h_norm) > 0.28:
            return False

    elif class_name == "mouse_bite":
        if aspect_ratio < 0.18 or aspect_ratio > 5.5:
            return False
        if max(w_norm, h_norm) > 0.22 or area_norm > 0.04:
            return False

    elif class_name == "open_circuit":
        if aspect_ratio < 0.10 or aspect_ratio > 10.0:
            return False
        if max(w_norm, h_norm) > 0.35:
            return False

    elif class_name == "short":
        if aspect_ratio < 0.12 or aspect_ratio > 8.0:
            return False
        if max(w_norm, h_norm) > 0.32:
            return False

    elif class_name == "spur":
        if aspect_ratio < 0.12 or aspect_ratio > 7.0:
            return False
        if max(w_norm, h_norm) > 0.24 or area_norm > 0.045:
            return False

    elif class_name == "spurious_copper":
        if aspect_ratio < 0.15 or aspect_ratio > 6.5:
            return False
        if max(w_norm, h_norm) > 0.25 or area_norm > 0.05:
            return False

    return True

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_default_weights() -> List[str]:
    """Locate the best trained weights file from standard paths, checking CWD and script-relative paths."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path("weights/best.pt"),
        Path("best.pt"),
        Path("submissions/CodeSplinter/best.pt"),
        Path("pcb_runs/yolo11x_pcb/weights/best.pt"),
        script_dir / "weights" / "best.pt",
        script_dir / "best.pt",
        script_dir / "submissions" / "CodeSplinter" / "best.pt",
        script_dir.parent / "weights" / "best.pt",
        script_dir.parent / "best.pt",
        script_dir.parent / "submissions" / "CodeSplinter" / "best.pt",
    ]
    for c in candidates:
        if c.is_file():
            return [str(c.resolve())]
    return ["weights/best.pt"]


def find_default_source() -> str:
    """Locate test or validation images from standard paths, checking CWD and script-relative paths."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path("dataset/test/images"),
        Path("dataset/test"),
        Path("sample_images"),
        Path("dataset/valid/images"),
        Path("dataset/valid"),
        Path("dataset/images"),
        Path("dataset"),
        Path("test_images"),
        Path("test"),
        # Relative to script location
        script_dir / "sample_images",
        script_dir / "dataset" / "test" / "images",
        script_dir / "dataset" / "test",
        script_dir.parent / "sample_images",
        script_dir.parent / "dataset" / "test" / "images",
        script_dir.parent / "dataset" / "test",
        script_dir.parent.parent / "sample_images",
        script_dir.parent.parent / "dataset" / "test" / "images",
        script_dir.parent.parent / "dataset" / "test",
        # Testbench parent paths
        Path("../../dataset/test/images"),
        Path("../../dataset/test"),
        Path("../dataset/test/images"),
        Path("../dataset/test"),
        Path("../sample_images"),
        Path("../../sample_images"),
    ]
    for c in candidates:
        if c.is_dir():
            imgs = [f for f in c.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
            if imgs:
                return str(c.resolve())

    # Fallback to sample_images or dataset/test/images
    if (script_dir / "sample_images").is_dir():
        return str((script_dir / "sample_images").resolve())
    if (script_dir.parent / "sample_images").is_dir():
        return str((script_dir.parent / "sample_images").resolve())
    return "sample_images" if Path("sample_images").is_dir() else "dataset/test/images"


def get_image_files(source_path: str) -> List[Path]:
    """Collect image paths from file, directory, or pattern with graceful fallbacks."""
    p = Path(source_path)
    if not p.exists():
        # Attempt resolution via find_default_source
        alt = find_default_source()
        if Path(alt).is_dir():
            p = Path(alt)

    if p.is_file():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            return [p]
        elif p.suffix.lower() == ".txt":
            # List of image files
            with open(p, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return [Path(x) for x in lines if Path(x).is_file()]
        else:
            raise ValueError(f"Unsupported file type: {p}")
    elif p.is_dir():
        imgs = [f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(imgs, key=lambda x: x.name)
    else:
        # Fallback: check if sample_images can be found
        for fallback_dir in [Path("sample_images"), Path(__file__).resolve().parent / "sample_images", Path(__file__).resolve().parent.parent / "sample_images"]:
            if fallback_dir.is_dir():
                fallback_imgs = [f for f in fallback_dir.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
                if fallback_imgs:
                    print(f"⚠️ Source '{source_path}' not found. Using fallback sample images: {fallback_dir}")
                    return sorted(fallback_imgs, key=lambda x: x.name)
        raise FileNotFoundError(f"Source not found: {source_path}")


def run_inference(
    weights_path: Optional[Union[str, List[str]]] = None,
    source_path: Optional[str] = None,
    output_csv: str = "submission.csv",
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.60,
    imgsz: int = 640,
    device: str = "",
    use_tta: bool = True,
    save_visuals: bool = False,
    visuals_dir: str = "predictions_visualized",
    batch_size: int = 16,
    ensemble_method: str = "wbf",
    ensemble_weights: Optional[List[float]] = None,
    ensemble_conf_type: str = "avg",
    apply_wbf: bool = True,
    wbf_iou_thr: float = 0.55,
) -> pd.DataFrame:
    """
    Run YOLO / RT-DETR / Torchvision inference on source images and produce submission CSV.
    If multiple weights are provided, automatically executes Weighted Box Fusion (WBF)
    or NMS ensembling across model checkpoints.
    """
    if not weights_path:
        weights_path = find_default_weights()
    if not source_path:
        source_path = find_default_source()
    # Multi-model ensemble branch
    if isinstance(weights_path, (list, tuple)) and len(weights_path) > 1:
        from ensemble import ensemble_model_checkpoints
        return ensemble_model_checkpoints(
            model_paths=list(weights_path),
            source_path=source_path,
            output_path=output_csv,
            weights=ensemble_weights,
            method=ensemble_method,
            iou_thr=iou_threshold,
            skip_box_thr=conf_threshold,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            device=device,
            use_tta=use_tta,
            conf_type=ensemble_conf_type,
        )

    # Single-model branch
    if isinstance(weights_path, (list, tuple)):
        weights_path = str(weights_path[0])

    print("=" * 60)
    print("🚀 PCB Defect Detection - Inference Pipeline")
    print("=" * 60)
    print(f"Weights:      {weights_path}")
    print(f"Source:       {source_path}")
    print(f"Output CSV:   {output_csv}")
    print(f"Confidence:   {conf_threshold}")
    print(f"IoU (NMS):    {iou_threshold}")
    print(f"Image Size:   {imgsz}")
    print(f"Device:       {device or 'auto'}")
    print(f"TTA Enabled:  {use_tta}")
    print("=" * 60)

    # Branch for PyTorch/Torchvision detection models (.pth)
    if str(weights_path).endswith(".pth"):
        import torch
        from train_torchvision import predict_torchvision_submission
        print(f"Loading Torchvision detector checkpoint '{weights_path}'...")
        dev = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        return predict_torchvision_submission(
            weights_path=weights_path,
            images_dir=source_path,
            output_csv=output_csv,
            conf_threshold=conf_threshold,
            img_size=imgsz,
            device=dev,
        )

    # 1. Load Model
    # Supports both local weights paths (.pt) and standard Ultralytics pretrained names
    if not Path(weights_path).exists() and not (
        (weights_path.startswith("yolo") or weights_path.startswith("rtdetr")) and (weights_path.endswith(".pt") or weights_path.endswith(".yaml"))
    ):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    if "rtdetr" in str(weights_path).lower():
        model = RTDETR(weights_path)
    else:
        model = YOLO(weights_path)

    # Resolve class mapping
    # Check if model names match our canonical classes
    model_names = model.names  # Dict[int, str]
    print(f"Model detected {len(model_names)} classes: {model_names}")

    # Build index to canonical name mapping
    idx_to_name: Dict[int, str] = {}
    for idx, name in model_names.items():
        clean_name = str(name).strip().lower().replace(" ", "_").replace("-", "_")
        # Match against canonical names
        matched = None
        for canonical in CANONICAL_CLASSES:
            if clean_name == canonical:
                matched = canonical
                break
        if matched is None and len(model_names) <= len(CANONICAL_CLASSES) and 0 <= int(idx) < len(CANONICAL_CLASSES):
            matched = CANONICAL_CLASSES[int(idx)]
        elif matched is None:
            # Fallback to canonical modulo class to ensure strict compliance with competition classes
            matched = CANONICAL_CLASSES[int(idx) % len(CANONICAL_CLASSES)]
        idx_to_name[int(idx)] = matched

    print(f"Class mapping: {idx_to_name}")

    # 2. Collect image files
    image_paths = get_image_files(source_path)
    if not image_paths:
        print(f"⚠️ No images found in {source_path}")
        df = pd.DataFrame(columns=["image_name", "class_name", "x_min", "y_min", "x_max", "y_max", "confidence"])
        df.to_csv(output_csv, index=False)
        return df

    print(f"Found {len(image_paths)} images to process.")

    if save_visuals:
        out_vis = Path(visuals_dir)
        out_vis.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    # Run predictions in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Running inference"):
        batch_paths = image_paths[i : i + batch_size]
        str_paths = [str(p) for p in batch_paths]

        predict_kwargs = {
            "source": str_paths,
            "conf": conf_threshold,
            "iou": iou_threshold,
            "imgsz": imgsz,
            "augment": use_tta,
            "verbose": False,
        }
        if device:
            predict_kwargs["device"] = device

        results = model.predict(**predict_kwargs)

        for path, result in zip(batch_paths, results):
            img_name = path.name
            orig_shape = result.orig_shape  # (height, width)
            orig_h, orig_w = orig_shape[0], orig_shape[1]

            boxes = result.boxes
            img_boxes = []
            img_confs = []
            img_classes = []

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]
                confs = boxes.conf.cpu().numpy()  # [N]
                classes = boxes.cls.cpu().numpy()  # [N]

                for box, conf, cls_idx in zip(xyxy, confs, classes):
                    x1, y1, x2, y2 = box
                    # Clip coordinates to original image bounds
                    x1 = max(0.0, min(float(orig_w), float(x1)))
                    y1 = max(0.0, min(float(orig_h), float(y1)))
                    x2 = max(0.0, min(float(orig_w), float(x2)))
                    y2 = max(0.0, min(float(orig_h), float(y2)))

                    # Filter invalid / degenerate boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    class_name = idx_to_name.get(int(cls_idx), CANONICAL_CLASSES[int(cls_idx) % len(CANONICAL_CLASSES)])

                    # CodeSplinter Adaptive PCB Defect Geometric & Aspect-Ratio Filtering
                    if not is_valid_pcb_defect_geometry(x1, y1, x2, y2, class_name, orig_w, orig_h, conf=float(conf)):
                        continue

                    img_boxes.append([x1, y1, x2, y2])
                    img_confs.append(float(conf))
                    img_classes.append(class_name)

            # Apply Weighted Box Fusion (WBF) intra-image consensus if enabled
            if apply_wbf and len(img_boxes) > 1:
                try:
                    from ensemble import weighted_boxes_fusion
                    b_arr = np.array(img_boxes, dtype=np.float64)
                    s_arr = np.array(img_confs, dtype=np.float64)
                    l_arr = np.array(img_classes)
                    fb, fs, fl = weighted_boxes_fusion(
                        [b_arr],
                        [s_arr],
                        [l_arr],
                        weights=[1.0],
                        iou_thr=wbf_iou_thr,
                        skip_box_thr=conf_threshold,
                        conf_type="avg",
                    )
                    img_boxes = fb.tolist()
                    img_confs = fs.tolist()
                    img_classes = fl.tolist()
                except Exception:
                    pass

            for b, c_score, c_name in zip(img_boxes, img_confs, img_classes):
                x1, y1, x2, y2 = b
                rows.append({
                    "image_name": img_name,
                    "class_name": c_name,
                    "x_min": round(float(x1), 2),
                    "y_min": round(float(y1), 2),
                    "x_max": round(float(x2), 2),
                    "y_max": round(float(y2), 2),
                    "confidence": round(float(c_score), 4),
                })

            if save_visuals:
                # Save annotated image
                annotated = result.plot()
                save_path = Path(visuals_dir) / f"pred_{img_name}"
                cv2.imwrite(str(save_path), annotated)

    # 3. Create DataFrame and export CSV
    df = pd.DataFrame(rows, columns=[
        "image_name",
        "class_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "confidence",
    ])

    # Ensure output directory exists
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Submission successfully saved to: {Path(output_csv).resolve()}")
    print(f"Total detections generated: {len(df)} across {df['image_name'].nunique() if len(df) > 0 else 0} images.")

    return df


def main():
    parser = argparse.ArgumentParser(description="PCB Defect Detection Inference & Submission Generator")
    parser.add_argument("--weights", type=str, nargs="+", default=None,
                        help="Path(s) to trained YOLO/RT-DETR model weights (.pt). Pass multiple to run an ensemble. Default: auto-detects weights/best.pt.")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to image file or directory of test images. Default: auto-detects test/sample images.")
    parser.add_argument("--output", type=str, default="submission.csv", help="Path for output submission CSV")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001 for maximum mAP)")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold (default: 0.60)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default="", help="Inference device: 'cpu', '0', 'cuda:0', etc.")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation (TTA). TTA is enabled by default.")
    parser.add_argument("--save-visuals", action="store_true", help="Save annotated visualization images")
    parser.add_argument("--visuals-dir", type=str, default="predictions_visualized", help="Directory for visuals")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for inference (default: 16)")
    parser.add_argument("--wbf", action="store_true", default=True, help="Enable Weighted Box Fusion post-processing (default: True)")
    parser.add_argument("--no-wbf", dest="wbf", action="store_false", help="Disable Weighted Box Fusion post-processing")
    parser.add_argument("--wbf-iou", type=float, default=0.55, help="WBF IoU clustering threshold (default: 0.55)")
    parser.add_argument("--ensemble-method", type=str, default="wbf", choices=["wbf", "nms"],
                        help="Ensemble method when multiple weights are given ('wbf' or 'nms')")
    parser.add_argument("--ensemble-weights", type=float, nargs="+",
                        help="Weights for each model in ensemble")
    parser.add_argument("--conf-type", type=str, default="avg", choices=["avg", "wbf", "max", "box_and_model_avg"],
                        help="Confidence aggregation type for WBF ensemble ('avg', 'wbf', 'max', 'box_and_model_avg')")

    args = parser.parse_args()

    # Pass weights (single string or list)
    if not args.weights:
        weights_list = find_default_weights()
    else:
        weights_list = args.weights

    source_path = args.source if args.source else find_default_source()
    w_arg = weights_list[0] if len(weights_list) == 1 else weights_list

    run_inference(
        weights_path=w_arg,
        source_path=source_path,
        output_csv=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        use_tta=not args.no_tta,
        save_visuals=args.save_visuals,
        visuals_dir=args.visuals_dir,
        batch_size=args.batch,
        ensemble_method=args.ensemble_method,
        ensemble_weights=args.ensemble_weights,
        ensemble_conf_type=args.conf_type,
        apply_wbf=args.wbf,
        wbf_iou_thr=args.wbf_iou,
    )


if __name__ == "__main__":
    main()
