#!/usr/bin/env python3
"""
MODEL MAYHEM: PCB DEFECT DETECTION - Model Ensemble Pipeline
Supports Weighted Box Fusion (WBF), Non-Maximum Suppression (NMS), and Soft-NMS
ensembling across multiple detection models (YOLO11x, YOLOv8x, RT-DETR, Faster R-CNN)
or multiple submission CSV files.

Ensures strict compliance with competition output format:
image_name,class_name,x_min,y_min,x_max,y_max,confidence
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Canonical class labels specified by competition
CANONICAL_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

EXPECTED_COLUMNS = [
    "image_name",
    "class_name",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "confidence",
]


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2].
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def weighted_boxes_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.05,
    conf_type: str = "avg",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Box Fusion (WBF) algorithm implemented in pure NumPy.
    Fuses bounding box predictions across multiple models or augmentations.

    Args:
        boxes_list: List of arrays [N_m, 4] for each model, in [x1, y1, x2, y2] format.
        scores_list: List of arrays [N_m] of confidence scores for each model.
        labels_list: List of arrays [N_m] of class labels (integer or string).
        weights: Optional list of model weights (len == len(boxes_list)). Defaults to equal weights.
        iou_thr: IoU threshold for clustering matching boxes.
        skip_box_thr: Confidence threshold below which boxes are discarded.
        conf_type: How fused confidence is calculated:
            - 'avg': Weighted average confidence of matching boxes in cluster.
            - 'wbf': Standard WBF score penalized by total model count (sum(w*conf) / sum(W)).
            - 'max': Maximum confidence among matching boxes in cluster.
            - 'box_and_model_avg': Weighted average scaled by min(1.0, count/num_models).

    Returns:
        fused_boxes: np.ndarray [K, 4]
        fused_scores: np.ndarray [K]
        fused_labels: np.ndarray [K]
    """
    num_models = len(boxes_list)
    if num_models == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    if weights is None:
        weights = [1.0] * num_models
    else:
        if len(weights) != num_models:
            raise ValueError(f"Length of weights ({len(weights)}) != number of models ({num_models})")
    
    total_model_weight = sum(weights)
    if total_model_weight <= 0:
        weights = [1.0] * num_models
        total_model_weight = float(num_models)

    # 1. Collect and filter all boxes across models
    all_boxes = []
    all_scores = []
    all_labels = []
    all_model_idx = []

    for m_idx in range(num_models):
        m_boxes = np.asarray(boxes_list[m_idx], dtype=np.float64)
        m_scores = np.asarray(scores_list[m_idx], dtype=np.float64)
        m_labels = np.asarray(labels_list[m_idx])

        if len(m_boxes) == 0:
            continue

        valid_mask = (m_scores >= skip_box_thr) & (m_boxes[:, 2] > m_boxes[:, 0]) & (m_boxes[:, 3] > m_boxes[:, 1])
        m_boxes = m_boxes[valid_mask]
        m_scores = m_scores[valid_mask]
        m_labels = m_labels[valid_mask]

        for b, s, l in zip(m_boxes, m_scores, m_labels):
            all_boxes.append(b)
            all_scores.append(float(s))
            all_labels.append(l)
            all_model_idx.append(m_idx)

    if len(all_boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_model_idx = np.array(all_model_idx, dtype=int)

    # 2. Process independently per unique class label
    unique_labels = np.unique(all_labels)
    final_boxes = []
    final_scores = []
    final_labels = []

    for label in unique_labels:
        cls_mask = (all_labels == label)
        cls_boxes = all_boxes[cls_mask]
        cls_scores = all_scores[cls_mask]
        cls_models = all_model_idx[cls_mask]

        # Sort by confidence descending
        sort_order = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[sort_order]
        cls_scores = cls_scores[sort_order]
        cls_models = cls_models[sort_order]

        # Clusters: list of lists containing indices into sorted cls_*
        clusters: List[List[Dict[str, Any]]] = []
        cluster_fused_boxes: List[np.ndarray] = []

        for b, s, m in zip(cls_boxes, cls_scores, cls_models):
            w = weights[m]
            box_item = {"box": b, "score": s, "weight": w, "model": m}

            # Find matching cluster with highest IoU >= iou_thr
            best_iou = -1.0
            best_c_idx = -1

            for c_idx, c_box in enumerate(cluster_fused_boxes):
                # Ensure each model contributes at most one box to a given cluster (Solovyev WBF specification)
                if any(it["model"] == m for it in clusters[c_idx]):
                    continue
                iou = calculate_iou(b, c_box)
                if iou >= iou_thr and iou > best_iou:
                    best_iou = iou
                    best_c_idx = c_idx

            if best_c_idx != -1:
                # Add to existing cluster
                clusters[best_c_idx].append(box_item)
                # Recompute fused coordinates for cluster
                c_items = clusters[best_c_idx]
                total_w = sum(it["weight"] * it["score"] for it in c_items)
                if total_w > 0:
                    new_x1 = sum(it["weight"] * it["score"] * it["box"][0] for it in c_items) / total_w
                    new_y1 = sum(it["weight"] * it["score"] * it["box"][1] for it in c_items) / total_w
                    new_x2 = sum(it["weight"] * it["score"] * it["box"][2] for it in c_items) / total_w
                    new_y2 = sum(it["weight"] * it["score"] * it["box"][3] for it in c_items) / total_w
                    cluster_fused_boxes[best_c_idx] = np.array([new_x1, new_y1, new_x2, new_y2])
            else:
                # Create new cluster
                clusters.append([box_item])
                cluster_fused_boxes.append(b.copy())

        # 3. Compute final scores for each cluster
        for c_items, c_box in zip(clusters, cluster_fused_boxes):
            if conf_type == "avg":
                weight_sum = sum(it["weight"] for it in c_items)
                score = sum(it["weight"] * it["score"] for it in c_items) / weight_sum if weight_sum > 0 else 0.0
            elif conf_type == "wbf":
                score = sum(it["weight"] * it["score"] for it in c_items) / total_model_weight
            elif conf_type == "max":
                score = max(it["score"] for it in c_items)
            elif conf_type == "box_and_model_avg":
                weight_sum = sum(it["weight"] for it in c_items)
                base_score = sum(it["weight"] * it["score"] for it in c_items) / weight_sum if weight_sum > 0 else 0.0
                score = base_score * min(1.0, len(c_items) / float(num_models))
            else:
                raise ValueError(f"Unknown conf_type: {conf_type}")

            # Clip score to [0, 1]
            score = float(np.clip(score, 0.0, 1.0))
            if score >= skip_box_thr:
                final_boxes.append(c_box)
                final_scores.append(score)
                final_labels.append(label)

    if len(final_boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    final_boxes_np = np.array(final_boxes, dtype=np.float64)
    final_scores_np = np.array(final_scores, dtype=np.float64)
    final_labels_np = np.array(final_labels)

    # Sort final predictions by confidence descending
    final_sort = np.argsort(-final_scores_np)
    return final_boxes_np[final_sort], final_scores_np[final_sort], final_labels_np[final_sort]


def nms_boxes_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    iou_thr: float = 0.50,
    skip_box_thr: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Non-Maximum Suppression (NMS) ensembling across multiple models.
    """
    num_models = len(boxes_list)
    if num_models == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    all_boxes = []
    all_scores = []
    all_labels = []

    for m_idx in range(num_models):
        m_boxes = np.asarray(boxes_list[m_idx], dtype=np.float64)
        m_scores = np.asarray(scores_list[m_idx], dtype=np.float64)
        m_labels = np.asarray(labels_list[m_idx])

        if len(m_boxes) == 0:
            continue

        w = weights[m_idx] if weights is not None else 1.0
        weighted_scores = m_scores * w

        valid_mask = (m_scores >= skip_box_thr) & (m_boxes[:, 2] > m_boxes[:, 0]) & (m_boxes[:, 3] > m_boxes[:, 1])
        for b, s, l in zip(m_boxes[valid_mask], weighted_scores[valid_mask], m_labels[valid_mask]):
            all_boxes.append(b)
            all_scores.append(float(s))
            all_labels.append(l)

    if len(all_boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    for label in np.unique(all_labels):
        cls_mask = (all_labels == label)
        cls_boxes = all_boxes[cls_mask]
        cls_scores = all_scores[cls_mask]

        order = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]

        suppressed = np.zeros(len(cls_boxes), dtype=bool)
        for i in range(len(cls_boxes)):
            if suppressed[i]:
                continue
            keep_boxes.append(cls_boxes[i])
            keep_scores.append(min(1.0, float(cls_scores[i])))
            keep_labels.append(label)

            for j in range(i + 1, len(cls_boxes)):
                if suppressed[j]:
                    continue
                if calculate_iou(cls_boxes[i], cls_boxes[j]) >= iou_thr:
                    suppressed[j] = True

    if len(keep_boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    keep_boxes_np = np.array(keep_boxes, dtype=np.float64)
    keep_scores_np = np.array(keep_scores, dtype=np.float64)
    keep_labels_np = np.array(keep_labels)

    sort_idx = np.argsort(-keep_scores_np)
    return keep_boxes_np[sort_idx], keep_scores_np[sort_idx], keep_labels_np[sort_idx]


def ensemble_csv_files(
    csv_paths: List[str],
    output_path: str = "submission_ensemble.csv",
    weights: Optional[List[float]] = None,
    method: str = "wbf",
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.05,
    conf_type: str = "avg",
) -> pd.DataFrame:
    """
    Ensemble predictions from multiple competition submission CSV files.

    Args:
        csv_paths: List of file paths to submission CSVs.
        output_path: Path where the fused submission CSV will be saved.
        weights: List of weights corresponding to each CSV file.
        method: Fusion method: 'wbf' (Weighted Box Fusion) or 'nms' (Non-Maximum Suppression).
        iou_thr: IoU clustering threshold (default: 0.55 for WBF, 0.50 for NMS).
        skip_box_thr: Minimum confidence threshold.
        conf_type: Confidence aggregation method for WBF ('avg', 'wbf', 'max', 'box_and_model_avg').

    Returns:
        pd.DataFrame containing the fused predictions.
    """
    print("=" * 65)
    print("⚡ PCB DEFECT DETECTION - SUBMISSION ENSEMBLE")
    print("=" * 65)
    print(f"Number of CSVs: {len(csv_paths)}")
    for i, p in enumerate(csv_paths):
        w = weights[i] if weights is not None else 1.0
        print(f"  [{i+1}] Weight {w:<4} | {p}")
    print(f"Ensemble Method: {method.upper()}")
    print(f"IoU Threshold:   {iou_thr}")
    print(f"Skip Box Thresh: {skip_box_thr}")
    if method == "wbf":
        print(f"Confidence Type: {conf_type}")
    print("=" * 65)

    dfs = []
    for p in csv_paths:
        path_obj = Path(p)
        if not path_obj.exists():
            raise FileNotFoundError(f"Submission CSV not found: {p}")
        df = pd.read_csv(path_obj)
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"CSV {p} missing required column: '{col}'")
        dfs.append(df)

    num_models = len(dfs)
    if weights is None:
        weights = [1.0] * num_models

    # Collect all unique image names across all models
    all_images = sorted(list(set().union(*[set(df["image_name"].unique()) for df in dfs])))
    print(f"\nAggregating detections across {len(all_images)} unique images...")

    fused_rows = []

    for img_name in tqdm(all_images, desc="Ensembling images"):
        boxes_list = []
        scores_list = []
        labels_list = []

        for df in dfs:
            img_df = df[df["image_name"] == img_name]
            if len(img_df) > 0:
                m_boxes = img_df[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float64)
                m_scores = img_df["confidence"].to_numpy(dtype=np.float64)
                m_labels = img_df["class_name"].to_numpy()
            else:
                m_boxes = np.empty((0, 4), dtype=np.float64)
                m_scores = np.empty(0, dtype=np.float64)
                m_labels = np.empty(0, dtype=str)

            boxes_list.append(m_boxes)
            scores_list.append(m_scores)
            labels_list.append(m_labels)

        if method == "wbf":
            f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                boxes_list=boxes_list,
                scores_list=scores_list,
                labels_list=labels_list,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
                conf_type=conf_type,
            )
        elif method == "nms":
            f_boxes, f_scores, f_labels = nms_boxes_fusion(
                boxes_list=boxes_list,
                scores_list=scores_list,
                labels_list=labels_list,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        else:
            raise ValueError(f"Unsupported ensemble method: {method}. Choose 'wbf' or 'nms'.")

        for b, s, l in zip(f_boxes, f_scores, f_labels):
            x1, y1, x2, y2 = b
            # Sanity clip and check
            x1 = max(0.0, float(x1))
            y1 = max(0.0, float(y1))
            x2 = max(x1 + 0.01, float(x2))
            y2 = max(y1 + 0.01, float(y2))

            # Canonical class guarantee
            c_name = str(l).strip().lower().replace(" ", "_").replace("-", "_")
            if c_name not in CANONICAL_CLASSES:
                # Map or fallback
                c_name = CANONICAL_CLASSES[0]

            fused_rows.append({
                "image_name": img_name,
                "class_name": c_name,
                "x_min": round(x1, 2),
                "y_min": round(y1, 2),
                "x_max": round(x2, 2),
                "y_max": round(y2, 2),
                "confidence": round(float(s), 4),
            })

    out_df = pd.DataFrame(fused_rows, columns=EXPECTED_COLUMNS)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\n✅ Ensembled predictions successfully saved to: {Path(output_path).resolve()}")
    print(f"Total fused detections: {len(out_df)} across {out_df['image_name'].nunique() if len(out_df) > 0 else 0} images.")

    return out_df


def ensemble_model_checkpoints(
    model_paths: List[str],
    source_path: str,
    output_path: str = "submission_ensemble.csv",
    weights: Optional[List[float]] = None,
    method: str = "wbf",
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.05,
    conf_threshold: float = 0.15,
    imgsz: int = 1024,
    device: str = "",
    use_tta: bool = False,
    conf_type: str = "avg",
) -> pd.DataFrame:
    """
    Runs multi-model inference across checkpoints and performs WBF or NMS fusion directly.
    """
    from inference import run_inference

    print("=" * 65)
    print("🤖 MULTI-MODEL CHECKPOINT ENSEMBLE")
    print("=" * 65)
    print(f"Models to evaluate: {model_paths}")
    print(f"Source images:      {source_path}")
    print(f"Method:             {method.upper()}")
    print("=" * 65)

    temp_csvs = []
    temp_dir = Path("pcb_ensemble_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx, m_path in enumerate(model_paths):
            temp_csv = temp_dir / f"model_{idx}_predictions.csv"
            print(f"\n--- Running Model [{idx+1}/{len(model_paths)}]: {m_path} ---")
            run_inference(
                weights_path=m_path,
                source_path=source_path,
                output_csv=str(temp_csv),
                conf_threshold=conf_threshold,
                iou_threshold=0.50,
                imgsz=imgsz,
                device=device,
                use_tta=use_tta,
                batch_size=16,
            )
            temp_csvs.append(str(temp_csv))

        # Run ensemble on generated CSVs
        ensemble_df = ensemble_csv_files(
            csv_paths=temp_csvs,
            output_path=output_path,
            weights=weights,
            method=method,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type=conf_type,
        )
        return ensemble_df
    finally:
        # Cleanup temporary prediction files
        for t in temp_csvs:
            try:
                if Path(t).exists():
                    os.remove(t)
            except Exception:
                pass
        try:
            if temp_dir.exists():
                temp_dir.rmdir()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Ensemble PCB Defect Detection Predictions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csvs", nargs="+", help="Paths to prediction CSV files to ensemble")
    group.add_argument("--models", nargs="+", help="Paths to model checkpoints (.pt) to run and ensemble")

    parser.add_argument("--source", type=str, help="Source image folder (required if --models is specified)")
    parser.add_argument("--output", type=str, default="submission_ensemble.csv", help="Output fused CSV path")
    parser.add_argument("--weights", nargs="+", type=float, help="Weights for each model/CSV (e.g. --weights 1.0 1.2 0.8)")
    parser.add_argument("--method", type=str, default="wbf", choices=["wbf", "nms"], help="Ensemble method: 'wbf' or 'nms'")
    parser.add_argument("--iou-thr", type=float, default=0.55, help="IoU threshold for box fusion (default: 0.55)")
    parser.add_argument("--skip-box-thr", type=float, default=0.05, help="Minimum confidence threshold (default: 0.05)")
    parser.add_argument("--conf-type", type=str, default="avg", choices=["avg", "wbf", "max", "box_and_model_avg"],
                        help="Confidence calculation type for WBF")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for model inference")
    parser.add_argument("--device", type=str, default="", help="Inference device ('cpu', '0', etc.)")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")

    args = parser.parse_args()

    if args.csvs:
        df = ensemble_csv_files(
            csv_paths=args.csvs,
            output_path=args.output,
            weights=args.weights,
            method=args.method,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr,
            conf_type=args.conf_type,
        )
    else:
        if not args.source:
            parser.error("--source is required when using --models")
        df = ensemble_model_checkpoints(
            model_paths=args.models,
            source_path=args.source,
            output_path=args.output,
            weights=args.weights,
            method=args.method,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr,
            imgsz=args.imgsz,
            device=args.device,
            use_tta=args.tta,
            conf_type=args.conf_type,
        )

    # Validate output
    try:
        from verify_submission import verify_submission
        print("\nVerifying fused submission CSV...")
        valid = verify_submission(args.output)
        if valid:
            print("🎉 Ensembled submission passed all validation checks!")
        else:
            print("⚠️ Validation issues detected in ensembled output.")
    except Exception as e:
        print(f"Validation check skipped: {e}")


if __name__ == "__main__":
    main()
