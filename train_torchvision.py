#!/usr/bin/env python3
"""
MODEL MAYHEM: PCB DEFECT DETECTION - Torchvision Multi-Model Training Pipeline
Supports training two-stage (Faster R-CNN) and one-stage (RetinaNet) detectors with
classic deep learning backbones:
- ResNet-50 V2 (fasterrcnn_resnet50_fpn_v2 / retinanet_resnet50_fpn_v2)
- MobileNet V3 / V2 (fasterrcnn_mobilenet_v3_large_fpn)

Allows direct training on dataset.yaml (YOLO format), validation on mAP@50:90,
and exporting competition-compliant submission CSVs.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    retinanet_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    FasterRCNN,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    densenet121, DenseNet121_Weights,
)
import yaml
from tqdm import tqdm

CANONICAL_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

NUM_DEFECT_CLASSES = len(CANONICAL_CLASSES)
# Note: Torchvision detectors reserve class 0 for background.
TOTAL_CLASSES = NUM_DEFECT_CLASSES + 1


class PCBDefectDataset(Dataset):
    """
    PyTorch Dataset loading images and YOLO-formatted bounding box annotations.
    """
    def __init__(self, images_dir: Union[str, Path], labels_dir: Optional[Union[str, Path]] = None, img_size: int = 1024):
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.image_files = sorted([f for f in self.images_dir.iterdir() if f.suffix.lower() in valid_exts])

        # Automatically locate labels directory if not specified
        if labels_dir is None:
            # Common YOLO layout: images/ -> labels/
            candidate = self.images_dir.parent / "labels"
            if candidate.exists():
                self.labels_dir = candidate
            else:
                self.labels_dir = self.images_dir
        else:
            self.labels_dir = Path(labels_dir)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_files[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise IOError(f"Could not read image: {img_path}")

        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize if specified
        if self.img_size and (orig_w != self.img_size or orig_h != self.img_size):
            resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            cur_w, cur_h = self.img_size, self.img_size
        else:
            resized = img_rgb
            scale_x = 1.0
            scale_y = 1.0
            cur_w, cur_h = orig_w, orig_h

        # Convert image to tensor [3, H, W] in [0, 1]
        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        # Read YOLO label file: <class_id> <x_center> <y_center> <width> <height>
        boxes = []
        labels = []
        label_file = self.labels_dir / f"{img_path.stem}.txt"

        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1]) * orig_w * scale_x
                        y_center = float(parts[2]) * orig_h * scale_y
                        box_w = float(parts[3]) * orig_w * scale_x
                        box_h = float(parts[4]) * orig_h * scale_y

                        x1 = max(0.0, x_center - box_w / 2.0)
                        y1 = max(0.0, y_center - box_h / 2.0)
                        x2 = min(float(cur_w), x_center + box_w / 2.0)
                        y2 = min(float(cur_h), y_center + box_h / 2.0)

                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            # In torchvision: label 0 is background, defect labels are 1..6
                            labels.append((cls_id % NUM_DEFECT_CLASSES) + 1)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([orig_h, orig_w]),
            "file_name": img_path.name,
        }

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


class CustomBackboneWithFPN(nn.Module):
    """
    Wraps PyTorch classification backbones (MobileNetV2, EfficientNet, DenseNet)
    with a Feature Pyramid Network (FPN) and an extra maxpool block for Faster R-CNN detection.
    """
    def __init__(self, body: nn.Module, return_nodes: Dict[str, str], in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.body = create_feature_extractor(body, return_nodes=return_nodes)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.fpn(self.body(x))


def load_torchvision_checkpoint(
    weights_path: Union[str, Path],
    device: torch.device = torch.device("cpu"),
) -> Tuple[str, Dict[str, Any]]:
    """
    Loads a Torchvision model checkpoint, handling both state_dict dictionaries
    with metadata and raw state_dicts.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint weights file not found: {weights_path}")

    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model_name = ckpt.get("model_name", "fasterrcnn_resnet50_v2")
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any("roi_heads" in k or "backbone" in k for k in ckpt.keys()):
        state_dict = ckpt
        fn = path.name.lower()
        if "mobilenet_v2" in fn:
            model_name = "fasterrcnn_mobilenet_v2"
        elif "mobilenet" in fn:
            model_name = "fasterrcnn_mobilenet_v3"
        elif "efficientnet" in fn:
            model_name = "fasterrcnn_efficientnet_b0"
        elif "densenet" in fn:
            model_name = "fasterrcnn_densenet121"
        elif "retinanet" in fn:
            model_name = "retinanet_resnet50_v2"
        else:
            model_name = "fasterrcnn_resnet50_v2"
    else:
        raise ValueError(f"Unrecognized checkpoint format in: {weights_path}")

    return model_name, state_dict


def build_torchvision_detector(
    model_name: str = "fasterrcnn_resnet50_v2",
    num_classes: int = TOTAL_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Builds torchvision object detection model with specified backbone.
    Supports 6 architectures:
      - fasterrcnn_resnet50_v2
      - fasterrcnn_mobilenet_v2
      - fasterrcnn_mobilenet_v3
      - fasterrcnn_efficientnet_b0
      - fasterrcnn_densenet121
      - retinanet_resnet50_v2
    """
    name = model_name.lower().replace("-", "_")

    if "retinanet" in name:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        return retinanet_resnet50_fpn_v2(weights=weights, weights_backbone=None, num_classes=num_classes)

    elif "resnet50" in name:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    elif "mobilenet_v3" in name:
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    elif "mobilenet_v2" in name or "mobilenet" in name:
        w = MobileNet_V2_Weights.DEFAULT if pretrained else None
        mb = mobilenet_v2(weights=w).features
        backbone = CustomBackboneWithFPN(mb, {"6": "0", "13": "1", "18": "2"}, [32, 96, 1280])
        ag = AnchorGenerator(sizes=((32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 4)
        roi = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "pool"], output_size=7, sampling_ratio=2)
        return FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=ag, box_roi_pool=roi)

    elif "efficientnet" in name:
        w = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        eb = efficientnet_b0(weights=w).features
        backbone = CustomBackboneWithFPN(eb, {"3": "0", "5": "1", "7": "2"}, [40, 112, 320])
        ag = AnchorGenerator(sizes=((32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 4)
        roi = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "pool"], output_size=7, sampling_ratio=2)
        return FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=ag, box_roi_pool=roi)

    elif "densenet" in name:
        w = DenseNet121_Weights.DEFAULT if pretrained else None
        dn = densenet121(weights=w).features
        backbone = CustomBackboneWithFPN(dn, {"denseblock1": "0", "denseblock2": "1", "denseblock3": "2"}, [256, 512, 1024])
        ag = AnchorGenerator(sizes=((32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 4)
        roi = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "pool"], output_size=7, sampling_ratio=2)
        return FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=ag, box_roi_pool=roi)

    else:
        raise ValueError(
            f"Unsupported model_name: '{model_name}'. "
            f"Choose from: 'fasterrcnn_resnet50_v2', 'fasterrcnn_mobilenet_v2', 'fasterrcnn_mobilenet_v3', "
            f"'fasterrcnn_efficientnet_b0', 'fasterrcnn_densenet121', 'retinanet_resnet50_v2'"
        )


def compute_voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """11-point or continuous VOC/COCO AP calculation."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def evaluate_torchvision_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Evaluates detector on validation set and computes mAP@50:90, mAP@50, and Macro Recall.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(x, 2) for x in np.arange(0.50, 0.95, 0.05)]

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating validation set"):
            images = [img.to(device) for img in images]
            preds = model(images)
            for p, t in zip(preds, targets):
                all_predictions.append({
                    "boxes": p["boxes"].cpu().numpy(),
                    "scores": p["scores"].cpu().numpy(),
                    "labels": p["labels"].cpu().numpy(),
                })
                all_targets.append({
                    "boxes": t["boxes"].cpu().numpy(),
                    "labels": t["labels"].cpu().numpy(),
                })

    from ensemble import calculate_iou

    # Compute AP across IoU thresholds and classes
    ap_per_iou = {iou_thr: [] for iou_thr in iou_thresholds}
    recall_per_class = {c: [] for c in range(1, TOTAL_CLASSES)}

    for iou_thr in iou_thresholds:
        for c in range(1, TOTAL_CLASSES):
            # Gather all detections and ground truths for class c
            n_positives = 0
            scores = []
            matches = []

            for pred, gt in zip(all_predictions, all_targets):
                gt_c_mask = (gt["labels"] == c)
                gt_boxes = gt["boxes"][gt_c_mask]
                n_positives += len(gt_boxes)

                pred_c_mask = (pred["labels"] == c)
                p_boxes = pred["boxes"][pred_c_mask]
                p_scores = pred["scores"][pred_c_mask]

                gt_matched = [False] * len(gt_boxes)
                # Sort detections by score
                p_order = np.argsort(-p_scores)
                p_boxes = p_boxes[p_order]
                p_scores = p_scores[p_order]

                for pb, ps in zip(p_boxes, p_scores):
                    scores.append(ps)
                    best_iou = -1.0
                    best_gt_idx = -1
                    for g_idx, gb in enumerate(gt_boxes):
                        iou = calculate_iou(pb, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx
                    if best_iou >= iou_thr and not gt_matched[best_gt_idx]:
                        matches.append(1)
                        gt_matched[best_gt_idx] = True
                    else:
                        matches.append(0)

            if n_positives == 0:
                continue

            if len(scores) == 0:
                ap_per_iou[iou_thr].append(0.0)
                if iou_thr == 0.50:
                    recall_per_class[c].append(0.0)
                continue

            order = np.argsort(-np.array(scores))
            sorted_matches = np.array(matches)[order]
            tp = np.cumsum(sorted_matches)
            fp = np.cumsum(1 - sorted_matches)
            rec = tp / float(n_positives)
            prec = tp / np.maximum(tp + fp, 1e-8)
            ap = compute_voc_ap(rec, prec)
            ap_per_iou[iou_thr].append(ap)
            if iou_thr == 0.50:
                recall_per_class[c].append(float(rec[-1]) if len(rec) > 0 else 0.0)

    # Average AP across classes and IoU thresholds
    map_at_each_iou = [float(np.mean(aps)) if len(aps) > 0 else 0.0 for aps in ap_per_iou.values()]
    map50_90 = float(np.mean(map_at_each_iou)) if len(map_at_each_iou) > 0 else 0.0
    map50 = float(np.mean(ap_per_iou[0.50])) if len(ap_per_iou[0.50]) > 0 else 0.0
    recalls = [np.mean(r) for r in recall_per_class.values() if len(r) > 0]
    macro_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0

    print("\n" + "=" * 60)
    print("🎯 TORCHVISION EVALUATION SUMMARY:")
    print("=" * 60)
    print(f"  • Primary Metric (mAP@50:90): {map50_90:.4f}")
    print(f"  • Tie-Breaker 1  (mAP@50):    {map50:.4f}")
    print(f"  • Tie-Breaker 2  (Recall):    {macro_recall:.4f}")
    print("=" * 60)

    return {
        "map50_90": map50_90,
        "map50": map50,
        "macro_recall": macro_recall,
    }


def predict_torchvision_submission(
    model: Optional[nn.Module] = None,
    weights_path: Optional[Union[str, Path]] = None,
    images_dir: Union[str, Path] = "dataset/test/images",
    output_csv: str = "submission_torchvision.csv",
    conf_threshold: float = 0.25,
    img_size: int = 1024,
    device: Optional[torch.device] = None,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Runs inference on test images and produces competition-compliant CSV.
    Can be called with an instantiated model or directly with a weights_path (.pth).
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model is None:
        if weights_path is None:
            raise ValueError("Either 'model' or 'weights_path' must be provided to predict_torchvision_submission.")
        detected_name, state_dict = load_torchvision_checkpoint(weights_path, device=device)
        m_name = model_name or detected_name
        model = build_torchvision_detector(model_name=m_name, pretrained=False)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    p = Path(images_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if p.is_file():
        image_paths = [p]
    elif p.is_dir():
        image_paths = sorted([f for f in p.iterdir() if f.suffix.lower() in valid_exts])
    else:
        image_paths = []

    rows = []
    print(f"Predicting on {len(image_paths)} test images...")

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Torchvision Inference"):
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                continue
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            scale_x = img_size / orig_w
            scale_y = img_size / orig_h
            resized = cv2.resize(img_rgb, (img_size, img_size))
            img_t = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img_t = img_t.to(device)

            pred = model(img_t)[0]
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            for b, s, l in zip(boxes, scores, labels):
                if s < conf_threshold:
                    continue
                # Unscale back to original image coordinates
                x1 = max(0.0, float(b[0]) / scale_x)
                y1 = max(0.0, float(b[1]) / scale_y)
                x2 = min(float(orig_w), float(b[2]) / scale_x)
                y2 = min(float(orig_h), float(b[3]) / scale_y)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Map label 1..6 back to canonical classes 0..5
                cls_idx = (int(l) - 1) % NUM_DEFECT_CLASSES
                cls_name = CANONICAL_CLASSES[cls_idx]

                rows.append({
                    "image_name": path.name,
                    "class_name": cls_name,
                    "x_min": round(x1, 2),
                    "y_min": round(y1, 2),
                    "x_max": round(x2, 2),
                    "y_max": round(y2, 2),
                    "confidence": round(float(s), 4),
                })

    df = pd.DataFrame(rows, columns=[
        "image_name", "class_name", "x_min", "y_min", "x_max", "y_max", "confidence"
    ])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved predictions to: {Path(output_csv).resolve()} ({len(df)} detections)")
    return df


def train_torchvision(
    model_name: str = "fasterrcnn_resnet50_v2",
    data_yaml: str = "dataset.yaml",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 0.0005,
    img_size: int = 1024,
    device_str: str = "",
    output_weights: str = "pcb_runs/torchvision_best.pth",
    pretrained: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Main training function for Torchvision detection models.
    """
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_yaml}")

    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg.get("path", "."))
    train_rel = data_cfg.get("train", "train/images")
    val_rel = data_cfg.get("val", "valid/images")

    train_img_dir = dataset_root / train_rel if not Path(train_rel).is_absolute() else Path(train_rel)
    val_img_dir = dataset_root / val_rel if not Path(val_rel).is_absolute() else Path(val_rel)

    device = torch.device(
        device_str if device_str else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print("=" * 65)
    print("🚀 TRAINING TORCHVISION DETECTION MODEL")
    print("=" * 65)
    print(f"Model:        {model_name}")
    print(f"Device:       {device}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Epochs:       {epochs}")
    print(f"Batch Size:   {batch_size}")
    print(f"Image Size:   {img_size}")
    print(f"Learning Rate:{lr}")
    print("=" * 65)

    model = build_torchvision_detector(model_name=model_name, pretrained=pretrained)
    model.to(device)

    train_dataset = PCBDefectDataset(train_img_dir, img_size=img_size)
    val_dataset = PCBDefectDataset(val_img_dir, img_size=img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0005,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_map = 0.0
    metrics = {"map50_90": 0.0, "map50": 0.0, "macro_recall": 0.0}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_losses.append(losses.item())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            pbar.set_postfix(loss=f"{losses.item():.4f}")

            if dry_run:
                break

        lr_scheduler.step()
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

        if dry_run:
            break

    # Final validation evaluation
    metrics = evaluate_torchvision_model(model, val_loader, device=device)

    checkpoint_data = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "img_size": img_size,
        "classes": CANONICAL_CLASSES,
        "map50_90": metrics["map50_90"],
        "map50": metrics["map50"],
        "macro_recall": metrics["macro_recall"],
    }
    Path(output_weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_data, output_weights)
    print(f"✅ Model weights saved to: {Path(output_weights).resolve()}")

    return {
        "best_weights": str(Path(output_weights).resolve()),
        "map50_90": metrics["map50_90"],
        "map50": metrics["map50"],
        "macro_recall": metrics["macro_recall"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train / Evaluate Torchvision Detection Models for PCB Defect Detection")
    parser.add_argument("--model", type=str, default="fasterrcnn_resnet50_v2",
                        choices=[
                            "fasterrcnn_resnet50_v2",
                            "fasterrcnn_mobilenet_v2",
                            "fasterrcnn_mobilenet_v3",
                            "fasterrcnn_efficientnet_b0",
                            "fasterrcnn_densenet121",
                            "retinanet_resnet50_v2",
                        ],
                        help="Torchvision model architecture")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image resolution")
    parser.add_argument("--device", type=str, default="", help="Device: '0', 'cpu', 'cuda'")
    parser.add_argument("--output-weights", type=str, default="pcb_runs/torchvision_best.pth", help="Path to save weights")
    parser.add_argument("--predict-source", type=str, help="Directory of test images for inference")
    parser.add_argument("--output-csv", type=str, default="submission_torchvision.csv", help="Output submission CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Perform quick 1-batch execution test")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load pre-trained ImageNet/COCO weights")

    args = parser.parse_args()

    results = train_torchvision(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        img_size=args.imgsz,
        device_str=args.device,
        output_weights=args.output_weights,
        pretrained=not args.no_pretrained,
        dry_run=args.dry_run,
    )

    if args.predict_source:
        device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        predict_torchvision_submission(
            weights_path=args.output_weights,
            images_dir=args.predict_source,
            output_csv=args.output_csv,
            img_size=args.imgsz,
            device=device,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
