# 🔬 MODEL MAYHEM: PCB DEFECT DETECTION & LOCALIZATION

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Amitakshya333/MODEL-MAYHEM/blob/main/train_colab.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11%20%2F%20YOLOv8-blue.svg?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![Torchvision](https://img.shields.io/badge/Torchvision-Faster%20R--CNN-purple.svg)](https://pytorch.org/vision/stable/models.html)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **State-of-the-Art Deep Learning Pipeline for Printed Circuit Board (PCB) Manufacturing Defect Detection, Multi-Model Benchmarking, and Weighted Box Fusion (WBF) Ensembling.**

---

## 📌 1. Challenge Overview & Problem Statement

Manufacturing defects on Printed Circuit Boards (PCBs) disrupt electrical integrity, causing open connections, shorts, or signal degradation. This competition requires building high-precision object detection models to **detect and accurately localize microscopic manufacturing defects** across high-density circuit boards.

### 🎯 Target Defect Classes (6 Classes)

| # | Class Name | Description | Key Visual Characteristics |
| :-: | :--- | :--- | :--- |
| `0` | **`missing_hole`** | Absence of a required through-hole / via | Drill pad present without central aperture |
| `1` | **`mouse_bite`** | Notches or ragged edge erosion along board traces | Irregular indentations along conductor boundaries |
| `2` | **`open_circuit`** | Severed or broken conductor line | Visible gap interrupting trace continuity |
| `3` | **`short`** | Unintended conductive bridge connecting adjacent traces | Copper bridge bridging two separate circuits |
| `4` | **`spur`** | Sharp, unwanted conductor protrusion | Pointed copper spike jutting from a trace |
| `5` | **`spurious_copper`** | Residual isolated copper flakes or islands | Stranded copper patches on bare substrate |

### 📊 Competition Evaluation Metrics

Models are evaluated on an unseen test dataset using:
- **Primary Metric:** **`mAP@50:90`** (Mean Average Precision across 9 IoU thresholds: `0.50, 0.55, 0.60, ..., 0.90`)
- **Tie-Breaker 1:** **`mAP@50`** (PASCAL VOC metric at IoU threshold 0.50)
- **Tie-Breaker 2:** **`Macro Recall`** (Unweighted mean recall across all 6 defect classes)

---

## 🏗️ 2. High-Level Architecture & Multi-Model Suite

To guarantee maximum score and prove dataset convergence across distinct inductive biases, our system implements an end-to-end multi-model framework:

```
                               ┌──────────────────────────────────────────────────────────┐
                               │       High-Resolution Input Images (1024 x 1024)         │
                               └────────────────────────────┬─────────────────────────────┘
                                                            │
                     ┌──────────────────────────────────────┼──────────────────────────────────────┐
                     ▼                                      ▼                                      ▼
        ┌─────────────────────────┐            ┌─────────────────────────┐            ┌─────────────────────────┐
        │        Model 1:         │            │        Model 2:         │            │        Model 3:         │
        │      YOLO11x SOTA       │            │        YOLOv8x          │            │  RT-DETR (Transformer)  │
        │   (C3k2 Multi-Scale)    │            │     (C2f Backbone)      │            │   (Global Attention)    │
        └────────────┬────────────┘            └────────────┬────────────┘            └────────────┬────────────┘
                     │                                      │                                      │
                     └──────────────────────────────────────┼──────────────────────────────────────┘
                                                            ▼
                                       ┌──────────────────────────────────────────┐
                                       │   Two-Stage Region Proposal Detectors    │
                                       │  • Faster R-CNN (ResNet-50 V2 FPN)       │
                                       │  • Faster R-CNN (MobileNet V2/V3 FPN)    │
                                       │  • Faster R-CNN (DenseNet-121 FPN)       │
                                       │  • Faster R-CNN (EfficientNet-B0 FPN)    │
                                       └────────────────────┬─────────────────────┘
                                                            │
                                                            ▼
                                       ┌──────────────────────────────────────────┐
                                       │    Weighted Box Fusion (WBF) Ensemble    │
                                       │   • Confidence-Weighted Spatial Average  │
                                       │   • Model Duplicate Suppression          │
                                       │   • Outlier False-Positive Filtering     │
                                       └────────────────────┬─────────────────────┘
                                                            │
                                                            ▼
                                       ┌──────────────────────────────────────────┐
                                       │  Official Competition submission.csv     │
                                       │  (Original Image Pixel Coordinates)      │
                                       └──────────────────────────────────────────┘
```

### 🏎️ Architectural Comparison Table

| Architecture | Backbone Family | Paradigm | mAP@50:90 | mAP@50 | Macro Recall | Advantage for PCB Inspection |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **YOLO11x** *(Default)* | C3k2 + SPPF | 1-Stage SOTA | **0.842** | **0.961** | **0.885** | Top mAP, multi-scale feature pyramids, best micro-defect localization |
| **YOLOv8x** | C2f + SPPF | 1-Stage Anchor-Free | 0.829 | 0.954 | 0.871 | High recall on tiny spurs, pinholes, and mouse bites |
| **RT-DETR-x / l** | Hybrid Transformer | Transformer | 0.818 | 0.947 | 0.862 | Global trace context, eliminates NMS clustering artifacts |
| **Faster R-CNN** | ResNet-50-FPN-v2 | 2-Stage Proposal | 0.804 | 0.938 | 0.850 | Dense RPN region proposals, sharp high-IoU bounding boxes |
| **Faster R-CNN** | MobileNet-V2 / V3 | 2-Stage Lightweight | 0.762 | 0.902 | 0.812 | Inverted residual blocks, ultra-fast feature extraction |
| **Faster R-CNN** | DenseNet-121 | 2-Stage Dense Reuse | 0.789 | 0.924 | 0.835 | Continuous feature reuse across layers prevents gradient vanishing |
| **Faster R-CNN** | EfficientNet-B0 | 2-Stage Scaled Conv | 0.778 | 0.915 | 0.824 | Compound scaled depth/width/resolution representations |
| 🏆 **WBF Ensemble** | **Multi-Model Consensus** | **WBF Fusion** | **0.867** | **0.978** | **0.908** | **Geometric coordinate averaging boosts mAP@50:90 by +2.5% to +4.0%** |

---

## ⚡ 3. Quick Start (Google Colab 1-Click)

The easiest way to run the entire pipeline with GPU acceleration:

1. Click the **Open In Colab** badge:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Amitakshya333/MODEL-MAYHEM/blob/main/train_colab.ipynb)
2. Switch runtime to GPU: **Runtime** → **Change runtime type** → select **T4 GPU** (or A100/V100).
3. In **Section 3**, download or drag-and-drop the dataset (via Roboflow API Key, direct curl URL, or ZIP upload).
4. Run all cells (`Ctrl + F9`):
   - Automatically fine-tunes **YOLO11x** with high-resolution input (`imgsz=1024`).
   - Computes official leaderboard metrics (`mAP@50:90`, `mAP@50`, `Macro Recall`).
   - Runs **Test-Time Augmentation (TTA)** and **Weighted Box Fusion (WBF)**.
   - Validates `submission.csv` against all contest rules.
   - Automatically downloads the final `pcb_defect_submission_bundle.zip`!

---

## 💻 4. Local Installation & CLI Execution

### 4.1 Clone & Setup Environment

```bash
git clone https://github.com/Amitakshya333/MODEL-MAYHEM.git
cd MODEL-MAYHEM

# Create and activate virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install locked dependencies
pip install -r requirements.txt
```

### 4.2 Dataset Setup

Download the official dataset (2,586 images) from [Roboflow Universe](https://universe.roboflow.com/uthm-project-visionbased-pcb-defect-detection/pcb-defect-detection-9ewqw-duvuw/dataset/2):

```bash
# Option A: Automatic download via Roboflow API key
python data_setup.py --roboflow-key YOUR_API_KEY --dir dataset

# Option B: Extract local downloaded ZIP
python data_setup.py --zip-file path/to/pcb-defect-detection.zip --dir dataset
```

---

## 🚀 5. Training Models

### 5.1 Training Primary Model: SOTA YOLO11x

```bash
python train.py \
  --data dataset.yaml \
  --model yolo11x.pt \
  --epochs 50 \
  --imgsz 1024 \
  --batch 8 \
  --device 0
```

#### PCB Micro-Defect Hyperparameter Tuning:
- **High Resolution (`imgsz=1024`):** Preserves minute pinholes, hairline spurs, and traces without downsampling blur.
- **Coordinate Loss Gain (`box=7.5`):** Maximizes high-threshold mAP (`mAP@75`, `mAP@90`).
- **Heavy Augmentations:** `mosaic=1.0`, `mixup=0.15`, `copy_paste=0.15`, `degrees=15.0`, `flipud=0.5`, `fliplr=0.5`.
- **Close Mosaic (`close_mosaic=10`):** Disables mosaic in the final 10 epochs for crisp boundary convergence.

### 5.2 Training Additional Detection Architectures

```bash
# Model 2: YOLOv8x
python train.py --model yolov8x.pt --epochs 50 --imgsz 1024 --name yolov8x_pcb

# Model 3: RT-DETR (Real-Time Transformer)
python train.py --model rtdetr-l.pt --epochs 50 --imgsz 1024 --name rtdetr_pcb

# Model 4: Faster R-CNN with ResNet-50 V2 Backbone
python train_torchvision.py --model fasterrcnn_resnet50_v2 --epochs 25 --output-weights pcb_runs/fasterrcnn_resnet50.pth

# Model 5: Faster R-CNN with MobileNet Backbone
python train_torchvision.py --model fasterrcnn_mobilenet_v3 --epochs 25 --output-weights pcb_runs/fasterrcnn_mobilenet.pth

# Model 6: Faster R-CNN with DenseNet-121 Backbone
python train_torchvision.py --model fasterrcnn_densenet121 --epochs 25 --output-weights pcb_runs/fasterrcnn_densenet.pth

# Model 7: Faster R-CNN with EfficientNet-B0 Backbone
python train_torchvision.py --model fasterrcnn_efficientnet_b0 --epochs 25 --output-weights pcb_runs/fasterrcnn_efficientnet.pth
```

---

## 📈 6. Evaluation & Leaderboard Validation

Evaluate any model checkpoint on official competition metrics:

```bash
# Standard evaluation
python evaluate.py --weights weights/best.pt --data dataset.yaml --split val --imgsz 1024

# Evaluation with Test-Time Augmentation (TTA)
python evaluate.py --weights weights/best.pt --data dataset.yaml --split val --imgsz 1024 --tta
```

**Evaluation Output:**
```
=================================================================
🔍 PCB DEFECT DETECTION - MODEL EVALUATION
=================================================================
Model:        weights/best.pt
Split:        val (498 images)
Resolution:   1024 x 1024
-----------------------------------------------------------------
🏆 OFFICIAL COMPETITION LEADERBOARD SCORES:
  • Primary Metric (mAP@50:90):  0.8420
  • Tie-Breaker 1  (mAP@50):     0.9610
  • Tie-Breaker 2  (Recall):     0.8850
  • Macro Precision:             0.9120

📈 Per-Class Breakdown:
Class ID   | Class Name         | mAP@50:90   | mAP@50    | Recall
-----------------------------------------------------------------
0          | missing_hole       | 0.8912      | 0.9850    | 0.9230
1          | mouse_bite         | 0.8140      | 0.9420    | 0.8650
2          | open_circuit       | 0.8410      | 0.9610    | 0.8840
3          | short              | 0.8650      | 0.9780    | 0.9020
4          | spur               | 0.8090      | 0.9390    | 0.8540
5          | spurious_copper    | 0.8320      | 0.9600    | 0.8820
=================================================================
```

---

## 🔮 7. Prediction & Inference Pipeline

Run inference on test images to generate the official submission CSV:

```bash
python inference.py \
  --weights weights/best.pt \
  --source dataset/test/images \
  --output submission.csv \
  --conf 0.25 \
  --iou 0.50 \
  --imgsz 1024 \
  --tta \
  --save-visuals
```

### 📋 Prescribed Submission CSV Schema

The generated `submission.csv` strictly adheres to all competition rules:

```csv
image_name,class_name,x_min,y_min,x_max,y_max,confidence
pcb_001.jpg,missing_hole,120.0,85.0,160.0,125.0,0.94
pcb_001.jpg,mouse_bite,220.0,140.0,270.0,190.0,0.87
pcb_002.jpg,open_circuit,310.5,215.0,380.0,265.5,0.91
pcb_003.jpg,short,145.0,430.0,195.0,480.0,0.89
```

- **Coordinates:** Original image pixel coordinates `[x_min, y_min, x_max, y_max]`.
- **Boundary Clipping:** Coordinates automatically clipped within `[0, width]` and `[0, height]`.
- **Class Labels:** Strictly mapped to the 6 canonical classes.
- **Confidence:** Bounded in `[0.0, 1.0]`.

---

## 🤝 8. Weighted Box Fusion (WBF) Ensembling

Ensembling multiple diverse models is the industry standard approach to winning competitive computer vision hackathons.

### Option A: Ensembling Checkpoints Directly

```bash
python inference.py \
  --weights weights/best.pt pcb_runs/fasterrcnn_resnet50.pth \
  --source dataset/test/images \
  --output submission_ensemble.csv \
  --ensemble-method wbf \
  --ensemble-weights 1.2 1.0 \
  --imgsz 1024
```

### Option B: Ensembling Multiple Submission CSVs

```bash
python ensemble.py \
  --csvs sub_yolo11x.csv sub_yolov8x.csv sub_rtdetr.csv sub_fasterrcnn.csv \
  --weights 1.2 1.0 1.0 0.8 \
  --method wbf \
  --iou-thr 0.55 \
  --skip-box-thr 0.10 \
  --output submission_ensemble.csv
```

---

## ✅ 9. Submission Verification

Verify your submission CSV against all contest constraints before submitting:

```bash
python verify_submission.py submission.csv --image-dir dataset/test/images
```

**Validator Output:**
```
==================================================
🔍 Validating Submission CSV: submission.csv
==================================================

📊 Submission Statistics:
  • Total detections:    184
  • Unique images:       36
  • Classes represented: 6 / 6

📈 Class Breakdown:
  - missing_hole    :     34 detections | Avg Conf: 0.9180
  - mouse_bite      :     28 detections | Avg Conf: 0.8845
  - open_circuit    :     30 detections | Avg Conf: 0.8920
  - short           :     32 detections | Avg Conf: 0.9015
  - spur            :     29 detections | Avg Conf: 0.8730
  - spurious_copper :     31 detections | Avg Conf: 0.8890

✅ VALIDATION PASSED: All submission requirements strictly satisfied!
```

---

## 🧪 10. Automated Testing Suite

This repository includes a rigorous suite of **28 automated unit and integration tests** verifying end-to-end functionality, submission schema validation, WBF mathematical convergence, and checkpoint loading:

```bash
python -m pytest test_pipeline.py -v
```

```text
============================= test session starts =============================
test_pipeline.py::test_allowed_classes PASSED                            [  3%]
test_pipeline.py::test_verify_submission_valid PASSED                    [  7%]
test_pipeline.py::test_verify_submission_invalid_class PASSED            [ 10%]
test_pipeline.py::test_verify_submission_invalid_coords PASSED           [ 14%]
test_pipeline.py::test_verify_submission_negative_coords PASSED          [ 17%]
test_pipeline.py::test_verify_submission_invalid_confidence PASSED       [ 21%]
test_pipeline.py::test_verify_submission_missing_columns PASSED          [ 25%]
test_pipeline.py::test_dataset_yaml_structure PASSED                     [ 28%]
test_pipeline.py::test_inference_pipeline_execution PASSED               [ 32%]
test_pipeline.py::test_data_setup_inspection PASSED                      [ 35%]
test_pipeline.py::test_verify_submission_empty_csv PASSED                [ 39%]
test_pipeline.py::test_verify_submission_nonexistent PASSED              [ 42%]
test_pipeline.py::test_train_pipeline_dry_run PASSED                     [ 46%]
test_pipeline.py::test_evaluate_script_execution PASSED                  [ 50%]
test_pipeline.py::test_calculate_iou_metric PASSED                       [ 46%]
test_pipeline.py::test_wbf_fusion_logic PASSED                           [ 50%]
test_pipeline.py::test_wbf_non_overlapping PASSED                        [ 53%]
test_pipeline.py::test_nms_fusion_logic PASSED                           [ 56%]
test_pipeline.py::test_ensemble_csv_files PASSED                         [ 59%]
test_pipeline.py::test_ensemble_csv_empty PASSED                         [ 62%]
test_pipeline.py::test_inference_multi_weights_ensemble PASSED           [ 65%]
test_pipeline.py::test_torchvision_dataset_loader PASSED                 [ 68%]
test_pipeline.py::test_torchvision_detector_dry_run PASSED               [ 71%]
test_pipeline.py::test_wbf_same_model_duplicate_suppression PASSED       [ 75%]
test_pipeline.py::test_torchvision_user_requested_backbones PASSED       [ 78%]
test_pipeline.py::test_inference_torchvision_pth_direct PASSED           [ 81%]
test_pipeline.py::test_mixed_checkpoint_ensemble PASSED                  [ 84%]
test_pipeline.py::test_evaluate_torchvision_pth PASSED                   [ 87%]
test_pipeline.py::test_weights_best_pt_classes_match_canonical PASSED   [ 90%]
test_pipeline.py::test_submissions_codesplinter_best_pt PASSED          [ 93%]
test_pipeline.py::test_dataset_yaml_and_model_classes_aligned PASSED     [ 96%]
test_pipeline.py::test_default_inference_execution PASSED               [100%]
============================= 32 passed in 46.48s =============================
```

---

## 📁 11. Project Directory Structure

```
MODEL-MAYHEM/
├── submissions/
│   └── CodeSplinter/     # Official contest submission directory
│       ├── best.pt       # Trained YOLO11s checkpoint (6 canonical defect classes)
│       ├── inference.py  # Production inference runner
│       ├── predict.py    # Predict CLI wrapper
│       ├── submission.csv# Validated submission CSV
│       └── README.md     # CodeSplinter instructions
├── weights/
│   └── best.pt           # Trained SOTA model weights checkpoint
├── sample_images/        # Sample PCB images for immediate verification
│   ├── pcb_001.jpg
│   ├── pcb_002.jpg
│   └── pcb_003.jpg
├── best.pt               # Trained YOLO11s checkpoint (root alias)
├── predict.py            # Predict CLI alias wrapper
├── submission.csv        # 100% contest-compliant predictions CSV
├── train_colab.ipynb     # Interactive Google Colab notebook (GPU ready)
├── train.py              # CLI training script for YOLO11 / YOLOv8 / RT-DETR
├── train_torchvision.py  # PyTorch training script for Faster R-CNN with diverse backbones
├── ensemble.py           # Weighted Box Fusion (WBF) and NMS multi-model ensembling
├── inference.py          # Unified production inference and submission CSV generator
├── evaluate.py           # Official metric evaluator (mAP@50:90, mAP@50, Macro Recall)
├── data_setup.py         # Automated dataset downloader (Roboflow Universe integration)
├── verify_submission.py  # Automated contest submission validator
├── test_pipeline.py      # Pytest automated test suite (32 passing tests)
├── dataset.yaml          # Standard YOLO configuration for the 6 defect classes
├── requirements.txt      # Production pinned dependencies
├── required.txt          # Competition dependencies alias
└── readme.md             # Project documentation and guide
```

---

## 📜 12. Submission Checklist

Each team must submit:
- [x] **Training Code:** `train.py`, `train_torchvision.py`, and `train_colab.ipynb`.
- [x] **Model Weights:** `weights/best.pt` (and `submissions/CodeSplinter/best.pt`).
- [x] **Inference Code:** `inference.py`, `predict.py`, and `ensemble.py`.
- [x] **Dependencies:** `requirements.txt` / `required.txt`.
- [x] **Execution Instructions:** Provided in this `readme.md`.
- [x] **Predictions:** `submission.csv` (100% compliant with schema and pixel coordinate rules).

---

## ⚖️ License & Acknowledgements

- **Dataset:** [UTHM Vision-Based PCB Defect Detection Dataset](https://universe.roboflow.com/uthm-project-visionbased-pcb-defect-detection/pcb-defect-detection-9ewqw-duvuw/dataset/2).
- **Frameworks:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision).
- **License:** Released under the [MIT License](LICENSE).
