# MODEL MAYHEM : PCB DEFECT DETECTION

## 1. Description

Build an object detection model to identify and localise manufacturing defects in Printed Circuit Boards (PCBs).

**Target Classes:**
1. `missing_hole`
2. `mouse_bite`
3. `open_circuit`
4. `short`
5. `spur`
6. `spurious_copper`

Participants may use any framework, including YOLO, PyTorch, TensorFlow, or Keras.

## 2. Dataset

**Dataset:** [PCB Defect Detection Dataset Download](https://universe.roboflow.com/uthm-project-visionbased-pcb-defect-detection/pcb-defect-detection-9ewqw-duvuw/dataset/2)

- Images: 2,586
- Classes: 6
- Task: Defect Classification and Localisation ( Bounding box )
- Version: 2

**Download Guide:**
1. Open the dataset link.
2. Click **Download Dataset**.
3. Select the format compatible with your framework (YOLOv8, COCO JSON, etc.).
4. Download and extract the ZIP file.


## 3. Evaluation Criteria

Models will be evaluated on an unseen test dataset.

**Primary Metric: mAP@50:90**

Average Precision across all six classes and nine IoU thresholds: 0.50, 0.55, 0.60, ... 0.90.

**Tie-Breakers (in order):**
1. mAP@50
2. Macro Recall (average recall across all six classes)

All submissions will be evaluated using the same scoring script and settings, irrespective of the framework used.

## 4. Submission

Each team must submit:

- **Training Code:** Model development and training scripts.
- **Model Weights:** Final trained checkpoint.
- **Inference Code:** Script to generate predictions.
- **Dependencies:** `requirements.txt` or equivalent.
- **Execution Instructions:** Steps to run the model.
- **Predictions:** CSV file in the prescribed format.

**CSV Format:**

```csv
image_name,class_name,x_min,y_min,x_max,y_max,confidence
pcb_001.jpg,missing_hole,120,85,160,125,0.94
pcb_001.jpg,mouse_bite,220,140,270,190,0.87
```

**Submission Rules:**
- Coordinates must be in original image pixels (`x_min`, `y_min`, `x_max`, `y_max`).
- Class names must match the prescribed labels.
- Confidence scores must be between 0 and 1.
- Each detection must occupy a separate row.
- Ensure the submitted model and inference code are executable.