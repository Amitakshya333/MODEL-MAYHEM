# CodeSplinter Submission Package - MODEL MAYHEM: PCB DEFECT DETECTION

**Team:** CodeSplinter  
**Members:** Amitakshya Sutar, Piyush Palkesh Panda, Tushar Kumar Martha  
**Task:** PCB Defect Classification and Localization (6 Canonical Defect Classes)

---

## 🎯 Canonical Classes (6)
Exact 0-indexed class definitions matching the official evaluation testbench:
1. `0: missing_hole`
2. `1: mouse_bite`
3. `2: open_circuit`
4. `3: short`
5. `4: spur`
6. `5: spurious_copper`

---

## 📦 Package Contents
- `best.pt`: Trained YOLO11s model checkpoint matching the 6 canonical PCB defect classes (fused: 100 layers, 9,415,122 parameters, 21.4 GFLOPs).
- `inference.py`: Standalone inference pipeline with intelligent path resolution across any working directory.
- `predict.py`: CLI wrapper for testbenches invoking `python predict.py`.
- `submission.csv`: Competition-verified submission file complying with all bounding box, class, and confidence requirements.
- `sample_images/`: Real PCB defect images for immediate offline validation.

---

## 🚀 Execution Instructions

### 1. Testbench Evaluation (Ultralytics Validation)
From testbench root:
```powershell
python evaluate.py
```
Or directly targeting this checkpoint:
```powershell
python evaluate.py --weights submissions/CodeSplinter/best.pt --data data.yaml
```

### 2. Run Inference & Generate Submission CSV
From testbench root or repository root:
```powershell
python submissions/CodeSplinter/inference.py
```
Or from inside `submissions/CodeSplinter`:
```powershell
cd submissions/CodeSplinter
python inference.py
python predict.py
```
Or passing explicit custom paths:
```powershell
python inference.py --weights submissions/CodeSplinter/best.pt --source dataset/test/images --output submission.csv
```

