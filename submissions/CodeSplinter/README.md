# CodeSplinter Submission Package - MODEL MAYHEM: PCB DEFECT DETECTION

**Team:** CodeSplinter  
**Members:** Amitakshya Sutar, Piyush Palkesh Panda, Tushar Kumar Martha  
**Task:** PCB Defect Classification and Localization (6 Defect Classes)

---

## 🎯 Canonical Classes (6)
1. `0: missing_hole`
2. `1: mouse_bite`
3. `2: open_circuit`
4. `3: short`
5. `4: spur`
6. `5: spurious_copper`

---

## 📦 Package Contents
- `best.pt`: Trained YOLO11s model checkpoint with 6 PCB defect classes (fused: 100 layers, 9,415,122 parameters, 21.4 GFLOPs).
- `inference.py`: Production inference pipeline with automatic weights & test-image resolution.
- `predict.py`: Inference CLI wrapper.
- `submission.csv`: Competition-verified submission file complying with all bounding box, class, and confidence requirements.

---

## 🚀 Execution Instructions

### Run Evaluation (Ultralytics Testbench)
```bash
python evaluate.py --weights submissions/CodeSplinter/best.pt
```

### Run Inference
```bash
python inference.py --weights submissions/CodeSplinter/best.pt --source <IMAGE_DIR> --output submission.csv
```
