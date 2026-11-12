#!/usr/bin/env python3
"""
MODEL MAYHEM: PCB DEFECT DETECTION - Prediction CLI
Alias for inference.py to support testbenches invoking `python predict.py`.
"""
import sys
from pathlib import Path

# Ensure script directory and parent directory are on sys.path
script_dir = str(Path(__file__).resolve().parent)
parent_dir = str(Path(__file__).resolve().parent.parent)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from inference import main, run_inference

if __name__ == "__main__":
    main()

