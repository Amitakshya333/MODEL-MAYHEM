#!/usr/bin/env python3
"""
MODEL MAYHEM: PCB DEFECT DETECTION - Prediction CLI
Alias for inference.py to support testbenches invoking `python predict.py`.
"""
import sys
from inference import main, run_inference

if __name__ == "__main__":
    main()
