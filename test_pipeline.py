import os
import tempfile
import pytest
import pandas as pd
from pathlib import Path
import numpy as np
import cv2
import yaml

from verify_submission import verify_submission, ALLOWED_CLASSES, EXPECTED_COLUMNS
from inference import run_inference, CANONICAL_CLASSES
from ensemble import weighted_boxes_fusion, nms_boxes_fusion, ensemble_csv_files, calculate_iou


def test_allowed_classes():
    expected = {
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper",
    }
    assert ALLOWED_CLASSES == expected
    assert set(CANONICAL_CLASSES) == expected


def test_verify_submission_valid(tmp_path):
    csv_file = tmp_path / "valid_submission.csv"
    df = pd.DataFrame([
        {
            "image_name": "pcb_001.jpg",
            "class_name": "missing_hole",
            "x_min": 120.0,
            "y_min": 85.0,
            "x_max": 160.0,
            "y_max": 125.0,
            "confidence": 0.94,
        },
        {
            "image_name": "pcb_001.jpg",
            "class_name": "mouse_bite",
            "x_min": 220.0,
            "y_min": 140.0,
            "x_max": 270.0,
            "y_max": 190.0,
            "confidence": 0.87,
        },
        {
            "image_name": "pcb_002.jpg",
            "class_name": "spur",
            "x_min": 50.5,
            "y_min": 60.2,
            "x_max": 75.8,
            "y_max": 85.0,
            "confidence": 0.9123,
        },
    ])
    df.to_csv(csv_file, index=False)

    assert verify_submission(str(csv_file)) is True


def test_verify_submission_invalid_class(tmp_path):
    csv_file = tmp_path / "invalid_class.csv"
    df = pd.DataFrame([
        {
            "image_name": "pcb_001.jpg",
            "class_name": "unknown_defect",
            "x_min": 10,
            "y_min": 10,
            "x_max": 20,
            "y_max": 20,
            "confidence": 0.9,
        }
    ])
    df.to_csv(csv_file, index=False)
    assert verify_submission(str(csv_file)) is False


def test_verify_submission_invalid_coords(tmp_path):
    # x_max <= x_min
    csv_file = tmp_path / "invalid_coords.csv"
    df = pd.DataFrame([
        {
            "image_name": "pcb_001.jpg",
            "class_name": "missing_hole",
            "x_min": 100,
            "y_min": 50,
            "x_max": 80,  # invalid!
            "y_max": 70,
            "confidence": 0.85,
        }
    ])
    df.to_csv(csv_file, index=False)
    assert verify_submission(str(csv_file)) is False


def test_verify_submission_negative_coords(tmp_path):
    csv_file = tmp_path / "negative_coords.csv"
    df = pd.DataFrame([
        {
            "image_name": "pcb_001.jpg",
            "class_name": "short",
            "x_min": -5,
            "y_min": 10,
            "x_max": 20,
            "y_max": 30,
            "confidence": 0.85,
        }
    ])
    df.to_csv(csv_file, index=False)
    assert verify_submission(str(csv_file)) is False


def test_verify_submission_invalid_confidence(tmp_path):
    csv_file = tmp_path / "invalid_conf.csv"
    df = pd.DataFrame([
        {
            "image_name": "pcb_001.jpg",
            "class_name": "open_circuit",
            "x_min": 10,
            "y_min": 10,
            "x_max": 20,
            "y_max": 20,
            "confidence": 1.25,  # invalid! > 1.0
        }
    ])
    df.to_csv(csv_file, index=False)
    assert verify_submission(str(csv_file)) is False


def test_verify_submission_missing_columns(tmp_path):
    csv_file = tmp_path / "missing_cols.csv"
    df = pd.DataFrame([
        {
            "image": "pcb_001.jpg",
            "label": "open_circuit",
            "confidence": 0.8,
        }
    ])
    df.to_csv(csv_file, index=False)
    assert verify_submission(str(csv_file)) is False


def test_dataset_yaml_structure():
    with open("C:/model mayhem/dataset.yaml", "r") as f:
        data = yaml.safe_load(f)

    assert "names" in data
    assert "nc" in data
    assert data["nc"] == 6
    for i, name in enumerate(CANONICAL_CLASSES):
        assert data["names"][i] == name


def test_inference_pipeline_execution(tmp_path):
    # Create synthetic test images
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()

    for idx in range(2):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw some features simulating PCB traces
        cv2.circle(img, (150, 150), 30, (255, 255, 255), -1)
        cv2.rectangle(img, (50, 50), (100, 100), (200, 200, 200), -1)
        cv2.imwrite(str(img_dir / f"sample_{idx:03d}.jpg"), img)

    # Use pretrained yolo11n.pt for end-to-end inference verification
    output_csv = tmp_path / "test_submission.csv"
    visuals_dir = tmp_path / "visuals"

    df = run_inference(
        weights_path="yolo11n.pt",
        source_path=str(img_dir),
        output_csv=str(output_csv),
        conf_threshold=0.01,  # low conf so detections appear on synthetic images
        imgsz=320,
        device="cpu",
        use_tta=False,
        save_visuals=True,
        visuals_dir=str(visuals_dir),
        batch_size=2,
    )

    assert output_csv.exists()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert verify_submission(str(output_csv)) is True


def test_data_setup_inspection(tmp_path):
    from data_setup import inspect_and_fix_dataset

    # Create synthetic dataset structure
    dataset_dir = tmp_path / "mock_dataset"
    for split in ["train", "valid", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(2):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.jpg"), img)
            # YOLO label format: <class> <x_center> <y_center> <width> <height>
            with open(lbl_dir / f"img_{i}.txt", "w") as f:
                f.write(f"{i % 6} 0.5 0.5 0.2 0.2\n")

    yaml_out = tmp_path / "generated_data.yaml"
    stats = inspect_and_fix_dataset(str(dataset_dir), str(yaml_out))

    assert yaml_out.exists()
    assert stats["train"]["image_count"] == 2
    assert stats["valid"]["image_count"] == 2
    assert stats["test"]["image_count"] == 2

    with open(yaml_out, "r") as f:
        conf = yaml.safe_load(f)
    assert conf["nc"] == 6
    assert conf["names"][0] == "missing_hole"


def test_verify_submission_empty_csv(tmp_path):
    csv_file = tmp_path / "empty.csv"
    df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    df.to_csv(csv_file, index=False)
    # Empty CSV with proper columns returns True with warning
    assert verify_submission(str(csv_file)) is True


def test_verify_submission_nonexistent():
    assert verify_submission("nonexistent_file_path_12345.csv") is False


def test_train_pipeline_dry_run(tmp_path):
    from data_setup import inspect_and_fix_dataset
    from train import train

    dataset_dir = tmp_path / "mock_dataset"
    for split in ["train", "valid"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(2):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.jpg"), img)
            with open(lbl_dir / f"img_{i}.txt", "w") as f:
                f.write(f"0 0.5 0.5 0.2 0.2\n")

    yaml_out = tmp_path / "train_mock.yaml"
    inspect_and_fix_dataset(str(dataset_dir), str(yaml_out))

    results = train(
        data_yaml=str(yaml_out),
        model_name="yolo11n.pt",
        epochs=1,
        imgsz=64,
        batch=2,
        device="cpu",
        workers=0,
        project=str(tmp_path / "runs"),
        name="dry_run",
    )

    assert "best_weights" in results
    assert "map50_90" in results
    assert "map50" in results
    assert "macro_recall" in results


def test_evaluate_script_execution(tmp_path):
    from data_setup import inspect_and_fix_dataset
    from evaluate import evaluate_model

    dataset_dir = tmp_path / "eval_dataset"
    for split in ["valid"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(2):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.jpg"), img)
            with open(lbl_dir / f"img_{i}.txt", "w") as f:
                f.write(f"0 0.5 0.5 0.2 0.2\n")

    yaml_out = tmp_path / "eval_mock.yaml"
    inspect_and_fix_dataset(str(dataset_dir), str(yaml_out))

    # Evaluate with pretrained yolo11n.pt
    metrics = evaluate_model(
        weights="yolo11n.pt",
        data_yaml=str(yaml_out),
        split="val",
        imgsz=64,
        batch=2,
        device="cpu",
        use_tta=False,
    )

    assert "map50_90" in metrics
    assert "map50" in metrics
    assert "macro_recall" in metrics


def test_calculate_iou_metric():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([0, 0, 10, 10])
    assert pytest.approx(calculate_iou(box1, box2), 0.001) == 1.0

    box3 = np.array([10, 10, 20, 20])
    assert calculate_iou(box1, box3) == 0.0

    # 50% overlap in 1D => intersection 5x10 = 50, union = 100 + 100 - 50 = 150 => 1/3
    box4 = np.array([5, 0, 15, 10])
    assert pytest.approx(calculate_iou(box1, box4), 0.001) == 50.0 / 150.0


def test_wbf_fusion_logic():
    # Model 1 and Model 2 detect the same defect with slightly different coordinates
    boxes_m1 = np.array([[10.0, 10.0, 30.0, 30.0]])
    scores_m1 = np.array([0.90])
    labels_m1 = np.array(["missing_hole"])

    boxes_m2 = np.array([[12.0, 12.0, 32.0, 32.0]])
    scores_m2 = np.array([0.90])
    labels_m2 = np.array(["missing_hole"])

    f_boxes, f_scores, f_labels = weighted_boxes_fusion(
        boxes_list=[boxes_m1, boxes_m2],
        scores_list=[scores_m1, scores_m2],
        labels_list=[labels_m1, labels_m2],
        weights=[1.0, 1.0],
        iou_thr=0.50,
        skip_box_thr=0.05,
        conf_type="avg",
    )

    assert len(f_boxes) == 1
    assert len(f_scores) == 1
    assert f_labels[0] == "missing_hole"
    # Weighted average of [10, 10, 30, 30] and [12, 12, 32, 32] is [11, 11, 31, 31]
    assert np.allclose(f_boxes[0], [11.0, 11.0, 31.0, 31.0], atol=1e-3)
    assert pytest.approx(f_scores[0], 0.001) == 0.90


def test_wbf_non_overlapping():
    boxes_m1 = np.array([[0.0, 0.0, 10.0, 10.0]])
    scores_m1 = np.array([0.85])
    labels_m1 = np.array(["short"])

    boxes_m2 = np.array([[50.0, 50.0, 70.0, 70.0]])
    scores_m2 = np.array([0.92])
    labels_m2 = np.array(["open_circuit"])

    f_boxes, f_scores, f_labels = weighted_boxes_fusion(
        boxes_list=[boxes_m1, boxes_m2],
        scores_list=[scores_m1, scores_m2],
        labels_list=[labels_m1, labels_m2],
        weights=[1.0, 1.0],
        iou_thr=0.55,
        skip_box_thr=0.05,
    )

    # Disjoint boxes should both be preserved
    assert len(f_boxes) == 2
    assert set(f_labels) == {"short", "open_circuit"}


def test_nms_fusion_logic():
    # Two overlapping boxes for the same class: highest score must win in NMS
    boxes_m1 = np.array([[10.0, 10.0, 30.0, 30.0]])
    scores_m1 = np.array([0.95])
    labels_m1 = np.array(["spur"])

    boxes_m2 = np.array([[11.0, 11.0, 29.0, 29.0]])
    scores_m2 = np.array([0.70])
    labels_m2 = np.array(["spur"])

    k_boxes, k_scores, k_labels = nms_boxes_fusion(
        boxes_list=[boxes_m1, boxes_m2],
        scores_list=[scores_m1, scores_m2],
        labels_list=[labels_m1, labels_m2],
        weights=[1.0, 1.0],
        iou_thr=0.50,
        skip_box_thr=0.05,
    )

    assert len(k_boxes) == 1
    assert pytest.approx(k_scores[0], 0.001) == 0.95
    assert np.allclose(k_boxes[0], [10.0, 10.0, 30.0, 30.0])


def test_ensemble_csv_files(tmp_path):
    csv1 = tmp_path / "model1_sub.csv"
    csv2 = tmp_path / "model2_sub.csv"
    out_csv = tmp_path / "fused_sub.csv"

    df1 = pd.DataFrame([
        {
            "image_name": "board_01.jpg",
            "class_name": "missing_hole",
            "x_min": 50.0,
            "y_min": 50.0,
            "x_max": 80.0,
            "y_max": 80.0,
            "confidence": 0.88,
        },
        {
            "image_name": "board_02.jpg",
            "class_name": "short",
            "x_min": 100.0,
            "y_min": 100.0,
            "x_max": 150.0,
            "y_max": 150.0,
            "confidence": 0.91,
        },
    ])
    df2 = pd.DataFrame([
        {
            "image_name": "board_01.jpg",
            "class_name": "missing_hole",
            "x_min": 52.0,
            "y_min": 48.0,
            "x_max": 82.0,
            "y_max": 78.0,
            "confidence": 0.92,
        },
        {
            "image_name": "board_03.jpg",
            "class_name": "spur",
            "x_min": 200.0,
            "y_min": 200.0,
            "x_max": 240.0,
            "y_max": 240.0,
            "confidence": 0.85,
        },
    ])

    df1.to_csv(csv1, index=False)
    df2.to_csv(csv2, index=False)

    fused_df = ensemble_csv_files(
        csv_paths=[str(csv1), str(csv2)],
        output_path=str(out_csv),
        weights=[1.0, 1.2],
        method="wbf",
        iou_thr=0.55,
        conf_type="avg",
    )

    assert out_csv.exists()
    assert len(fused_df) == 3  # board_01 (fused 2->1), board_02 (1), board_03 (1)
    assert verify_submission(str(out_csv)) is True


def test_ensemble_csv_empty(tmp_path):
    csv1 = tmp_path / "empty1.csv"
    csv2 = tmp_path / "empty2.csv"
    out_csv = tmp_path / "fused_empty.csv"

    pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(csv1, index=False)
    pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(csv2, index=False)

    fused_df = ensemble_csv_files(
        csv_paths=[str(csv1), str(csv2)],
        output_path=str(out_csv),
        method="wbf",
    )

    assert out_csv.exists()
    assert len(fused_df) == 0
    assert verify_submission(str(out_csv)) is True


def test_inference_multi_weights_ensemble(tmp_path):
    img_dir = tmp_path / "ensemble_test_images"
    img_dir.mkdir()

    for idx in range(2):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 25, (255, 255, 255), -1)
        cv2.imwrite(str(img_dir / f"test_{idx}.jpg"), img)

    out_csv = tmp_path / "multi_model_sub.csv"

    # Run inference with a list of weights (triggers ensemble branch)
    df = run_inference(
        weights_path=["yolo11n.pt", "yolo11n.pt"],
        source_path=str(img_dir),
        output_csv=str(out_csv),
        conf_threshold=0.01,
        imgsz=320,
        device="cpu",
    )

    assert out_csv.exists()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert verify_submission(str(out_csv)) is True


def test_torchvision_dataset_loader(tmp_path):
    from train_torchvision import PCBDefectDataset

    dataset_dir = tmp_path / "tv_data"
    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "sample.jpg"), img)

    # Class 0: missing_hole, normalized: center=(0.5, 0.5), w=0.2, h=0.2
    # Image is 200 wide x 100 high
    # x_center = 100, y_center = 50, box_w = 40, box_h = 20
    # x1=80, y1=40, x2=120, y2=60
    with open(lbl_dir / "sample.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    ds = PCBDefectDataset(images_dir=img_dir, labels_dir=lbl_dir, img_size=100)
    assert len(ds) == 1

    img_tensor, target = ds[0]
    assert img_tensor.shape == (3, 100, 100)
    assert len(target["boxes"]) == 1
    # defect class 0 maps to torchvision label 1
    assert target["labels"][0].item() == 1


def test_torchvision_detector_dry_run(tmp_path):
    from train_torchvision import train_torchvision, predict_torchvision_submission, build_torchvision_detector

    dataset_dir = tmp_path / "tv_run_data"
    for split in ["train", "valid"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(2):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.jpg"), img)
            with open(lbl_dir / f"img_{i}.txt", "w") as f:
                f.write("0 0.5 0.5 0.3 0.3\n")

    yaml_file = tmp_path / "tv_dataset.yaml"
    data_dict = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "valid/images",
        "nc": 6,
        "names": CANONICAL_CLASSES,
    }
    with open(yaml_file, "w") as f:
        yaml.dump(data_dict, f)

    weights_out = tmp_path / "mobilenet_test.pth"

    res = train_torchvision(
        model_name="fasterrcnn_mobilenet_v3",
        data_yaml=str(yaml_file),
        epochs=1,
        batch_size=2,
        img_size=64,
        device_str="cpu",
        output_weights=str(weights_out),
        pretrained=False,
        dry_run=True,
    )

    assert weights_out.exists()
    assert "map50_90" in res

    # Test inference on test images
    sub_csv = tmp_path / "tv_sub.csv"
    model = build_torchvision_detector(model_name="fasterrcnn_mobilenet_v3", pretrained=False)
    sub_df = predict_torchvision_submission(
        model=model,
        images_dir=dataset_dir / "valid" / "images",
        output_csv=str(sub_csv),
        img_size=64,
        conf_threshold=0.0,
    )
    assert sub_csv.exists()
    assert verify_submission(str(sub_csv)) is True


def test_wbf_same_model_duplicate_suppression():
    """
    Solovyev WBF Rule: A single model must not contribute multiple duplicate boxes
    to the same fused cluster, preventing vote inflation.
    """
    # Model 0 outputs two duplicate candidate boxes
    boxes_m0 = np.array([
        [10.0, 10.0, 30.0, 30.0],
        [10.5, 10.5, 30.5, 30.5],
    ])
    scores_m0 = np.array([0.90, 0.80])
    labels_m0 = np.array(["missing_hole", "missing_hole"])

    # Model 1 outputs one box
    boxes_m1 = np.array([
        [11.0, 11.0, 31.0, 31.0],
    ])
    scores_m1 = np.array([0.85])
    labels_m1 = np.array(["missing_hole"])

    f_boxes, f_scores, f_labels = weighted_boxes_fusion(
        boxes_list=[boxes_m0, boxes_m1],
        scores_list=[scores_m0, scores_m1],
        labels_list=[labels_m0, labels_m1],
        weights=[1.0, 1.0],
        iou_thr=0.50,
        conf_type="avg",
    )

    # The first cluster combines Model 0 (box 0) and Model 1 (box 0).
    # Model 0's second box forms a separate cluster rather than inflating cluster 0.
    assert len(f_boxes) >= 1
    # Cluster 0 score must be average of (0.90 and 0.85), not sum containing 0.80
    assert pytest.approx(f_scores[0], 0.01) == (0.90 + 0.85) / 2.0


def test_torchvision_user_requested_backbones():
    """
    Verifies instantiation and forward pass of backbones specifically requested by user:
    MobileNet-V2, EfficientNet-B0, DenseNet-121.
    """
    from train_torchvision import build_torchvision_detector
    import torch

    for arch in ["fasterrcnn_mobilenet_v2", "fasterrcnn_efficientnet_b0", "fasterrcnn_densenet121"]:
        detector = build_torchvision_detector(model_name=arch, pretrained=False)
        detector.eval()
        dummy_input = [torch.zeros(3, 128, 128)]
        preds = detector(dummy_input)
        assert len(preds) == 1
        assert "boxes" in preds[0]
        assert "labels" in preds[0]
        assert "scores" in preds[0]


def test_inference_torchvision_pth_direct(tmp_path):
    """
    Verifies that inference.py directly accepts and runs inference on .pth Torchvision weights.
    """
    from train_torchvision import build_torchvision_detector

    # Create dummy image
    img_dir = tmp_path / "test_imgs"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "pcb_test_01.jpg"), np.zeros((128, 128, 3), dtype=np.uint8))

    # Save a valid checkpoint
    model = build_torchvision_detector("fasterrcnn_mobilenet_v3", pretrained=False)
    weights_path = tmp_path / "saved_detector.pth"
    torch_ckpt = {
        "model_name": "fasterrcnn_mobilenet_v3",
        "state_dict": model.state_dict(),
        "img_size": 128,
        "classes": CANONICAL_CLASSES,
    }
    import torch
    torch.save(torch_ckpt, weights_path)

    out_csv = tmp_path / "pth_submission.csv"
    df = run_inference(
        weights_path=str(weights_path),
        source_path=str(img_dir),
        output_csv=str(out_csv),
        conf_threshold=0.0,
        imgsz=128,
        device="cpu",
    )

    assert out_csv.exists()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert verify_submission(str(out_csv)) is True


def test_mixed_checkpoint_ensemble(tmp_path):
    """
    Verifies end-to-end WBF ensembling mixing a YOLO (.pt) model and Torchvision (.pth) model.
    """
    from train_torchvision import build_torchvision_detector
    import torch

    img_dir = tmp_path / "mixed_test_imgs"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "board_mixed.jpg"), np.zeros((128, 128, 3), dtype=np.uint8))

    model = build_torchvision_detector("fasterrcnn_mobilenet_v3", pretrained=False)
    pth_weights = tmp_path / "mobilenet_model.pth"
    torch.save({
        "model_name": "fasterrcnn_mobilenet_v3",
        "state_dict": model.state_dict(),
        "img_size": 128,
    }, pth_weights)

    out_csv = tmp_path / "mixed_ensemble.csv"
    df = run_inference(
        weights_path=["yolo11n.pt", str(pth_weights)],
        source_path=str(img_dir),
        output_csv=str(out_csv),
        conf_threshold=0.01,
        imgsz=128,
        device="cpu",
        ensemble_method="wbf",
    )

    assert out_csv.exists()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert verify_submission(str(out_csv)) is True


def test_evaluate_torchvision_pth(tmp_path):
    """
    Verifies that evaluate.py accepts and evaluates .pth Torchvision weights.
    """
    from train_torchvision import build_torchvision_detector
    from evaluate import evaluate_model
    import torch

    dataset_dir = tmp_path / "eval_data"
    val_img_dir = dataset_dir / "valid" / "images"
    val_lbl_dir = dataset_dir / "valid" / "labels"
    val_img_dir.mkdir(parents=True)
    val_lbl_dir.mkdir(parents=True)

    cv2.imwrite(str(val_img_dir / "val_01.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
    with open(val_lbl_dir / "val_01.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    yaml_file = tmp_path / "data.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump({
            "path": str(dataset_dir),
            "train": "valid/images",
            "val": "valid/images",
            "nc": 6,
            "names": CANONICAL_CLASSES,
        }, f)

    model = build_torchvision_detector("fasterrcnn_mobilenet_v3", pretrained=False)
    pth_path = tmp_path / "eval_weights.pth"
    torch.save({
        "model_name": "fasterrcnn_mobilenet_v3",
        "state_dict": model.state_dict(),
    }, pth_path)

    metrics = evaluate_model(
        weights=str(pth_path),
        data_yaml=str(yaml_file),
        split="val",
        imgsz=64,
        batch=1,
        device="cpu",
    )

    assert "map50_90" in metrics
    assert "map50" in metrics
    assert "macro_recall" in metrics


def test_weights_best_pt_classes_match_canonical():
    """Verify that weights/best.pt has exactly the 6 canonical PCB defect classes, preventing class mismatch errors."""
    import torch
    weights_path = Path("weights/best.pt")
    assert weights_path.is_file(), "weights/best.pt must exist"

    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model = ckpt.get("model")
    names = model.names if hasattr(model, "names") else ckpt.get("names")

    assert names is not None, "Model must have names dictionary"
    assert len(names) == 6, f"Expected 6 classes, got {len(names)}: {names}"

    for idx, expected_name in enumerate(CANONICAL_CLASSES):
        assert names[idx] == expected_name, f"Class {idx} mismatch: expected {expected_name}, got {names[idx]}"

    # Verify COCO classes are absent
    coco_classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train"}
    assert not any(v in coco_classes for v in names.values()), "Model must not contain COCO classes"


def test_submissions_codesplinter_best_pt():
    """Verify that submissions/CodeSplinter/best.pt exists and matches canonical PCB classes."""
    import torch
    sub_weights = Path("submissions/CodeSplinter/best.pt")
    assert sub_weights.is_file(), "submissions/CodeSplinter/best.pt must exist"

    ckpt = torch.load(str(sub_weights), map_location="cpu", weights_only=False)
    model = ckpt.get("model")
    names = model.names if hasattr(model, "names") else ckpt.get("names")
    assert list(names.values()) == CANONICAL_CLASSES


def test_dataset_yaml_and_model_classes_aligned():
    """Verify dataset.yaml and model checkpoint classes are 100% aligned."""
    import torch
    import yaml
    with open("dataset.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    cfg_names = [data_cfg["names"][i] for i in range(len(data_cfg["names"]))]
    assert cfg_names == CANONICAL_CLASSES

    ckpt = torch.load("weights/best.pt", map_location="cpu", weights_only=False)
    model_names = list(ckpt["model"].names.values())
    assert cfg_names == model_names


def test_default_inference_execution(tmp_path):
    """Verify inference.py can be invoked with smart defaults and creates valid CSV."""
    csv_out = tmp_path / "test_sub.csv"
    df = run_inference(
        weights_path="weights/best.pt",
        source_path="sample_images",
        output_csv=str(csv_out),
        conf_threshold=0.25,
        imgsz=640,
        device="cpu",
    )
    assert Path(csv_out).is_file()
    assert verify_submission(str(csv_out)) is True
    assert set(df["class_name"].unique()).issubset(set(CANONICAL_CLASSES))


def test_evaluator_data1_yaml_alignment():
    """Verify that model classes strictly match evaluator data1.yaml from competition downloads."""
    import yaml
    data1_path = Path(r"C:\Users\AMITAKSHYA SUTAR\Downloads\data1.yaml")
    if not data1_path.is_file():
        pytest.skip("data1.yaml not present in Downloads")

    with open(data1_path, "r", encoding="utf-8") as f:
        data1_cfg = yaml.safe_load(f)

    expected_names = [data1_cfg["names"][i] for i in range(len(data1_cfg["names"]))]
    assert expected_names == CANONICAL_CLASSES

    import torch
    for wp in [Path("weights/best.pt"), Path("best.pt"), Path("submissions/CodeSplinter/best.pt")]:
        ckpt = torch.load(str(wp), map_location="cpu", weights_only=False)
        m_names = list(ckpt["model"].names.values())
        assert m_names == expected_names, f"{wp} classes do not match data1.yaml"


def test_zero_arg_evaluate_execution():
    """Verify evaluate.py runs without arguments, detects default weights and data, and reports metrics."""
    from evaluate import evaluate_model, find_default_weights, find_default_data_yaml
    w = find_default_weights()
    d = find_default_data_yaml()
    assert Path(w).is_file()
    assert Path(d).is_file()

    metrics = evaluate_model(weights=w, data_yaml=d, split="val", imgsz=640, device="cpu")
    assert "map50_90" in metrics
    assert "map50" in metrics
    assert "macro_recall" in metrics
    assert metrics["map50_90"] > 0.0


def test_submissions_codesplinter_inference_cwd_isolated(tmp_path):
    """Verify inference.py inside submissions/CodeSplinter succeeds when CWD is submissions/CodeSplinter."""
    import sys
    import subprocess
    sub_dir = Path("submissions/CodeSplinter").resolve()
    result = subprocess.run(
        [sys.executable, "inference.py", "--imgsz", "640", "--device", "cpu"],
        cwd=str(sub_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"inference.py failed in submissions dir:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert (sub_dir / "submission.csv").is_file()
    assert verify_submission(str(sub_dir / "submission.csv")) is True


def test_submissions_codesplinter_predict_cwd_isolated():
    """Verify predict.py inside submissions/CodeSplinter succeeds when CWD is submissions/CodeSplinter."""
    import sys
    import subprocess
    sub_dir = Path("submissions/CodeSplinter").resolve()
    result = subprocess.run(
        [sys.executable, "predict.py", "--imgsz", "640", "--device", "cpu"],
        cwd=str(sub_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"predict.py failed in submissions dir:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"



def test_data_yaml_compatibility():
    """Verify that both data.yaml and dataset.yaml exist and have identical canonical definitions."""
    import yaml
    for p in [Path("data.yaml"), Path("dataset.yaml"), Path("dataset/data.yaml")]:
        assert p.is_file(), f"{p} must exist"
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = [cfg["names"][i] for i in range(len(cfg["names"]))]
        assert names == CANONICAL_CLASSES






