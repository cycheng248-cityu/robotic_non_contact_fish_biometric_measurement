# -*- coding: utf-8 -*-
"""
M0-only YOLO pose evaluation for underwater fish measurements.

Outputs:
- M0 evaluation summary CSV.
- Checkpoint manifest CSV with SHA-256.
- Raw prediction CSV.
- Matched-fish/keypoint error CSV.
- Per-image detection summary CSV.
- Annotated diagnostic images.
- Ultralytics validation outputs.

Evaluated landmarks:
0 = HEAD
1 = TAIL_BASE
2 = UPPER
3 = LOWER

Additional landmarks, if present in data.yaml, are retained in raw
predictions but are not used for the four-landmark evaluation.

Two separate confidence settings:
- MAP_CONFIDENCE: low detection threshold for validation/mAP.
- PREDICTION_CONFIDENCE_THRESHOLD: operating detection threshold.

KEYPOINT_CONFIDENCE_THRESHOLD is used only for a separate measurement
acceptance diagnostic. It does not filter the main keypoint-error metrics.

Distance errors are measured in original-image pixels, not centimetres.
"""

from __future__ import annotations

import hashlib
import math
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
import ultralytics
from ultralytics import YOLO


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    r"D:\2nd_project_AUV_deepwater_fish_size_estimation"
)

DATA_YAML = PROJECT_DIR / "data" / "data.yaml"

TEST_IMAGES_DIR = PROJECT_DIR / "data" / "test" / "images"
TEST_LABELS_DIR = PROJECT_DIR / "data" / "test" / "labels"

MODEL_NAME = "A3_1280_AdamW_cosine_250"

# Change this if your new corrected-label training used another folder.
MODEL_WEIGHTS = (
    PROJECT_DIR
    / "runs_AdamW"
    / MODEL_NAME
    / "weights"
    / "best.pt"
)

OUTPUT_ROOT = PROJECT_DIR / "evaluation_AdamW"


# ============================================================
# 2. FIXED EVALUATION PARAMETERS
# ============================================================

IMG_SIZE = 1280

MAP_CONFIDENCE = 0.001
PREDICTION_CONFIDENCE_THRESHOLD = 0.25

IOU_THRESHOLD = 0.50

# Preserves the setting in the attached evaluator.
# Use 10 to match the previous Pi deployment benchmark.
MAX_DETECTIONS = 300

MATCH_IOU_THRESHOLD = 0.50
PCK_THRESHOLD_PERCENT_BOX_DIAGONAL = 5.0

KEYPOINT_CONFIDENCE_THRESHOLD = 0.50

HEAD = 0
TAIL_BASE = 1
UPPER = 2
LOWER = 3

REQUIRED_KEYPOINTS = (HEAD, TAIL_BASE, UPPER, LOWER)

KEYPOINT_NAMES = {
    HEAD: "head",
    TAIL_BASE: "tail_base",
    UPPER: "upper",
    LOWER: "lower",
}

CLASS_FISH_SIDE = 0
CLASS_FISH_TOP = 1

DEVICE = 0 if torch.cuda.is_available() else "cpu"

VALIDATION_BATCH = 1
WORKERS = 0

SAVE_DIAGNOSTIC_IMAGES = True
N_DIAGNOSTIC_IMAGES = 20

# Missing label files are errors by default.
# For intentional negative images, use empty .txt label files.
ALLOW_MISSING_LABEL_FILES = False


# ============================================================
# 3. BASIC HELPERS
# ============================================================

def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_keypoint_shape() -> tuple[int, int]:
    with DATA_YAML.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    shape = config.get("kpt_shape")

    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError(
            "data.yaml must define kpt_shape, "
            "for example kpt_shape: [4, 3]."
        )

    count, dimensions = map(int, shape)

    if count <= max(REQUIRED_KEYPOINTS):
        raise ValueError(
            f"kpt_shape={shape} does not include all four "
            "required measurement landmarks."
        )

    if dimensions != 3:
        raise ValueError(
            "This evaluator requires x, y, visibility labels: "
            "kpt_shape must have second value 3."
        )

    return count, dimensions


def find_test_images() -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    images = sorted(
        path
        for path in TEST_IMAGES_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    if not images:
        raise RuntimeError(
            f"No test images found:\n{TEST_IMAGES_DIR}"
        )

    return images


def image_to_label_path(image_path: Path) -> Path:
    relative_path = image_path.relative_to(TEST_IMAGES_DIR)
    return (TEST_LABELS_DIR / relative_path).with_suffix(".txt")


def rate_or_nan(numerator: float, denominator: float) -> float:
    return (
        float(numerator / denominator)
        if denominator > 0
        else math.nan
    )


def f1_score(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return math.nan

    return (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall > 0
        else 0.0
    )


def numeric_values(values) -> np.ndarray:
    series = pd.Series(values, dtype="object")
    array = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    return array[np.isfinite(array)]


def describe_values(values) -> dict:
    array = numeric_values(values)

    if array.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
            "iqr": math.nan,
        }

    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "iqr": float(
            np.percentile(array, 75)
            - np.percentile(array, 25)
        ),
    }


def column_values(frame: pd.DataFrame, name: str):
    if name not in frame.columns:
        return []
    return frame[name]


def metric_value(results: dict, key: str) -> float:
    try:
        return float(results.get(key, math.nan))
    except (TypeError, ValueError):
        return math.nan


# ============================================================
# 4. LABEL PARSING
# ============================================================

def parse_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    keypoint_count: int,
) -> list[dict]:
    if not label_path.is_file():
        if ALLOW_MISSING_LABEL_FILES:
            return []

        raise FileNotFoundError(
            f"Label file not found:\n{label_path}\n"
            "Use an empty label file for a negative image."
        )

    ground_truth = []
    expected_fields = 5 + 3 * keypoint_count

    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        fields = raw_line.split()

        if len(fields) != expected_fields:
            raise ValueError(
                f"{label_path}, line {line_number}: "
                f"expected {expected_fields} fields, "
                f"found {len(fields)}."
            )

        values = np.asarray(
            [float(value) for value in fields],
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite label value: "
                f"{label_path}, line {line_number}"
            )

        class_value = values[0]

        if class_value < 0 or class_value != int(class_value):
            raise ValueError(
                f"Invalid class ID: {label_path}, "
                f"line {line_number}"
            )

        cx, cy, width, height = values[1:5]

        if (
            np.any(values[1:5] < 0)
            or np.any(values[1:5] > 1)
            or width <= 0
            or height <= 0
        ):
            raise ValueError(
                f"Invalid normalized box: {label_path}, "
                f"line {line_number}"
            )

        box = np.array(
            [
                (cx - width / 2) * image_width,
                (cy - height / 2) * image_height,
                (cx + width / 2) * image_width,
                (cy + height / 2) * image_height,
            ],
            dtype=float,
        )

        pose = values[5:].reshape(keypoint_count, 3)
        visibility_values = pose[:, 2]

        if not np.all(
            np.isin(visibility_values, [0, 1, 2])
        ):
            raise ValueError(
                f"Visibility must be 0, 1, or 2: "
                f"{label_path}, line {line_number}"
            )

        visibility = visibility_values.astype(int)
        annotated = visibility > 0

        if np.any(
            (pose[annotated, :2] < 0)
            | (pose[annotated, :2] > 1)
        ):
            raise ValueError(
                f"Annotated keypoint outside normalized range: "
                f"{label_path}, line {line_number}"
            )

        keypoints = np.full(
            (keypoint_count, 2),
            np.nan,
            dtype=float,
        )

        keypoints[annotated] = (
            pose[annotated, :2]
            * np.array([image_width, image_height])
        )

        ground_truth.append(
            {
                "class_id": int(class_value),
                "box_xyxy": box,
                "keypoints_xy": keypoints,
                "visibility": visibility,
                "line_number": line_number,
            }
        )

    return ground_truth


# ============================================================
# 5. GEOMETRY AND MATCHING
# ============================================================

def box_iou_xyxy(box_a, box_b) -> float:
    top_left = np.maximum(box_a[:2], box_b[:2])
    bottom_right = np.minimum(box_a[2:], box_b[2:])

    intersection_size = np.maximum(
        bottom_right - top_left,
        0.0,
    )

    intersection = float(np.prod(intersection_size))

    area_a = float(
        np.prod(np.maximum(box_a[2:] - box_a[:2], 0.0))
    )
    area_b = float(
        np.prod(np.maximum(box_b[2:] - box_b[:2], 0.0))
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def greedy_class_consistent_match(
    ground_truth: list[dict],
    predictions: list[dict],
) -> list[tuple[int, int, float]]:
    candidates = []

    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            if gt["class_id"] != pred["class_id"]:
                continue

            overlap = box_iou_xyxy(
                gt["box_xyxy"],
                pred["box_xyxy"],
            )

            if overlap >= MATCH_IOU_THRESHOLD:
                candidates.append(
                    (overlap, gt_index, pred_index)
                )

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_gt = set()
    used_predictions = set()
    matches = []

    for overlap, gt_index, pred_index in candidates:
        if (
            gt_index in used_gt
            or pred_index in used_predictions
        ):
            continue

        used_gt.add(gt_index)
        used_predictions.add(pred_index)

        matches.append(
            (gt_index, pred_index, overlap)
        )

    return matches


def valid_predicted_point(
    point,
    image_width: int,
    image_height: int,
) -> bool:
    point = np.asarray(point)

    return bool(
        point.shape == (2,)
        and np.all(np.isfinite(point))
        and 0 <= point[0] < image_width
        and 0 <= point[1] < image_height
    )


def derived_distance(
    gt: dict,
    pred: dict,
    first_index: int,
    second_index: int,
    image_width: int,
    image_height: int,
    vertical_only: bool,
) -> tuple[float, float, float]:
    indices = (first_index, second_index)

    if not all(gt["visibility"][index] > 0 for index in indices):
        return math.nan, math.nan, math.nan

    if not all(
        valid_predicted_point(
            pred["keypoints_xy"][index],
            image_width,
            image_height,
        )
        for index in indices
    ):
        return math.nan, math.nan, math.nan

    gt_delta = (
        gt["keypoints_xy"][first_index]
        - gt["keypoints_xy"][second_index]
    )

    pred_delta = (
        pred["keypoints_xy"][first_index]
        - pred["keypoints_xy"][second_index]
    )

    if vertical_only:
        gt_distance = float(abs(gt_delta[1]))
        pred_distance = float(abs(pred_delta[1]))
    else:
        gt_distance = float(np.linalg.norm(gt_delta))
        pred_distance = float(np.linalg.norm(pred_delta))

    return (
        gt_distance,
        pred_distance,
        abs(pred_distance - gt_distance),
    )


# ============================================================
# 6. PREDICTION EXTRACTION
# ============================================================

def extract_predictions(
    result,
    keypoint_count: int,
) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy().astype(float)
    confidence = result.boxes.conf.cpu().numpy().astype(float)
    classes = result.boxes.cls.cpu().numpy().astype(int)

    pose_xy = np.empty((len(boxes), 0, 2), dtype=float)
    pose_conf = np.empty((len(boxes), 0), dtype=float)

    if result.keypoints is not None:
        pose_xy = result.keypoints.xy.cpu().numpy().astype(float)

        if pose_xy.shape[1] != keypoint_count:
            raise ValueError(
                f"Model predicts {pose_xy.shape[1]} keypoints, "
                f"but data.yaml specifies {keypoint_count}."
            )

        if result.keypoints.conf is not None:
            pose_conf = (
                result.keypoints.conf.cpu().numpy().astype(float)
            )

    predictions = []

    for index, box in enumerate(boxes):
        if (
            not np.all(np.isfinite(box))
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            continue

        keypoints = np.full(
            (keypoint_count, 2),
            np.nan,
            dtype=float,
        )

        keypoint_confidence = np.full(
            keypoint_count,
            np.nan,
            dtype=float,
        )

        if index < pose_xy.shape[0] and pose_xy.shape[1]:
            keypoints[:] = pose_xy[index]

        if index < pose_conf.shape[0] and pose_conf.shape[1]:
            keypoint_confidence[:] = pose_conf[index]

        predictions.append(
            {
                "class_id": int(classes[index]),
                "confidence": float(confidence[index]),
                "box_xyxy": box,
                "keypoints_xy": keypoints,
                "keypoint_confidence": keypoint_confidence,
                "prediction_index": index,
            }
        )

    return predictions


# ============================================================
# 7. MATCHED LANDMARK SCORING
# ============================================================

def score_match(
    gt: dict,
    pred: dict,
    image_width: int,
    image_height: int,
) -> dict:
    diagonal = float(
        np.linalg.norm(
            gt["box_xyxy"][2:] - gt["box_xyxy"][:2]
        )
    )

    threshold_px = (
        diagonal
        * PCK_THRESHOLD_PERCENT_BOX_DIAGONAL
        / 100.0
    )

    row = {
        "gt_box_diagonal_px": diagonal,
        "pck_5pct_threshold_px": threshold_px,
    }

    all_annotated = True
    all_scored = True
    all_correct = True
    all_confident = True

    for index in REQUIRED_KEYPOINTS:
        name = KEYPOINT_NAMES[index]
        annotated = bool(gt["visibility"][index] > 0)

        point = pred["keypoints_xy"][index]
        confidence = pred["keypoint_confidence"][index]

        valid = valid_predicted_point(
            point,
            image_width,
            image_height,
        )

        confident = bool(
            valid
            and np.isfinite(confidence)
            and confidence >= KEYPOINT_CONFIDENCE_THRESHOLD
        )

        row.update(
            {
                f"{name}_gt_visible": int(annotated),
                f"{name}_pred_valid": int(valid),
                f"{name}_passes_confidence_gate": int(confident),
                f"{name}_gt_x": gt["keypoints_xy"][index, 0],
                f"{name}_gt_y": gt["keypoints_xy"][index, 1],
                f"{name}_pred_x": point[0],
                f"{name}_pred_y": point[1],
                f"{name}_pred_conf": confidence,
            }
        )

        error = math.nan
        normalized_error = math.nan
        conditional_pck = math.nan

        # PCK over annotated matched landmarks counts missing/invalid
        # predictions as failures, rather than excluding them.
        annotated_pck = math.nan

        if annotated:
            annotated_pck = 0

            if valid:
                error = float(
                    np.linalg.norm(
                        point - gt["keypoints_xy"][index]
                    )
                )

                normalized_error = rate_or_nan(
                    100.0 * error,
                    diagonal,
                )

                conditional_pck = int(error <= threshold_px)
                annotated_pck = conditional_pck

        row[f"{name}_error_px"] = error
        row[f"{name}_error_percent_box_diagonal"] = normalized_error
        row[f"{name}_pck_5pct"] = conditional_pck
        row[f"{name}_pck_5pct_annotated"] = annotated_pck

        all_annotated = all_annotated and annotated
        all_scored = all_scored and annotated and valid
        all_correct = (
            all_correct
            and annotated
            and valid
            and conditional_pck == 1
        )
        all_confident = all_confident and confident

    row["all_4_gt_annotated"] = int(all_annotated)
    row["all_4_keypoints_scored"] = int(all_scored)

    row["all_4_keypoints_pck_5pct"] = (
        int(all_correct) if all_scored else math.nan
    )

    row["all_4_keypoints_pck_5pct_annotated"] = (
        int(all_correct) if all_annotated else math.nan
    )

    row["all_4_pass_confidence_gate"] = int(all_confident)

    distance_specs = (
        (
            "standard_length",
            HEAD,
            TAIL_BASE,
            False,
            "standard_length_distance_error_px",
        ),
        (
            "body_height",
            UPPER,
            LOWER,
            True,
            "body_height_distance_error_px",
        ),
        (
            "thickness_proxy",
            UPPER,
            LOWER,
            False,
            "thickness_proxy_distance_error_px",
        ),
    )

    for prefix, first, second, vertical_only, error_column in distance_specs:
        gt_distance, pred_distance, error = derived_distance(
            gt,
            pred,
            first,
            second,
            image_width,
            image_height,
            vertical_only,
        )

        row[f"gt_{prefix}_px"] = gt_distance
        row[f"pred_{prefix}_px"] = pred_distance
        row[error_column] = error

    return row


# ============================================================
# 8. EVALUATE M0
# ============================================================

def evaluate_m0(
    test_images: list[Path],
    output_dir: Path,
    keypoint_count: int,
    checkpoint_hash: str,
) -> dict:
    diagnostic_dir = output_dir / "diagnostic_images"

    if SAVE_DIAGNOSTIC_IMAGES:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_WEIGHTS), task="pose")

    validation = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=VALIDATION_BATCH,
        conf=MAP_CONFIDENCE,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        project=str(output_dir),
        name="ultralytics_test_validation",
        exist_ok=False,
        verbose=True,
    )

    validation_results = dict(validation.results_dict)

    prediction_rows = []
    matched_rows = []
    image_rows = []

    total_gt = 0
    total_predictions = 0
    total_matches = 0

    for image_index, image_path in enumerate(test_images, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Cannot read image:\n{image_path}"
            )

        image_height, image_width = image.shape[:2]
        label_path = image_to_label_path(image_path)

        ground_truth = parse_label_file(
            label_path,
            image_width,
            image_height,
            keypoint_count,
        )

        result = model.predict(
            source=image,
            imgsz=IMG_SIZE,
            conf=PREDICTION_CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            device=DEVICE,
            verbose=False,
        )[0]

        predictions = extract_predictions(
            result,
            keypoint_count,
        )

        matches = greedy_class_consistent_match(
            ground_truth,
            predictions,
        )

        total_gt += len(ground_truth)
        total_predictions += len(predictions)
        total_matches += len(matches)

        image_rows.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "image_width": image_width,
                "image_height": image_height,
                "n_gt_fish": len(ground_truth),
                "n_predictions": len(predictions),
                "n_matches": len(matches),
                "n_unmatched_gt": len(ground_truth) - len(matches),
                "n_unmatched_predictions": (
                    len(predictions) - len(matches)
                ),
            }
        )

        for pred in predictions:
            row = {
                "model": MODEL_NAME,
                "weights_path": str(MODEL_WEIGHTS),
                "checkpoint_sha256": checkpoint_hash,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "image_width": image_width,
                "image_height": image_height,
                "prediction_index": pred["prediction_index"],
                "class_id": pred["class_id"],
                "confidence": pred["confidence"],
            }

            for coordinate, value in zip(
                ("x1", "y1", "x2", "y2"),
                pred["box_xyxy"],
            ):
                row[f"box_{coordinate}"] = value

            for index in range(keypoint_count):
                row[f"kp{index}_x"] = pred["keypoints_xy"][index, 0]
                row[f"kp{index}_y"] = pred["keypoints_xy"][index, 1]
                row[f"kp{index}_conf"] = (
                    pred["keypoint_confidence"][index]
                )

            prediction_rows.append(row)

        for gt_index, pred_index, overlap in matches:
            gt = ground_truth[gt_index]
            pred = predictions[pred_index]

            row = {
                "model": MODEL_NAME,
                "weights_path": str(MODEL_WEIGHTS),
                "checkpoint_sha256": checkpoint_hash,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "image_width": image_width,
                "image_height": image_height,
                "gt_index": gt_index,
                "gt_label_line": gt["line_number"],
                "pred_index": pred["prediction_index"],
                "class_id": gt["class_id"],
                "prediction_confidence": pred["confidence"],
                "match_iou": overlap,
            }

            row.update(
                score_match(
                    gt,
                    pred,
                    image_width,
                    image_height,
                )
            )

            matched_rows.append(row)

        if (
            SAVE_DIAGNOSTIC_IMAGES
            and image_index <= N_DIAGNOSTIC_IMAGES
        ):
            diagnostic_path = (
                diagnostic_dir
                / f"{image_index:04d}_{image_path.stem}.jpg"
            )

            if not cv2.imwrite(
                str(diagnostic_path),
                result.plot(),
            ):
                raise RuntimeError(
                    f"Cannot save diagnostic:\n{diagnostic_path}"
                )

        if image_index % 25 == 0 or image_index == len(test_images):
            print(
                f"M0: {image_index}/{len(test_images)} images | "
                f"GT={total_gt} | predictions={total_predictions} | "
                f"matched={total_matches}"
            )

    prediction_df = pd.DataFrame(prediction_rows)
    matched_df = pd.DataFrame(matched_rows)
    image_df = pd.DataFrame(image_rows)

    prediction_csv = output_dir / "m0_raw_predictions.csv"
    matched_csv = output_dir / "m0_matched_keypoint_errors.csv"
    image_csv = output_dir / "m0_per_image_summary.csv"

    # Keep an identifiable header even when no predictions/matches exist.
    if prediction_df.empty:
        prediction_df = pd.DataFrame(
            columns=["model", "image_path", "prediction_index"]
        )

    if matched_df.empty:
        matched_df = pd.DataFrame(
            columns=["model", "image_path", "gt_index", "pred_index"]
        )

    prediction_df.to_csv(
        prediction_csv,
        index=False,
        encoding="utf-8-sig",
    )

    matched_df.to_csv(
        matched_csv,
        index=False,
        encoding="utf-8-sig",
    )

    image_df.to_csv(
        image_csv,
        index=False,
        encoding="utf-8-sig",
    )

    preprocess_ms = float(
        validation.speed.get("preprocess", math.nan)
    )
    inference_ms = float(
        validation.speed.get("inference", math.nan)
    )
    postprocess_ms = float(
        validation.speed.get("postprocess", math.nan)
    )

    total_latency_ms = (
        preprocess_ms + inference_ms + postprocess_ms
    )

    matched_precision = rate_or_nan(
        total_matches,
        total_predictions,
    )
    matched_recall = rate_or_nan(total_matches, total_gt)

    summary = {
        "model": MODEL_NAME,
        "weights": str(MODEL_WEIGHTS),
        "checkpoint_sha256": checkpoint_hash,
        "status": "COMPLETED",
        "error": "",
        "data_yaml": str(DATA_YAML),
        "data_yaml_sha256": sha256_file(DATA_YAML),
        "test_images_dir": str(TEST_IMAGES_DIR),
        "test_labels_dir": str(TEST_LABELS_DIR),
        "split": "test",
        "imgsz": IMG_SIZE,
        "total_label_keypoints": keypoint_count,
        "conf_for_map": MAP_CONFIDENCE,
        "prediction_conf_for_keypoint_accuracy": (
            PREDICTION_CONFIDENCE_THRESHOLD
        ),
        "keypoint_confidence_threshold": (
            KEYPOINT_CONFIDENCE_THRESHOLD
        ),
        "iou": IOU_THRESHOLD,
        "match_iou": MATCH_IOU_THRESHOLD,
        "pck_threshold_percent_box_diagonal": (
            PCK_THRESHOLD_PERCENT_BOX_DIAGONAL
        ),
        "max_det": MAX_DETECTIONS,
        "device": str(DEVICE),
        "validation_batch": VALIDATION_BATCH,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "n_test_images": len(test_images),
        "n_gt_fish": total_gt,
        "n_predicted_fish_at_operating_threshold": total_predictions,
        "n_iou_matched_fish": total_matches,
        "matched_gt_rate": matched_recall,
        "matched_prediction_precision": matched_precision,
        "matched_detection_f1": f1_score(
            matched_precision,
            matched_recall,
        ),
        "raw_prediction_csv": str(prediction_csv),
        "matched_error_csv": str(matched_csv),
        "per_image_csv": str(image_csv),
        "diagnostic_dir": str(diagnostic_dir),
        "model_size_bytes": MODEL_WEIGHTS.stat().st_size,
        "model_size_mb": MODEL_WEIGHTS.stat().st_size / 1024**2,
        "model_parameters": sum(
            parameter.numel()
            for parameter in model.model.parameters()
        ),
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_latency_ms": total_latency_ms,
        "fps": (
            1000.0 / total_latency_ms
            if total_latency_ms > 0
            else math.nan
        ),
        "speed_scope": (
            "Ultralytics validation stage timings; "
            "not full video-pipeline throughput"
        ),
        "physical_measurement_units": (
            "pixels only; no centimetre calibration applied"
        ),
    }

    for prefix, suffix in (("box", "B"), ("pose", "P")):
        for name, key in (
            ("precision", "precision"),
            ("recall", "recall"),
            ("map50", "mAP50"),
            ("map50_95", "mAP50-95"),
        ):
            summary[f"{prefix}_{name}"] = metric_value(
                validation_results,
                f"metrics/{key}({suffix})",
            )

        summary[f"{prefix}_f1"] = f1_score(
            summary[f"{prefix}_precision"],
            summary[f"{prefix}_recall"],
        )

    all_errors = []

    for name in KEYPOINT_NAMES.values():
        errors = column_values(
            matched_df,
            f"{name}_error_px",
        )

        stats = describe_values(errors)
        all_errors.extend(numeric_values(errors).tolist())

        summary[f"{name}_n_scored"] = stats["n"]
        summary[f"{name}_mean_error_px"] = stats["mean"]
        summary[f"{name}_median_error_px"] = stats["median"]
        summary[f"{name}_iqr_error_px"] = stats["iqr"]
        summary[f"{name}_p95_error_px"] = stats["p95"]

        summary[f"{name}_median_error_pct_box_diagonal"] = (
            describe_values(
                column_values(
                    matched_df,
                    f"{name}_error_percent_box_diagonal",
                )
            )["median"]
        )

        # Legacy conditional PCK: valid predictions only.
        summary[f"{name}_pck_5pct_box_diagonal"] = (
            describe_values(
                column_values(
                    matched_df,
                    f"{name}_pck_5pct",
                )
            )["mean"]
        )

        # Includes invalid/missing predictions as failures,
        # conditional on a matched fish and an annotated GT landmark.
        summary[f"{name}_pck_5pct_annotated_matched"] = (
            describe_values(
                column_values(
                    matched_df,
                    f"{name}_pck_5pct_annotated",
                )
            )["mean"]
        )

    summary["overall_keypoint_mae_px"] = (
        describe_values(all_errors)["mean"]
    )

    summary["all_4_keypoints_scored_n"] = int(
        numeric_values(
            column_values(
                matched_df,
                "all_4_keypoints_scored",
            )
        ).sum()
    )

    summary["all_4_keypoints_pck_5pct_rate_scored"] = (
        describe_values(
            column_values(
                matched_df,
                "all_4_keypoints_pck_5pct",
            )
        )["mean"]
    )

    summary["all_4_keypoints_pck_5pct_rate_annotated_matched"] = (
        describe_values(
            column_values(
                matched_df,
                "all_4_keypoints_pck_5pct_annotated",
            )
        )["mean"]
    )

    confidence_passes = int(
        numeric_values(
            column_values(
                matched_df,
                "all_4_pass_confidence_gate",
            )
        ).sum()
    )

    summary["all_4_confidence_gate_pass_n"] = confidence_passes
    summary["all_4_confidence_gate_pass_rate_matched"] = (
        rate_or_nan(confidence_passes, total_matches)
    )

    for prefix in (
        "standard_length",
        "body_height",
        "thickness_proxy",
    ):
        # Standard length and body height use side-view fish.
        # The thickness proxy uses top-view fish.
        expected_class = (
            CLASS_FISH_TOP
            if prefix == "thickness_proxy"
            else CLASS_FISH_SIDE
        )

        if matched_df.empty:
            relevant = matched_df
        else:
            relevant = matched_df.loc[
                matched_df["class_id"] == expected_class
            ]

        stats = describe_values(
            column_values(
                relevant,
                f"{prefix}_distance_error_px",
            )
        )

        summary[f"{prefix}_distance_class_id"] = expected_class
        summary[f"{prefix}_distance_n_scored"] = stats["n"]

        for statistic, label in (
            ("mean", "mean"),
            ("median", "median"),
            ("iqr", "iqr"),
            ("p95", "p95"),
        ):
            summary[
                f"{prefix}_distance_{label}_absolute_error_px"
            ] = stats[statistic]

    return summary


# ============================================================
# 9. MAIN
# ============================================================

def main() -> None:
    require_path(PROJECT_DIR, "Project directory")
    require_path(DATA_YAML, "Dataset YAML")
    require_path(TEST_IMAGES_DIR, "Test image directory")
    require_path(TEST_LABELS_DIR, "Test label directory")
    require_path(MODEL_WEIGHTS, "M0 checkpoint")

    if not MODEL_WEIGHTS.is_file():
        raise ValueError(
            f"Expected a .pt checkpoint file:\n{MODEL_WEIGHTS}"
        )

    keypoint_count, _ = read_keypoint_shape()
    test_images = find_test_images()

    # Parse labels before expensive model validation starts.
    for image_path in test_images:
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Cannot read image:\n{image_path}"
            )

        height, width = image.shape[:2]

        parse_label_file(
            image_to_label_path(image_path),
            width,
            height,
            keypoint_count,
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = OUTPUT_ROOT / f"{MODEL_NAME}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_hash = sha256_file(MODEL_WEIGHTS)

    manifest_path = output_dir / "m0_checkpoint_manifest.csv"
    summary_path = output_dir / "m0_evaluation_summary.csv"

    pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "weights_path": str(MODEL_WEIGHTS),
                "checkpoint_sha256": checkpoint_hash,
                "file_size_bytes": MODEL_WEIGHTS.stat().st_size,
                "data_yaml": str(DATA_YAML),
                "data_yaml_sha256": sha256_file(DATA_YAML),
                "run_id": run_id,
            }
        ]
    ).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("M0 BASELINE POSE EVALUATION")
    print("=" * 72)
    print(f"Model: {MODEL_NAME}")
    print(f"Weights: {MODEL_WEIGHTS}")
    print(f"SHA-256: {checkpoint_hash}")
    print(f"Test images: {len(test_images)}")
    print(f"Label keypoints: {keypoint_count}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Maximum detections: {MAX_DETECTIONS}")
    print(f"Output directory: {output_dir}")

    try:
        summary = evaluate_m0(
            test_images,
            output_dir,
            keypoint_count,
            checkpoint_hash,
        )

    except Exception as exc:
        summary = {
            "model": MODEL_NAME,
            "weights": str(MODEL_WEIGHTS),
            "checkpoint_sha256": checkpoint_hash,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }

        pd.DataFrame([summary]).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        (output_dir / "error_traceback.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )

        print(f"\nEvaluation failed. Details saved to:\n{output_dir}")
        raise

    pd.DataFrame([summary]).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("FINAL M0 RESULTS")
    print("=" * 72)

    print("\nBOUNDING-BOX METRICS")
    print(f"Precision: {summary['box_precision']:.4f}")
    print(f"Recall:    {summary['box_recall']:.4f}")
    print(f"F1:        {summary['box_f1']:.4f}")
    print(f"mAP50:     {summary['box_map50']:.4f}")
    print(f"mAP50-95:  {summary['box_map50_95']:.4f}")

    print("\nPOSE METRICS")
    print(f"Precision: {summary['pose_precision']:.4f}")
    print(f"Recall:    {summary['pose_recall']:.4f}")
    print(f"F1:        {summary['pose_f1']:.4f}")
    print(f"mAP50:     {summary['pose_map50']:.4f}")
    print(f"mAP50-95:  {summary['pose_map50_95']:.4f}")

    print("\nOPERATING-THRESHOLD DETECTION MATCHING")
    print(
        f"Matched fish: {summary['n_iou_matched_fish']} / "
        f"{summary['n_gt_fish']} GT fish"
    )
    print(
        "Matched prediction precision: "
        f"{summary['matched_prediction_precision']:.4f}"
    )
    print(
        f"Matched GT recall: {summary['matched_gt_rate']:.4f}"
    )

    print("\nREQUIRED LANDMARKS")
    for name in KEYPOINT_NAMES.values():
        print(
            f"{name}: "
            f"median error="
            f"{summary[f'{name}_median_error_px']:.3f} px | "
            f"PCK@5% annotated/matched="
            f"{summary[f'{name}_pck_5pct_annotated_matched']:.4f}"
        )

    print(
        "\nOverall scored landmark MAE: "
        f"{summary['overall_keypoint_mae_px']:.3f} px"
    )
    print(
        "Four-landmark confidence acceptance among matched fish: "
        f"{summary['all_4_confidence_gate_pass_rate_matched']:.4f}"
    )

    print("\nVALIDATION SPEED")
    print(f"Inference: {summary['inference_ms']:.3f} ms/image")
    print(
        "Preprocess + inference + postprocess: "
        f"{summary['total_latency_ms']:.3f} ms/image"
    )
    print(
        "Reciprocal stage-time FPS: "
        f"{summary['fps']:.3f}"
    )

    print("\nEvaluation complete.")
    print(f"Summary:\n{summary_path}")
    print(f"Checkpoint manifest:\n{manifest_path}")
    print(f"Detailed outputs:\n{output_dir}")


if __name__ == "__main__":
    main()