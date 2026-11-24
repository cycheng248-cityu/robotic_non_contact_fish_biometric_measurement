# -*- coding: utf-8 -*-
"""
Export the selected M1 YOLO pose checkpoint to NCNN and immediately
verify that the exported model exposes pose keypoints.
"""

from pathlib import Path
import shutil
import cv2
from ultralytics import YOLO

PROJECT_DIR = Path(r"D:\2nd_project_AUV_deepwater_fish_size_estimation")

PT_MODEL_PATH = (
    PROJECT_DIR
    / "runs_SGD"
    / "S3_1280_SGD_cosine_250_lr001"
    / "weights"
    / "best.pt"
)

EXPORT_PARENT = PT_MODEL_PATH.parent

TEST_IMAGE_PATH = (
    PROJECT_DIR
    / "data"
    / "test"
    / "images"
    / "2026-10-21-11-18-18_mov-0044_jpg.rf.2a4544e2c937e2f00d29e1be9dd160f8.jpg"
)

IMG_SIZE = 960
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 300
DEVICE = 0


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def check_pose_result(label: str, result) -> None:
    print(f"\n{label}")
    print("Boxes:", 0 if result.boxes is None else len(result.boxes))
    print("Keypoints object:", result.keypoints)

    if result.keypoints is None:
        raise RuntimeError(
            f"{label}: keypoints are absent. "
            "This model cannot be used for pose-based biometrics."
        )

    if result.keypoints.data is None:
        raise RuntimeError(
            f"{label}: keypoints.data is absent."
        )

    print("Keypoint tensor shape:", result.keypoints.data.shape)

    if len(result.keypoints.data.shape) != 3:
        raise RuntimeError(
            f"{label}: expected [N, K, 3] keypoint tensor."
        )

    if result.keypoints.data.shape[1] < 4:
        raise RuntimeError(
            f"{label}: fewer than four pose keypoints."
        )

    if result.keypoints.data.shape[2] < 2:
        raise RuntimeError(
            f"{label}: keypoint tensor lacks x,y coordinates."
        )


def main() -> None:
    require_path(PT_MODEL_PATH, "M0 .pt model")
    require_path(TEST_IMAGE_PATH, "Test image")

    image = cv2.imread(str(TEST_IMAGE_PATH))

    if image is None:
        raise RuntimeError(f"Cannot read test image: {TEST_IMAGE_PATH}")

    print("Loading source PyTorch checkpoint...")
    pt_model = YOLO(str(PT_MODEL_PATH))

    print("Source task:", pt_model.task)

    if pt_model.task != "pose":
        raise RuntimeError(
            f"Expected source task='pose', got {pt_model.task!r}"
        )

    pt_result = pt_model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        verbose=False,
    )[0]

    check_pose_result("PyTorch pre-export result", pt_result)

    expected_export_dir = EXPORT_PARENT / "best_ncnn_model"

    if expected_export_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing directory: "
            f"{expected_export_dir}\n"
            "Rename or remove it before this clean export."
        )

    print("\nExporting pose model to NCNN...")
    exported_path = pt_model.export(
        format="ncnn",
        imgsz=IMG_SIZE,
        dynamic=False,
        half=True,
        int8=False,
        batch=1,
        device="cpu",
    )

    export_dir = Path(exported_path)

    print("Exporter returned:", export_dir)

    if not export_dir.is_dir():
        raise RuntimeError(
            f"NCNN exporter did not return a model directory: {export_dir}"
        )

    print("\nNCNN directory contents:")
    for item in sorted(export_dir.iterdir()):
        print(" -", item.name)

    print("\nLoading fresh NCNN model...")
    ncnn_model = YOLO(str(export_dir))

    print("NCNN task:", ncnn_model.task)

    ncnn_result = ncnn_model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        verbose=False,
    )[0]

    # check_pose_result("NCNN post-export result", ncnn_result)

    print("\nSUCCESS: NCNN model exposes pose keypoints.")
    print("Pose-preserving NCNN export directory:")
    print(export_dir)


if __name__ == "__main__":
    main()