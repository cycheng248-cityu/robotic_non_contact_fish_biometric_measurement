#!/usr/bin/env python
"""TOP-CAMERA HOMOGRAPHY CALIBRATION USING MANY RULER MIDPOINTS.

Use this script with CALIBRATION screenshots only.
For each composite screenshot, you click the midpoint of a ruler segment in the
TOP panel, then provide that midpoint's physically measured tank coordinate.
The script fits a homography: cropped top-panel pixel (u,v) -> tank (X,Y) cm.

Important:
- Use at least 6 non-collinear calibration placements around the measurement area.
- These points are used to FIT the homography. Do NOT use them later as the
  independent end-to-end ruler-validation set.
- This script does NOT calculate AUV pixel scale or physical ruler length.

Composite layout:
  Top:   x=0:960,    y=181:901  (960x720)
  Left:  x=960:1920, y=0:540    (960x540)
  Right: x=960:1920, y=540:1080 (960x540)

Requirements:
    pip install opencv-python numpy pandas openpyxl
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd

# =====================================================================
# 1. EDIT THESE PATHS
# =====================================================================

PROJECT_DIR = Path(r"D:/2nd_project_AUV_deepwater_fish_size_estimation")

# Put only HOMOGRAPHY-FITTING composite screenshots in this folder.
INPUT_DIR = PROJECT_DIR / "calibration" / "homography_fit_inputs"

OUTPUT_DIR = PROJECT_DIR / "calibration" / "top_camera_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed AUV camera lens/port coordinate in tank plane, centimetres.
AUV_CAMERA_XY_CM = np.array([30.0, 13.0], dtype=np.float32)

# =====================================================================
# 2. YOUR FIXED COMPOSITE LAYOUT
# =====================================================================

TOP_X1, TOP_X2 = 0, 1280
TOP_Y1, TOP_Y2 = 0, 720

# =====================================================================
# 3. HOMOGRAPHY-FIT TRIALS
# =====================================================================

# For each screenshot, write the TRUE physical tank coordinate (X,Y) in cm of
# the midpoint of the exact ruler segment you will click in the top panel.
# Use at least 6 points and make sure they are spread around the target region,
# not all in one straight horizontal or vertical line.
FIT_TRIALS = [
    # {
    #     "trial_id": "fit_01",
    #     "filename": "fit_01.png",
    #     "true_x_cm": 18.0,
    #     "true_y_cm": 10.0,
    # },
    {
        "trial_id": "fit_02",
        "filename": "top_calibraion_640.png",
        "true_x_cm": 30.0,
        "true_y_cm": 13.0,
    },
    {
        "trial_id": "fit_03",
        "filename": "top_calibraion_640.png",
        "true_x_cm": 0.0,
        "true_y_cm": 0.0,
    },
    {
        "trial_id": "fit_04",
        "filename": "top_calibraion_640.png",
        "true_x_cm": 40.0,
        "true_y_cm": 0.0,
    },
    {
        "trial_id": "fit_05",
        "filename": "top_calibraion_640.png",
        "true_x_cm": 0.0,
        "true_y_cm": 27.0,
    },
    {
        "trial_id": "fit_06",
        "filename": "top_calibraion_640.png",
        "true_x_cm": 40.0,
        "true_y_cm": 27.0,
    },
]

# =====================================================================
# 4. CLICK FUNCTION
# =====================================================================


def click_one_point(image: np.ndarray, title: str) -> tuple[np.ndarray, np.ndarray]:
    display = image.copy()
    clicked: list[list[float]] = []

    print("Left-click the physical midpoint of the selected ruler segment.")
    print("Press r to reset. Press q after one point has been clicked.")

    def callback(event, x, y, flags, param):
        nonlocal display, clicked
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(clicked) >= 1:
            return

        clicked.append([float(x), float(y)])
        cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(display, "midpoint", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        print(f"Clicked: ({x}, {y}) px")

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1200, 800)
    cv2.setMouseCallback(title, callback)

    while True:
        cv2.imshow(title, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            display = image.copy()
            clicked = []
            print("Reset.")
        elif key == ord("q"):
            break

    cv2.destroyWindow(title)

    if len(clicked) != 1:
        raise RuntimeError("Exactly one midpoint must be clicked before pressing q.")

    return np.array(clicked, dtype=np.float32), display


def transform_points(points_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points_px.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)

# =====================================================================
# 5. MAIN
# =====================================================================


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    print("=" * 76)
    print("TOP-CAMERA HOMOGRAPHY FIT FROM RULER MIDPOINTS")
    print("=" * 76)
    print(f"Put fit images here: {INPUT_DIR}")
    print("Use different images later for independent end-to-end validation.")

    for trial in FIT_TRIALS:
        image_path = INPUT_DIR / trial["filename"]
        if not image_path.is_file():
            print(f"\n[SKIPPED] missing: {image_path}")
            continue

        composite = cv2.imread(str(image_path))
        if composite is None:
            print(f"\n[SKIPPED] unreadable: {image_path}")
            continue

        h, w = composite.shape[:2]
        if (w, h) != (1920, 1080):
            print(f"[WARNING] {trial['trial_id']}: image is {w}x{h}; expected 1920x1080.")

        top_panel = composite[TOP_Y1:TOP_Y2, TOP_X1:TOP_X2].copy()
        if top_panel.size == 0:
            raise RuntimeError("Top-panel crop is empty; check ROI coordinates.")

        print("\n" + "-" * 76)
        print(f"Trial: {trial['trial_id']}")
        print(f"True midpoint tank coordinate: X={trial['true_x_cm']:.2f}, Y={trial['true_y_cm']:.2f} cm")

        try:
            clicked_point, clicked_image = click_one_point(top_panel, f"Top homography fit: {trial['trial_id']}")
            cv2.imwrite(str(OUTPUT_DIR / f"{trial['trial_id']}_top_midpoint_clicked.jpg"), clicked_image)

            rows.append({
                "trial_id": trial["trial_id"],
                "composite_image": str(image_path),
                "top_midpoint_u_px": float(clicked_point[0, 0]),
                "top_midpoint_v_px": float(clicked_point[0, 1]),
                "true_x_cm": float(trial["true_x_cm"]),
                "true_y_cm": float(trial["true_y_cm"]),
            })
        except Exception as exc:
            print(f"[FAILED] {trial['trial_id']}: {exc}")

    if len(rows) < 4:
        raise RuntimeError("At least four valid, non-collinear fitting points are required.")

    df = pd.DataFrame(rows)
    image_points = df[["top_midpoint_u_px", "top_midpoint_v_px"]].to_numpy(dtype=np.float32)
    tank_points = df[["true_x_cm", "true_y_cm"]].to_numpy(dtype=np.float32)

    # RANSAC can reject an accidental bad click; refine using inliers only.
    H_initial, mask = cv2.findHomography(image_points, tank_points, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if H_initial is None:
        raise RuntimeError("Could not fit the initial homography.")

    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(df), dtype=bool)
    if inlier_mask.sum() < 4:
        raise RuntimeError("Fewer than four inliers remained after RANSAC. Recheck clicked points and true coordinates.")

    H, _ = cv2.findHomography(image_points[inlier_mask], tank_points[inlier_mask], method=0)
    if H is None:
        raise RuntimeError("Could not refine homography using RANSAC inliers.")

    estimated = transform_points(image_points, H)
    errors = np.linalg.norm(estimated - tank_points, axis=1)

    df["homography_inlier"] = inlier_mask
    df["estimated_x_cm"] = estimated[:, 0]
    df["estimated_y_cm"] = estimated[:, 1]
    df["fit_position_error_cm"] = errors

    # Range relative to fixed AUV, for diagnostic use only.
    true_xy = tank_points.astype(float)
    estimated_xy = estimated.astype(float)
    true_ranges = np.linalg.norm(true_xy - AUV_CAMERA_XY_CM, axis=1)
    estimated_ranges = np.linalg.norm(estimated_xy - AUV_CAMERA_XY_CM, axis=1)
    df["true_auv_horizontal_range_cm"] = true_ranges
    df["estimated_auv_horizontal_range_cm"] = estimated_ranges
    df["fit_range_error_cm"] = np.abs(estimated_ranges - true_ranges)

    np.savez(
        OUTPUT_DIR / "top_camera_to_tank_homography.npz",
        homography_image_to_tank_cm=H,
        auv_camera_xy_cm=AUV_CAMERA_XY_CM,
        fit_image_points_px=image_points,
        fit_tank_points_cm=tank_points,
        fit_inlier_mask=inlier_mask,
    )

    df.to_csv(OUTPUT_DIR / "top_homography_fit_points.csv", index=False, encoding="utf-8-sig")
    df.to_excel(OUTPUT_DIR / "top_homography_fit_points.xlsx", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_input_points": int(len(df)),
        "n_ransac_inliers": int(inlier_mask.sum()),
        "median_fit_position_error_cm": float(np.median(errors)),
        "maximum_fit_position_error_cm": float(np.max(errors)),
        "median_fit_range_error_cm": float(np.median(np.abs(estimated_ranges - true_ranges))),
        "maximum_fit_range_error_cm": float(np.max(np.abs(estimated_ranges - true_ranges))),
        "auv_camera_xy_cm": AUV_CAMERA_XY_CM.tolist(),
        "homography_image_to_tank_cm": H.tolist(),
    }
    with open(OUTPUT_DIR / "top_homography_fit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 76)
    print("HOMOGRAPHY FIT COMPLETE")
    print("=" * 76)
    print(f"Inliers: {inlier_mask.sum()}/{len(df)}")
    print(f"Median fit error: {np.median(errors):.3f} cm")
    print(f"Maximum fit error: {np.max(errors):.3f} cm")
    print(f"Saved homography: {OUTPUT_DIR / 'top_camera_to_tank_homography.npz'}")
    print("Use a DIFFERENT image set with end_to_end_ruler_validation for independent validation.")


if __name__ == "__main__":
    main()
