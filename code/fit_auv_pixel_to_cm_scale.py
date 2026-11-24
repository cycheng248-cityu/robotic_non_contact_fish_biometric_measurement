#!/usr/bin/env python3
"""Fit AUV pixel-to-centimetre calibration from separate source images.

Each trial uses one TOP source image and one matching underwater source image
of the SAME stationary ruler placement. A configurable 1280x720 region is
cropped from each 1920x1080 source before clicking.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================================================================
# 1. PROJECT PATHS
# =====================================================================

PROJECT_DIR = Path(r"D:/2nd_project_AUV_deepwater_fish_size_estimation")

TOP_INPUT_DIR = (
    PROJECT_DIR / "calibration" / "auv_scale_fit_inputs"
)
LEFT_INPUT_DIR = (
    PROJECT_DIR / "calibration" / "auv_scale_fit_inputs"
)
RIGHT_INPUT_DIR = (
    PROJECT_DIR / "calibration" / "auv_scale_fit_inputs"
)

METRIC_AUV_PANEL = "left"  # Change to "right" for right-camera calibration.

OUTPUT_DIR = (
    PROJECT_DIR
    / "calibration"
    / "auv_scale_fit_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOMOGRAPHY_FILE = (
    PROJECT_DIR
    / "calibration"
    / "top_camera_results"
    / "top_camera_to_tank_homography.npz"
)

# =====================================================================
# 2. SOURCE SIZE AND 1280x720 CROPS
# =====================================================================

SOURCE_WIDTH = 1920
SOURCE_HEIGHT = 1080
CROP_WIDTH = 1280
CROP_HEIGHT = 720

# Format: (x1, y1, x2, y2). Each must produce exactly 1280x720.
# Defaults extract the upper-left 1280x720 region. Edit per camera as needed.
TOP_CROP = (0, 0, 1280, 720)
LEFT_CROP = (0, 0, 1280, 720)
RIGHT_CROP = (0, 0, 1280, 720)

AUV_CAMERA_XY_CM = np.array([30.0, 13.0], dtype=np.float32)

# Keep identical in calibration and the final fish pipeline.
RANGE_MODE = "x_axis"  # "x_axis" or "euclidean"

# =====================================================================
# 3. MATCHED TOP/UNDERWATER TRIALS
# =====================================================================

SCALE_FIT_TRIALS = [
    {
        "trial_id": "scale_fit_02",
        "top_filename": "scale_fit_1280_1_top.png",
        "left_filename": "scale_fit_1280_1_w.png",
        "right_filename": None,
        "known_reference_length_cm": 5.0,
        "true_target_x_cm": 20.0,
        "true_target_y_cm": 13.0,
    },
    {
        "trial_id": "scale_fit_03",
        "top_filename": "scale_fit_1280_2_top.png",
        "left_filename": "scale_fit_1280_2_w.png",
        "right_filename": None,
        "known_reference_length_cm": 5.0,
        "true_target_x_cm": 18.0,
        "true_target_y_cm": 13.0,
    },
    {
        "trial_id": "scale_fit_04",
        "top_filename": "scale_fit_1280_3_top.png",
        "left_filename": "scale_fit_1280_3_w.png",
        "right_filename": None,
        "known_reference_length_cm": 5.0,
        "true_target_x_cm": 16.0,
        "true_target_y_cm": 13.0,
    },
    {
        "trial_id": "scale_fit_05",
        "top_filename": "scale_fit_1280_4_top.png",
        "left_filename": "scale_fit_1280_4_w.png",
        "right_filename": None,
        "known_reference_length_cm": 5.0,
        "true_target_x_cm": 10.0,
        "true_target_y_cm": 13.0,
    },
    {
        "trial_id": "scale_fit_06",
        "top_filename": "scale_fit_1280_5_top.png",
        "left_filename": "scale_fit_1280_5_w.png",
        "right_filename": None,
        "known_reference_length_cm": 5.0,
        "true_target_x_cm": 5.0,
        "true_target_y_cm": 13.0,
    },
]

POLYNOMIAL_DEGREE = 2
MINIMUM_TRIALS_TO_FIT = 5

# =====================================================================
# 4. IMAGE AND CLICK HELPERS
# =====================================================================


def read_and_crop_1280(
    path: Path,
    label: str,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    """Read 1920x1080 and return configured 1280x720 ROI.

    An already-cropped 1280x720 image is also accepted unchanged.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} image: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read {label} image: {path}")

    height, width = image.shape[:2]

    if (width, height) == (CROP_WIDTH, CROP_HEIGHT):
        print(f"{label}: already {width}x{height}; no crop applied.")
        return image.copy()

    if (width, height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
        raise RuntimeError(
            f"{label} is {width}x{height}; expected either "
            f"{SOURCE_WIDTH}x{SOURCE_HEIGHT} or "
            f"{CROP_WIDTH}x{CROP_HEIGHT}: {path}"
        )

    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RuntimeError(
            f"Invalid {label} ROI {roi} for {width}x{height}: {path}"
        )

    cropped = image[y1:y2, x1:x2].copy()
    crop_height, crop_width = cropped.shape[:2]

    if (crop_width, crop_height) != (CROP_WIDTH, CROP_HEIGHT):
        raise RuntimeError(
            f"{label} ROI {roi} produced {crop_width}x{crop_height}; "
            f"expected {CROP_WIDTH}x{CROP_HEIGHT}."
        )

    print(
        f"{label}: cropped {width}x{height} with ROI {roi} "
        f"to {crop_width}x{crop_height}."
    )
    return cropped


def get_metric_path_and_roi(trial: dict) -> tuple[Path, tuple[int, int, int, int]]:
    if METRIC_AUV_PANEL == "left":
        filename = trial.get("left_filename")
        input_dir = LEFT_INPUT_DIR
        roi = LEFT_CROP
    elif METRIC_AUV_PANEL == "right":
        filename = trial.get("right_filename")
        input_dir = RIGHT_INPUT_DIR
        roi = RIGHT_CROP
    else:
        raise ValueError("METRIC_AUV_PANEL must be 'left' or 'right'.")

    if not filename:
        raise ValueError(
            f"{trial['trial_id']} has no {METRIC_AUV_PANEL}_filename."
        )
    return input_dir / filename, roi


def click_points(
    image: np.ndarray,
    n_points: int,
    title: str,
    instructions: str,
    color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    display = image.copy()
    points: list[list[float]] = []

    print("\n" + instructions)
    print("Left-click point(s). Press r to reset. Press q when complete.")

    def callback(event, x, y, flags, param):
        nonlocal display, points
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= n_points:
            return
        points.append([float(x), float(y)])
        index = len(points)
        cv2.circle(display, (x, y), 7, color, -1)
        cv2.putText(
            display,
            str(index),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        print(f"Clicked {index}/{n_points}: ({x}, {y}) px")

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1200, 675)
    cv2.setMouseCallback(title, callback)

    while True:
        cv2.imshow(title, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            display = image.copy()
            points = []
            print("Reset.")
        elif key == ord("q"):
            break

    cv2.destroyWindow(title)

    if len(points) != n_points:
        raise RuntimeError(
            f"Expected {n_points} point(s), received {len(points)}."
        )

    return np.asarray(points, dtype=np.float32), display


def top_pixel_to_tank(point_px: np.ndarray, homography: np.ndarray) -> np.ndarray:
    transformed = cv2.perspectiveTransform(
        point_px.reshape(1, 1, 2).astype(np.float32),
        homography,
    )
    return transformed[0, 0]


def range_to_auv(tank_xy_cm: np.ndarray) -> float:
    if RANGE_MODE == "x_axis":
        return abs(float(AUV_CAMERA_XY_CM[0]) - float(tank_xy_cm[0]))
    if RANGE_MODE == "euclidean":
        return float(np.linalg.norm(tank_xy_cm - AUV_CAMERA_XY_CM))
    raise ValueError("RANGE_MODE must be 'x_axis' or 'euclidean'.")


def save_click_annotation(image: np.ndarray, path: Path, lines: list[str]) -> None:
    annotated = image.copy()
    y = 30
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30

    if not cv2.imwrite(str(path), annotated):
        raise RuntimeError(f"Could not save annotation: {path}")

# =====================================================================
# 5. FIT AND SAVE CALIBRATION
# =====================================================================


def fit_and_save_curve(df: pd.DataFrame) -> None:
    df = df.sort_values("estimated_auv_range_cm").reset_index(drop=True)
    x = df["estimated_auv_range_cm"].to_numpy(dtype=float)
    y = df["observed_cm_per_pixel"].to_numpy(dtype=float)

    if np.unique(np.round(x, 6)).size <= POLYNOMIAL_DEGREE:
        raise RuntimeError("Insufficient distinct ranges for the selected fit degree.")

    coefficients = np.polyfit(x, y, POLYNOMIAL_DEGREE)
    y_fit = np.polyval(coefficients, x)
    residuals = y - y_fit
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2)))

    df["fitted_cm_per_pixel"] = y_fit
    df["fit_residual_cm_per_pixel"] = residuals
    df["fit_residual_percent"] = 100.0 * residuals / y

    trials_csv = OUTPUT_DIR / "auv_scale_fit_trials.csv"
    trials_xlsx = OUTPUT_DIR / "auv_scale_fit_trials.xlsx"
    df.to_csv(trials_csv, index=False, encoding="utf-8-sig")
    df.to_excel(trials_xlsx, index=False)

    lookup = (
        df.assign(range_bin_cm=df["estimated_auv_range_cm"].round(2))
        .groupby("range_bin_cm", as_index=False)
        .agg(
            n=("observed_cm_per_pixel", "size"),
            median_cm_per_pixel=("observed_cm_per_pixel", "median"),
            mean_cm_per_pixel=("observed_cm_per_pixel", "mean"),
            std_cm_per_pixel=("observed_cm_per_pixel", "std"),
        )
        .sort_values("range_bin_cm")
    )
    lookup.to_csv(
        OUTPUT_DIR / "auv_scale_lookup_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lookup.to_excel(OUTPUT_DIR / "auv_scale_lookup_table.xlsx", index=False)

    x_line = np.linspace(float(x.min()), float(x.max()), 300)
    y_line = np.polyval(coefficients, x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=60, label="Calibration measurements")
    plt.plot(
        x_line,
        y_line,
        linewidth=2,
        label=f"Degree {POLYNOMIAL_DEGREE}; R²={r_squared:.4f}",
    )
    plt.xlabel(f"Top-view-estimated range (cm), {RANGE_MODE}")
    plt.ylabel("Conversion scale (cm/pixel)")
    plt.title(
        f"{METRIC_AUV_PANEL.capitalize()} Pixel-to-cm calibration: 1280x720 crop"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "auv_range_to_cm_per_pixel_fit.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(x, df["pixels_per_cm"].to_numpy(dtype=float), s=60)
    plt.xlabel(f"Top-view-estimated range (cm), {RANGE_MODE}")
    plt.ylabel("Conversion scale (cm/pixel)")
    plt.title(
        f"{METRIC_AUV_PANEL.capitalize()} Pixel-to-cm calibration: 1280x720 crop"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "auv_range_to_pixels_per_cm.png", dpi=200)
    plt.close()

    calibration = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_image_size": [SOURCE_WIDTH, SOURCE_HEIGHT],
        "cropped_image_size": [CROP_WIDTH, CROP_HEIGHT],
        "top_crop_xyxy": list(TOP_CROP),
        "left_crop_xyxy": list(LEFT_CROP),
        "right_crop_xyxy": list(RIGHT_CROP),
        "metric_auv_panel": METRIC_AUV_PANEL,
        "range_mode": RANGE_MODE,
        "auv_camera_xy_cm": AUV_CAMERA_XY_CM.tolist(),
        "top_homography_file": str(HOMOGRAPHY_FILE),
        "number_of_trials": int(len(df)),
        "range_min_cm": float(x.min()),
        "range_max_cm": float(x.max()),
        "curve_type": f"polynomial_degree_{POLYNOMIAL_DEGREE}",
        "polynomial_coefficients_highest_power_first": coefficients.tolist(),
        "r_squared": r_squared,
        "rmse_cm_per_pixel": rmse,
        "warning": (
            "Use the same crops and range definition in the final pipeline. "
            "Do not extrapolate beyond the fitted range."
        ),
    }

    with open(
        OUTPUT_DIR / "auv_range_to_scale_calibration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(calibration, file, indent=2)

    np.savez(
        OUTPUT_DIR / "auv_range_to_scale_calibration.npz",
        observed_ranges_cm=x,
        observed_cm_per_pixel=y,
        polynomial_coefficients=coefficients,
        polynomial_degree=POLYNOMIAL_DEGREE,
        range_min_cm=float(x.min()),
        range_max_cm=float(x.max()),
        auv_camera_xy_cm=AUV_CAMERA_XY_CM,
        source_image_size=np.array([SOURCE_WIDTH, SOURCE_HEIGHT], dtype=int),
        cropped_image_size=np.array([CROP_WIDTH, CROP_HEIGHT], dtype=int),
        top_crop_xyxy=np.array(TOP_CROP, dtype=int),
        left_crop_xyxy=np.array(LEFT_CROP, dtype=int),
        right_crop_xyxy=np.array(RIGHT_CROP, dtype=int),
        metric_auv_panel=np.array(METRIC_AUV_PANEL),
        range_mode=np.array(RANGE_MODE),
    )

    print("\n" + "=" * 82)
    print("AUV PIXEL-TO-CENTIMETRE SCALE FIT COMPLETE")
    print("=" * 82)
    print(f"Metric panel: {METRIC_AUV_PANEL}")
    print(f"Range mode: {RANGE_MODE}")
    print(f"Observed range: {x.min():.3f} to {x.max():.3f} cm")
    print(f"Curve coefficients: {coefficients}")
    print(f"R²: {r_squared:.5f}")
    print(f"RMSE: {rmse:.8f} cm/pixel")
    print(f"Trials CSV: {trials_csv}")
    print(f"Calibration JSON: {OUTPUT_DIR / 'auv_range_to_scale_calibration.json'}")
    print("Use different ruler placements for independent validation.")

# =====================================================================
# 6. MAIN
# =====================================================================


def main() -> None:
    if METRIC_AUV_PANEL not in {"left", "right"}:
        raise ValueError("METRIC_AUV_PANEL must be 'left' or 'right'.")
    if RANGE_MODE not in {"x_axis", "euclidean"}:
        raise ValueError("RANGE_MODE must be 'x_axis' or 'euclidean'.")
    if not HOMOGRAPHY_FILE.is_file():
        raise FileNotFoundError(
            f"Top homography not found:\n{HOMOGRAPHY_FILE}\n"
            "Fit the homography using the same TOP_CROP coordinates."
        )

    for directory in (TOP_INPUT_DIR, LEFT_INPUT_DIR, RIGHT_INPUT_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    homography_data = np.load(HOMOGRAPHY_FILE, allow_pickle=False)
    if "homography_image_to_tank_cm" not in homography_data.files:
        raise KeyError("Homography file lacks 'homography_image_to_tank_cm'.")

    homography = homography_data["homography_image_to_tank_cm"]
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        raise RuntimeError("Homography must be a finite 3x3 matrix.")

    print("=" * 82)
    print("FIT AUV SCALE FROM 1280x720 CROPS")
    print("=" * 82)
    print(f"Top directory: {TOP_INPUT_DIR}")
    print(f"Left directory: {LEFT_INPUT_DIR}")
    print(f"Right directory: {RIGHT_INPUT_DIR}")
    print(f"Metric panel: {METRIC_AUV_PANEL}")
    print(f"Top crop: {TOP_CROP}")
    print(f"Left crop: {LEFT_CROP}")
    print(f"Right crop: {RIGHT_CROP}")
    print("Each top/underwater pair must show the same ruler placement.")

    rows: list[dict] = []

    for trial in SCALE_FIT_TRIALS:
        trial_id = trial["trial_id"]
        top_path = TOP_INPUT_DIR / trial["top_filename"]

        try:
            metric_path, metric_roi = get_metric_path_and_roi(trial)
            top = read_and_crop_1280(top_path, "top", TOP_CROP)
            metric = read_and_crop_1280(
                metric_path,
                f"{METRIC_AUV_PANEL} underwater",
                metric_roi,
            )
        except Exception as exc:
            print(f"\n[SKIPPED] {trial_id}: {exc}")
            continue

        cv2.imwrite(str(OUTPUT_DIR / f"{trial_id}_top_crop.png"), top)
        cv2.imwrite(
            str(OUTPUT_DIR / f"{trial_id}_{METRIC_AUV_PANEL}_crop.png"),
            metric,
        )

        print("\n" + "-" * 82)
        print(f"Trial: {trial_id}")
        print(f"Top image: {top_path.name}")
        print(f"Underwater image: {metric_path.name}")
        print(
            f"Known ruler segment: "
            f"{float(trial['known_reference_length_cm']):.3f} cm"
        )

        try:
            top_point, top_clicked = click_points(
                top,
                1,
                f"Top midpoint: {trial_id}",
                "TOP CROP: Click the midpoint of the ruler segment.",
                (0, 255, 0),
            )

            estimated_xy = top_pixel_to_tank(top_point[0], homography)
            estimated_range = range_to_auv(estimated_xy)

            save_click_annotation(
                top_clicked,
                OUTPUT_DIR / f"{trial_id}_top_clicked.png",
                [
                    f"Estimated X={estimated_xy[0]:.2f} cm",
                    f"Estimated Y={estimated_xy[1]:.2f} cm",
                    f"Estimated AUV range={estimated_range:.2f} cm",
                ],
            )

            endpoints, metric_clicked = click_points(
                metric,
                2,
                f"{METRIC_AUV_PANEL.capitalize()} endpoints: {trial_id}",
                (
                    f"UNDERWATER CROP: Click the two endpoints of the "
                    f"{float(trial['known_reference_length_cm']):.3f}-cm segment."
                ),
                (0, 0, 255),
            )

            pixel_length = float(np.linalg.norm(endpoints[0] - endpoints[1]))
            known_length = float(trial["known_reference_length_cm"])

            if not np.isfinite(pixel_length) or pixel_length <= 0:
                raise RuntimeError("The clicked pixel length is zero or invalid.")
            if not np.isfinite(known_length) or known_length <= 0:
                raise ValueError("known_reference_length_cm must be positive.")

            observed_cm_per_pixel = known_length / pixel_length
            pixels_per_cm = pixel_length / known_length

            p1 = tuple(np.rint(endpoints[0]).astype(int))
            p2 = tuple(np.rint(endpoints[1]).astype(int))
            cv2.line(metric_clicked, p1, p2, (0, 255, 0), 2)

            save_click_annotation(
                metric_clicked,
                OUTPUT_DIR
                / f"{trial_id}_{METRIC_AUV_PANEL}_endpoints_clicked.png",
                [
                    f"Range={estimated_range:.2f} cm",
                    f"Pixels={pixel_length:.2f}",
                    f"Scale={observed_cm_per_pixel:.8f} cm/px",
                ],
            )

            true_x = trial.get("true_target_x_cm", np.nan)
            true_y = trial.get("true_target_y_cm", np.nan)

            if pd.notna(true_x) and pd.notna(true_y):
                true_xy = np.array([float(true_x), float(true_y)], dtype=float)
                true_range = range_to_auv(true_xy)
                position_error = float(np.linalg.norm(estimated_xy - true_xy))
                range_error = abs(estimated_range - true_range)
            else:
                true_range = np.nan
                position_error = np.nan
                range_error = np.nan

            rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "trial_id": trial_id,
                    "top_source_image": str(top_path),
                    "underwater_source_image": str(metric_path),
                    "source_width_px": SOURCE_WIDTH,
                    "source_height_px": SOURCE_HEIGHT,
                    "crop_width_px": CROP_WIDTH,
                    "crop_height_px": CROP_HEIGHT,
                    "top_crop_xyxy": str(TOP_CROP),
                    "metric_crop_xyxy": str(metric_roi),
                    "metric_auv_panel": METRIC_AUV_PANEL,
                    "range_mode": RANGE_MODE,
                    "known_reference_length_cm": known_length,
                    "top_midpoint_u_px": float(top_point[0, 0]),
                    "top_midpoint_v_px": float(top_point[0, 1]),
                    "estimated_target_x_cm": float(estimated_xy[0]),
                    "estimated_target_y_cm": float(estimated_xy[1]),
                    "true_target_x_cm": true_x,
                    "true_target_y_cm": true_y,
                    "top_position_error_cm": position_error,
                    "estimated_auv_range_cm": estimated_range,
                    "true_auv_range_cm": true_range,
                    "top_range_error_cm": range_error,
                    "metric_endpoint1_x_px": float(endpoints[0, 0]),
                    "metric_endpoint1_y_px": float(endpoints[0, 1]),
                    "metric_endpoint2_x_px": float(endpoints[1, 0]),
                    "metric_endpoint2_y_px": float(endpoints[1, 1]),
                    "metric_pixel_length_px": pixel_length,
                    "observed_cm_per_pixel": observed_cm_per_pixel,
                    "pixels_per_cm": pixels_per_cm,
                }
            )

            print(
                f"Estimated top location: X={estimated_xy[0]:.3f}, "
                f"Y={estimated_xy[1]:.3f} cm"
            )
            print(f"Estimated AUV range: {estimated_range:.3f} cm")
            print(f"Ruler pixel length: {pixel_length:.3f} px")
            print(
                f"Observed scale: {observed_cm_per_pixel:.8f} cm/px "
                f"({pixels_per_cm:.3f} px/cm)"
            )

        except Exception as exc:
            print(f"[FAILED] {trial_id}: {exc}")

    if len(rows) < MINIMUM_TRIALS_TO_FIT:
        raise RuntimeError(
            f"Only {len(rows)} trials completed. Need at least "
            f"{MINIMUM_TRIALS_TO_FIT}."
        )

    fit_and_save_curve(pd.DataFrame(rows))


if __name__ == "__main__":
    main()