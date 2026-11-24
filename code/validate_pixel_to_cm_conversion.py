#!/usr/bin/env python
"""Independent validation of AUV pixel-to-centimetre conversion from calibration screenshots.

PURPOSE
-------
Use this BEFORE applying the conversion to new single-fish videos.
For each composite ruler screenshot, you manually click two endpoints of a known
ruler segment in the selected AUV panel. The script calculates the scale needed
to recover that known segment and compares it with a candidate pixel-to-cm
conversion model.

This script is deliberately simple:
- It does NOT run YOLO.
- It does NOT calculate fish length.
- It does NOT require top-camera homography.
- It validates the AUV panel pixel-to-centimetre scale directly.

You must provide one candidate conversion method:
A. Fixed scale: same cm/pixel for every accepted frame, OR
B. Range lookup table: a measured cm/pixel value at each known target range, OR
C. Polynomial curve: cm/pixel = f(range_cm).

Recommended approach:
- Use RANGE_LOOKUP for the first prototype.
- Use different screenshots from those used to fit the calibration lookup/curve.
- Keep ruler flat, straight, and at the same vertical/depth plane as fish.
- Use one AUV panel only (left or right), consistently.

Composite layout:
  Top:   x=0:960,    y=181:901  (960x720)
  Left:  x=960:1920, y=0:540    (960x540)
  Right: x=960:1920, y=540:1080 (960x540)

Requirements:
    pip install opencv-python numpy pandas openpyxl matplotlib
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================================================================
# 1. EDIT PROJECT PATHS
# =====================================================================

PROJECT_DIR = Path(r"D:/2nd_project_AUV_deepwater_fish_size_estimation")

# Put only INDEPENDENT ruler screenshots here.
# Do NOT use screenshots used to construct the candidate scale model.
INPUT_DIR = PROJECT_DIR / "calibration" / "pixel_to_cm_validation_inputs"

OUTPUT_DIR = PROJECT_DIR / "calibration" / "pixel_to_cm_validation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 2. COMPOSITE SCREEN LAYOUT
# =====================================================================

TOP_X1, TOP_X2 = 0, 1280
TOP_Y1, TOP_Y2 = 0, 720

LEFT_X1, LEFT_X2 = 0, 1280
LEFT_Y1, LEFT_Y2 = 0, 720

RIGHT_X1, RIGHT_X2 = 640, 1280
RIGHT_Y1, RIGHT_Y2 = 360, 840

# Use the exact AUV camera panel that will be used for fish length inference.
METRIC_AUV_PANEL = "left"  # "left" or "right"

# =====================================================================
# 3. SELECT THE CANDIDATE CONVERSION YOU WANT TO VALIDATE
# =====================================================================

# Choose ONE: "fixed", "lookup", or "polynomial".
CONVERSION_METHOD = "polynomial"

# Option A: one constant conversion factor for all accepted frames.
FIXED_CM_PER_PIXEL = 0.0120  # Replace if CONVERSION_METHOD = "fixed"

# Option B: piecewise linear interpolation between measured scale points.
# Replace these arrays with YOUR fitted calibration results.
# Values must refer to the same metric AUV panel and same composite/camera setup.
LOOKUP_RANGE_CM = np.array([5.0, 6.5, 8.5, 11.5, 16.0], dtype=float)
LOOKUP_CM_PER_PIXEL = np.array([0.0243, 0.0190, 0.0143, 0.0123, 0.0100], dtype=float)

# Option C: polynomial conversion model.
# Enter coefficients from np.polyfit(range_cm, cm_per_pixel, degree).
# Example: [a, b, c] means a*D^2 + b*D + c.
POLYNOMIAL_COEFFICIENTS = np.array([ -1.3369e-05,   0.0012516,  -0.0056337], dtype=float)

# Whether estimated target range is allowed outside the lookup/fit range.
# Keep False. Do not extrapolate during validation or fish analysis.
ALLOW_EXTRAPOLATION = False

# =====================================================================
# 4. INDEPENDENT RULER VALIDATION TRIALS
# =====================================================================

# For each independent ruler screenshot, record:
# - filename: full 1920x1080 composite screenshot in INPUT_DIR
# - known_reference_length_cm: distance between the two ruler marks you click
# - known_target_range_cm: independently measured camera-to-ruler distance
#
# This script validates pixel-to-cm conversion directly using the known range.
# It does NOT estimate range from the top camera.
VALIDATION_TRIALS = [
    {
        "trial_id": "pixel_scale_validation_01",
        "filename": "scale_fit_1280_1_w.png",
        "known_reference_length_cm": 5.0,
        "known_target_range_cm": 10.0,
    },
    {
        "trial_id": "pixel_scale_validation_02",
        "filename": "scale_fit_1280_2_w.png",
        "known_reference_length_cm": 5.0,
        "known_target_range_cm": 12.0,
    },
    {
        "trial_id": "pixel_scale_validation_03",
        "filename": "scale_fit_1280_3_w.png",
        "known_reference_length_cm": 5.0,
        "known_target_range_cm": 14.0,
    },
    {
        "trial_id": "pixel_scale_validation_04",
        "filename": "scale_fit_1280_4_w.png",
        "known_reference_length_cm": 5.0,
        "known_target_range_cm": 20.0,
    },
    {
        "trial_id": "pixel_scale_validation_05",
        "filename": "scale_fit_1280_5_w.png",
        "known_reference_length_cm": 5.0,
        "known_target_range_cm": 25.0,
    },
]

# =====================================================================
# 5. HELPERS
# =====================================================================


def crop_metric_panel(composite: np.ndarray) -> np.ndarray:
    if METRIC_AUV_PANEL == "left":
        panel = composite[LEFT_Y1:LEFT_Y2, LEFT_X1:LEFT_X2].copy()
    elif METRIC_AUV_PANEL == "right":
        panel = composite[RIGHT_Y1:RIGHT_Y2, RIGHT_X1:RIGHT_X2].copy()
    else:
        raise ValueError("METRIC_AUV_PANEL must be 'left' or 'right'.")

    if panel.size == 0:
        raise RuntimeError("Metric AUV crop is empty. Check ROI coordinates.")
    return panel


def click_endpoints(image: np.ndarray, title: str, known_length_cm: float) -> tuple[np.ndarray, np.ndarray]:
    display = image.copy()
    points: list[list[float]] = []

    print(f"\nClick the two endpoints of the known {known_length_cm:.3f} cm ruler segment.")
    print("Click the same physical ruler marks consistently across trials.")
    print("Press r to reset. Press q only after both points are clicked.")

    def callback(event, x, y, flags, param):
        nonlocal display, points
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 2:
            return
        points.append([float(x), float(y)])
        point_number = len(points)
        cv2.circle(display, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(display, str(point_number), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        print(f"Clicked {point_number}/2: ({x}, {y}) px")

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1200, 800)
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

    if len(points) != 2:
        raise RuntimeError("Exactly two ruler endpoints are required.")

    return np.asarray(points, dtype=np.float32), display


def predicted_scale_cm_per_pixel(range_cm: float) -> float:
    if CONVERSION_METHOD == "fixed":
        return float(FIXED_CM_PER_PIXEL)

    if CONVERSION_METHOD == "lookup":
        lo = float(LOOKUP_RANGE_CM.min())
        hi = float(LOOKUP_RANGE_CM.max())
        if not ALLOW_EXTRAPOLATION and not (lo <= range_cm <= hi):
            raise ValueError(
                f"Known range {range_cm:.3f} cm is outside lookup range {lo:.3f}-{hi:.3f} cm."
            )
        return float(np.interp(range_cm, LOOKUP_RANGE_CM, LOOKUP_CM_PER_PIXEL))

    if CONVERSION_METHOD == "polynomial":
        if len(POLYNOMIAL_COEFFICIENTS) < 2:
            raise ValueError("POLYNOMIAL_COEFFICIENTS must contain at least two values.")
        return float(np.polyval(POLYNOMIAL_COEFFICIENTS, range_cm))

    raise ValueError("CONVERSION_METHOD must be 'fixed', 'lookup', or 'polynomial'.")

# =====================================================================
# 6. MAIN VALIDATION
# =====================================================================


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if METRIC_AUV_PANEL not in {"left", "right"}:
        raise ValueError("METRIC_AUV_PANEL must be 'left' or 'right'.")

    if CONVERSION_METHOD == "lookup":
        if len(LOOKUP_RANGE_CM) != len(LOOKUP_CM_PER_PIXEL):
            raise ValueError("LOOKUP_RANGE_CM and LOOKUP_CM_PER_PIXEL must have the same length.")
        if np.any(np.diff(LOOKUP_RANGE_CM) <= 0):
            raise ValueError("LOOKUP_RANGE_CM must be strictly increasing.")

    print("=" * 78)
    print("INDEPENDENT AUV PIXEL-TO-CENTIMETRE CONVERSION VALIDATION")
    print("=" * 78)
    print(f"Input folder: {INPUT_DIR}")
    print(f"Metric AUV panel: {METRIC_AUV_PANEL}")
    print(f"Conversion method: {CONVERSION_METHOD}")
    if CONVERSION_METHOD == "lookup":
        print(f"Lookup range: {LOOKUP_RANGE_CM.min():.2f}-{LOOKUP_RANGE_CM.max():.2f} cm")

    rows = []

    for trial in VALIDATION_TRIALS:
        image_path = INPUT_DIR / trial["filename"]
        if not image_path.is_file():
            print(f"\n[SKIPPED] Missing: {image_path}")
            continue

        composite = cv2.imread(str(image_path))
        if composite is None:
            print(f"\n[SKIPPED] Cannot read: {image_path}")
            continue

        image_h, image_w = composite.shape[:2]
        if (image_w, image_h) != (1920, 1080):
            print(f"[WARNING] {trial['trial_id']}: image is {image_w}x{image_h}; expected 1920x1080.")

        metric_panel = crop_metric_panel(composite)
        cv2.imwrite(str(OUTPUT_DIR / f"{trial['trial_id']}_{METRIC_AUV_PANEL}_crop.png"), metric_panel)

        known_length_cm = float(trial["known_reference_length_cm"])
        known_range_cm = float(trial["known_target_range_cm"])

        print("\n" + "-" * 78)
        print(f"Trial: {trial['trial_id']}")
        print(f"Known target range: {known_range_cm:.3f} cm")
        print(f"Known ruler segment: {known_length_cm:.3f} cm")

        try:
            endpoints, annotated = click_endpoints(
                metric_panel,
                f"{METRIC_AUV_PANEL.capitalize()} panel: {trial['trial_id']}",
                known_length_cm,
            )

            pixel_length = float(np.linalg.norm(endpoints[0] - endpoints[1]))
            observed_scale = known_length_cm / pixel_length
            model_scale = predicted_scale_cm_per_pixel(known_range_cm)
            estimated_length = pixel_length * model_scale
            signed_error = estimated_length - known_length_cm
            absolute_error = abs(signed_error)
            absolute_percentage_error = 100.0 * absolute_error / known_length_cm
            scale_error_pct = 100.0 * (model_scale - observed_scale) / observed_scale

            p1 = tuple(endpoints[0].astype(int))
            p2 = tuple(endpoints[1].astype(int))
            cv2.line(annotated, p1, p2, (0, 255, 0), 2)
            cv2.putText(annotated, f"Known={known_length_cm:.3f} cm; estimated={estimated_length:.3f} cm", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Error={signed_error:+.3f} cm ({absolute_percentage_error:.2f}% abs)", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(OUTPUT_DIR / f"{trial['trial_id']}_{METRIC_AUV_PANEL}_endpoints_clicked.png"), annotated)

            rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "trial_id": trial["trial_id"],
                "composite_image": str(image_path),
                "metric_auv_panel": METRIC_AUV_PANEL,
                "conversion_method": CONVERSION_METHOD,
                "known_target_range_cm": known_range_cm,
                "known_reference_length_cm": known_length_cm,
                "endpoint1_x_px": float(endpoints[0, 0]),
                "endpoint1_y_px": float(endpoints[0, 1]),
                "endpoint2_x_px": float(endpoints[1, 0]),
                "endpoint2_y_px": float(endpoints[1, 1]),
                "pixel_length_px": pixel_length,
                "observed_cm_per_pixel": observed_scale,
                "candidate_cm_per_pixel": model_scale,
                "scale_error_percent": scale_error_pct,
                "estimated_reference_length_cm": estimated_length,
                "signed_length_error_cm": signed_error,
                "absolute_length_error_cm": absolute_error,
                "absolute_percentage_length_error": absolute_percentage_error,
            })

            print(f"Measured ruler pixels: {pixel_length:.3f} px")
            print(f"Observed scale: {observed_scale:.8f} cm/px")
            print(f"Candidate scale: {model_scale:.8f} cm/px")
            print(f"Estimated length: {estimated_length:.3f} cm")
            print(f"Error: {signed_error:+.3f} cm ({absolute_percentage_error:.2f}% absolute)")

        except Exception as exc:
            print(f"[FAILED] {trial['trial_id']}: {exc}")

    if not rows:
        raise RuntimeError("No validation trials completed. Check paths and trial filenames.")

    df = pd.DataFrame(rows)
    trials_csv = OUTPUT_DIR / "pixel_to_cm_conversion_validation_trials.csv"
    trials_xlsx = OUTPUT_DIR / "pixel_to_cm_conversion_validation_trials.xlsx"
    df.to_csv(trials_csv, index=False, encoding="utf-8-sig")
    df.to_excel(trials_xlsx, index=False)

    abs_cm = df["absolute_length_error_cm"].to_numpy(dtype=float)
    abs_pct = df["absolute_percentage_length_error"].to_numpy(dtype=float)
    signed = df["signed_length_error_cm"].to_numpy(dtype=float)

    summary = pd.DataFrame([{
        "n_independent_validation_trials": len(df),
        "metric_auv_panel": METRIC_AUV_PANEL,
        "conversion_method": CONVERSION_METHOD,
        "median_absolute_length_error_cm": float(np.median(abs_cm)),
        "mean_absolute_length_error_cm": float(np.mean(abs_cm)),
        "maximum_absolute_length_error_cm": float(np.max(abs_cm)),
        "median_absolute_percentage_error": float(np.median(abs_pct)),
        "mean_absolute_percentage_error": float(np.mean(abs_pct)),
        "maximum_absolute_percentage_error": float(np.max(abs_pct)),
        "mean_signed_length_bias_cm": float(np.mean(signed)),
    }])
    summary_csv = OUTPUT_DIR / "pixel_to_cm_conversion_validation_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 5))
    plt.scatter(df["known_target_range_cm"], df["absolute_percentage_length_error"], s=65)
    plt.xlabel("Known camera-to-ruler range (cm)")
    plt.ylabel("Absolute reference-length error (%)")
    plt.title("Independent pixel-to-centimetre conversion validation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pixel_to_cm_validation_error_by_range.png", dpi=200)
    plt.close()

    print("\n" + "=" * 78)
    print("PIXEL-TO-CENTIMETRE VALIDATION COMPLETE")
    print("=" * 78)
    print(summary.to_string(index=False))
    print(f"\nTrial results: {trials_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Error plot: {OUTPUT_DIR / 'pixel_to_cm_validation_error_by_range.png'}")


if __name__ == "__main__":
    main()
