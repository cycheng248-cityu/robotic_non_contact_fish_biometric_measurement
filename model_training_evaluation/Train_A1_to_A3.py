# -*- coding: utf-8 -*-
"""
Automatically train four YOLO11n-pose schedule configurations.

Fixed for all runs:
- Alpha-CIoU + Wing loss.
- Corrected dataset.
- Initial pretrained YOLO11n-pose checkpoint.
- Loss gains.
- Augmentation settings.
- Seed.

Run this file normally or press Run in your IDE.
No command-line arguments are required.

Keep fish_fixed_losses.py in the same folder.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import inspect
import json
import math
import random
import time
import traceback

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import ultralytics

from ultralytics import YOLO
from ultralytics.utils.loss import BboxLoss

from fish_fixed_losses import (
    ALPHA,
    WING_W,
    WING_EPSILON,
    FixedFishPoseLoss,
)


# ============================================================
# 1. USER SETTINGS
# ============================================================

PROJECT_DIR = Path(
    r"D:\2nd_project_AUV_deepwater_fish_size_estimation"
)

DATA_YAML = PROJECT_DIR / "data" / "data.yaml"

OUTPUT_ROOT = PROJECT_DIR / "runs_AdamW"

PRETRAINED_MODEL = "yolo11n-pose.pt"

# One seed gives four training runs.
# Change to [0, 1, 2] for twelve runs.
SEEDS = [0]

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# A failed run is recorded and the next schedule is attempted.
CONTINUE_AFTER_FAILURE = True

# Confirm the loss names/formula before starting.
LOSS_DESCRIPTION = "alpha-CIoU(alpha=2.0) + Wing(w=10, epsilon=2)"


# ============================================================
# 2. FOUR TRAINING SCHEDULES
# ============================================================

SCHEDULES = {

    "A1_1280_AdamW_linear_250_lr001": {
        "imgsz": 1280,
        "batch": 8,
        "optimizer": "AdamW",
        "epochs": 150,
        "cos_lr": False,
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 5.0,
        "patience": 50,
    },

    "A2_1280_AdamW_cosine_250_lr001": {
        "imgsz": 1280,
        "batch": 8,
        "optimizer": "AdamW",
        "epochs": 250,
        "cos_lr": False,
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 5.0,
        "patience": 50,
    },

    "A3_1280_AdamW_linear_250_lr0001": {
        "imgsz": 1280,
        "batch": 8,
        "optimizer": "AdamW",
        "epochs": 250,
        "cos_lr": True,
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 5.0,
        "patience": 50,
    },
}


# ============================================================
# 3. COMMON TRAINING SETTINGS
# ============================================================

COMMON_ARGS = {
    "task": "pose",
    "multi_scale": False,

    "nbs": 64,
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # Fixed loss gains for every schedule.
    # Wing and OKS have different scales; do not interpret these gains
    # as demonstrating that their contributions are balanced.
    "box": 9.0,
    "pose": 12.0,
    "kobj": 3.0,
    "cls": 0.5,
    "dfl": 1.5,

    # Moderate mosaic, followed by unmixed final epochs.
    "mosaic": 0.25,
    "close_mosaic": 30,
    "mixup": 0.0,
    "copy_paste": 0.0,

    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.20,
    "shear": 0.0,
    "perspective": 0.0,

    "fliplr": 0.5,
    "flipud": 0.0,

    "hsv_h": 0.015,
    "hsv_s": 0.25,
    "hsv_v": 0.25,

    "workers": 0,
    "cache": False,
    "deterministic": True,
    "amp": True,

    "val": True,
    "save": True,
    "save_period": 10,
    "plots": True,
    "verbose": True,
    "exist_ok": False,
}


# ============================================================
# 4. BASIC HELPERS
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path, content):
    Path(path).write_text(
        json.dumps(content, indent=2, default=str),
        encoding="utf-8",
    )


def write_manifest(path, rows):
    if not rows:
        return

    fields = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 5. PREFLIGHT CHECKS
# ============================================================

def check_loss_api():
    expected = [
        "self",
        "pred_dist",
        "pred_bboxes",
        "anchor_points",
        "target_bboxes",
        "target_scores",
        "target_scores_sum",
        "fg_mask",
    ]

    actual = list(
        inspect.signature(BboxLoss.forward).parameters
    )

    if actual != expected:
        raise RuntimeError(
            "The installed Ultralytics BboxLoss API differs from "
            "the interface supported by this script.\n"
            f"Ultralytics version: {ultralytics.__version__}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}\n"
            "Do not start the long training runs until the custom "
            "loss adapter has been checked for this version."
        )


def check_dataset():
    if not PROJECT_DIR.is_dir():
        raise FileNotFoundError(PROJECT_DIR)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(DATA_YAML)

    with DATA_YAML.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("Invalid dataset YAML.")

    for split in ("train", "val"):
        if not data.get(split):
            raise ValueError(
                f"Dataset YAML is missing '{split}'."
            )

    shape = data.get("kpt_shape")

    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or int(shape[0]) < 4
        or int(shape[1]) != 3
    ):
        raise ValueError(
            "Expected at least four keypoints with x/y/visibility, "
            "for example kpt_shape: [5, 3]."
        )

    count = int(shape[0])

    # Identity mapping is appropriate only for anatomical landmark IDs
    # that remain the same under horizontal reflection.
    expected_flip = list(range(count))
    flip_idx = data.get("flip_idx")

    if COMMON_ARGS["fliplr"] > 0:
        if flip_idx is None:
            raise ValueError(
                "Horizontal pose augmentation requires flip_idx.\n"
                "For anatomical indices head/tail-base/upper/lower/"
                "middle, verify whether the identity mapping is "
                f"appropriate, then add flip_idx: {expected_flip} "
                "to data.yaml.\n"
                "Alternatively set fliplr=0.0."
            )

        if list(flip_idx) != expected_flip:
            raise ValueError(
                f"Found flip_idx={flip_idx}.\n"
                "This script assumes anatomical landmark IDs that "
                "are unchanged by horizontal reflection.\n"
                "Verify your annotation convention before proceeding."
            )

    return data


# ============================================================
# 6. INSTALL AND VERIFY THE CUSTOM LOSS
# ============================================================

def install_fixed_loss(trainer):
    """
    on_train_start executes after model/device setup.
    A new loss object is created for each independent run.
    """

    model = trainer.model

    if hasattr(model, "module"):
        model = model.module

    model.criterion = FixedFishPoseLoss(model)

    print("\n" + "=" * 72)
    print("CUSTOM LOSS INSTALLED")
    print(f"Criterion: {type(model.criterion).__name__}")
    print(f"Box: {type(model.criterion.bbox_loss).__name__}")
    print(f"Pose: {type(model.criterion.keypoint_loss).__name__}")
    print("=" * 72)

    write_json(
        Path(trainer.save_dir) / "fixed_loss_configuration.json",
        {
            "loss": LOSS_DESCRIPTION,
            "alpha": ALPHA,
            "wing_w": WING_W,
            "wing_epsilon": WING_EPSILON,
            "box_gain": trainer.args.box,
            "pose_gain": trainer.args.pose,
            "kobj_gain": trainer.args.kobj,
            "ultralytics_version": ultralytics.__version__,
        },
    )


def verify_loss_calls(trainer):
    """
    Check actual custom-component execution after each training epoch.
    Stops training if no foreground reaches either component.
    """

    model = trainer.model

    if hasattr(model, "module"):
        model = model.module

    criterion = getattr(model, "criterion", None)

    if not isinstance(criterion, FixedFishPoseLoss):
        raise RuntimeError(
            "The intended custom criterion is no longer installed."
        )

    box_calls = criterion.bbox_loss.forward_calls
    wing_calls = criterion.keypoint_loss.forward_calls

    if box_calls == 0 or wing_calls == 0:
        raise RuntimeError(
            "Custom loss verification failed: "
            f"box calls={box_calls}, Wing calls={wing_calls}. "
            "Check the installed loss API and dataset annotations."
        )

    loss_items = getattr(trainer, "loss_items", None)

    if (
        loss_items is not None
        and not bool(torch.isfinite(loss_items).all())
    ):
        raise RuntimeError(
            "Non-finite training loss detected."
        )

    if trainer.epoch == 0:
        print(
            "\nCustom loss execution verified: "
            f"box calls={box_calls}, Wing calls={wing_calls}.\n"
        )

    write_json(
        Path(trainer.save_dir) / "loss_execution_check.json",
        {
            "epoch": int(trainer.epoch) + 1,
            "box_forward_calls": int(box_calls),
            "wing_forward_calls": int(wing_calls),
        },
    )


# ============================================================
# 7. RUN ALL FOUR SCHEDULES
# ============================================================

def main():
    check_loss_api()
    dataset = check_dataset()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    experiment_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    experiment_dir = (
        OUTPUT_ROOT / f"fixed_loss_schedules_{experiment_id}"
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest_path = experiment_dir / "run_manifest.csv"
    rows = []

    write_json(
        experiment_dir / "experiment_configuration.json",
        {
            "created": datetime.now().isoformat(),
            "loss": LOSS_DESCRIPTION,
            "pretrained_model": PRETRAINED_MODEL,
            "data_yaml": str(DATA_YAML),
            "data_yaml_sha256": sha256_file(DATA_YAML),
            "dataset_configuration": dataset,
            "schedules": SCHEDULES,
            "common_arguments": COMMON_ARGS,
            "seeds": SEEDS,
            "device": DEVICE,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
        },
    )

    print("\n" + "=" * 72)
    print("AUTOMATIC FOUR-SCHEDULE TRAINING")
    print("=" * 72)
    print(f"Dataset: {DATA_YAML}")
    print(f"Loss: {LOSS_DESCRIPTION}")
    print(f"Device: {DEVICE}")
    print(f"Total runs: {len(SCHEDULES) * len(SEEDS)}")
    print(f"Output: {experiment_dir}")

    if DEVICE == "cpu":
        print(
            "\nWARNING: CUDA is unavailable. "
            "These training runs may be very slow on CPU."
        )

    for schedule_name, schedule_args in SCHEDULES.items():
        for seed in SEEDS:
            clean_memory()
            set_seed(seed)

            run_name = f"{schedule_name}_seed{seed}"
            started = time.perf_counter()

            wrapper = None
            actual_save_dir = experiment_dir / run_name

            row = {
                "schedule": schedule_name,
                "seed": seed,
                "status": "RUNNING",
                "error": "",
                **schedule_args,
                "mosaic": COMMON_ARGS["mosaic"],
                "close_mosaic": COMMON_ARGS["close_mosaic"],
                "loss": LOSS_DESCRIPTION,
                "run_directory": str(actual_save_dir),
                "best_weights": "",
                "best_sha256": "",
                "last_weights": "",
                "elapsed_hours": math.nan,
            }

            rows.append(row)
            write_manifest(manifest_path, rows)

            print("\n" + "=" * 72)
            print(f"STARTING: {run_name}")
            print("=" * 72)

            try:
                # Each run starts from the same pretrained checkpoint.
                # No schedule resumes or fine-tunes another schedule.
                wrapper = YOLO(
                    PRETRAINED_MODEL,
                    task="pose",
                )

                wrapper.add_callback(
                    "on_train_start",
                    install_fixed_loss,
                )

                wrapper.add_callback(
                    "on_train_epoch_end",
                    verify_loss_calls,
                )

                train_args = dict(COMMON_ARGS)
                train_args.update(schedule_args)

                train_args.update(
                    {
                        "data": str(DATA_YAML),
                        "device": DEVICE,
                        "seed": seed,
                        "project": str(experiment_dir),
                        "name": run_name,
                        "resume": False,
                    }
                )

                wrapper.train(**train_args)

                actual_save_dir = Path(
                    wrapper.trainer.save_dir
                )

                best_path = (
                    actual_save_dir / "weights" / "best.pt"
                )

                last_path = (
                    actual_save_dir / "weights" / "last.pt"
                )

                if not best_path.is_file():
                    raise RuntimeError(
                        f"Training returned without best.pt: "
                        f"{best_path}"
                    )

                row.update(
                    {
                        "status": "COMPLETED",
                        "run_directory": str(actual_save_dir),
                        "best_weights": str(best_path),
                        "best_sha256": sha256_file(best_path),
                        "last_weights": str(last_path),
                    }
                )

                print(f"\nCOMPLETED: {run_name}")
                print(f"Best checkpoint:\n{best_path}")

            except Exception as exc:
                row.update(
                    {
                        "status": "FAILED",
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )

                error_path = (
                    experiment_dir
                    / f"{run_name}_error.txt"
                )

                error_path.write_text(
                    traceback.format_exc(),
                    encoding="utf-8",
                )

                print(f"\nFAILED: {run_name}")
                print(row["error"])

                if not CONTINUE_AFTER_FAILURE:
                    raise

            finally:
                row["elapsed_hours"] = (
                    time.perf_counter() - started
                ) / 3600.0

                write_manifest(manifest_path, rows)

                if wrapper is not None:
                    del wrapper

                clean_memory()

    completed = sum(
        row["status"] == "COMPLETED"
        for row in rows
    )

    failed = sum(
        row["status"] == "FAILED"
        for row in rows
    )

    print("\n" + "=" * 72)
    print("FOUR-SCHEDULE EXPERIMENT FINISHED")
    print("=" * 72)
    print(f"Completed runs: {completed}")
    print(f"Failed runs: {failed}")
    print(f"Manifest:\n{manifest_path}")
    print(f"Results:\n{experiment_dir}")


if __name__ == "__main__":
    main()