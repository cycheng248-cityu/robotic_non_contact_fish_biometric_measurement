# -*- coding: utf-8 -*-
"""
Fixed alpha-CIoU + Wing loss for Ultralytics YOLO11 pose.

Designed for the v8PoseLoss/BboxLoss API used by Ultralytics 8.3.x.
The training script checks the BboxLoss signature before training.

Preserved from the attached implementation:
- Alpha-CIoU exponent: 2.0
- Wing w: 10.0
- Wing epsilon: 2.0

DFL and the parent pose criterion's classification/keypoint-objectness
terms are retained.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.utils.loss import BboxLoss, v8PoseLoss


ALPHA = 2.0
WING_W = 10.0
WING_EPSILON = 2.0


class AlphaCIoUBboxLoss(BboxLoss):
    """Replace the IoU term while retaining the parent's DFL calculation."""

    def __init__(self, original_loss):
        reg_max = (
            original_loss.dfl_loss.reg_max
            if original_loss.dfl_loss is not None
            else 1
        )

        super().__init__(reg_max=reg_max)

        self.alpha = ALPHA
        self.forward_calls = 0

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        self.forward_calls += 1

        # The parent calculates DFL using the installed library version.
        _, loss_dfl = super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
        )

        predicted = pred_bboxes[fg_mask]
        target = target_bboxes[fg_mask]

        # Perform geometry in float32 for numerical stability under AMP.
        predicted = predicted.float()
        target = target.float()

        eps = 1e-7

        pwh = (predicted[:, 2:] - predicted[:, :2]).clamp_min(eps)
        gwh = (target[:, 2:] - target[:, :2]).clamp_min(eps)

        pc = (predicted[:, :2] + predicted[:, 2:]) / 2.0
        gc = (target[:, :2] + target[:, 2:]) / 2.0

        intersection_wh = (
            torch.minimum(predicted[:, 2:], target[:, 2:])
            - torch.maximum(predicted[:, :2], target[:, :2])
        ).clamp_min(0.0)

        intersection = intersection_wh.prod(dim=-1)

        union = (
            pwh.prod(dim=-1)
            + gwh.prod(dim=-1)
            - intersection
        ).clamp_min(eps)

        iou = (intersection / union).clamp(0.0, 1.0)

        enclosing_wh = (
            torch.maximum(predicted[:, 2:], target[:, 2:])
            - torch.minimum(predicted[:, :2], target[:, :2])
        )

        enclosing_diagonal_sq = (
            enclosing_wh.square().sum(dim=-1).clamp_min(eps)
        )

        centre_distance_sq = (pc - gc).square().sum(dim=-1)

        v = (4.0 / math.pi**2) * (
            torch.atan(gwh[:, 0] / gwh[:, 1])
            - torch.atan(pwh[:, 0] / pwh[:, 1])
        ).square()

        with torch.no_grad():
            ciou_modifier = v / (1.0 - iou + v + eps)

        alpha_ciou_loss = (
            1.0
            - iou.clamp_min(1e-6).pow(self.alpha)
            + (
                centre_distance_sq / enclosing_diagonal_sq
            ).clamp_min(0.0).pow(self.alpha)
            + (ciou_modifier * v).clamp_min(0.0).pow(self.alpha)
        )

        weights = target_scores.sum(dim=-1)[fg_mask].float()

        if torch.is_tensor(target_scores_sum):
            denominator = target_scores_sum.float().clamp_min(eps)
        else:
            denominator = max(float(target_scores_sum), eps)

        loss_box = (
            alpha_ciou_loss * weights
        ).sum() / denominator

        return loss_box, loss_dfl


class WingKeypointLoss(nn.Module):
    """Masked coordinate Wing loss, preserving the supplied formulation."""

    def __init__(self):
        super().__init__()

        self.w = WING_W
        self.epsilon = WING_EPSILON

        self.c = (
            self.w
            - self.w * math.log1p(self.w / self.epsilon)
        )

        self.forward_calls = 0

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        self.forward_calls += 1

        # Coordinates reaching this component are the parent criterion's
        # decoded training coordinates, not original-image pixels.
        difference = (
            pred_kpts[..., :2].float()
            - gt_kpts[..., :2].float()
        ).abs()

        loss = torch.where(
            difference < self.w,
            self.w * torch.log1p(difference / self.epsilon),
            difference - self.c,
        )

        mask = kpt_mask.to(dtype=loss.dtype)

        loss = loss.sum(dim=-1) * mask

        return loss.sum() / mask.sum().clamp_min(1.0)


class FixedFishPoseLoss(v8PoseLoss):
    """Use the parent's pose pipeline with explicit custom components."""

    def __init__(self, model):
        super().__init__(model)

        self.bbox_loss = AlphaCIoUBboxLoss(
            self.bbox_loss
        ).to(self.device)

        self.keypoint_loss = WingKeypointLoss().to(self.device)