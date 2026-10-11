"""
Loss function for the Attention U-Net flood head: a weighted combination
of binary cross-entropy and Dice loss, per proposal Sec 'Model Training' --
"to handle the class imbalance between flooded and non-flooded pixels."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation. Operates on logits directly
    (applies sigmoid internally) so it composes cleanly with BCEWithLogitsLoss
    in CombinedLoss without double-applying sigmoid.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    """
    weighted_loss = bce_weight * BCE + dice_weight * Dice.

    Defaults to equal weighting (0.5/0.5), a reasonable starting point per
    the proposal's own framing of this as "a weighted combination" without
    specifying exact weights -- tune bce_weight/dice_weight if validation
    metrics show one failure mode dominating (e.g. raise dice_weight if the
    model predicts mostly-empty masks, since Dice penalises that more
    directly than BCE does under heavy class imbalance).
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)
