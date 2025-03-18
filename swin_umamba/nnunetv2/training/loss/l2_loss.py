import torch
from torch import nn, Tensor
import numpy as np

#Source : https://github.com/Phyrise/nnUNet_translation/blob/b07d03cd456762455877a2c0005fba0eb7fec982/nnunetv2/training/loss/mse.py

class L2(nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Compute per-element MSE loss
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(input, target)

        # Create a mask that is 1 where target < 0.1, 0 elsewhere
        mask = (target < 0.1).float()

        # Apply mask element-wise and sum the losses
        masked_loss = loss * mask
        total_loss = masked_loss.sum()

        # Count the number of selected elements to avoid dividing by zero
        non_zero_elements = mask.sum()
        mse_loss_val = total_loss / (non_zero_elements + 1e-8)  # small constant to avoid division by zero

        return mse_loss_val
