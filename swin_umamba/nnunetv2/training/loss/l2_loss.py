import torch
from torch import nn, Tensor
import numpy as np

#Source : https://github.com/Phyrise/nnUNet_translation/blob/b07d03cd456762455877a2c0005fba0eb7fec982/nnunetv2/training/loss/mse.py
class L2(nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(input, target)
        mask = np.ones_like(target)[target < 0.1]
        loss = (loss * mask.float()).sum() 
        non_zero_elements = mask.sum()
        mse_loss_val = loss / non_zero_elements
        #print("TARGET", len(target))
        print(input.shape , target.shape, loss.shape)
        return mse_loss_val  #super().forward(input[target > 0.5], target[target > 0.5]) 