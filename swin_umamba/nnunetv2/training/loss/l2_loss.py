import torch
from torch import nn, Tensor
import numpy as np

#Source : https://github.com/Phyrise/nnUNet_translation/blob/b07d03cd456762455877a2c0005fba0eb7fec982/nnunetv2/training/loss/mse.py
class L2(nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        #print("TARGET", len(target))
        print(input.shape , target.shape)
        return super().forward(input[target > 0.5], target[target > 0.5]) 