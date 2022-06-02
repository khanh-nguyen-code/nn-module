from typing import List

import torch
from torch import nn

from package.nn_module import Module


class ResNet(Module):
    def __init__(self, *block_list: nn.Module):
        super().__init__()
        self.block_list = block_list
        for i, block in enumerate(block_list):
            self.register_module(f"block_{i}", block)

    def forward(self, y0: torch.Tensor) -> torch.Tensor:
        y = y0
        for block in self.block_list:
            y = y + block.forward(y)
        return y
