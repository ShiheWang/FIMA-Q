import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from quantizers._ste import *


class Log2Quantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1)
        self.inited = False
        self.drop_prob = 1.0
        self.channel_wise = channel_wise
        self.training_mode = False

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False
        
    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited
        if self.training_mode and self.drop_prob < 1.0:
            x_orig = x
        scaled_x = (x / self.scale).clamp(min=1e-15, max=1.0)
        x_quant = round_ste(-scaled_x.log2()) if self.training_mode else torch.round(-scaled_x.log2())
        mask = x_quant < 2 * self.n_levels
        x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
        x_dequant = 2 ** (-1 * x_quant) * self.scale
        x_dequant = x_dequant * mask
        if self.training_mode and self.drop_prob < 1.0:
            x_prob = torch.where(torch.rand_like(x) < self.drop_prob, x_dequant, x_orig)
            return x_prob
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise})'

