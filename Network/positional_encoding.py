import torch
import torch.nn as nn
import math

class PositionalEncoding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        self.register_buffer('div_term', div_term)
        
    def forward(self, x):
        """x: [batch, depth, height, width, channels]"""
        b, d, h, w, _ = x.shape
        device = x.device
        
        # Create position encodings for each dimension
        pos_d = torch.arange(d, device=device).float() / d
        pos_h = torch.arange(h, device=device).float() / h
        pos_w = torch.arange(w, device=device).float() / w
        
        # Create grid
        grid_d, grid_h, grid_w = torch.meshgrid(pos_d, pos_h, pos_w, indexing='ij')
        
        # Compute positional encodings
        pe = torch.zeros(d, h, w, self.dim, device=device)
        pe[..., 0::2] = torch.sin(grid_d.unsqueeze(-1) * self.div_term)
        pe[..., 1::2] = torch.cos(grid_w.unsqueeze(-1) * self.div_term)
        pe = pe.permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, d, h, w]
        
        return x + pe
