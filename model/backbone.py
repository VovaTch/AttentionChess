from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from .chess_conv_attention import hollow_chess_kernel

class ChessFeatureExpander(nn.Module):
    """
    Convolutional expanding chess block for backbone for attachess.
    """
    
    def __init__(self, hidden_size) -> nn.Module:
        
        super().__init__()
        
        self.conv_expand = nn.ModuleList([
            nn.ConvTranspose2d(hidden_size * 2, hidden_size * 2, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(hidden_size * 2, hidden_size * 4, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        ])
        self.conv_process = nn.ModuleList([
            ResConv2dBlock(hidden_size * 2, 5, 1),
            ResConv2dBlock(hidden_size * 2, 3, 2)
        ])
        
    def forward(self, x):
        for conv_idx, (conv_e, conv_p) in enumerate(zip(self.conv_expand, self.conv_process)):
            x = F.gelu(conv_p(x))
            x = F.gelu(conv_e(x)) if conv_idx < len(self.conv_expand) - 1 else conv_e(x)
            
        return x
            
        
class ResConv2dBlock(nn.Module):
    """
    Sub residual block for ChessFeatureExpander
    """
    
    def __init__(self, num_kernels: int, num_conv_layers: int, dim_multiplier: int) -> nn.Module:
        super().__init__()
        self.conv_block = nn.ModuleList()
        
        for layer_idx in range(num_conv_layers):
            
            if layer_idx > 0:
                self.conv_block.append(nn.GELU())
                self.conv_block.append(nn.LayerNorm([num_kernels, 8 * dim_multiplier, 8 * dim_multiplier]))
            self.conv_block.append(nn.Conv2d(num_kernels, num_kernels, kernel_size=(3, 3), padding=1))
            
    def forward(self, x):
        x_bypass = x
        for layer in self.conv_block:
            x = layer(x)
        x += x_bypass
        return x

class ResChessConvBlock(nn.Module):
    """
    Sub residual block for ChessFeatureExpander
    """
    
    def __init__(self, num_kernels: int, num_conv_layers: int, dim_multiplier: int) -> nn.Module:
        super().__init__()
        self.conv_block = nn.ModuleList([])
        
        for layer_idx in range(num_conv_layers):
            
            if layer_idx > 0:
                self.conv_block.append(nn.GELU())
                self.conv_block.append(nn.LayerNorm([num_kernels, 8 * dim_multiplier, 8 * dim_multiplier]))
            self.conv_block.append(nn.Conv2d(num_kernels, 8 * dim_multiplier, kernel_size=(15, 15), padding=7))
            hollow_chess_kernel(self.conv_block[-1].weight)
            
    def forward(self, x):
        x_bypass = x
        for layer in self.conv_block:
            x = layer(x)
        x += x_bypass
        return x
        