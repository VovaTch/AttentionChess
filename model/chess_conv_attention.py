import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Prototype the double attention. Maybe convert to convolutions
class MultiHeadDoubleAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        # chess conv layers
        self.k_conv = nn.Conv2d(d_model, d_model, (15, 15), padding=7)
        hollow_chess_kernel(self.k_conv.weight)
        self.k2_conv = nn.Conv2d(d_model, d_model, (15, 15), padding=7)
        hollow_chess_kernel(self.k2_conv.weight)
        self.q_conv = nn.Conv2d(d_model, d_model, (15, 15), padding=7)
        hollow_chess_kernel(self.q_conv.weight)
        self.q2_conv = nn.Conv2d(d_model, d_model, (15, 15), padding=7)
        hollow_chess_kernel(self.q2_conv.weight)
        self.v_conv = nn.Conv2d(d_model, d_model, (15, 15), padding=7)
        hollow_chess_kernel(self.v_conv.weight)
        self.v2_conv = nn.Conv2d(d_model, d_model, (15, 15), padding=7)
        hollow_chess_kernel(self.v2_conv.weight)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()

    def forward(self, q, k, v, mask=None):
        """Dimensions of q,k,v in this method: bs x 8 x 8 x8"""
        batch_size = q.size(0)

        # perform chess conv operation and split into h heads for keys
        k = self.k_conv(k.permute(0, 3, 1, 2))
        k = self.relu(k)
        k = self.k_conv(k).permute(0, 2, 3, 1).flatten(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k)

        # perform chess conv operation and split into h heads for queries
        q = self.q_conv(q.permute(0, 3, 1, 2))
        q = self.relu(q)
        q = self.k_conv(q).permute(0, 2, 3, 1).flatten(1, 2)
        q = q.view(batch_size, -1, self.h, self.d_k)

        # perform chess conv operation and split into h heads for values
        v = self.v_conv(v.permute(0, 3, 1, 2))
        v = self.relu(v)
        v = self.v_conv(v).permute(0, 2, 3, 1).flatten(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out(concat)
        output = output.view(batch_size, 8, 8, self.d_model)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


def hollow_chess_kernel(weights_r):
    """Takes a 15x15 kernel and makes it a chess kernel"""

    with torch.no_grad():
        weights_r[:, :, 1:7, 0] = 0
        weights_r[:, :, 8:14, 0] = 0
        weights_r[:, :, 2:7, 1] = 0
        weights_r[:, :, 8:13, 1] = 0
        weights_r[:, :, 3:7, 2] = 0
        weights_r[:, :, 8:12, 2] = 0
        weights_r[:, :, 4:7, 3] = 0
        weights_r[:, :, 8:11, 3] = 0
        weights_r[:, :, 5:7, 4] = 0
        weights_r[:, :, 8:10, 4] = 0

        weights_r[:, :, 0, 1:7] = 0
        weights_r[:, :, 0, 8:14] = 0
        weights_r[:, :, 1, 2:7] = 0
        weights_r[:, :, 1, 8:13] = 0
        weights_r[:, :, 2, 3:7] = 0
        weights_r[:, :, 2, 8:12] = 0
        weights_r[:, :, 3, 4:7] = 0
        weights_r[:, :, 3, 8:11] = 0
        weights_r[:, :, 4, 5:7] = 0
        weights_r[:, :, 4, 8:10] = 0

        weights_r[:, :, 1:7, 14] = 0
        weights_r[:, :, 8:14, 14] = 0
        weights_r[:, :, 2:7, 13] = 0
        weights_r[:, :, 8:13, 13] = 0
        weights_r[:, :, 3:7, 12] = 0
        weights_r[:, :, 8:12, 12] = 0
        weights_r[:, :, 4:7, 11] = 0
        weights_r[:, :, 8:11, 11] = 0
        weights_r[:, :, 5:7, 10] = 0
        weights_r[:, :, 8:10, 10] = 0

        weights_r[:, :, 14, 1:7] = 0
        weights_r[:, :, 14, 8:14] = 0
        weights_r[:, :, 13, 2:7] = 0
        weights_r[:, :, 13, 8:13] = 0
        weights_r[:, :, 12, 3:7] = 0
        weights_r[:, :, 12, 8:12] = 0
        weights_r[:, :, 11, 4:7] = 0
        weights_r[:, :, 11, 8:11] = 0
        weights_r[:, :, 10, 5:7] = 0
        weights_r[:, :, 10, 8:10] = 0


def create_chess_kernel(out_channels=8, in_channels=8):
    """Creates the kernel with an appropriate shape"""

    weights_r = torch.randn(out_channels, in_channels, 15, 15, requires_grad=True)
    hollow_chess_kernel(weights_r)

    return weights_r


class FeedForward(nn.Module):
    """Feed forward network for the conv"""

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    """Norm for the transformer"""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        # batch_size = x.size()[0]
        # x = x_in.view(batch_size, -1, 64)
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        # norm = norm.view(batch_size, -1, 8, 8)
        return norm


class ChessEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadDoubleAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask)).squeeze()
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))

        return x
