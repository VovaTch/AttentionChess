import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class TrigoLinear(nn.Module):
    """
    A simple trigonometric linear layer composed of trigonometric neurons; experimental 
    neuron type to avoid segmenting the classification field to piece-wise linear segments.
    Should work exactly like the regular input layer, but this time we have ~3n parameters with biases
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TrigoLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.weight = nn.Parameter(torch.empty((out_features, in_features, 2), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features, in_features + 1), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
            
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor):
        
        return ripple_linear(input, self.out_features, self.weight, self.bias)
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

@torch.jit.script
def ripple_linear(input:torch.Tensor, out_features:int, weight:torch.Tensor, bias:torch.Tensor):
    
    # Register output sizes
    input_size = input.size()
    output_size = list(input_size)
    output_size[-1] = out_features
    
    # perform operation w1 * sin(w2 * x + b2) + b1
    input_flattened = input.view(-1, input_size[-1])
    super_batch_size = input_flattened.size()[0]
    
    sin_weight_block = weight[:, :, 1].unsqueeze(2).repeat(1, 1, super_batch_size)
    sin_input_block = input_flattened.permute((1, 0)).unsqueeze(0).repeat(out_features, 1, 1)
    sin_bias_block = bias[:, :-1].unsqueeze(2).repeat(1, 1, super_batch_size)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    
    weighted_trigo_out = torch.sin(sin_weight_block * sin_input_block + sin_bias_block)
    out_weight_block = weight[:, :, 0].unsqueeze(2).repeat(1, 1, super_batch_size)
    out_bias_block = bias[:, -1].unsqueeze(1).repeat(1, super_batch_size)
    
    trigo_out = torch.sum(out_weight_block * weighted_trigo_out, dim=1) + out_bias_block
    trigo_out = trigo_out.permute((1, 0))
        
    # return unflattened tensor
    return trigo_out.reshape(output_size)
        
class TrigoAttention(nn.Module):
    def __init__(self, heads:int, d_model:int, dropout:float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = TrigoLinear(d_model, d_model)
        self.v_linear = TrigoLinear(d_model, d_model)
        self.k_linear = TrigoLinear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.out = TrigoLinear(d_model, d_model)
    
    def forward(self, q, k, v, mask:Optional[torch.Tensor]=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout_p)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output
    
    
def attention(q, k, v, d_k:int, mask:Optional[torch.Tensor] = None, dropout_p:float=0.0):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    
    dropout_layer = nn.Dropout(p=dropout_p)
    scores = dropout_layer(scores)
            
    output = torch.matmul(scores, v)
    return output


class TrigoFeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int=32, dropout:float = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = TrigoLinear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = TrigoLinear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
    
class Norm(nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
    
class TrigoEncoderLayer(nn.Module):
    def __init__(self, d_model:int, nhead:int, dropout:float = 0.1, batch_first:bool = False):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = TrigoAttention(nhead, d_model, dropout=dropout)
        self.ff = TrigoFeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, src_mask:Optional[torch.Tensor]=None,
                src_key_padding_mask:Optional[torch.Tensor]=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,src_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
    
class TrigoDecoderLayer(nn.Module):
    def __init__(self, d_model:int, nhead:int, dropout=0.1, batch_first:bool = False):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = TrigoAttention(nhead, d_model, dropout=dropout)
        self.attn_2 = TrigoAttention(nhead, d_model, dropout=dropout)
        self.ff = TrigoFeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask:Optional[torch.Tensor]=None, tgt_mask:Optional[torch.Tensor]=None, 
                tgt_key_padding_mask:Optional[torch.Tensor]=None, memory_mask:Optional[torch.Tensor]=None, 
                memory_key_padding_mask:Optional[torch.Tensor]=None):
        # x2 = self.norm_1(x)
        # x = x + self.dropout_1(self.attn_1(x2, x2, x2, tgt_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x