import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast


model = GPT2LMHeadModel.from_pretrained("gpt2")


def embed(x):

    i = torch.arange(x.shape[-1])

    tok_weight = model.transformer.wte.weight
    pos_weight = model.transformer.wpe.weight

    return tok_weight[x] + pos_weight[i]


def layer_norm(x, layer, block):

    if layer == 1:
        weight = model.transformer.h[block].ln_1.weight
        bias = model.transformer.h[block].ln_1.bias
    if layer == 2:
        weight = model.transformer.h[block].ln_2.weight
        bias = model.transformer.h[block].ln_2.bias

    ln = (x - x.mean(2, keepdim=True)) / (x.var(2, keepdim=True, unbiased=False) + 1e-5).sqrt()

    return ln * weight + bias


def attn(x, block):

    B, T, C = x.shape

    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

    c_attn_weight = model.transformer.h[block].attn.c_attn.weight
    c_attn_bias = model.transformer.h[block].attn.c_attn.bias
    c_attn_out = x @ c_attn_weight + c_attn_bias

    Q, K, V = c_attn_out.split(768, dim=2)

    Q = Q.view(B, T, 12, -1).transpose(1, 2)
    K = K.view(B, T, 12, -1).transpose(1, 2)
    V = V.view(B, T, 12, -1).transpose(1, 2)

    scores = Q @ K.transpose(2, 3) / 8
    scores = scores.masked_fill(~mask, float('-inf'))
    scores = scores.softmax(dim=-1)

    out = (scores @ V).transpose(1, 2).reshape(B, T, C)

    c_proj_weight = model.transformer.h[block].attn.c_proj.weight
    c_proj_bias = model.transformer.h[block].attn.c_proj.bias

    return out @ c_proj_weight + c_proj_bias


def mlp(x, block):

    fc_weight = model.transformer.h[block].mlp.c_fc.weight
    fc_bias = model.transformer.h[block].mlp.c_fc.bias
    proj_weight = model.transformer.h[block].mlp.c_proj.weight
    proj_bias = model.transformer.h[block].mlp.c_proj.bias

    fc_out = x @ fc_weight + fc_bias

    fc_out = F.gelu(fc_out, approximate='tanh')

    return fc_out @ proj_weight + proj_bias