import math

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


model = GPT2LMHeadModel.from_pretrained("gpt2")
config = model.config


def embed(x):

    i = torch.arange(x.shape[-1])

    tok_weight = model.transformer.wte.weight
    pos_weight = model.transformer.wpe.weight

    return tok_weight[x] + pos_weight[i]


def layer_norm(x, layer, block):

    if layer == 0:
        weight = model.transformer.ln_f.weight
        bias = model.transformer.ln_f.bias
    elif layer == 1:
        weight = model.transformer.h[block].ln_1.weight
        bias = model.transformer.h[block].ln_1.bias
    elif layer == 2:
        weight = model.transformer.h[block].ln_2.weight
        bias = model.transformer.h[block].ln_2.bias
    else:
        raise ValueError("Layer must be 0, 1, or 2.")

    ln = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True, unbiased=False) + config.layer_norm_epsilon).sqrt()

    return ln * weight + bias


def attn(x, block):

    B, T, C = x.shape
    H = config.n_head

    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

    c_attn_weight = model.transformer.h[block].attn.c_attn.weight
    c_attn_bias = model.transformer.h[block].attn.c_attn.bias
    c_attn_out = x @ c_attn_weight + c_attn_bias

    Q, K, V = c_attn_out.split(C, dim=2)

    Q = Q.view(B, T, H, -1).transpose(1, 2)
    K = K.view(B, T, H, -1).transpose(1, 2)
    V = V.view(B, T, H, -1).transpose(1, 2)

    scores = Q @ K.transpose(2, 3) / math.sqrt(C // H)
    scores = scores.masked_fill(~mask, float("-inf"))
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

    fc_out = F.gelu(fc_out, approximate="tanh")

    return fc_out @ proj_weight + proj_bias


def block(x, b):

    ln_1_out = layer_norm(x, 1, b)
    attn_out = x + attn(ln_1_out, b)
    ln_2_out = layer_norm(attn_out, 2, b)

    return attn_out + mlp(ln_2_out, b)


def forward(x):

    x = embed(x)

    for b in range(config.n_layer):
        x = block(x, b)

    x = layer_norm(x, 0, 0)

    out = x @ model.transformer.wte.weight.T

    return out
