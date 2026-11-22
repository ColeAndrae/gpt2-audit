import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast


TEST_SHAPES = [(3, 4, 768),
               (4, 2, 768),
               (8, 10, 768),
               (9, 2, 768),
               (8, 8, 768)]


def test_attention():

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    for B, T, C in TEST_SHAPES:

            ln_1_out = torch.rand(B, T, C)
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

            c_attn_weight = model.transformer.h[0].attn.c_attn.weight
            c_attn_bias = model.transformer.h[0].attn.c_attn.bias
            c_attn_out = ln_1_out @ c_attn_weight + c_attn_bias

            Q, K, V = c_attn_out.split(768, dim=2)

            Q = Q.view(B, T, 12, -1).transpose(1, 2)
            K = K.view(B, T, 12, -1).transpose(1, 2)
            V = V.view(B, T, 12, -1).transpose(1, 2)

            scores = Q @ K.transpose(2, 3) / 8
            scores = scores.masked_fill(~mask, float('-inf'))
            scores = scores.softmax(dim=-1)

            out = (scores @ V).transpose(1, 2).reshape(B, T, C)

            actual = out @ model.transformer.h[0].attn.c_proj.weight + model.transformer.h[0].attn.c_proj.bias
            expected = model.transformer.h[0].attn(ln_1_out)[0]

            assert torch.allclose(actual, expected, atol=1e-4)             
    

