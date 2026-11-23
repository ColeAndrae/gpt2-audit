import torch
import torch.nn as nn
import numpy as np
from audit.model import attn
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

        x = torch.rand(B, T, C)

        actual = attn(x, 0)
        expected = model.transformer.h[0].attn(x)[0]

        assert torch.allclose(actual, expected, atol=1e-4)             
