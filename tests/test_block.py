import torch
import numpy as np
from audit.model import embed, block
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast


TEST_SHAPES = [(3, 4),
               (4, 2),
               (8, 10),
               (9, 2),
               (8, 8)]


def test_block():

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    for B, T in TEST_SHAPES:

        x = torch.randint(50257, (B, T))

        embed_out = embed(x)
        actual = block(embed_out, 0)

        expected = model.transformer.h[0](embed_out)

        assert torch.allclose(actual, expected, atol=1e-4)   