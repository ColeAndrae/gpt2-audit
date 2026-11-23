import torch
import numpy as np
from audit.model import embed
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast


TEST_SHAPES = [(3, 4),
               (4, 2),
               (8, 10),
               (9, 2),
               (8, 8)]


def test_embed():

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    for B, T in TEST_SHAPES:

        x = torch.randint(50257, (B, T))

        actual = embed(x)
        expected = model(x, output_hidden_states=True).hidden_states[0]
        
        assert torch.allclose(expected, actual, rtol=1e-05, atol=1e-06)
    
    for B, T in TEST_SHAPES:
    
        x = torch.randint(50257, (B, T))
        indices = torch.arange(x.shape[-1] - 1, -1, -1)

        expected = model(x, output_hidden_states=True).hidden_states[0]
        actual = model.transformer.wte(x) + model.transformer.wpe(indices)
        
        assert not torch.allclose(expected, actual, rtol=1e-05, atol=1e-06)

