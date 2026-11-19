import torch
import numpy as np
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

        input = torch.randint(50257, (B, T))
        indices = torch.arange(input.shape[-1])

        expected = model(input, output_hidden_states=True).hidden_states[0]
        actual = model.transformer.wte(input) + model.transformer.wpe(indices)
        
        assert torch.allclose(expected, actual, rtol=1e-05, atol=1e-06)
    
    for B, T in TEST_SHAPES:
    
        input = torch.randint(50257, (B, T))
        indices = torch.arange(input.shape[-1] - 1, -1, -1)

        expected = model(input, output_hidden_states=True).hidden_states[0]
        actual = model.transformer.wte(input) + model.transformer.wpe(indices)
        
        assert not torch.allclose(expected, actual, rtol=1e-05, atol=1e-06)

def test_layer_norm():

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    for B, T in TEST_SHAPES:
    
        input = torch.randint(50257, (B, T))
        indices = torch.arange(input.shape[-1])

        tok_emb = model.transformer.wte(input)
        pos_emb = model.transformer.wpe(indices)

        x = tok_emb + pos_emb

        expected = F.layer_norm(x, (768,))
        actual = (x - x.mean(2, keepdim=True)) / (x.var(2, keepdim=True, unbiased=False) + 1e-5).sqrt()

        assert torch.allclose(expected, actual, rtol=1e-05, atol=1e-06)

