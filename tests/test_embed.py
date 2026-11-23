import torch
from audit.model import embed


TEST_SHAPES = [(3, 4),
               (4, 2),
               (8, 10),
               (9, 2),
               (8, 8)]


def test_embed(gpt2):

    for B, T in TEST_SHAPES:

        x = torch.randint(50257, (B, T))

        actual = embed(x)
        expected = gpt2(x, output_hidden_states=True).hidden_states[0]

        assert torch.allclose(expected, actual, rtol=1e-05, atol=1e-06)


def test_embed_reversed_positions_fail(gpt2):

    for B, T in TEST_SHAPES:

        x = torch.randint(50257, (B, T))
        indices = torch.arange(x.shape[-1] - 1, -1, -1)

        expected = gpt2(x, output_hidden_states=True).hidden_states[0]
        actual = gpt2.transformer.wte(x) + gpt2.transformer.wpe(indices)

        assert not torch.allclose(expected, actual, rtol=1e-05, atol=1e-06)
