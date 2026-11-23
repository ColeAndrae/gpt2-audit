import torch
from audit.model import mlp


TEST_SHAPES = [(3, 4, 768),
               (4, 2, 768),
               (8, 10, 768),
               (9, 2, 768),
               (8, 8, 768)]


def test_mlp(gpt2):

    for B, T, C in TEST_SHAPES:

        x = torch.rand(B, T, C)

        actual = mlp(x, 0)
        expected = gpt2.transformer.h[0].mlp(x)

        assert torch.allclose(actual, expected, atol=1e-4)
