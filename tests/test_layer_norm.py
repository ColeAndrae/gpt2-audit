import torch
from audit.model import layer_norm


TEST_SHAPES = [(3, 4, 768),
               (4, 2, 768),
               (8, 10, 768),
               (9, 2, 768),
               (8, 8, 768)]


def test_layer_norm(gpt2):

    for B, T, C in TEST_SHAPES:

        x = torch.rand(B, T, C)

        actual = layer_norm(x, 1, 0)
        expected = gpt2.transformer.h[0].ln_1(x)

        assert torch.allclose(expected, actual, rtol=1e-05, atol=1e-06)
