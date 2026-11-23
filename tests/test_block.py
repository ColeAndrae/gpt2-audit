import torch
from audit.model import embed, block


TEST_SHAPES = [(3, 4),
               (4, 2),
               (8, 10),
               (9, 2),
               (8, 8)]


def test_block(gpt2):

    for B, T in TEST_SHAPES:

        x = torch.randint(50257, (B, T))

        embed_out = embed(x)
        actual = block(embed_out, 0)

        expected = gpt2.transformer.h[0](embed_out)

        assert torch.allclose(actual, expected, atol=1e-4)
