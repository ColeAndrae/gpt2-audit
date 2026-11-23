import torch
from audit.model import forward


TEST_PROMPTS = [
    "What are the three main differences between aerobic and anaerobic cellular respiration?",
    "Write a short, suspenseful paragraph about an astronaut discovering an unexpected footprint on Mars.",
    "Write a Python function that takes a list of integers and returns only the even numbers.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?",
    "Explain how blockchain technology works to a ten-year-old using a simple playground analogy.",
]


def test_forward(gpt2, tokenizer):

    for prompt in TEST_PROMPTS:

        x = torch.tensor([tokenizer.encode(prompt)])

        expected = gpt2(x).logits
        actual = forward(x)

        assert torch.allclose(actual, expected, atol=1e-4)
