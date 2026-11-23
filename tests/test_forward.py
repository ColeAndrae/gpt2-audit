import torch
import numpy as np
from audit.model import forward
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast


TEST_PROMPTS = prompts = [
    "What are the three main differences between aerobic and anaerobic cellular respiration?",
    "Write a short, suspenseful paragraph about an astronaut discovering an unexpected footprint on Mars.",
    "Write a Python function that takes a list of integers and returns only the even numbers.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?",
    "Explain how blockchain technology works to a ten-year-old using a simple playground analogy."
]


def test_forward():

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    for PROMPT in TEST_PROMPTS:

        x = torch.tensor(tokenizer.encode([PROMPT]))

        expected = model.forward(x).logits
        actual = forward(x)

        assert torch.allclose(actual, expected, atol=1e-4)   