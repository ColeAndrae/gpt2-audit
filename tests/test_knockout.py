import torch
import random
from audit.model import forward


TEST_PROMPTS = [
    "What are the three main differences between aerobic and anaerobic cellular respiration?",
    "Write a short, suspenseful paragraph about an astronaut discovering an unexpected footprint on Mars.",
    "Write a Python function that takes a list of integers and returns only the even numbers.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?",
    "Explain how blockchain technology works to a ten-year-old using a simple playground analogy.",
]


def test_knockout(gpt2, tokenizer):

    torch.manual_seed(42)

    for prompt in TEST_PROMPTS:

        target_block = torch.randint(0, 11, (1,)).item()
        target_head = torch.randint(0, 11, (1,)).item()

        cache = {}

        x = torch.tensor([tokenizer.encode(prompt)])

        round_1_logits = forward(x, target_block, target_head, cache=cache)
        round_2_logits = forward(x, target_block, target_head, cache[(target_block, target_head)])

        assert torch.equal(round_1_logits, round_2_logits)

        round_3_logits = forward(x, target_block, target_head, cache=cache)
        round_4_logits = forward(x, target_block, target_head, torch.zeros_like(cache[(target_block, target_head)]))

        assert not torch.equal(round_3_logits, round_4_logits)

        replacement_head = torch.ones_like(cache[(target_block, target_head)])
        round_5_logits = forward(x, target_block, target_head, replacement_head, cache=cache)

        assert not torch.equal(replacement_head, cache[(target_block, target_head)])

