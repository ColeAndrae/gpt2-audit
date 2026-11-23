import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@pytest.fixture(scope="session")
def gpt2():
    return GPT2LMHeadModel.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
