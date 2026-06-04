from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from hw4.models.logprobs import (
    approx_kl_from_logprobs,
    build_completion_mask,
    compute_per_token_logprobs,
)


class TinyCausalLM(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("base_logits", logits)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, *, input_ids, attention_mask, use_cache):
        self.calls.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
            }
        )
        batch_size, seq_len = input_ids.shape
        logits = self.base_logits[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        return SimpleNamespace(logits=logits * self.scale)


def test_compute_per_token_logprobs_scores_each_token_from_previous_logits():
    logits = torch.tensor(
        [
            [0.0, 2.0, -1.0, 0.5, 1.0],
            [1.0, -2.0, 3.0, 0.0, 0.5],
            [-1.5, 0.5, 2.5, -0.5, 1.5],
            [2.0, 1.0, 0.0, -1.0, -2.0],
        ]
    )
    model = TinyCausalLM(logits)
    input_ids = torch.tensor([[3, 2, 4, 1], [0, 1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)

    actual = compute_per_token_logprobs(model, input_ids, attention_mask)

    expected = -F.cross_entropy(
        logits[:-1].unsqueeze(0).expand(2, -1, -1).reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(2, -1)
    assert actual.shape == (2, 3)
    torch.testing.assert_close(actual, expected)
    assert model.calls[-1]["use_cache"] is False


def test_compute_per_token_logprobs_respects_enable_grad_flag():
    logits = torch.randn(3, 4)
    model = TinyCausalLM(logits)
    input_ids = torch.tensor([[0, 1, 2]])
    attention_mask = torch.ones_like(input_ids)

    with_grad = compute_per_token_logprobs(model, input_ids, attention_mask, enable_grad=True)
    without_grad = compute_per_token_logprobs(model, input_ids, attention_mask, enable_grad=False)

    assert with_grad.requires_grad
    assert not without_grad.requires_grad


@pytest.mark.parametrize(
    ("input_ids", "attention_mask", "prompt_input_len", "expected"),
    [
        (
            torch.tensor([[11, 12, 21, 22, 0, 0], [13, 14, 31, 0, 0, 0]]),
            torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]]),
            2,
            torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]),
        ),
        (
            torch.tensor([[11, 12, 13, 21, 0]]),
            torch.tensor([[1, 1, 1, 1, 0]]),
            3,
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        ),
    ],
)
def test_build_completion_mask_selects_completion_targets_and_excludes_padding(
    input_ids, attention_mask, prompt_input_len, expected
):
    actual = build_completion_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_input_len=prompt_input_len,
        pad_token_id=0,
    )

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float
    assert actual.device == input_ids.device
    torch.testing.assert_close(actual, expected)


def test_approx_kl_from_logprobs_returns_scalar_masked_mean_with_clamped_delta():
    new_logprobs = torch.tensor([[-2.0, -1.0, -3.0], [-4.0, -2.5, -1.5]])
    ref_logprobs = torch.tensor([[-1.5, -2.0, 50.0], [-5.0, -1.5, -2.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    actual = approx_kl_from_logprobs(
        new_logprobs,
        ref_logprobs,
        mask,
        log_ratio_clip=1.0,
    )

    delta = torch.clamp(ref_logprobs - new_logprobs, min=-1.0, max=1.0)
    per_token = torch.exp(delta) - delta - 1.0
    expected = (per_token * mask).sum() / mask.sum()
    assert actual.ndim == 0
    torch.testing.assert_close(actual, expected)
