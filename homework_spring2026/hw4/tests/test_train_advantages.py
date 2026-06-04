import pytest
import torch

from hw4.train import compute_group_advantages, maybe_normalize_advantages


def test_compute_group_advantages_normalizes_each_prompt_major_group_with_population_std():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 14.0])

    advantages = compute_group_advantages(rewards, group_size=3, eps=0.0)

    grouped = rewards.reshape(2, 3)
    expected = (grouped - grouped.mean(dim=1, keepdim=True)) / grouped.std(
        dim=1,
        keepdim=True,
        unbiased=False,
    )
    torch.testing.assert_close(advantages, expected.reshape(-1))


def test_compute_group_advantages_handles_singleton_and_constant_groups_without_nonfinite_values():
    singleton = compute_group_advantages(torch.tensor([1.0, -2.0, 5.0]), group_size=1)
    constant = compute_group_advantages(torch.tensor([7.0, 7.0, 3.0, 3.0]), group_size=2)

    torch.testing.assert_close(singleton, torch.zeros_like(singleton))
    torch.testing.assert_close(constant, torch.zeros_like(constant))
    assert torch.isfinite(singleton).all()
    assert torch.isfinite(constant).all()


def test_compute_group_advantages_rejects_non_divisible_reward_count():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.raises(ValueError, match="divisible"):
        compute_group_advantages(rewards, group_size=2)


def test_maybe_normalize_advantages_disabled_returns_input_unchanged():
    advantages = torch.tensor([3.0, -1.0, 2.0])

    normalized = maybe_normalize_advantages(advantages, enabled=False)

    assert normalized is advantages
    torch.testing.assert_close(normalized, advantages)


def test_maybe_normalize_advantages_enabled_uses_full_vector_population_std_and_preserves_shape():
    advantages = torch.tensor([[1.0, 2.0], [4.0, 8.0]])

    normalized = maybe_normalize_advantages(advantages, enabled=True, eps=0.0)

    expected = (advantages - advantages.mean()) / advantages.std(unbiased=False)
    assert normalized.shape == advantages.shape
    torch.testing.assert_close(normalized, expected)


def test_maybe_normalize_advantages_enabled_handles_zero_std_without_nonfinite_values():
    advantages = torch.tensor([5.0, 5.0, 5.0])

    normalized = maybe_normalize_advantages(advantages, enabled=True)

    torch.testing.assert_close(normalized, torch.zeros_like(advantages))
    assert torch.isfinite(normalized).all()
