from types import SimpleNamespace

import pytest
import torch

from hw4.models.logprobs import (
    approx_kl_from_logprobs,
    compute_per_token_logprobs,
    masked_mean,
    masked_mean_per_row,
)
from hw4.rl.base import AlgoConfig
from hw4.rl.reinforce import Reinforce
from hw4.rollout.rollout_buffer import RolloutBatch


class TinyPolicy(torch.nn.Module):
    def __init__(self, seq_len: int, vocab_size: int):
        super().__init__()
        values = torch.arange(seq_len * vocab_size, dtype=torch.float).reshape(seq_len, vocab_size)
        self.logits_by_pos = torch.nn.Parameter(values / 10.0)
        self.config = SimpleNamespace(use_cache=True)
        self.calls = 0

    def forward(self, *, input_ids, attention_mask, use_cache):
        self.calls += 1
        batch_size, seq_len = input_ids.shape
        logits = self.logits_by_pos[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        return SimpleNamespace(logits=logits)


def make_rollout(model: TinyPolicy) -> RolloutBatch:
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ]
    )
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    with torch.no_grad():
        new_logp = compute_per_token_logprobs(model, input_ids, attention_mask, enable_grad=False)
    ref_logprobs = new_logp + torch.tensor(
        [
            [0.0, 0.2, -0.3],
            [0.0, -0.4, 0.0],
        ]
    )
    return RolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_logprobs=torch.full_like(new_logp, 123.0),
        ref_logprobs=ref_logprobs,
        rewards=torch.tensor([0.0, 0.0]),
        advantages=torch.tensor([2.0, -1.0]),
    )


def test_reinforce_update_computes_sequence_policy_loss_kl_entropy_and_optimizer_step():
    model = TinyPolicy(seq_len=4, vocab_size=5)
    rollout = make_rollout(model)
    cfg = AlgoConfig(
        ppo_epochs=1,
        minibatch_size=2,
        kl_coef=0.25,
        max_grad_norm=100.0,
        adv_clip=10.0,
        seed=0,
    )
    algo = Reinforce(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    expected_logp = compute_per_token_logprobs(model, rollout.input_ids, rollout.attention_mask)
    expected_seq_logp = masked_mean_per_row(expected_logp, rollout.completion_mask)
    expected_pg_loss = -(rollout.advantages * expected_seq_logp).mean()
    expected_kl = approx_kl_from_logprobs(expected_logp, rollout.ref_logprobs, rollout.completion_mask)
    expected_entropy = -masked_mean(expected_logp, rollout.completion_mask)
    expected_loss = expected_pg_loss + cfg.kl_coef * expected_kl

    stats = algo.update(model, optimizer, rollout)

    assert model.training
    assert model.config.use_cache is False
    assert stats["train/count_optimizer_steps_per_training_iteration"] == 1.0
    assert stats["train/count_minibatches_skipped_because_completion_mask_had_no_tokens"] == 0.0
    assert stats["train/count_update_attempts_skipped_due_to_nonfinite_loss_or_gradients"] == 0.0
    assert stats["train/policy_loss_with_kl_penalty_mean_over_minibatches"] == pytest.approx(
        float(expected_loss.detach())
    )
    assert stats["train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches"] == pytest.approx(
        float(expected_kl.detach())
    )
    assert stats["train/policy_token_entropy_mean_over_minibatches"] == pytest.approx(
        float(expected_entropy.detach())
    )


def test_reinforce_update_skips_minibatches_with_empty_completion_mask():
    model = TinyPolicy(seq_len=3, vocab_size=4)
    input_ids = torch.tensor([[0, 1, 2], [1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    per_token = torch.zeros(2, 2)
    rollout = RolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=torch.zeros(2, 2),
        old_logprobs=per_token,
        ref_logprobs=per_token,
        rewards=torch.zeros(2),
        advantages=torch.ones(2),
    )
    cfg = AlgoConfig(minibatch_size=2, max_grad_norm=100.0, seed=0)
    algo = Reinforce(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    before = model.logits_by_pos.detach().clone()
    stats = algo.update(model, optimizer, rollout)

    torch.testing.assert_close(model.logits_by_pos, before)
    assert model.calls == 0
    assert stats["train/count_minibatches_skipped_because_completion_mask_had_no_tokens"] == 1.0
    assert stats["train/count_optimizer_steps_per_training_iteration"] == 0.0
    assert stats["train/policy_loss_with_kl_penalty_mean_over_minibatches"] == 0.0
