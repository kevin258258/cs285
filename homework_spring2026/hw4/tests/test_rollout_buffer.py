import torch

from hw4.rollout.rollout_buffer import RolloutBatch, iter_minibatches


def make_rollout_batch(num_rows: int = 5, seq_len: int = 4) -> RolloutBatch:
    ids = torch.arange(num_rows * seq_len).reshape(num_rows, seq_len)
    per_token = torch.arange(num_rows * (seq_len - 1), dtype=torch.float).reshape(num_rows, seq_len - 1)
    row_values = torch.arange(num_rows, dtype=torch.float)
    return RolloutBatch(
        input_ids=ids,
        attention_mask=ids + 100,
        completion_mask=per_token + 200,
        old_logprobs=per_token + 300,
        ref_logprobs=per_token + 400,
        rewards=row_values + 500,
        advantages=row_values + 600,
        task_names=[f"task-{i}" for i in range(num_rows)],
        completion_texts=[f"completion-{i}" for i in range(num_rows)],
    )


def assert_minibatch_matches_indices(minibatch: RolloutBatch, source: RolloutBatch, indices: torch.Tensor):
    torch.testing.assert_close(minibatch.input_ids, source.input_ids[indices])
    torch.testing.assert_close(minibatch.attention_mask, source.attention_mask[indices])
    torch.testing.assert_close(minibatch.completion_mask, source.completion_mask[indices])
    torch.testing.assert_close(minibatch.old_logprobs, source.old_logprobs[indices])
    torch.testing.assert_close(minibatch.ref_logprobs, source.ref_logprobs[indices])
    torch.testing.assert_close(minibatch.rewards, source.rewards[indices])
    torch.testing.assert_close(minibatch.advantages, source.advantages[indices])
    assert minibatch.task_names == [source.task_names[int(i)] for i in indices]
    assert minibatch.completion_texts == [source.completion_texts[int(i)] for i in indices]


def test_iter_minibatches_without_shuffle_preserves_order_and_keeps_remainder():
    batch = make_rollout_batch(num_rows=5)

    minibatches = list(iter_minibatches(batch, minibatch_size=2, shuffle=False))

    assert [mb.input_ids.shape[0] for mb in minibatches] == [2, 2, 1]
    assert_minibatch_matches_indices(minibatches[0], batch, torch.tensor([0, 1]))
    assert_minibatch_matches_indices(minibatches[1], batch, torch.tensor([2, 3]))
    assert_minibatch_matches_indices(minibatches[2], batch, torch.tensor([4]))


def test_iter_minibatches_with_shuffle_uses_generator_and_keeps_fields_aligned():
    batch = make_rollout_batch(num_rows=6)
    seed = 1234
    generator = torch.Generator().manual_seed(seed)
    expected_order = torch.randperm(6, generator=torch.Generator().manual_seed(seed))

    minibatches = list(iter_minibatches(batch, minibatch_size=4, shuffle=True, generator=generator))

    assert [mb.input_ids.shape[0] for mb in minibatches] == [4, 2]
    assert_minibatch_matches_indices(minibatches[0], batch, expected_order[:4])
    assert_minibatch_matches_indices(minibatches[1], batch, expected_order[4:])
    observed = torch.cat([mb.rewards - 500 for mb in minibatches]).long()
    torch.testing.assert_close(torch.sort(observed).values, torch.arange(6))


def test_iter_minibatches_moves_tensor_fields_to_requested_device_and_preserves_optional_none():
    batch = make_rollout_batch(num_rows=3)
    batch.task_names = None
    batch.completion_texts = None

    minibatches = list(iter_minibatches(batch, minibatch_size=2, shuffle=False, device=torch.device("cpu")))
    minibatch = minibatches[0]

    tensor_fields = (
        minibatch.input_ids,
        minibatch.attention_mask,
        minibatch.completion_mask,
        minibatch.old_logprobs,
        minibatch.ref_logprobs,
        minibatch.rewards,
        minibatch.advantages,
    )
    assert all(t.device.type == "cpu" for t in tensor_fields)
    assert minibatch.task_names is None
    assert minibatch.completion_texts is None
