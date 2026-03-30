import unittest

import torch

from hw1_imitation.model import MSEPolicy


class MSEPolicyTests(unittest.TestCase):
    def test_sample_actions_matches_forward_shape(self) -> None:
        policy = MSEPolicy(
            state_dim=5,
            action_dim=2,
            chunk_size=3,
            hidden_dims=(7, 11, 13),
        )
        state = torch.randn(4, 5)

        sampled = policy.sample_actions(state)
        forwarded = policy.forward(state)

        self.assertEqual(sampled.shape, (4, 3, 2))
        self.assertTrue(torch.allclose(sampled, forwarded))

    def test_hidden_dims_builds_all_hidden_layers(self) -> None:
        policy = MSEPolicy(
            state_dim=5,
            action_dim=2,
            chunk_size=3,
            hidden_dims=(7, 11, 13),
        )

        linear_layers = [
            layer for layer in policy.net if isinstance(layer, torch.nn.Linear)
        ]
        linear_shapes = [
            (layer.in_features, layer.out_features) for layer in linear_layers
        ]

        self.assertEqual(linear_shapes, [(5, 7), (7, 11), (11, 13), (13, 6)])


if __name__ == "__main__":
    unittest.main()
