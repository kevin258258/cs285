import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """基础的 MLP 策略网络，可以接收一个观测值并输出动作分布。

    该类应该实现 `forward` 和 `get_action` 方法。`update` 方法应该在子类中实现，
    因为不同算法的策略更新规则不同。
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """接收单个观测值（numpy 数组）并返回单个动作（numpy 数组）。"""
        # TODO: 实现 get_action
        action = self.forward(ptu.from_numpy(obs).float()).sample()
        action = ptu.to_numpy(action)

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        该函数定义了网络的前向传播。你可以返回任何你想要的内容，但它需要是可微的。
        例如，你可以返回一个 torch.FloatTensor。你也可以返回更灵活的对象，例如
        `torch.distributions.Distribution` 对象。这取决于你！
        """
        if self.discrete:
            # TODO: 为离散动作空间的策略定义前向传播。
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            # TODO: 为连续动作空间的策略定义前向传播。
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            action_distribution = distributions.Normal(mean, std)
            return action_distribution

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        对提供的一批数据执行一次梯度下降迭代。你不需要在基类中实现这个方法，
        但你需要在子类中实现它。
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """策略梯度算法的策略子类。"""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """实现策略梯度的 actor 更新。"""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: 计算策略梯度 actor 损失
        log_prob = self.forward(obs).log_prob(actions)
        if not self.discrete:
            log_prob = log_prob.sum(dim=-1)
        loss = -(log_prob * advantages).mean()

        # TODO: 执行一次优化器步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
