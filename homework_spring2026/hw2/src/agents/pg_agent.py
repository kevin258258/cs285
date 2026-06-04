from typing import Optional, Sequence
import numpy as np
import torch

from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # 创建 actor（策略）网络
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # 如果需要，创建 critic（基线）网络
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # 其他智能体参数
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """PG 的训练步骤涉及使用给定的观测/动作以及根据所见奖励计算出的
        qvals/advantages 来更新其 actor。

        每个输入都是 NumPy 数组的列表，其中每个数组对应一条轨迹。批次大小是
        所有轨迹中样本的总数（即所有数组长度的总和）。
        """

        # 步骤 1: 使用奖励 (r_0, ..., r_t, ..., r_T) 计算每个 (s_t, a_t) 点的 Q 值
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: 将数组列表展平为单个数组，以便其余代码可以以向量化的方式编写。
        # 从此处开始，obs、actions、rewards、terminals 和 q_values 都应该是具有
        # 前导维度 `batch_size` 的数组。
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)

        # 步骤 2: 从 Q 值计算 advantages
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # 步骤 3: 使用所有数据点 (s_t, a_t, adv_t) 更新 PG actor/策略
        # TODO: 使用 advantages 更新一次 PG actor/策略网络
        info: dict = self.actor.update(obs, actions, advantages)

        # 步骤 4: 如果需要，使用所有数据点 (s_t, a_t, q_t) 更新 PG critic/基线
        if self.critic is not None:
            # 对 critic/基线网络执行 `self.baseline_gradient_steps` 次更新
            critic_info = {}
            for _ in range(self.baseline_gradient_steps):
                step_info = self.critic.update(obs, q_values)
                for k, v in step_info.items():
                    if k not in critic_info:
                        critic_info[k] = []
                    critic_info[k].append(v)
            # 取平均
            critic_info = {k: np.mean(v) for k, v in critic_info.items()}
            info.update(critic_info)

        return info

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        辅助函数，接收一个奖励列表 {r_0, r_1, ..., r_t', ... r_T} 并返回一个列表，
        其中每个索引 t 包含 sum_{t'=0}^T gamma^t' r_{t'}
        

        注意，输出列表的所有条目应该完全相同，因为每个求和都是从 0 到 T（不涉及 t）！
        """
        sum_reward = 0
        for i in range(len(rewards)):
            sum_reward += rewards[i] * (self.gamma ** i)

        return [sum_reward] * len(rewards)

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        辅助函数，接收一个奖励列表 {r_0, r_1, ..., r_t', ... r_T} 并返回一个列表，
        其中每个索引 t 的条目是 sum_{t'=t}^T gamma^(t'-t) * r_{t'}。
        """
        discounted_rewards = []
        for i in range(len(rewards)):
            sum_reward = 0
            for j in range(i, len(rewards)):
                sum_reward += rewards[j] * (self.gamma ** (j - i))
            discounted_rewards.append(sum_reward)
        return discounted_rewards

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Q 函数的蒙特卡洛估计。"""

        q_values = []
        for traj_rewards in rewards:
            if not self.use_reward_to_go:
                # 情况 1: 在基于轨迹的 PG 中，我们忽略时间步，而是在每个点使用整个轨迹的折扣回报。
                # 换句话说: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
                q_values.append(np.array(self._discounted_return(traj_rewards)))
            else:
                # 情况 2: 在 reward-to-go PG 中，我们只使用时间步 t 之后的奖励来估计 (s_t, a_t) 的 Q 值。
                # 换句话说: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
                q_values.append(np.array(self._discounted_reward_to_go(traj_rewards)))

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """通过（可能）从估计的 Q 值中减去价值基线来计算 advantages。

        操作在扁平的 1D NumPy 数组上。
        """
        if self.critic is None:
            # TODO: 如果没有基线，那么 advantages 是什么？
            advantages = q_values
        else:
            # TODO: 运行 critic 并将其用作基线
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs).float()).squeeze())
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: 如果使用基线，但不使用 GAE，那么 advantages 是什么？
                advantages = q_values - values
            else:
                # TODO: 实现 GAE
                batch_size = obs.shape[0]

                # 提示: 追加一个虚拟的 T+1 值以简化递归计算
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)


                for i in reversed(range(batch_size)):
                    # 从时间步 T 开始递归计算 advantage 估计。
                    # 使用 terminals 处理边界情况。如果状态是其轨迹中的最后一个状态，
                    # terminals[i] 为 1，否则为 0。
                    if terminals[i] == 1:
                        # terminal 状态: V(s_{t+1}) = 0，后续 advantage = 0
                        delta = rewards[i] - values[i]
                        advantages[i] = delta
                    else:
                        delta = rewards[i] + self.gamma * values[i + 1] - values[i]
                        advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i + 1]

                # 移除虚拟 advantage
                advantages = advantages[:-1]

        # TODO: 将 advantages 归一化为在批次内均值为零、标准差为一
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages
