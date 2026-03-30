"""Push-T 模仿策略的模型定义。"""

from __future__ import annotations

import abc
from logging import raiseExceptions
from posix import stat
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from modal import forward
from torch import chunk, nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """动作分块策略的基类。"""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """计算一个 batch 的训练损失。"""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # 仅对 flow 策略生效
    ) -> torch.Tensor:
        """生成形状为 `(batch, chunk_size, action_dim)` 的动作块。"""


class MSEPolicy(BasePolicy):
    """使用 MSE 损失预测动作块。"""

    ### TODO: 在这里实现 MSEPolicy ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.out_dim = action_dim * chunk_size
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], self.out_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.net(state)
        x = x.view(-1, self.chunk_size, self.action_dim)

        return x

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        x = self.forward(state)
        loss = torch.mean(
            torch.sum(((x - action_chunk).view(-1, self.out_dim) ** 2), dim=-1)
        )

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.forward(state)


class FlowMatchingPolicy(BasePolicy):
    """使用 flow matching 损失预测动作块。"""

    # 大概看了一下，flow matching主要就是学习一个向量场，这里依旧按照典型的MLP实现

    ### TODO: 在这里实现 FlowMatchingPolicy ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.out_dim = chunk_size * action_dim
        self.hidden_dims = hidden_dims
        self.net = self.net = nn.Sequential(
            nn.Linear(state_dim + self.out_dim + 1, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], self.out_dim),
        )

    def forward(
        self, state: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        Batch = state.shape[0]
        x_t_flat = xt.reshape(Batch, -1)
        if t.dim() == 1:
            t.unsqueeze(-1)
        x = torch.cat([state, x_t_flat, t], dim=-1)
        v = self.net(x)
        return v.view(Batch, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        device = state.device
        Batch = state.shape[0]
        x1 = action_chunk
        x0 = torch.randn_like(x1)
        t = torch.rand(Batch, 1, device=device)  # [B, 1]
        t_expand = t.view(Batch, 1, 1)
        xt = (1 - t_expand) * x0 + t_expand * x1
        v = self.forward(state, xt, t)
        real = x1 - x0
        return F.mse_loss(v, real)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # 这里采用A(t +dt) = A(t) + dt * v
        B = state.shape[0]
        device = state.device

        x = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        dt = 1.0 / num_steps

        for k in range(num_steps):
            t = torch.full(
                (B, 1),
                fill_value=k / num_steps,
                device=device,
                dtype=state.dtype,
            )
            v = self.forward(state, x, t)
            x = x + dt * v

        return x


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
