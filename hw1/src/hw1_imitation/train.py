"""训练并评估 Push-T 模仿策略。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.evaluation import Logger, evaluate_policy
from hw1_imitation.model import PolicyType, build_policy

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # Push-T 数据集下载到的路径。
    data_dir: Path = Path("data")

    # 策略类型，可选 MSE 或 flow。
    policy_type: PolicyType = "mse"
    # flow 策略使用的去噪步数（对 MSE 策略无影响）。
    flow_num_steps: int = 10
    # 动作块大小。
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # 训练的 epoch 数。
    num_epochs: int = 400
    # 运行评估的频率，以训练步数计。
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # 记录训练指标的频率，以训练步数计。
    log_interval: int = 100
    # 随机种子。
    seed: int = 42
    # WandB 项目名。
    wandb_project: str = "hw1-imitation"
    # 用于日志和 WandB 的实验名称后缀。
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "训练一个 Push-T MLP 策略。",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    global_step = 0
    last_eval_step = 0

    for _epoch in range(config.num_epochs):
        model.train()
        for state, action_chunk in loader:
            state = state.to(device)
            action_chunk = action_chunk.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(state, action_chunk)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % config.log_interval == 0:
                logger.log({"train/loss": float(loss.item())}, step=global_step)

            if global_step % config.eval_interval == 0:
                evaluate_policy(
                    model=model,
                    normalizer=normalizer,
                    device=device,
                    chunk_size=config.chunk_size,
                    video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=global_step,
                    logger=logger,
                )
                last_eval_step = global_step

    if global_step > 0 and last_eval_step != global_step:
        evaluate_policy(
            model=model,
            normalizer=normalizer,
            device=device,
            chunk_size=config.chunk_size,
            video_size=config.video_size,
            num_video_episodes=config.num_video_episodes,
            flow_num_steps=config.flow_num_steps,
            step=global_step,
            logger=logger,
        )

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
