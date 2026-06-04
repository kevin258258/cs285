"""Push-T 策略的评估工具。"""

from __future__ import annotations

import copy
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import wandb
from PIL import Image

from hw1_imitation.data import Normalizer
from hw1_imitation.model import BasePolicy

ENV_ID = "gym_pusht/PushT-v0"
NUM_EVAL_EPISODES = 100


class Logger:
    """用于记录指标的日志器。"""

    CSV_DISALLOWED_TYPES = (wandb.Image, wandb.Video, wandb.Histogram)

    def __init__(self, path: Path):
        if path.exists():
            raise FileExistsError(f"Log directory {path} already exists.")
        path.mkdir(parents=True)
        self.path = path
        self.csv_path = path / "log.csv"
        self.header = None
        self.rows = []

    def log(self, row: dict[str, Any], step: int) -> None:
        row["step"] = step
        if self.header is None:
            self.header = [
                k
                for k, v in row.items()
                if not isinstance(v, self.CSV_DISALLOWED_TYPES)
            ]
            with self.csv_path.open("w") as f:
                f.write(",".join(self.header) + "\n")
        filtered_row = {
            k: v for k, v in row.items() if not isinstance(v, self.CSV_DISALLOWED_TYPES)
        }
        with self.csv_path.open("a") as f:
            f.write(
                ",".join([str(filtered_row.get(k, "")) for k in self.header]) + "\n"
            )
        wandb.log(row, step=step)
        self.rows.append(copy.deepcopy(row))

    def dump_for_grading(self) -> None:
        wandb_dir = Path(wandb.run.dir).parent
        wandb.finish()
        shutil.copytree(wandb_dir, self.path / "wandb")


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(frame)
    resized = image.resize(size, resample=Image.BILINEAR)
    return np.asarray(resized)


def encode_video(frames: list[np.ndarray], fps: int = 20) -> wandb.Video | None:
    if not frames:
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with imageio.get_writer(
            tmp_path,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        return wandb.Video(io.BytesIO(video_bytes), format="mp4")
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def log_checkpoint_artifact(model: BasePolicy, step: int) -> None:
    if wandb.run is None:
        raise RuntimeError("wandb.init did not create a run.")

    run_dir = Path(wandb.run.dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pkl"
    torch.save(model, checkpoint_path)

    artifact = wandb.Artifact(
        name=f"policy-checkpoint-{wandb.run.id}",
        type="model",
        metadata={"step": step},
    )
    artifact.add_file(checkpoint_path.as_posix(), name=checkpoint_path.name)
    wandb.log_artifact(artifact)


def evaluate_policy(
    model: BasePolicy,
    normalizer: Normalizer,
    device: torch.device,
    chunk_size: int,
    video_size: tuple[int, int],
    num_video_episodes: int,
    flow_num_steps: int,
    step: int,
    logger: Logger,
) -> None:
    """在 Push-T 环境中评估策略，并将结果记录到 Weights & Biases。

    该函数会在 Push-T gym 环境中，使用给定策略运行固定数量的评估回合。
    它会用给定的 normalizer 对观测做归一化，从策略中请求一段动作序列
    （对于 flow 策略可选地使用多步采样），并在环境中依次执行这些动作，
    直到每个回合终止。

    指标：
        - 记录每个回合最大奖励的平均值。
        - 可选地记录前 ``num_video_episodes`` 个回合的渲染视频。

    检查点：
        - 将策略保存为 ``.pkl`` 文件，并作为 W&B artifact 上传，
          同时附带当前训练步数标签。

    参数：
        model: 要评估的策略。
        normalizer: 用于缩放状态和动作的归一化器。
        device: 执行策略推理所用的设备。
        chunk_size: 每次调用策略时生成的动作数。
        video_size: 渲染视频的 `(width, height)`。
        num_video_episodes: 需要录制视频的回合数。
        flow_num_steps: flow 策略使用的去噪步数。
        step: 用于日志和 artifact 元数据的训练步数。
        logger: 用于记录指标的日志器。
    """
    model.eval()
    rewards: list[float] = []
    videos: list[wandb.Video] = []

    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high

    for ep_idx in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep_idx)
        done = False
        chunk_index = chunk_size
        action_chunk: np.ndarray | None = None
        frames: list[np.ndarray] = []
        max_reward = 0.0
        save_video = ep_idx < num_video_episodes

        while not done:
            if action_chunk is None or chunk_index >= chunk_size:
                state = (
                    torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                )
                with torch.no_grad():
                    pred_chunk = (
                        model.sample_actions(
                            state.unsqueeze(0), num_steps=flow_num_steps
                        )
                        .cpu()
                        .numpy()[0]
                    )
                action_chunk = normalizer.denormalize_action(pred_chunk)
                action_chunk = np.clip(action_chunk, action_low, action_high)
                chunk_index = 0

            action = action_chunk[chunk_index]
            obs, reward, terminated, truncated, info = env.step(
                action.astype(np.float32)
            )
            if save_video:
                frame = env.render()
                frame = resize_frame(frame, video_size)
                frames.append(frame)
            max_reward = max(max_reward, float(reward))
            done = terminated or truncated
            chunk_index += 1

        rewards.append(max_reward)
        if save_video:
            video = encode_video(frames, fps=20)
            if video is not None:
                videos.append(video)

    env.close()
    log_data: dict[str, float | wandb.Video] = {
        "eval/mean_reward": float(np.mean(rewards))
    }
    for idx, video in enumerate(videos):
        log_data[f"eval/rollout_ep{idx}"] = video
    logger.log(log_data, step=step)
    log_checkpoint_artifact(model, step=step)
