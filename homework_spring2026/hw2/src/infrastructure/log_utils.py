import copy
import json
import os
import pickle
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import torch
from torch import nn
import wandb
from PIL import Image, ImageEnhance


class Logger:
    """用于记录指标的记录器。"""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)
        self.rows = []

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

        wandb.log(row, step=step)
        self.rows.append(copy.deepcopy(row))

    def log_trajs_as_videos(self, trajs, step, max_videos_to_save=2, fps=10, video_title='video'):
        videos = [traj['image_obs'] for traj in trajs][:max_videos_to_save]
        video = get_wandb_video(videos, fps=fps)
        wandb.log({video_title: video}, step=step)

    def close(self):
        if self.file is not None:
            self.file.close()


def remove_functions(obj):
    if isinstance(obj, dict):
        return {
            k: remove_functions(v)
            for k, v in obj.items()
            if not callable(v)
        }
    elif isinstance(obj, list):
        return [remove_functions(v) for v in obj if not callable(v)]
    elif callable(obj):
        return None
    else:
        return obj


def dump_log(agent: nn.Module, logger: Logger, args, save_dir: str):
    """将日志转储到 pkl 文件。"""
    cur_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    config = vars(args)
    config = remove_functions(config)

    data = {
        'log': logger.rows,
        'log_hash': hash(json.dumps(str(logger.rows), sort_keys=True)),
        'config': config,
        'config_hash': hash(json.dumps(str(config), sort_keys=True)),
        'time': cur_time,
    }

    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(config, f)
    with open(os.path.join(save_dir, f'log.pkl'), 'wb') as f:
        pickle.dump(data, f)

    torch.save(agent.state_dict(), os.path.join(save_dir, 'agent.pt'))


def get_flag_dict():
    """返回 flags 的字典。"""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
    config=None,
):
    """设置 Weights & Biases 用于日志记录。"""
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=config,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """辅助函数，用于调整视频形状。"""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # 将 n_cols 设置为视频数量的平方根。
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """返回一个 Weights & Biases 视频。

    它接收一个视频列表，并将它们重塑为具有指定列数的单个视频。

    参数:
        renders: 视频列表。每个视频应该是一个形状为 (t, h, w, c) 的 numpy 数组。
        n_cols: 重塑后视频的列数。如果为 None，则设置为视频数量的平方根。
    """
    # 将视频填充到相同长度。
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # 降低填充帧的亮度。
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # 添加边框。
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')
