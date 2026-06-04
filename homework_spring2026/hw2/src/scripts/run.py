import argparse
import os
from datetime import datetime
import time

import gym
import numpy as np
import torch
import tqdm

from agents.pg_agent import PGAgent
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log

MAX_NVIDEO = 2


def run_training_loop(logger, args):
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # 创建 gym 环境
    env = gym.make(args.env_name, render_mode=None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    max_ep_len = args.ep_len or env.spec.max_episode_steps

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # 模拟时间步，将用于保存视频
    if hasattr(env, "model"):
        fps = 1 / env.dt
    else:
        fps = env.env.metadata["render_fps"]

    # 初始化智能体
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter):
        print(f"\n********** 迭代 {itr} ************")
        # 使用 utils.sample_trajectories 采样 `args.batch_size` 个转移样本
        # 确保使用 `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env, agent.actor, args.batch_size, max_ep_len
        )
        total_envsteps += envsteps_this_batch

        # trajs 应该是一个由 NumPy 数组字典组成的列表，每个字典对应一条轨迹。
        # 这一行将其转换为单个字典，其中每个键对应一个 NumPy 数组列表。
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        # 使用采样的轨迹和智能体的 update 函数训练智能体
        train_info = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
            trajs_dict["terminal"],
        )

        if itr % args.scalar_log_freq == 0:
            # 保存评估指标
            print("\n正在为评估收集数据...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # 计算额外指标
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # 执行日志记录
            for key, value in logs.items():
                print("{} : {}".format(key, value))
            logger.log(logs, itr)
            print("日志记录完成...\n\n", flush=True)

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\n正在收集视频 rollout...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

    dump_log(agent, logger, args, args.save_dir)


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--exp_name", type=str, default='exp')
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # 每次训练迭代收集的步数
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # 每次评估迭代收集的步数

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # 学生不应将此参数从环境的默认值更改
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    args = parser.parse_args(args=args)

    return args


def main(args):
    # 创建日志目录
    logdir_prefix = "exp"  # 为自动评分器保留

    exp_name = f"{args.env_name}_{args.exp_name}_sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = vars(args)
    setup_wandb(project='cs285_hw2', name=exp_name, config=config)
    args.save_dir = os.path.join(logdir_prefix, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(os.path.join(args.save_dir, 'log.csv'))

    run_training_loop(logger, args)


if __name__ == "__main__":
    args = setup_arguments()
    main(args)
