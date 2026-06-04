# 作业 2: 策略梯度 (Policy Gradients)

## 环境配置

关于通用环境配置和 Modal 的说明，请参见作业 1 的 README。

## 示例命令

以下是一些示例命令。请在 `hw2` 目录下运行它们。

* 在本地机器上运行:
  ```bash
  uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
  ```


* 在 Modal 上运行:
  ```bash
  uv run modal run src/scripts/modal_run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
  ```
  * 注意，对于本次作业，Modal 可能不是必需的。
在测试中，在本地笔记本电脑 CPU 上的训练速度比在 Modal 上快得多。
但如果你愿意，仍然可以使用 Modal。
  * 你可以通过在 `src/scripts/modal_run.py` 中更改变量来请求不同的 GPU 类型、CPU 核心数和内存大小
  * 使用 `modal run --detach` 可让你的任务在后台持续运行。

## 故障排除

* 如果在安装 `box2d-py` 时看到关于 `swig` 的错误，你可能需要在机器上安装 `swig` 和 `cmake`。
如果你使用的是 Mac 并且已安装 Homebrew，可以运行 `brew install swig cmake`。
在 Modal 上，这些应该已经安装好了。
