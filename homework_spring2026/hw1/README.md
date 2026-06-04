# 作业 1：模仿学习

## 环境配置

本项目使用 `uv` 进行包管理。`uv` 是 [Astral](https://astral.sh) 推出的 Python 包与环境管理工具。它用一个简单统一的接口替代了
`pip`、`pipx`、`conda` 和 `virtualenv` 等工具，而且速度也比以往的工具更快。

### 安装 `uv`

在终端中运行以下命令：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装完成后，重新打开一个终端，让 `uv` 出现在你的 `PATH` 中。

### 始终使用 `uv run`

**不要**直接运行 `python` 或 `pip`。请始终通过 `uv run` 执行脚本，这样依赖和环境会被自动处理。如果你想新增依赖，可以使用 `uv add`。它会把依赖写入 `pyproject.toml`，更新 `uv.lock`，并把包安装到你的虚拟环境中。

示例：

```bash
uv run src/hw1_imitation/train.py --help
```

使用提供的起始代码时，这条命令应该可以直接运行。

## Weights & Biases（wandb）登录

这些作业使用 [Weights & Biases（WandB）](https://wandb.ai) 来追踪实验。WandB 是一个用于记录和可视化机器学习实验的工具，学术用途可免费使用。在运行训练脚本之前，你需要使用自己的 API key 登录 WandB。

```bash
uv run wandb login
```

按照提示粘贴你的 API key。

## 使用 Modal

**注意：这个作业大概率不需要使用 Modal。测试中，本地笔记本 CPU 的训练速度比 Modal 更快。不过后续作业你可能会用到 Modal，所以如果你想先配置好环境，可以参考下面的说明：**

首先，注册一个 Modal 账号。你通常会获得 30 美元的免费额度，这对本次作业来说完全够用。之后，你可以用下面的命令在 Modal 上训练：

```bash
uv run modal run src/hw1_imitation/modal_train.py
```

这条命令会构建一个 Modal 容器并远程启动训练。你可以传入与本地训练脚本相同的参数。如果你已经在本地登录了 WandB，API key 会被自动转发到 Modal 容器中。

日志和检查点会被保存到名为 `hw1-imitation-volume` 的 Modal volume 中。你可以用下面的命令查看日志：

```bash
uv run modal volume ls hw1-imitation-volume exp
```

然后，你可以使用类似下面的命令把日志和检查点下载到本地机器：

```bash
uv run modal volume get hw1-imitation-volume exp/<experiment_name>
```
