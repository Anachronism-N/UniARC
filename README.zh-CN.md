# UniARC

本仓库整合论文 **Discrete vs. Continuous: A Comprehensive Study of Unified Audio
Understanding in LALMs** 的两条实验路线，比较语音、环境声和音乐中的连续特征与离散表征。

**两套代码已经放在这一个仓库中**：原 `UniARC/code` 整理到 `uniarc/`，
原 `UniARC_paper/xares-llm` 整理到 `xares-llm/`。只需获取本仓库，按下面对应路线使用，
无需再分别下载两个原仓库。

[English](README.md) · [论文与代码对应](docs/paper_mapping.md) ·
[数据和模型准备](docs/data_and_models.md) · [合并记录](docs/migration.md)

| 实验路线 | 代码目录 | 语言模型 | 训练方式 |
| --- | --- | --- | --- |
| 参数高效微调 | `xares-llm/` | SmolLM2 135M / 360M | 投影层与 LoRA，20 个任务、9 类编码器 |
| 冻结骨干探测 | `uniarc/` | Llama 1B / 8B | 冻结语言骨干，训练适配层，7 类任务覆盖 8 个数据集 |

论文以 **UniARC** 作为统一框架名称，XARES-LLM 是微调路线使用的上游框架。
源码中保留了 WavTokenizer 的冻结探测扩展，但论文表 2 没有报告这一编码器。

本版本包含源码、可配置入口、说明文档、来源记录与检查工具；数据集、预训练权重和
已训练适配层需要另行准备。具体实现与论文描述的差异见论文对应文档，不能把源码检查
或小规模测试视为论文结果已经复现。

## 快速开始

```bash
git clone https://github.com/Anachronism-N/UniARC.git
cd UniARC
```

建议在 Linux + NVIDIA GPU 环境训练。两条路线使用独立虚拟环境，安装步骤分别见
[冻结探测说明](uniarc/README.md) 和 [XARES-LLM 说明](xares-llm/README.md)。
不要将两套依赖强行安装到同一环境。

在仓库根目录，可先使用 Python 3.10+ 执行不需要下载模型的检查：

```bash
python run.py --help
python scripts/check_release.py
python -m unittest discover -s tests -v
python run.py probe --help
python run.py --show-command xares example/hubert-large/hubertlarge.py cremad cremad
```

`--show-command` 只展示要执行的命令与工作目录。

### 使用 UniARC 冻结探测代码

在仓库根目录创建独立环境（以下为 Bash 命令）：

```bash
python3.11 -m venv .venv-probe
source .venv-probe/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/probe.txt
```

按 [uniarc/README.md](uniarc/README.md) 准备数据和模型，并修改 JSON 配置中的位置。
检查配置及训练：

```bash
python run.py probe train --config uniarc/configs/hubert_asr.json --dry-run
python run.py probe train --config uniarc/configs/hubert_asr.json
```

推理前，在 `hubert_asr_infer.json` 中填写训练得到的 `checkpoint` 和 `data.test`：

```bash
python run.py probe infer --config uniarc/configs/hubert_asr_infer.json
```

推理结果为配置的输出目录下的 `predictions.jsonl`，评估方法见
[评估说明](docs/evaluation.md)。

### 使用 XARES-LLM 微调代码

另开一个终端，在同一仓库根目录创建 XARES 环境（Bash）：

```bash
python3.11 -m venv .venv-xares
source .venv-xares/bin/activate
python -m pip install --upgrade pip
python -m pip install -c xares-llm/requirements-constraints.txt -e ./xares-llm
```

需要与 CUDA 匹配的 PyTorch，以及供 TorchCodec 使用的 FFmpeg；具体安装和模型资源见
[xares-llm/README.md](xares-llm/README.md)。下面命令训练并评估 CREMA-D：

```bash
python run.py xares example/hubert-large/hubertlarge.py cremad cremad \
  --args '{"decoder_model_name":"HuggingFaceTB/SmolLM2-135M","projector_type":"mlp","lora_enabled":true}'
```

将模型名改为 `HuggingFaceTB/SmolLM2-360M` 可切换到另一规模。离散编码器还需要准备
对应码本或权重；详见后端说明。该命令仅运行一个实验，不会自动完成整篇论文的全部实验。

训练完成后自动评估，结果写入
`xares-llm/experiments/<config>/<decoder>/<encoder>/scores.tsv`。

统一入口会在 `xares-llm/` 目录内运行微调后端，其编码器和自定义 YAML 路径相对于该目录；
冻结后端的工作目录是仓库根目录。`--python` 可指定另一个虚拟环境的 Python 可执行文件。

## 本次整理内容

- 统一论文名称、双路线结构、启动入口和数据准备说明。
- 将研究脚本中的机器路径改为可配置资源，修复影响迁移的代码问题。
- 保留必要代码与任务配置，排除日志、缓存、旧输出、Git 备份和无关实验。
- 采用 Apache-2.0，保留上游版权与文件来源，提供软件引用信息。
- 增加检查脚本和持续集成配置，实际执行范围见[验证记录](docs/validation.md)。

## 引用与许可证

附件论文作者栏为匿名提交，因此 `CITATION.cff` 暂时仅记录软件引用。
正式论文作者顺序、DOI 和发表信息确认后再填写，旧仓库中的旧标题和引用未直接沿用。

项目代码采用 [Apache-2.0](LICENSE)。第三方声明见 [NOTICE](NOTICE)，数据与模型权重
遵循各自许可。本仓库只发布代码及配置，不附带训练数据或模型权重。
