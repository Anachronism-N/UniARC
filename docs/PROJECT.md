# UniARC: Unified Audio Representation Comparison

## 项目简介

UniARC（Unified Audio Representation Comparison）是一个统一的音频表征比较框架，源自论文 *"Comparative Analysis of Discrete and Continuous Space LLMs in Speech Recognition"*（投稿至 Interspeech 2024）。该框架系统地比较了**离散（Discrete）**与**连续（Continuous）**两种语音表征方案在大语言模型（LLM）驱动的语音识别及多种音频理解任务中的表现差异。

## 研究背景

近年来，大语言模型（LLM）在语音识别（ASR）等任务中展现出强大的能力。然而，语音信号输入 LLM 之前需要被编码为合适的表征形式，当前主要有两类方案：

- **连续表征（Continuous Representation）**：使用预训练语音编码器（如 HuBERT、WavLM）提取连续特征向量，通过线性投影层对齐到 LLM 的输入空间。
- **离散表征（Discrete Representation）**：使用语音编解码器（如 DAC、SpeechTokenizer、WavTokenizer）将语音量化为离散 Token 序列，直接作为 LLM 的输入。

本项目旨在通过统一的实验框架，公平地评估这两类方案在多项下游任务上的性能。

## 核心架构

### LLM 后端

| 模型 | 描述 |
|------|------|
| **JTFS LM** | 自定义 Transformer 解码器语言模型，基于 PyTorch Lightning 实现 |
| **Llama 3.2-1B** | Meta 发布的 1B 参数 Llama 模型 |
| **Llama 3.1-8B** | Meta 发布的 8B 参数 Llama 模型 |

### 语音编码器

#### 连续编码器

| 编码器 | 描述 | 输出维度 |
|--------|------|----------|
| **HuBERT Large** | Facebook 预训练自监督语音模型 | 1024维 |
| **WavLM** | Microsoft 预训练自监督语音模型 | 1024维 |

#### 离散编码器

| 编码器 | 描述 | 量化方式 |
|--------|------|----------|
| **DAC (Descript Audio Codec)** | 高质量神经音频编解码器，16kHz | 残差向量量化（RVQ） |
| **SpeechTokenizer** | 面向语音的离散 Token 编码器 | 语义+声学分层量化 |
| **WavTokenizer** | 基于 Codec 的波形 Token 化器 | 向量量化 |
| **XCodec** | 通用音频 Codec | 向量量化 |

### 架构设计

```
┌──────────────────────────────────────────────────────────┐
│                      UniARC Framework                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐          ┌───────────────────────────┐  │
│  │  Audio Input │──────────▶  Speech Encoder          │  │
│  └─────────────┘          │  (HuBERT/WavLM/DAC/...)  │  │
│                            └────────────┬──────────────┘  │
│                                         │                 │
│                            ┌────────────▼──────────────┐  │
│                            │  Projection Layer         │  │
│                            │  (Linear / Embedding)     │  │
│                            └────────────┬──────────────┘  │
│                                         │                 │
│                            ┌────────────▼──────────────┐  │
│  ┌─────────────┐          │  LLM Backbone             │  │
│  │ Task Prompt  │──────────▶  (JTFS LM / Llama)       │  │
│  └─────────────┘          └────────────┬──────────────┘  │
│                                         │                 │
│                            ┌────────────▼──────────────┐  │
│                            │  Task-Specific Output     │  │
│                            │  (Text / Label)           │  │
│                            └───────────────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## 支持的下游任务

| 任务类别 | 具体任务 | 数据集 |
|----------|----------|--------|
| 语音识别 (ASR) | 自动语音识别 | LibriSpeech (100h / 960h) |
| 情感识别 (ER) | 语音情感分类 | IEMOCAP (4类), MELD, CREMA-D |
| 意图分类 (IC) | 语音意图识别 | SLURP |
| 音乐分类 (MC) | 音乐体裁分类 | GTZAN |
| 音频描述 (AC) | 音频内容描述生成 | Clotho |
| 声音事件分类 (SEC) | 环境声音分类 | UrbanSound8K |
| 歌曲描述 (SD) | 歌曲描述生成 | Song Describer Dataset |

## 项目结构

```
ASRCompare/
├── code/                              # 源代码
│   ├── dataset/                       # 数据加载器
│   │   ├── dataloader_continus.py         # 连续表征数据加载（读取 .scp/.seq 二进制音频）
│   │   ├── dataloader_discrete.py         # 离散表征数据加载（读取预量化 Token .pt 文件）
│   │   └── dataloader_continus_wavtokenizer.py  # WavTokenizer/DAC 数据加载（读取原始音频）
│   ├── model/                         # 模型定义
│   │   ├── model_JTFS_continus.py         # JTFS LM + Whisper 连续编码器（463行）
│   │   ├── model_JTFS_discrete.py         # JTFS LM + 离散编码器
│   │   ├── model_llama2_continus.py       # Llama + HuBERT 连续编码器（714行）
│   │   ├── model_llama2_continus_prompt.py    # Llama + HuBERT + Prompt 模板
│   │   ├── model_llama2_discrete.py       # Llama + K-Means 离散编码（340行）
│   │   ├── model_llama2_DAC_prompt.py     # Llama + DAC Codec + Prompt（556行）
│   │   ├── model_llama2_speechtokenizer_prompt.py  # Llama + SpeechTokenizer
│   │   ├── model_llama2_wavlm_prompt.py   # Llama + WavLM + Prompt
│   │   ├── model_llama2_wavtokenizer_prompt.py  # Llama + WavTokenizer
│   │   ├── model_llama2_xcodec_prompt.py  # Llama + XCodec
│   │   ├── model_llama2_classifier.py     # 音频分类器模型（端到端分类头）
│   │   └── ckpt/                          # ⚠️ 训练好的 Checkpoint（.gitignore 排除）
│   ├── module/                        # 基础模块
│   │   ├── transformer.py                # 自定义 Transformer 编/解码器层
│   │   ├── embedding.py                   # 位置编码与 Token 嵌入
│   │   ├── layers.py                      # LinearNorm 等基础层
│   │   ├── codec.py                       # 编解码器工具
│   │   ├── modeling_llama.py              # 自定义 LLaMA 实现（含 KV Cache）
│   │   ├── modeling_llama_huggingface.py  # 修改版 HuggingFace LLaMA（禁用 lm_head）
│   │   └── utils.py                       # Padding Mask 工具函数
│   ├── train/                         # 训练脚本
│   │   ├── train_JTFS_continus.py         # JTFS + 连续编码器训练
│   │   ├── train_JTFS_discrete.py         # JTFS + 离散编码器训练
│   │   ├── train_llama_continus.py        # Llama + HuBERT 训练（4-GPU DDP, bf16）
│   │   ├── train_llama_continus_wavlm.py  # Llama + WavLM 训练
│   │   ├── train_llama_discrete_DAC.py    # Llama + DAC 训练（4-GPU DDP, fp32）
│   │   ├── train_llama_discrete_speechtokenizer.py  # Llama + SpeechTokenizer 训练
│   │   ├── train_llama_discrete_wavtokenizer.py     # Llama + WavTokenizer 训练
│   │   ├── train_llama_discrete_xcodec.py # Llama + XCodec 训练
│   │   ├── train_llama_discrete_supervised.py  # 有监督离散训练
│   │   ├── get_time.py                    # 训练耗时统计工具
│   │   ├── loss_fig.py                    # 损失曲线绘图工具
│   │   ├── check_ckpt_params.py           # Checkpoint 参数检查
│   │   ├── log/                           # TensorBoard 训练日志
│   │   └── 8B_log/                        # 8B 模型训练日志
│   ├── inference/                     # 推理脚本与结果
│   │   ├── inference_JTFS_continus.py     # JTFS 连续推理
│   │   ├── inference_JTFS_discrete.py     # JTFS 离散推理
│   │   ├── inference_llama_continus.py    # Llama + HuBERT 连续推理
│   │   ├── inference_llama_continus_wavlm.py  # Llama + WavLM 连续推理
│   │   ├── inference_llama_discrete.py    # Llama K-Means 离散推理
│   │   ├── inference_llama_discrete_supervised.py  # Llama 有监督离散推理
│   │   ├── inference_llama_distrete_DAC.py         # Llama + DAC 推理
│   │   ├── inference_llama_distrete_speechtokenizer.py  # Llama + SpeechTokenizer 推理
│   │   ├── inference_llama_distrete_wavtokenizer.py     # Llama + WavTokenizer 推理
│   │   ├── analyze_ckpt.py                # Checkpoint 参数定量分析
│   │   ├── check_ckpt.py                  # Checkpoint 参数核查
│   │   ├── verify_param_changes.py        # 参数变化验证
│   │   ├── 1B/                            # Llama-3.2-1B 推理结果（见下表）
│   │   └── 8B/                            # Llama-3.1-8B 推理结果（见下表）
│   └── test/                          # 评估脚本
│       ├── librispeech_test.py            # ASR 评估（WER/CER/iWER 三项指标）
│       ├── ER_test.py                     # 情感识别准确率评估
│       └── clotho_test.py                 # 音频描述评估
├── model/                             # 预训练模型（⚠️ 权重文件未上传，见下方说明）
│   ├── getDAC.py                          # DAC 模型自动下载脚本
│   ├── qwen3-omnii.py                     # Qwen3 模型下载脚本
│   ├── Llama-3.2-1B/                      # ⚠️ Llama 3.2-1B 模型权重（~2.4GB）
│   ├── Meta-Llama-3.1-8B/                 # ⚠️ Llama 3.1-8B 模型权重（~15GB）
│   ├── hubert-large-ls960-ft/             # ⚠️ HuBERT Large 模型权重（~2.4GB）
│   ├── wavlm/                             # ⚠️ WavLM 模型权重（~1.4GB）
│   └── speechtokenizer/                   # ⚠️ SpeechTokenizer 模型与源码（~462MB）
├── data/                              # ⚠️ 数据集（未上传，见下方说明）
├── fig/                               # 论文图表
├── docs/                              # 项目文档
├── requirements.txt                   # Python 依赖
└── readme.md                          # 原始 README
```

### 推理结果目录说明

`code/inference/` 下的子目录包含各 编码器×任务 组合的推理结果，每个目录下有 `target.txt`（标注）和 `output.txt`（预测）：

| 编码器 | ASR | ASR (other) | ER | IC | Music | Clotho | US | Song | CREMA-D |
|--------|-----|-------------|----|----|-------|--------|----|------|---------|
| **HuBERT** | `hubert_ASR/` | `hubert_ASR_other/` | `hubert_ER/` | `hubert_IC/` | `hubert_music/` | `hubert_clotho/` | `hubert_us/` | `hubert_song/` | `hubert_cremad/` |
| **WavLM** | `wavlm_ASR_clean/` | `wavlm_ASR_other/` | `wavlm_ER/` | `wavlm_IC/` | `wavlm_music/` | `wavlm_clotho/` | `wavlm_US/` | `wavlm_song/` | `wavlm_cremad/` |
| **DAC** | `DAC_ASR/` | `DAC_ASR_other/` | `DAC_ER/` | `DAC_IC/` | `DAC_music/` | `DAC_clotho/` | — | `DAC_song/` | `DAC_cremad/` |
| **SpeechTokenizer** | `speechtokenizer_ASR/` | `speechtokenizer_ASR_other/` | `speechtokenizer_ER/` | `speechtokenizer_IC/` | `speechtokenizer_music/` | `speechtokenizer_clotho/` | `speechtokenizer_us/` | `speechtokenizer_song/` | `speechtokenizer_cremad/` |
| **WavTokenizer** | `wavtokenizer_ASR_libri_test_clean/` | — | `wavtokenizer_ER/` | — | — | — | — | — | — |

- `1B/` 子目录：使用 Llama-3.2-1B 的推理结果，包含 `hubert_*` 和 `wavlm_*` 子目录
- `8B/` 子目录：使用 Llama-3.1-8B 的推理结果，包含 `DAC_*` 子目录

## 未上传到 GitHub 的文件说明

以下文件因体积过大（超过 GitHub 100MB 文件限制）而未上传至仓库：

### 预训练模型权重 (`model/`)

| 目录 | 大小 | 说明 | 获取方式 |
|------|------|------|----------|
| `Llama-3.2-1B/` | ~2.4 GB | Meta Llama 3.2 1B 参数模型 | [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-1B)（需申请许可） |
| `Meta-Llama-3.1-8B/` | ~15 GB | Meta Llama 3.1 8B 参数模型 | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)（需申请许可） |
| `hubert-large-ls960-ft/` | ~2.4 GB | Facebook HuBERT Large (Fine-tuned) | [HuggingFace](https://huggingface.co/facebook/hubert-large-ls960-ft) |
| `wavlm/` | ~1.4 GB | Microsoft WavLM Large | [HuggingFace](https://huggingface.co/microsoft/wavlm-large) |
| `speechtokenizer/` | ~462 MB | SpeechTokenizer 模型及源码 | [GitHub](https://github.com/ZhangXInFD/SpeechTokenizer) |

### 训练 Checkpoint (`code/model/ckpt/`)

各编码器+任务训练后的最佳模型 Checkpoint（`.ckpt` 文件），被 `.gitignore` 排除。如需复现实验结果，请参照训练脚本重新训练。

### 数据集 (`data/`)

整个 `data/` 目录被 `.gitignore` 排除，本地包含以下数据集：

| 目录 | 数据集 | 说明 | 获取方式 |
|------|--------|------|----------|
| `Librispeech/` | LibriSpeech | 英语 ASR 数据集（100h/960h） | [OpenSLR](https://www.openslr.org/12/) |
| `iemocap_4class_data/` | IEMOCAP | 情感识别（4类：happy/sad/angry/neutral） | 需学术申请 |
| `MELD/` | MELD | 多模态情感识别（Friends 影视对话） | [GitHub](https://github.com/declare-lab/MELD) |
| `cremad_final_split/` | CREMA-D | 语音情感识别（6类情感） | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D) |
| `GTZAN/` | GTZAN | 音乐体裁分类（10类） | [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) |
| `clotho/` | Clotho | 音频描述（Audio Captioning） | [Zenodo](https://zenodo.org/record/3490684) |
| `us8k/` | UrbanSound8K | 环境声音分类（10类） | [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) |
| `SLURP_intent/` | SLURP | 语音意图分类 | [GitHub](https://github.com/pswietojanski/slurp) |
| `sdd_final_split/` | Song Describer | 歌曲描述生成 | [Zenodo](https://zenodo.org/record/10072001) |
| `FSD50K/` | FSD50K | 大规模声音事件检测 | [Zenodo](https://zenodo.org/record/4060432) |
| `kmeans/` | K-Means 特征 | HuBERT 后的 K-Means 离散量化特征 | 由预处理脚本生成 |

`data/` 目录下还包含数据预处理脚本（`transdata_*.py`, `cremad_data.py`, `song_des.py` 等），用于将原始数据集转换为项目所需的 `.scp` + `.txt` 格式。

### 音频与二进制文件

`.wav`, `.mp3`, `.flac`, `.seq`, `.pt`, `.ckpt`, `.safetensors`, `.bin` 等格式文件均被 `.gitignore` 排除。

## 技术栈

- **深度学习框架**：PyTorch 2.5 + PyTorch Lightning 2.x
- **LLM**：HuggingFace Transformers 4.44
- **语音处理**：torchaudio, librosa, soundfile
- **离散编解码**：DAC, SpeechTokenizer, WavTokenizer, XCodec
- **评估指标**：jiwer (WER/CER/iWER), sacrebleu, torchmetrics
- **训练策略**：DDP 多卡分布式训练, bf16/fp32 混合精度, 梯度累积
- **可视化**：TensorBoard
