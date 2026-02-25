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
│   │   ├── dataloader_continus.py         # 连续表征数据加载
│   │   ├── dataloader_discrete.py         # 离散表征数据加载
│   │   └── dataloader_continus_wavtokenizer.py  # WavTokenizer/DAC 数据加载
│   ├── model/                         # 模型定义
│   │   ├── model_JTFS_continus.py         # JTFS LM + 连续编码器
│   │   ├── model_JTFS_discrete.py         # JTFS LM + 离散编码器
│   │   ├── model_llama2_continus.py       # Llama + 连续编码器 (HuBERT)
│   │   ├── model_llama2_continus_prompt.py    # Llama + HuBERT + Prompt
│   │   ├── model_llama2_discrete.py       # Llama + 离散编码器
│   │   ├── model_llama2_DAC_prompt.py     # Llama + DAC + Prompt
│   │   ├── model_llama2_speechtokenizer_prompt.py  # Llama + SpeechTokenizer
│   │   ├── model_llama2_wavlm_prompt.py   # Llama + WavLM + Prompt
│   │   ├── model_llama2_wavtokenizer_prompt.py  # Llama + WavTokenizer
│   │   ├── model_llama2_xcodec_prompt.py  # Llama + XCodec
│   │   └── model_llama2_classifier.py     # 分类器模型
│   ├── module/                        # 基础模块
│   │   ├── transformer.py                # Transformer 层实现
│   │   ├── embedding.py                   # 嵌入层
│   │   ├── layers.py                      # 基础层
│   │   ├── codec.py                       # 编解码器工具
│   │   ├── modeling_llama.py              # 自定义 LLaMA 实现
│   │   ├── modeling_llama_huggingface.py  # HuggingFace LLaMA
│   │   └── utils.py                       # 工具函数
│   ├── train/                         # 训练脚本
│   │   ├── train_JTFS_continus.py         # JTFS 连续训练
│   │   ├── train_JTFS_discrete.py         # JTFS 离散训练
│   │   ├── train_llama_continus.py        # Llama + HuBERT 训练
│   │   ├── train_llama_continus_wavlm.py  # Llama + WavLM 训练
│   │   ├── train_llama_discrete_DAC.py    # Llama + DAC 训练
│   │   ├── train_llama_discrete_speechtokenizer.py  # Llama + SpeechTokenizer 训练
│   │   ├── train_llama_discrete_wavtokenizer.py     # Llama + WavTokenizer 训练
│   │   └── train_llama_discrete_xcodec.py # Llama + XCodec 训练
│   ├── inference/                     # 推理脚本与结果
│   │   ├── inference_JTFS_continus.py     # JTFS 连续推理
│   │   ├── inference_JTFS_discrete.py     # JTFS 离散推理
│   │   ├── inference_llama_continus.py    # Llama 连续推理
│   │   ├── inference_llama_discrete.py    # Llama 离散推理
│   │   └── ...                            # 其他推理脚本 + 结果目录
│   └── test/                          # 评估脚本
│       ├── librispeech_test.py            # ASR 评估 (WER/CER/iWER)
│       ├── ER_test.py                     # 情感识别评估
│       └── clotho_test.py                 # 音频描述评估
├── model/                             # 预训练模型权重
│   ├── Llama-3.2-1B/                      # Llama 3.2-1B 模型
│   ├── Meta-Llama-3.1-8B/                 # Llama 3.1-8B 模型
│   ├── hubert-large-ls960-ft/             # HuBERT Large 模型
│   ├── wavlm/                             # WavLM 模型
│   └── speechtokenizer/                   # SpeechTokenizer 模型
├── data/                              # 数据集（已被 .gitignore 排除）
├── fig/                               # 论文图表
├── docs/                              # 项目文档
├── requirements.txt                   # Python 依赖
└── readme.md                          # 原始 README
```

## 技术栈

- **深度学习框架**：PyTorch 2.5 + PyTorch Lightning 2.x
- **LLM**：HuggingFace Transformers 4.44
- **语音处理**：torchaudio, librosa, soundfile
- **离散编解码**：DAC, SpeechTokenizer, WavTokenizer, XCodec
- **评估指标**：jiwer (WER/CER/iWER), sacrebleu, torchmetrics
- **训练策略**：DDP 多卡分布式训练, bf16/fp32 混合精度, 梯度累积
- **可视化**：TensorBoard
