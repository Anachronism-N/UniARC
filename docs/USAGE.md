# UniARC 使用说明

本文档详细介绍如何使用 UniARC 框架进行语音表征的对比实验，包括环境配置、数据准备、模型训练、推理和评估的完整流程。

## 目录

- [1. 环境配置](#1-环境配置)
- [2. 预训练模型准备](#2-预训练模型准备)
- [3. 数据准备](#3-数据准备)
- [4. 训练模型](#4-训练模型)
- [5. 推理评估](#5-推理评估)
- [6. 评估指标计算](#6-评估指标计算)
- [7. 常见问题](#7-常见问题)

---

## 1. 环境配置

### 1.1 基础环境

- Python 3.10+
- CUDA 12.1
- PyTorch 2.5+
- 推荐使用 NVIDIA A100/V100 GPU（至少需要 4 块 GPU 进行分布式训练）

### 1.2 安装依赖

```bash
# 使用 pip 安装
pip install -r requirements.txt

# 或者使用 conda（推荐）
conda env create -f code/environment.yml
```

### 1.3 关键依赖说明

| 包名 | 版本 | 说明 |
|------|------|------|
| `torch` | 2.5.1+cu121 | 深度学习框架 |
| `lightning` | 2.0.2 | PyTorch Lightning 训练框架 |
| `transformers` | 4.44.2 | HuggingFace Transformers |
| `dac` | - | Descript Audio Codec |
| `speechtokenizer` | - | 语音 Token 化工具 |
| `jiwer` | 4.0.0 | 语音识别评估 |
| `torchaudio` | 2.5.1 | 音频处理 |

---

## 2. 预训练模型准备

将以下预训练模型下载到 `model/` 目录下：

### 2.1 LLM 后端

```bash
# Llama 3.2-1B（需要 Meta 许可）
# 下载到: model/Llama-3.2-1B/

# Llama 3.1-8B（可选，需要更多显存）
# 下载到: model/Meta-Llama-3.1-8B/
```

### 2.2 语音编码器

```bash
# HuBERT Large（用于连续表征实验）
# 下载到: model/hubert-large-ls960-ft/
# 来源: https://huggingface.co/facebook/hubert-large-ls960-ft

# WavLM（用于连续表征实验）
# 下载到: model/wavlm/

# SpeechTokenizer（用于离散表征实验）
# 下载到: model/speechtokenizer/

# DAC 16kHz（自动下载）
# 代码中使用 dac.utils.download(model_type="16khz") 自动获取
```

---

## 3. 数据准备

### 3.1 数据格式

每个数据集需要准备两个文件：

- **`.scp` 文件**：音频索引文件（含有 utterance ID 和音频文件路径/偏移信息）
- **`.txt` 文件**：标签文件（每行格式为 `<utterance_id> <text/label>`）

`.scp` 文件格式：
```
<总样本数>
<utterance_id> <seek_position> <num_bytes>
<utterance_id> <seek_position> <num_bytes>
...
```

`.txt` 文件格式：
```
<utterance_id> <transcription or label>
<utterance_id> <transcription or label>
...
```

### 3.2 支持的数据集

将数据集放在 `data/` 目录下，目录结构如下：

```
data/
├── Librispeech/                 # ASR 数据集
│   ├── train/
│   │   ├── train-clean-100.scp
│   │   ├── train-clean-100.txt
│   │   ├── train-all-960.scp
│   │   └── train-all-960.txt
│   ├── dev_clean/
│   └── test_clean/
├── iemocap_4class_data/         # 情感识别 (4分类)
│   ├── train.scp / train_labels.txt
│   ├── val.scp / valid_labels.txt
│   └── test.scp / test_labels.txt
├── MELD/                        # 情感识别 (多模态)
├── cremad_final_split/          # 情感识别
├── GTZAN/                       # 音乐分类
├── clotho/                      # 音频描述
├── us8k/                        # UrbanSound8K
├── SLURP_intent/                # 意图分类
├── sdd_final_split/             # 歌曲描述
└── FSD50K/                      # 声音事件检测
```

### 3.3 数据预处理工具

`data/` 目录下提供了多个数据预处理脚本：

```bash
# LibriSpeech ASR 数据转换
python data/transdata_ASR.py

# IEMOCAP 情感数据转换
python data/transdata_ER_IEMOCAP.py

# MELD 情感数据转换
python data/transdata_ER_MELD.py

# GTZAN 音乐分类数据转换
python data/transdata_GTZAN.py

# SLURP 意图分类数据转换
python data/transdata_SLURP.py

# Clotho 测试数据转换
python data/transdata_clotho_test.py

# UrbanSound8K 数据转换
python data/transdata_urbansound.py
```

---

## 4. 训练模型

### 4.1 训练配置

所有训练脚本位于 `code/train/` 目录下。训练前需要修改脚本中的以下配置：

```python
# 1. 模型路径
hubert_ckpt_path = "/your/path/to/hubert-large-ls960-ft"
llama_ckpt_path = "/your/path/to/Llama-3.2-1B"

# 2. 数据集路径（取消注释需要的数据集）
trainset = ASRDataset("path/to/train.scp", "path/to/train.txt")
valset = ASRDataset("path/to/val.scp", "path/to/val.txt")

# 3. 训练参数
batchsize = 8                    # 批次大小
devices = [4, 5, 6, 7]          # GPU 设备 ID
precision = "bf16-mixed"         # 混合精度
accumulate_grad_batches = 4      # 梯度累积步数
max_epochs = 150                 # 最大训练轮数
```

### 4.2 训练示例

#### 连续表征 — HuBERT + Llama

```bash
# 修改 code/train/train_llama_continus.py 中的数据路径，然后执行：
cd /path/to/ASRCompare
python code/train/train_llama_continus.py
```

#### 连续表征 — WavLM + Llama

```bash
python code/train/train_llama_continus_wavlm.py
```

#### 离散表征 — DAC + Llama

```bash
python code/train/train_llama_discrete_DAC.py
```

#### 离散表征 — SpeechTokenizer + Llama

```bash
python code/train/train_llama_discrete_speechtokenizer.py
```

#### 离散表征 — WavTokenizer + Llama

```bash
python code/train/train_llama_discrete_wavtokenizer.py
```

#### 离散表征 — XCodec + Llama

```bash
python code/train/train_llama_discrete_xcodec.py
```

#### JTFS LM 训练

```bash
# 连续表征
python code/train/train_JTFS_continus.py

# 离散表征
python code/train/train_JTFS_discrete.py
```

### 4.3 训练监控

训练日志保存在 `code/train/log/` 目录下，使用 TensorBoard 查看：

```bash
tensorboard --logdir code/train/log/ --port 6006
```

### 4.4 Checkpoint 管理

- Checkpoint 保存在 `code/model/ckpt/` 下对应的子目录中
- 默认保存验证损失最低的 Top-2 模型
- 支持 EarlyStopping（默认 patience=5~30 epochs）

---

## 5. 推理评估

### 5.1 推理配置

推理脚本位于 `code/inference/` 目录下。需要修改：

```python
# 1. 模型路径
hubert_ckpt_path = "/your/path/to/hubert-large-ls960-ft"
llama_ckpt_path = "/your/path/to/Llama-3.2-1B"

# 2. Checkpoint 路径（加载训练好的模型）
model = IS.load_from_checkpoint(
    "/path/to/your/checkpoint.ckpt",
    hubert_ckpt_path=hubert_ckpt_path,
    llama_ckpt_path=llama_ckpt_path,
    layer=24,
    map_location=device,
    strict=False
)

# 3. 测试数据路径（取消注释需要的数据集）
test_set_clean = ASRDataset("path/to/test.scp", "path/to/test.txt")

# 4. 输出目录
fdir = "/path/to/output_dir"
```

### 5.2 推理示例

```bash
# 连续表征推理
python code/inference/inference_llama_continus.py

# 离散表征 - DAC 推理
python code/inference/inference_llama_distrete_DAC.py

# 离散表征 - SpeechTokenizer 推理
python code/inference/inference_llama_distrete_speechtokenizer.py
```

### 5.3 推理输出

推理结果保存在指定的输出目录中，包含两个文件：

- `target.txt`：Ground Truth 标签
- `output.txt`：模型预测结果

---

## 6. 评估指标计算

### 6.1 ASR 评估

使用 `code/test/librispeech_test.py` 计算语音识别相关指标：

```python
# 修改脚本末尾的路径，指向推理结果目录
compute_metrics("/path/to/inference/results")
```

**输出指标**：
- **WER (Word Error Rate)**：词错误率，越低越好
- **CER (Character Error Rate)**：字符错误率，越低越好
- **iWER (Information Word Error Rate)**：信息正确率 = max(0, 1 - WER)，越高越好

### 6.2 情感识别评估

```python
# 使用 code/test/ER_test.py
python code/test/ER_test.py
```

### 6.3 音频描述评估

```python
# 使用 code/test/clotho_test.py
python code/test/clotho_test.py
```

---

## 7. 常见问题

### Q1: 显存不足怎么办？

- 减小 `batchsize`（建议从 8 降到 4 或 2）
- 使用 `bf16-mixed` 混合精度训练
- 增大 `accumulate_grad_batches` 来保持等效 batch size
- 使用 1B 参数的 Llama 3.2 替代 8B 参数版本

### Q2: 如何切换不同任务的数据集？

在训练和推理脚本中，通过注释/取消注释来切换数据集路径：

```python
# 注释掉当前使用的数据集
# trainset = ASRDataset("path/to/current_train.scp", "path/to/current_train.txt")

# 取消注释目标数据集
trainset = ASRDataset("path/to/target_train.scp", "path/to/target_train.txt")
```

### Q3: 如何使用 8B 模型？

```python
# 修改 llama_ckpt_path 为 8B 模型路径
llama_ckpt_path = "/path/to/Meta-Llama-3.1-8B"

# 8B 模型的推理结果保存在 code/inference/8B/ 目录
# 8B 模型的训练日志保存在 code/train/8B_log/ 目录
```

### Q4: 如何从 Checkpoint 恢复训练？

```python
# 在训练脚本中，取消注释恢复训练代码
ckpt_path = "/path/to/your/checkpoint.ckpt"
trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
```

### Q5: DAC 模型路径如何获取？

DAC 模型会通过 `dac.utils.download()` 自动下载：

```python
import dac
DAC_model_path = dac.utils.download(model_type="16khz")
```

---

## 引用

如使用本项目代码，请引用：

```bibtex
@inproceedings{uniarc2024,
  title={Comparative Analysis of Discrete and Continuous Space LLMs in Speech Recognition},
  booktitle={Interspeech 2024},
  year={2024}
}
```
