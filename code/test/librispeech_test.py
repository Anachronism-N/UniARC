# import os
# import jiwer
# from jiwer import wer, cer
# import re

# def compute_metrics(filedir):
#     target_path = os.path.join(filedir, "target.txt")
#     output_path = os.path.join(filedir, "output.txt")

#     if not os.path.exists(target_path) or not os.path.exists(output_path):
#         print("错误：找不到 target.txt 或 output.txt")
#         return

#     # 1. 读取文件
#     with open(target_path, "r", encoding="utf-8") as f:
#         refs = f.readlines()
#     with open(output_path, "r", encoding="utf-8") as f:
#         hyps = f.readlines()

#     # 检查行数是否对应
#     if len(refs) != len(hyps):
#         print(f"警告：行数不匹配！Target: {len(refs)}, Output: {len(hyps)}")
#         # 截取到相同长度以避免报错，实际使用中应检查为什么不匹配
#         min_len = min(len(refs), len(hyps))
#         refs = refs[:min_len]
#         hyps = hyps[:min_len]

#     # 2. 文本标准化 (Text Normalization) - 非常重要！
#     # ASR 评估通常忽略大小写和标点符号
#     transform = jiwer.Compose([
#         jiwer.ToLowerCase(),
#         jiwer.RemovePunctuation(),
#         jiwer.RemoveMultipleSpaces(),
#         jiwer.Strip(),
#         jiwer.ExpandCommonEnglishContractions() # 可选：把 I'm 变成 I am
#     ])

#     # 3. 预处理文本
#     refs_clean = [transform(r) for r in refs]
#     hyps_clean = [transform(h) for h in hyps]

#     # 4. 计算指标
#     # 注意：如果有空行（比如模型没预测出结果），jiwer 可能会报错或给出 inf，需要处理
#     # 这里做一个简单的过滤，确保 reference 不为空
#     valid_refs = []
#     valid_hyps = []
#     for r, h in zip(refs_clean, hyps_clean):
#         if len(r.strip()) > 0:
#             valid_refs.append(r)
#             valid_hyps.append(h)
    
#     if len(valid_refs) == 0:
#         print("没有有效的 Reference 数据")
#         return

#     error_rate_word = wer(valid_refs, valid_hyps)
#     error_rate_char = cer(valid_refs, valid_hyps)

#     print("-" * 30)
#     print(f"评估样本数: {len(valid_refs)}")
#     print(f"WER (词错误率): {error_rate_word * 100:.2f}%")
#     print(f"CER (字错误率): {error_rate_char * 100:.2f}%")
#     print("-" * 30)
    
#     # 打印几个例子看看效果
#     print("Example 1:")
#     print(f"Ref: {valid_refs[0]}")
#     print(f"Hyp: {valid_hyps[0]}")


# # 使用方法
# compute_metrics("/commondocument/group2/ASRCompare/code/inference/wavlm_ASR_other")


import os
import jiwer
from jiwer import wer, cer
import re
import string
import unicodedata

# --- 严格复制自 XARES-LLM 源代码的预处理函数 ---
def preprocess_string(text: str, remove_punctuation: bool = True):
    # 1. Unicode NFKC 标准化 (处理全角半角等)
    text = unicodedata.normalize("NFKC", text)
    # 2. 转小写
    text = text.lower()
    # 3. 将连字符替换为空格 (iWER 的核心逻辑：拆分合成词)
    text = text.replace("-", " ")
    # 4. 移除标点
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    # 5. 合并多余空格并修整两端
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_metrics(filedir):
    target_path = os.path.join(filedir, "target.txt")
    output_path = os.path.join(filedir, "output.txt")

    if not os.path.exists(target_path) or not os.path.exists(output_path):
        print("错误：找不到 target.txt 或 output.txt")
        return

    # 读取文件
    with open(target_path, "r", encoding="utf-8") as f:
        refs = f.readlines()
    with open(output_path, "r", encoding="utf-8") as f:
        hyps = f.readlines()

    if len(refs) != len(hyps):
        print(f"警告：行数不匹配！Target: {len(refs)}, Output: {len(hyps)}")
        min_len = min(len(refs), len(hyps))
        refs = refs[:min_len]
        hyps = hyps[:min_len]

    # --- A. 用于标准 WER/CER 的转换器 (保留你原来的逻辑) ---
    standard_transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ExpandCommonEnglishContractions() 
    ])

    # --- B. 数据清洗 ---
    # 1. 标准清洗 (WER/CER)
    refs_std = [standard_transform(r) for r in refs]
    hyps_std = [standard_transform(h) for h in hyps]

    # 2. iWER 专用清洗 (严格匹配 preprocess_string)
    refs_iwer = [preprocess_string(r) for r in refs]
    hyps_iwer = [preprocess_string(h) for h in hyps]

    # 3. 过滤掉 Reference 为空的行
    valid_refs_std, valid_hyps_std = [], []
    valid_refs_iwer, valid_hyps_iwer = [], []

    for rs, hs, ri, hi in zip(refs_std, hyps_std, refs_iwer, hyps_iwer):
        if len(rs.strip()) > 0:
            valid_refs_std.append(rs)
            valid_hyps_std.append(hs)
            valid_refs_iwer.append(ri)
            valid_hyps_iwer.append(hi)

    if not valid_refs_std:
        print("没有有效的 Reference 数据")
        return

    # --- C. 计算三个指标 ---
    
    # 1. WER & CER (标准错误率模式)
    error_rate_word = wer(valid_refs_std, valid_hyps_std)
    error_rate_char = cer(valid_refs_std, valid_hyps_std)

    # 2. iWER (严格按照提供的逻辑：正确率模式)
    # 先算 iWER 预处理后的 WER 错误率
    i_wer_err = wer(valid_refs_iwer, valid_hyps_iwer)
    # 执行公式：max(0, 1.0 - wer_score)
    iwer_val = max(0, 1.0 - i_wer_err)

    # --- D. 输出结果 ---
    print("-" * 45)
    print(f"评估样本数: {len(valid_refs_std)}")
    print(f"WER (词错误率): {error_rate_word * 100:.2f}%  (越低越好)")
    print(f"CER (字错误率): {error_rate_char * 100:.2f}%  (越低越好)")
    print(f"iWER (信息正确率): {iwer_val * 100:.2f}%  (越高越好)")
    print("-" * 45)
    
    # 对比示例
    print("【预处理对比示例】")
    idx = 0
    print(f"原始 Reference:  {refs[idx].strip()}")
    print(f"原始 Prediction: {hyps[idx].strip()}")
    print(f"iWER 预处理后 Ref:  {valid_refs_iwer[idx]}")
    print(f"iWER 预处理后 Hyp:  {valid_hyps_iwer[idx]}")

# 执行
compute_metrics("/commondocument/group2/ASRCompare/code/inference/1B/hubert_ASR_LS100")