import json
import torch
from aac_metrics import Evaluate

def evaluate_clotho(pred_path, ref_path):
    """
    针对无 ID、按行对齐的文件进行评估
    pred_path: 推理结果文件，每行一条描述
    ref_path: 参考答案文件，每行格式为 "描述1|描述2|描述3..."
    """
    
    # 1. 加载推理结果 (Predictions)
    print(f"正在读取推理结果: {pred_path}")
    candidates = []
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                candidates.append(line)

    # 2. 加载参考标准 (References)
    print(f"正在读取参考答案: {ref_path}")
    multireferences = []
    with open(ref_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                # 使用 | 拆分，并过滤掉可能存在的空字符串
                refs = [r.strip() for r in line.split('|') if r.strip()]
                multireferences.append(refs)

    # 3. 检查行数是否匹配
    num_preds = len(candidates)
    num_refs = len(multireferences)
    
    print(f"预测样本数: {num_preds}")
    print(f"参考样本数: {num_refs}")

    if num_preds != num_refs:
        print(f"⚠️ 警告：行数不一致！将按最小行数 ({min(num_preds, num_refs)}) 进行截断对齐评估。")
        min_len = min(num_preds, num_refs)
        candidates = candidates[:min_len]
        multireferences = multireferences[:min_len]
    else:
        print("✅ 行数完全匹配，开始评估...")

    if not candidates:
        print("错误：没有可评估的数据。")
        return

    # 4. 设置运行设备
    # 注意：如果你的 GPU 2 显存满了，可以手动指定 device="cuda:0" 或 "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 5. 调用 aac-metrics 进行评估
    # 包含核心指标：SPIDEr, SPICE, CIDEr (SPIDEr会自动计算CIDEr), FENSE
    # 注意：第一次运行 FENSE 或 SPICE 可能会下载预训练模型或 Java 依赖
    print("正在计算指标 (SPIDEr, SPICE, FENSE, CIDEr)... 这可能需要 1-3 分钟")
    
    # 初始化评估器
    evaluate = Evaluate(metrics=["spider", "spice", "fense"], device=device)
    
    # 计算得分
    corpus_scores, _ = evaluate(candidates, multireferences)

    # 6. 格式化输出结果
    print("\n" + "="*40)
    print("         Clotho 评估报告")
    print("="*40)
    
    # 定义需要展示的指标
    # 注意：spider 内部包含 cider，可以直接取出
    display_metrics = ['spider', 'cider', 'spice', 'fense']
    
    for m in display_metrics:
        if m in corpus_scores:
            val = corpus_scores[m].item()
            print(f"{m.upper():<10}: {val:.4f}")
        else:
            # 有时某些指标名在输出中可能带后缀，这里做个兼容
            found = False
            for k in corpus_scores.keys():
                if m in k:
                    print(f"{k.upper():<10}: {corpus_scores[k].item():.4f}")
                    found = True
            if not found:
                print(f"{m.upper():<10}: 未计算")

    print("="*40)
    print("评估完成！")

if __name__ == "__main__":
    # --- 请确认文件路径 ---
    MY_PREDS = "/commondocument/group2/ASRCompare/code/inference/8B/DAC_song/output.txt" 
    GOLD_REFS = "/commondocument/group2/ASRCompare/code/inference/8B/DAC_song/target.txt" 
    
    try:
        evaluate_clotho(MY_PREDS, GOLD_REFS)
    except Exception as e:
        print(f"\n运行出错: {e}")
        print("提示：如果遇到 CUDA OOM，请尝试在脚本中设置 device='cpu' 或清空显存占用。")