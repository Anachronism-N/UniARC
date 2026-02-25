import os
from typing import List

def calculate_accuracy_from_files(predictions_path: str, truths_path: str) -> float:
    try:
        # 修改点：不再过滤空行，如果为空则赋值为 "[EMPTY]"
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predicted_labels = []
            for line in f:
                parts = line.split()
                # 如果这一行有内容，取第一个；否则给个占位符
                predicted_labels.append(parts[0] if parts else "[EMPTY_PRED]")

        with open(truths_path, 'r', encoding='utf-8') as f:
            true_labels = []
            for line in f:
                parts = line.split()
                true_labels.append(parts[0] if parts else "[EMPTY_TRUE]")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}")
        return 0.0

    # 现在行数应该能对上了
    if len(predicted_labels) != len(true_labels):
        print("错误：行数依然不匹配！")
        print(f"  - 预测文件提取出的标签数: {len(predicted_labels)}")
        print(f"  - 真实标签文件提取出的标签数: {len(true_labels)}")
        # 打印最后几行看看情况
        print(f"预测最后3个: {predicted_labels[-3:]}")
        print(f"真实最后3个: {true_labels[-3:]}")
        return 0.0
        
    total_count = len(true_labels)
    correct_count = sum(1 for p, t in zip(predicted_labels, true_labels) if p == t)
    
    return correct_count / total_count

if __name__ == "__main__":
    BASE_DIR = "/commondocument/group2/ASRCompare/code/inference/8B/DAC_IC"
    pred_file = os.path.join(BASE_DIR, "output.txt")
    true_file = os.path.join(BASE_DIR, "target.txt")
    
    accuracy_score = calculate_accuracy_from_files(pred_file, true_file)
    
    print(f"工作目录: {BASE_DIR}")
    print("-" * 20)
    print(f"计算出的正确率: {accuracy_score:.4f}")
    print(f"正确率 (百分比): {accuracy_score * 100:.2f}%")