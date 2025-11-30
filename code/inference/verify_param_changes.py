import torch
import sys
sys.path.append('/commondocument/group2/ASRCompare/code')
from model.model_llama2_continus import IS
import numpy as np

# 加载模型
layer = 24
hubert_ckpt_path = "/commondocument/group2/ASRCompare/model/hubert-large-ls960-ft"
llama_ckpt_path = "/commondocument/group2/ASRCompare/model/Llama-3.2-1B"

print("正在加载模型...")
model = IS(hubert_ckpt_path=hubert_ckpt_path, llama_ckpt_path=llama_ckpt_path, layer=layer)

# 选择一些关键参数进行跟踪
# 1. 自定义层参数 (应该会变化)
# 2. 一些预训练模型参数 (可能会变化)

# 定义要检查的参数路径列表
params_to_check = [
    # 自定义层参数
    "audio_embedding_last_Linear.0.weight",
    "audio_embedding_last_Linear.0.bias",
    "audio_embedding_last_Linear.2.weight",
    "audio_embedding_last_Linear.2.bias",
    # llama模型参数
    "llama.model.layers.0.self_attn.q_proj.weight",
    "llama.model.layers.0.self_attn.v_proj.weight",
    # audio_model参数
    "audio_model.encoder.layers.0.self_attn.q_proj.weight",
    "audio_model.encoder.layers.0.self_attn.v_proj.weight"
]

# 保存加载前的参数值
print("\n=== 记录加载前的参数值 ===")
params_before = {}
for param_name in params_to_check:
    try:
        # 递归查找参数
        param_parts = param_name.split('.')
        param = model
        for part in param_parts:
            param = getattr(param, part)
        
        # 保存参数的统计信息
        params_before[param_name] = {
            "mean": param.data.mean().item(),
            "std": param.data.std().item(),
            "sum": param.data.sum().item(),
            "shape": param.data.shape
        }
        print(f"参数 {param_name}:")
        print(f"  均值: {params_before[param_name]['mean']:.6f}")
        print(f"  标准差: {params_before[param_name]['std']:.6f}")
        print(f"  总和: {params_before[param_name]['sum']:.6f}")
    except Exception as e:
        print(f"警告: 无法访问参数 {param_name}: {e}")

# 加载ckpt文件
ckpt_path = "/commondocument/group2/ASRCompare/code/model/ckpt/epoch=09-train_loss=11.016-val_loss=0.593.ckpt"
print(f"\n正在加载ckpt文件: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt['state_dict']

# 加载权重
print("正在将ckpt权重加载到模型...")
model.load_state_dict(state_dict, strict=False)
print("权重加载完成")

# 保存加载后的参数值
print("\n=== 记录加载后的参数值 ===")
params_after = {}
for param_name in params_to_check:
    try:
        # 递归查找参数
        param_parts = param_name.split('.')
        param = model
        for part in param_parts:
            param = getattr(param, part)
        
        # 保存参数的统计信息
        params_after[param_name] = {
            "mean": param.data.mean().item(),
            "std": param.data.std().item(),
            "sum": param.data.sum().item(),
            "shape": param.data.shape
        }
        print(f"参数 {param_name}:")
        print(f"  均值: {params_after[param_name]['mean']:.6f}")
        print(f"  标准差: {params_after[param_name]['std']:.6f}")
        print(f"  总和: {params_after[param_name]['sum']:.6f}")
    except Exception as e:
        print(f"警告: 无法访问参数 {param_name}: {e}")

# 比较参数变化
print("\n=== 参数变化比较 ===")
param_changed = False
for param_name in params_to_check:
    if param_name in params_before and param_name in params_after:
        # 计算变化百分比
        mean_diff_percent = abs(params_after[param_name]['mean'] - params_before[param_name]['mean']) / max(1e-10, abs(params_before[param_name]['mean'])) * 100
        sum_diff_percent = abs(params_after[param_name]['sum'] - params_before[param_name]['sum']) / max(1e-10, abs(params_before[param_name]['sum'])) * 100
        
        # 判断是否发生显著变化（阈值设为1%）
        if mean_diff_percent > 1.0 or sum_diff_percent > 1.0:
            status = "✓ 已变化"
            param_changed = True
        else:
            status = "✗ 未变化"
        
        print(f"参数 {param_name}: {status}")
        print(f"  均值变化: {params_after[param_name]['mean'] - params_before[param_name]['mean']:.6f} ({mean_diff_percent:.2f}%)")
        print(f"  总和变化: {params_after[param_name]['sum'] - params_before[param_name]['sum']:.6f} ({sum_diff_percent:.2f}%)")
    else:
        print(f"参数 {param_name}: 无法比较")

# 验证加载是否成功
print("\n=== 加载成功验证 ===")

# 1. 检查自定义层是否变化（应该变化）
custom_layers_changed = False
for param_name in params_to_check:
    if "audio_embedding_last_Linear" in param_name and param_name in params_before and param_name in params_after:
        mean_diff_percent = abs(params_after[param_name]['mean'] - params_before[param_name]['mean']) / max(1e-10, abs(params_before[param_name]['mean'])) * 100
        if mean_diff_percent > 1.0:
            custom_layers_changed = True
            break

print(f"自定义层参数是否变化: {'是 ✓' if custom_layers_changed else '否 ✗'}")

# 2. 验证state_dict中参数是否确实加载到模型
mismatch_count = 0
total_checked = 0

print("\n=== 随机抽样验证部分参数精确值 ===")
# 随机选择10个参数进行精确值比较
sample_params = list(state_dict.keys())[:10]  # 取前10个参数作为示例
for param_name in sample_params:
    try:
        # 从state_dict获取值
        state_value = state_dict[param_name]
        
        # 从模型获取值
        param_parts = param_name.split('.')
        model_param = model
        for part in param_parts:
            model_param = getattr(model_param, part)
        
        # 比较是否相同
        is_same = torch.allclose(state_value, model_param.data, atol=1e-6)
        if not is_same:
            mismatch_count += 1
        total_checked += 1
        
        print(f"参数 {param_name}: {'相同' if is_same else '不同'}")
    except Exception as e:
        print(f"警告: 无法比较参数 {param_name}: {e}")

# 总结
print("\n=== 总结 ===")
print(f"1. 参数加载情况: {'部分参数成功加载' if param_changed else '未检测到参数变化'}")
print(f"2. 自定义层参数: {'已成功更新' if custom_layers_changed else '可能未更新'}")
print(f"3. 抽样验证: 检查了{total_checked}个参数，发现{mismatch_count}个不匹配")

if param_changed and custom_layers_changed:
    print("\n结论: 模型权重加载成功，参数确实发生了变化。")
else:
    print("\n结论: 可能存在加载问题，建议进一步检查。")
    print("可能的解决方案:")
    print("1. 确认使用了正确的ckpt文件路径")
    print("2. 检查模型定义是否与保存时一致")
    print("3. 尝试打印更详细的加载日志")
    print("4. 检查是否有参数冻结或命名不一致的问题")