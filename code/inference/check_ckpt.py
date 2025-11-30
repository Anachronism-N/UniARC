import torch
import numpy as np

def compare_ckpts(ckpt_path1, ckpt_path2, tolerance=1e-6):
    """对比两个checkpoint的参数差异"""
    
    # 加载checkpoint
    ckpt1 = torch.load(ckpt_path1, map_location='cpu')
    ckpt2 = torch.load(ckpt_path2, map_location='cpu')
    
    # 提取state_dict
    state_dict1 = ckpt1.get('state_dict', ckpt1)
    state_dict2 = ckpt2.get('state_dict', ckpt2)
    
    print(f"检查点1参数数量: {len(state_dict1)}")
    print(f"检查点2参数数量: {len(state_dict2)}")
    print("-" * 50)
    
    # 检查参数名一致性
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    common_keys = keys1.intersection(keys2)
    only_in_ckpt1 = keys1 - keys2
    only_in_ckpt2 = keys2 - keys1
    
    print(f"共同参数: {len(common_keys)}")
    print(f"仅在ckpt1中的参数: {len(only_in_ckpt1)}")
    print(f"仅在ckpt2中的参数: {len(only_in_ckpt2)}")
    
    if only_in_ckpt1:
        print("\n仅在ckpt1中的参数示例:")
        for key in list(only_in_ckpt1)[:5]:
            print(f"  - {key}")
    
    if only_in_ckpt2:
        print("\n仅在ckpt2中的参数示例:")
        for key in list(only_in_ckpt2)[:5]:
            print(f"  - {key}")
    
    # 对比共同参数的值
    print("\n对比共同参数的值...")
    different_params = []
    identical_params = []
    
    for key in common_keys:
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        # 检查形状是否相同
        if param1.shape != param2.shape:
            different_params.append((key, "形状不同", param1.shape, param2.shape))
            continue
        
        # 计算差异
        if param1.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            diff = torch.max(torch.abs(param1 - param2)).item()
            if diff > tolerance:
                different_params.append((key, f"值不同 (最大差异: {diff:.6e})", param1.shape))
            else:
                identical_params.append(key)
        else:
            # 对于整数类型参数
            if torch.any(param1 != param2):
                different_params.append((key, "整数值不同", param1.shape))
            else:
                identical_params.append(key)
    
    print(f"\n参数对比结果:")
    print(f"相同的参数: {len(identical_params)}")
    print(f"不同的参数: {len(different_params)}")
    
    if different_params:
        print("\n不同的参数详情 (前10个):")
        for i, (key, reason, shape) in enumerate(different_params[:10]):
            print(f"  {i+1}. {key}")
            print(f"     原因: {reason}")
            print(f"     形状: {shape}")
    
    return different_params, identical_params

# 使用示例
if __name__ == "__main__":
    ckpt1 = "/commondocument/group2/ASRCompare/code/model/ckpt/epoch=01-train_loss=19.30-val_loss=17.40.ckpt"
    ckpt2 = "/commondocument/group2/ASRCompare/code/model/ckpt/epoch=997-train_loss=0.00-val_loss=0.00.ckpt"
    
    different, identical = compare_ckpts(ckpt1, ckpt2)