import torch
import sys
sys.path.append('/commondocument/group2/ASRCompare/code')
from model.model_llama2_continus import IS

# 加载模型
layer = 24
hubert_ckpt_path = "/commondocument/group2/ASRCompare/model/hubert-large-ls960-ft"
llama_ckpt_path = "/commondocument/group2/ASRCompare/model/Llama-3.2-1B"

print("正在加载模型...")
model = IS(hubert_ckpt_path=hubert_ckpt_path, llama_ckpt_path=llama_ckpt_path, layer=layer)
print("模型加载完成")

# 加载ckpt文件
ckpt_path = "/commondocument/group2/ASRCompare/code/model/ckpt/epoch=09-train_loss=11.016-val_loss=0.593.ckpt"
print(f"正在加载ckpt文件: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

# 分析ckpt文件结构
print("\n=== ckpt文件结构分析 ===")
print(f"ckpt文件包含的键: {list(ckpt.keys())}")

# 检查是否包含state_dict
if 'state_dict' in ckpt:
    print("ckpt文件是标准的PyTorch Lightning检查点")
    state_dict = ckpt['state_dict']
else:
    print("ckpt文件是直接的state_dict")
    state_dict = ckpt

print(f"state_dict包含的参数数量: {len(state_dict)}")

# 分析模型的参数
print("\n=== 模型参数分析 ===")
model_params = {name: param for name, param in model.named_parameters()}
print(f"模型包含的参数数量: {len(model_params)}")

# 比较参数匹配情况
print("\n=== 参数匹配分析 ===")
matching_params = 0
missing_in_ckpt = []
missing_in_model = []

# 检查模型参数在ckpt中是否存在
for param_name in model_params:
    if param_name in state_dict:
        matching_params += 1
    else:
        missing_in_ckpt.append(param_name)

# 检查ckpt参数在模型中是否存在
for param_name in state_dict:
    if param_name not in model_params:
        missing_in_model.append(param_name)

print(f"匹配的参数数量: {matching_params}")
print(f"模型中存在但ckpt中缺失的参数数量: {len(missing_in_ckpt)}")
print(f"ckpt中存在但模型中缺失的参数数量: {len(missing_in_model)}")

# 显示部分缺失的参数示例
if len(missing_in_ckpt) > 0:
    print("\n模型中存在但ckpt中缺失的参数示例（前5个）:")
    for i, param_name in enumerate(missing_in_ckpt[:5]):
        print(f"  {i+1}. {param_name}")
    if len(missing_in_ckpt) > 5:
        print(f"  ... 还有{len(missing_in_ckpt)-5}个参数")

if len(missing_in_model) > 0:
    print("\nckpt中存在但模型中缺失的参数示例（前5个）:")
    for i, param_name in enumerate(missing_in_model[:5]):
        print(f"  {i+1}. {param_name}")
    if len(missing_in_model) > 5:
        print(f"  ... 还有{len(missing_in_model)-5}个参数")

# 分析audio_model参数
print("\n=== audio_model参数分析 ===")
audio_params_in_model = [name for name in model_params if 'audio_model' in name]
audio_params_in_ckpt = [name for name in state_dict if 'audio_model' in name]
print(f"模型中audio_model参数数量: {len(audio_params_in_model)}")
print(f"ckpt中audio_model参数数量: {len(audio_params_in_ckpt)}")

# 尝试使用strict=True加载，观察错误信息
print("\n=== 尝试使用strict=True加载 ===")
try:
    model.load_state_dict(state_dict, strict=True)
    print("使用strict=True加载成功")
except Exception as e:
    print(f"使用strict=True加载失败，错误信息:")
    print(f"{e}")

# 总结分析
print("\n=== 总结 ===")
print(f"1. ckpt文件大小: {ckpt_path.split('/')[-1]}")
print(f"2. 严格模式(strict=True)加载: {'成功' if matching_params == len(state_dict) and matching_params == len(model_params) else '失败'}")
print(f"3. 宽松模式(strict=False)加载: {'可能成功但部分参数未匹配' if missing_in_ckpt or missing_in_model else '完全匹配'}")
print(f"4. audio_model参数: {'部分或全部未保存' if len(audio_params_in_ckpt) < len(audio_params_in_model) else '全部保存'}")
print("\n建议: 由于使用了strict=False，即使部分参数未匹配也不会报错，但这可能导致模型未正确加载所有训练参数。")