import soundfile as sf
import torch
# 1. 导入 AutoModel 而不是具体的类
from transformers import WhisperFeatureExtractor, AutoModel

model_id = "Atotti/Qwen3-Omni-AudioTransformer"

# 2. 加载特征提取器（这个是标准的 Whisper 格式）
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)

# 3. 使用 AutoModel 加载，关键在于 trust_remote_code=True
# 这会自动下载仓库里的 modeling_qwen3_omni_moe.py 并在后台完成类定义
qwen_encoder = AutoModel.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
).eval()

# 如果有 GPU，建议移动到 GPU 提高速度
device = "cuda" if torch.cuda.is_available() else "cpu"
qwen_encoder.to(device)

# 读取音频
wav, sr = sf.read("audio_16k.wav", dtype="float32")

# 提取特征
features = feature_extractor(wav, sampling_rate=sr, return_tensors="pt", padding=False)

# 处理输入数据
# 注意：通常模型需要 batch 维度 [1, feature_dim, seq_len]
x = features["input_features"].to(device=device, dtype=qwen_encoder.dtype)
# 原代码用了 x[0]，如果后续报错可以尝试用 features["input_features"] (保持 [1, 80, 3000])

# 获取特征长度
feat_len = torch.tensor([x.shape[-1]], dtype=torch.long).to(device)

with torch.no_grad():
    # 这里的 qwen_encoder 实际上就是你要的 Qwen3OmniMoeAudioEncoder 实例
    out = qwen_encoder(input_features=x, feature_lens=feat_len)
    
    # 提取最后一层隐藏状态
    hid = out.last_hidden_state  # 形状通常为 [Batch, Time, Dimension]
    
    print(f"输出形状: {hid.shape}") 
    # 如果 hid 是 [1, T, D]，我们取第一个 batch
    if hid.ndim == 3:
        print(hid[0, 0, :10])
    else:
        print(hid[0, :10])