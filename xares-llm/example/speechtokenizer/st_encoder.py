# Modified for the UniARC unified source release; see README.md.
import torch
import torch.nn as nn
import os
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 修复点1: 将import提到最外层，异常捕获与类定义解耦
try:
    from speechtokenizer import SpeechTokenizer
except ImportError:
    logger.error("❌ Error: speechtokenizer library not found! Install via: pip install speechtokenizer")
    raise ImportError("speechtokenizer is required!")

# 修复点2: 核心编码器类 必须定义在【顶层全局作用域】，无任何缩进！！！
# 类名SpeechTokenizerEncoder 以Encoder结尾 ✔️ 完全符合框架要求
class SpeechTokenizerEncoder(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        model_path = model_path or os.environ.get("SPEECHTOKENIZER_MODEL_PATH")
        if not model_path:
            raise ValueError("Provide model_path or SPEECHTOKENIZER_MODEL_PATH (config.json and .pt checkpoint).")
        self.model_path = model_path
        self._check_model_files()
        
        logger.info(f"📂 Loading SpeechTokenizer -> config: {self.config_path}, ckpt: {self.ckpt_path}")
        self.model = SpeechTokenizer.load_from_checkpoint(self.config_path, self.ckpt_path)
        self.model.eval()  # 冻结权重，推理模式，不参与训练梯度更新
        
        if isinstance(self.model.transform, nn.Linear):
            self.output_dim = self.model.transform.out_features
        else:
            self.output_dim = self.model.quantizer.dimension
        self.max_length = 320000 
        self.sampling_rate = 16000
        logger.info(f"SpeechTokenizer loaded successfully! Output dimension fixed to: {self.output_dim}")

    def _check_model_files(self):
        """校验配置文件、权重文件是否存在"""
        self.config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"❌ Config file not found: {self.config_path}")
        
        ckpt_files = sorted(glob.glob(os.path.join(self.model_path, "*.pt")))
        if not ckpt_files:
            raise FileNotFoundError(f"❌ No .pt checkpoint found in {self.model_path}")
        self.ckpt_path = ckpt_files[0]

    def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        前向传播核心逻辑（适配多GPU，Accelerate自动分配设备）
        Input: audio - 支持 (T)/(B,T)/(B,1,T) 三种输入格式
        Output: (features: [B, T, 1024], mask: None) 完全适配下游LLM输入格式
        """
        assert isinstance(audio, torch.Tensor), f"Audio must be torch.Tensor, got {type(audio)}"
        
        # 统一输入维度为 (B,1,T)，满足SpeechTokenizer的输入要求
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(1)  # (T) → (1,1,T)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(1)  # (B,T) → (B,1,T)
        
        # OOM Fix: Chunking
        total_length = audio.shape[-1]
        
        with torch.no_grad():
            if total_length > self.max_length:
                input_list = torch.split(audio, self.max_length, dim=-1)
                output_list = []
                
                for chunk in input_list:
                    if chunk.shape[-1] < self.sampling_rate:
                        pad_len = self.sampling_rate - chunk.shape[-1]
                        chunk = torch.nn.functional.pad(chunk, (0, pad_len))
                    
                    quantized_list = self.model.forward_feature(chunk, layers=[0])
                    output_list.append(quantized_list[0])
                
                features = torch.cat(output_list, dim=-1) # (B, 1024, Total_T)
            else:
                 quantized_list = self.model.forward_feature(audio, layers=[0])
                 features = quantized_list[0]

            # 维度转置 → (B, T, 1024)，适配X-ARES-LLM的特征拼接格式
            if features.ndim == 3:
                features = features.permute(0, 2, 1)
                features = self.model.transform(features)
        
        return features, None

# ✅ 修复点3: 注释掉测试代码，框架加载时不会执行，避免冗余日志/显存占用
# if __name__ == "__main__":
#     encoder = SpeechTokenizerEncoder()
#     test_audio = torch.randn(2, 16000)  # 模拟输入
#     output, _ = encoder(test_audio)
#     logger.info(f"✅ Test Success! Input shape: {test_audio.shape}, Output shape: {output.shape}")
#     logger.info(f"✅ Real output dim confirm: {output.shape[-1]}")
