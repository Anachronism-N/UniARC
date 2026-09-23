# Modified for the UniARC unified source release; see README.md.
import torch
from transformers import WavLMModel

class WavLMEncoder(torch.nn.Module):
    def __init__(self, model_path="microsoft/wavlm-large", **kwargs) -> None:
        super().__init__()
        # 加载预训练模型
        # print("dtype:",dtype)

        self.model = WavLMModel.from_pretrained(model_path)
        
        # WavLM-Large 的输出维度是 1024
        self.output_dim = self.model.config.hidden_size 
        # print("self.output_dim:",self.output_dim)

    def forward(self, audio, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        audio: 输入的音频波形，形状为 (batch_size, num_samples)
        audio_attention_mask: 对应音频的 mask (1 为有效，0 为 padding)
        """

        outputs = self.model(
            input_values=audio, 
            attention_mask=audio_attention_mask
        )
        
        # 形状为 (batch_size, frame_seq_len, 1024)
        output = outputs.last_hidden_state
        
        # 注意：WavLM 对音频进行了下采样（通常是 320 倍），
        # 原始音频 16000 个采样点会变成 50 个特征向量。
        # 如果需要返回对应的 mask，通常需要对原始 audio_attention_mask 进行下采样，
        # 简单起见，这里可以参考 Transformer 的常规处理返回 None 或处理后的 mask。
        new_mask = None
        if audio_attention_mask is not None:
            # 这是一个简单的启发式下采样计算方式，对应 WavLM 的卷积层步长
            new_mask = audio_attention_mask[:, ::320] 
            # 这里的切片可能需要根据 output 的具体长度对齐进行 padding/cropping
            if new_mask.shape[1] > output.shape[1]:
                new_mask = new_mask[:, :output.shape[1]]
            elif new_mask.shape[1] < output.shape[1]:
                # 补齐长度
                padding = torch.zeros((new_mask.shape[0], output.shape[1] - new_mask.shape[1]), device=new_mask.device)
                new_mask = torch.cat([new_mask, padding], dim=1)

        return output, new_mask

# 测试代码
if __name__ == "__main__":
    # 模拟 1 秒 16kHz 的音频，batch_size=2
    mock_audio = torch.randn(2, 16000) 
    encoder = WavLMEncoder()
    features, mask = encoder(mock_audio)
    print(f"Feature shape: {features.shape}") # 预期: [2, 49~50, 1024]
