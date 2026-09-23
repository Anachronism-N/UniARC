# Modified for the UniARC unified source release; see README.md.

import torch
import torch.nn as nn
from transformers import AutoProcessor, WhisperModel

class WhisperLargeV3Encoder(nn.Module):
    def __init__(self, model_path: str = "openai/whisper-large-v3", sampling_rate: int = 16000):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = WhisperModel.from_pretrained(model_path)
        self.sampling_rate = sampling_rate
        self.output_dim = self.model.config.d_model

    def forward(self, audio: torch.Tensor, audio_attention_mask=None):
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        audio_list = [a.detach().cpu().float().numpy() for a in audio]
        features = self.processor(
            audio_list,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=480000, 
            return_attention_mask=True,
        )
        
        device = next(self.model.parameters()).device
        features = {k: v.to(device) for k, v in features.items()}

        with torch.no_grad():
            # WhisperModel forward returns last_hidden_state from encoder if decoder_input_ids is not provided?
            # Actually WhisperModel includes both Encoder and Decoder.
            # We only want the encoder outputs.
            encoder_outputs = self.model.encoder(**features)
            
        hidden = encoder_outputs.last_hidden_state # (B, 1500, D)
        
        # Handle attention mask
        # Processor returns mask for 3000 features (100Hz)
        # Encoder output is 1500 features (50Hz) -> Downsample by 2
        if "attention_mask" in features:
            mask = features["attention_mask"][:, ::2] # (B, 1500)
            
            # Optimization: Truncate to longest valid sequence in the batch
            # mask is 1 for valid, 0 for padded
            valid_lengths = mask.sum(dim=1)
            max_len = valid_lengths.max().item()
            if max_len < 1:
                max_len = 1
            
            # Ensure we don't exceed bounds (though max_len should be <= 1500)
            max_len = int(max_len)
            hidden = hidden[:, :max_len, :]
            mask = mask[:, :max_len]
            
            return hidden, mask
            
        return hidden, None

if __name__ == "__main__":
    enc = WhisperLargeV3Encoder()
    x = torch.randn(1, 16000)
    y, _ = enc(x)
    print(y.shape)
