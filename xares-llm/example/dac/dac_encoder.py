# Modified for the UniARC unified source release; see README.md.

import torch
import torch.nn as nn
from transformers import AutoProcessor, DacModel

class DACEncoder(nn.Module):
    def __init__(self, model_path: str = "descript/dac_16khz", sampling_rate: int = 16000):
        super().__init__()
        # Use transformers DacModel
        # Note: DacModel usually includes encoder, quantizer, decoder.
        # We need to extract the encoder output.
        # DacModel forward returns DacOutput which has 'audio_codes', 'latents', etc.
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = DacModel.from_pretrained(model_path)
        self.sampling_rate = sampling_rate
        self.output_dim = self.model.config.hidden_size

    def forward(self, audio: torch.Tensor, audio_attention_mask=None):
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        audio_list = [a.detach().cpu().float().numpy() for a in audio]
        features = self.processor(
            audio_list,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        
        device = next(self.model.parameters()).device
        features = {k: v.to(device) for k, v in features.items()}

        input_values = features.get("input_values")
        if input_values is None:
            raise ValueError("DAC processor must return input_values.")
        with torch.no_grad():
            # Transformers 4.57.3 returns quantized_representation, not latents.
            outputs = self.model.encode(input_values, return_dict=True)
            hidden = outputs.quantized_representation
        if hidden is None or hidden.ndim != 3 or hidden.shape[1] != self.output_dim:
            raise ValueError("DAC must return quantized features [batch, hidden_size, frames].")
        hidden = hidden.transpose(1, 2)

        # Add Normalization
        hidden = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],)) 
             
        return hidden, None

if __name__ == "__main__":
    enc = DACEncoder()
    x = torch.randn(1, 16000)
    y, _ = enc(x)
    print(y.shape)
