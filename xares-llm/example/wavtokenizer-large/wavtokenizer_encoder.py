# Modified for the UniARC unified source release; see README.md.

import torch
import torch.nn as nn
import os
import sys

# Add WavTokenizer root to path if needed, similar to wavtokenizer_large.py
wavtokenizer_root = os.environ.get(
    "WAVTOKENIZER_ROOT",
    "",
)
if os.path.isdir(wavtokenizer_root) and wavtokenizer_root not in sys.path:
    sys.path.append(wavtokenizer_root)

try:
    from decoder.pretrained import WavTokenizer
except ImportError:
    try:
        from wavtokenizer import WavTokenizer
    except ImportError:
        try:
             from decoder.experiment import WavTokenizer
        except ImportError:
             WavTokenizer = None

class WavTokenizerEncoder(nn.Module):
    def __init__(self, 
                 config_path: str | None = None, 
                 model_path: str | None = None, 
                 sampling_rate: int = 16000):
        super().__init__()
        
        if WavTokenizer is None:
            raise ImportError("Could not import WavTokenizer. Please set WAVTOKENIZER_ROOT environment variable correctly.")

        config_path = config_path or os.environ.get("WAVTOKENIZER_CONFIG")
        model_path = model_path or os.environ.get("WAVTOKENIZER_CHECKPOINT")
        if not config_path or not model_path:
            raise ValueError("Provide config_path/model_path or WAVTOKENIZER_CONFIG/WAVTOKENIZER_CHECKPOINT.")

        # Initialize WavTokenizer
        if hasattr(WavTokenizer, "from_pretrained0802"):
            self.tokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
        elif hasattr(WavTokenizer, "from_pretrained08"):
            self.tokenizer = WavTokenizer.from_pretrained08(config_path, model_path)
        elif hasattr(WavTokenizer, "load_from_checkpoint"):
             # Fallback or older version
             self.tokenizer = WavTokenizer.load_from_checkpoint(model_path)
        else:
             # Basic init
             self.tokenizer = WavTokenizer(config_path, model_path)
        
        self.sampling_rate = sampling_rate
        self.output_dim = 512 

    def forward(self, audio: torch.Tensor, audio_attention_mask=None):
        # audio: (B, T)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        device = next(self.tokenizer.parameters()).device
        audio = audio.to(device)
        
        # Resample from 16000 to 24000 if needed
        if self.sampling_rate != 24000:
             import torchaudio.functional as F
             audio = F.resample(audio, orig_freq=self.sampling_rate, new_freq=24000)
        
        with torch.no_grad():
             # We need to pass bandwidth_id. Typically 0 for the first bandwidth.
             bandwidth_id = torch.tensor([0], device=device)
             
             # The tokenizer (WavTokenizer) has a feature_extractor
             # which is an EncodecFeatures instance.
             # Its forward returns (quantized, codes, commit_loss)
             # quantized is what we want (the hidden features)
             features, _, _ = self.tokenizer.feature_extractor(audio, bandwidth_id=bandwidth_id)
             # features shape: (B, D, T)
             
        hidden = features.transpose(1, 2) # (B, T, D)
        
        # Normalize features (Crucial for Projector stability)
        hidden = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],))
        
        # Create attention mask (assuming all valid for now, as padding handling for resampled audio is complex)
        # B, T
        audio_attention_mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long)
        
        return hidden, audio_attention_mask

if __name__ == "__main__":
    enc = WavTokenizerEncoder()
    x = torch.randn(1, 16000)
    y, _ = enc(x)
    print(y.shape)
