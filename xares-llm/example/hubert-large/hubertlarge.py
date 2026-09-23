# Modified for the UniARC unified source release; see README.md.
import torch
from transformers import AutoModel, AutoProcessor, HubertModel

def _assert_finite(name: str, x: torch.Tensor):
    if not torch.isfinite(x).all():
        bad = x[~torch.isfinite(x)]
        raise RuntimeError(f"[NaN/Inf] {name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
                           f"bad_count={bad.numel()}")

class HubertLargeEncoder(torch.nn.Module):
    def __init__(self, model_path: str = "facebook/hubert-large-ls960-ft", sampling_rate: int = 16000):
        super().__init__()
        # Load in bfloat16 for memory efficiency (A100 supports it well)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = HubertModel.from_pretrained(model_path)
        self.sampling_rate = sampling_rate
        self.output_dim = getattr(self.model.config, "hidden_size", None)

        if hasattr(self.model.config, "apply_spec_augment"):
            self.model.config.apply_spec_augment = False

    def forward(self, audio: torch.Tensor, audio_attention_mask=None):
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        audio_list = [a.detach().cpu().float().numpy() for a in audio]
        features = self.processor(
            audio_list,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        if audio_attention_mask is not None:
            features["attention_mask"] = audio_attention_mask

        device = next(self.model.parameters()).device
        dtype = self.model.dtype
        # Cast input features to model dtype
        features = {k: v.to(device) if k == "attention_mask" else v.to(device).to(dtype) for k, v in features.items()}

        with torch.no_grad():
            out = self.model(**features)
        
        hidden = out.last_hidden_state
        return hidden, None
