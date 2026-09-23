# Modified for the UniARC unified source release; see README.md.
import torch
from transformers import AutoModel, AutoProcessor

def _assert_finite(name: str, x: torch.Tensor):
    if not torch.isfinite(x).all():
        bad = x[~torch.isfinite(x)]
        raise RuntimeError(f"[NaN/Inf] {name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
                           f"bad_count={bad.numel()}")

class Wav2Vec2Large960hDownEncoder(torch.nn.Module):
    def __init__(self, model_path: str = "facebook/wav2vec2-large-960h", sampling_rate: int = 16000):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.sampling_rate = sampling_rate
        self.output_dim = getattr(self.model.config, "hidden_size", None)

        # --- 可选：双保险禁用 spec augment（不影响推理/特征提取稳定性）
        if hasattr(self.model.config, "apply_spec_augment"):
            self.model.config.apply_spec_augment = False

        # --- 注册 hook：定位 NaN 从哪一块开始出现
        # self._hooks = []
        # def hook(name):
        #     def _h(module, inp, out):
        #         t = out[0] if isinstance(out, (tuple, list)) else out
        #         if isinstance(t, torch.Tensor):
        #             _assert_finite(name, t)
        #     return _h

        # # wav2vec2 常见模块名：feature_extractor / feature_projection / encoder
        # if hasattr(self.model, "feature_extractor"):
        #     self._hooks.append(self.model.feature_extractor.register_forward_hook(hook("feature_extractor.out")))
        # if hasattr(self.model, "feature_projection"):
        #     self._hooks.append(self.model.feature_projection.register_forward_hook(hook("feature_projection.out")))
        # if hasattr(self.model, "encoder"):
        #     self._hooks.append(self.model.encoder.register_forward_hook(hook("encoder.out")))

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
        features = {k: v.to(device) for k, v in features.items()}

        with torch.no_grad():
            out = self.model(**features)
        hidden = out.last_hidden_state  # (B, T', H)
        B, T, H = hidden.shape

        # pool_size = 2  # ⭐⭐⭐ 可调：8 / 10 / 16
        # new_T = T // pool_size

        # if new_T > 0:
        #     hidden = hidden[:, : new_T * pool_size, :]
        #     hidden = hidden.view(B, new_T, pool_size, H).mean(dim=2)

        # print(hidden.shape)
        if "attention_mask" in features:
             # Precise mask calculation using model architecture
             input_mask = features["attention_mask"]
             input_lengths = input_mask.sum(dim=1).long()
             
             if hasattr(self.model, "_get_feat_extract_output_lengths"):
                 output_lengths = self.model._get_feat_extract_output_lengths(input_lengths)
             else:
                 # Fallback to naive stride if helper not available (should not happen for Wav2Vec2)
                 output_lengths = (input_lengths - 1) // 320 + 1

             # Create new binary mask (B, T)
             mask = torch.zeros(B, T, dtype=input_mask.dtype, device=device)
             for i, length in enumerate(output_lengths):
                 # Clamp length to T (hidden size) to avoid out of bounds
                 valid_len = min(length.item(), T)
                 mask[i, :valid_len] = 1
                 
             return hidden, mask
             
        return hidden, None
if __name__ == "__main__":
    mdl = Wav2Vec2Large960hDownEncoder(model_path="facebook/wav2vec2-large-960h").eval()

    # 1 秒 16kHz 的假音频 (B, T)
    x = torch.randn(1, 16000)

    y, _ = mdl(x)
    print(y.shape)  # (B, T', H)
