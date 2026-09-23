"""No-download flow test using a real, tiny randomly initialized Llama.

The audio frontend/tokenizer are test doubles. The projector hidden layer is
reduced from 8192 to 32 units to test flow without allocating the research model.
This is an integration smoke test, not a paper-result or audio-encoder test.
"""

import importlib.util
import types
import unittest
from unittest.mock import patch

HAS_ML_PACKAGES = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "lightning"))


@unittest.skipUnless(HAS_ML_PACKAGES, "Install torch, transformers and lightning to run the CPU flow test")
class TinyLlamaFlowTests(unittest.TestCase):
    def test_hubert_train_backward_and_inference(self):
        import numpy as np
        import torch
        from torch import nn
        from transformers import BatchEncoding, BatchFeature, LlamaConfig, LlamaForCausalLM
        from uniarc.model import model_llama2_continus_prompt as implementation

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(3407)

        class FakeTokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 0

            def __len__(self):
                return 32

            def __call__(self, text, **kwargs):
                texts = [text] if isinstance(text, str) else text
                rows = []
                for item in texts:
                    words = item.replace(self.eos_token, "").split()
                    row = [4 + sum(word.encode("utf-8")) % 28 for word in words]
                    if item.endswith(self.eos_token):
                        row.append(self.eos_token_id)
                    rows.append(torch.tensor(row, dtype=torch.long))
                tokens = nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0)
                return BatchEncoding({"input_ids": tokens, "attention_mask": tokens.ne(0).long()})

            def batch_decode(self, sequences, **kwargs):
                return [" ".join(str(token) for token in row.tolist() if token not in (0, 1, 2)) for row in sequences]

        class FakeProcessor:
            def __call__(self, audio, **kwargs):
                rows = [torch.from_numpy(item.astype(np.float32)) for item in audio]
                return BatchFeature({"input_values": nn.utils.rnn.pad_sequence(rows, batch_first=True)})

        class FakeHubert(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.5))

            @staticmethod
            def _get_feat_extract_output_lengths(lengths):
                return (lengths - 400) // 320 + 1

            def forward(self, input_values, output_hidden_states=False):
                count = int(self._get_feat_extract_output_lengths(input_values.shape[1]))
                features = torch.linspace(-1, 1, count * 1024, device=input_values.device).reshape(1, count, 1024)
                features = features.expand(input_values.shape[0], -1, -1) * self.scale
                if output_hidden_states:
                    return {"hidden_states": (features,)}
                return types.SimpleNamespace(last_hidden_state=features)

        config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0, attention_dropout=0.0)
        llama = LlamaForCausalLM(config)
        frontend = FakeHubert()
        linear = nn.Linear

        def reduced_projector(in_features, out_features, *args, **kwargs):
            if out_features == 8192:
                out_features = 32
            if in_features == 8192:
                in_features = 32
            return linear(in_features, out_features, *args, **kwargs)

        with patch.object(implementation.HubertModel, "from_pretrained", return_value=frontend), \
             patch.object(implementation.Wav2Vec2Processor, "from_pretrained", return_value=FakeProcessor()), \
             patch.object(implementation.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()), \
             patch.object(implementation.LlamaForCausalLM, "from_pretrained", return_value=llama), \
             patch.object(implementation.nn, "Linear", side_effect=reduced_projector):
            model = implementation.IS(task_prompt="transcribe speech", learning_rate=1e-3)

        batch = ([np.linspace(-1000, 1000, 3601, dtype=np.float32), np.linspace(-500, 500, 7201, dtype=np.float32)], ["short", "longer reference"])
        frozen_before = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
        adapter_before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.audio_embedding_last_Linear.training)
        self.assertFalse(model.audio_model.training)
        self.assertFalse(model.llama.training)
        optimizer = model.configure_optimizers()
        with patch.object(model.llama, "forward", wraps=model.llama.forward) as forward:
            loss = model(batch)
        mask = forward.call_args.kwargs["attention_mask"]
        positions = forward.call_args.kwargs["position_ids"]
        self.assertTrue(mask[0].eq(0).any())
        for row_mask, row_positions in zip(mask, positions):
            valid = row_mask.bool()
            torch.testing.assert_close(row_positions[valid], torch.arange(int(valid.sum())))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [parameter.grad for parameter in model.audio_embedding_last_Linear.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(torch.count_nonzero(gradient).item() for gradient in gradients))
        self.assertTrue(all(parameter.grad is None for parameter in model.llama.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.audio_model.parameters()))
        optimizer.step()
        parameters = dict(model.named_parameters())
        self.assertTrue(any(not torch.equal(value, parameters[name]) for name, value in adapter_before.items()))
        for name, value in frozen_before.items():
            torch.testing.assert_close(parameters[name], value, rtol=0, atol=0)

        # Exercise real Hugging Face beam generation, limiting only output length.
        real_generate = model.llama.generate
        seen = {}

        def short_generate(**kwargs):
            seen.update(kwargs)
            kwargs["max_new_tokens"] = 4
            return real_generate(**kwargs)

        with patch.object(model.llama, "generate", side_effect=short_generate):
            predictions = model.inference(batch)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(isinstance(item, str) for item in predictions))
        self.assertEqual(seen["num_beams"], 5)
        self.assertTrue(seen["attention_mask"][:, -1].eq(1).all())
        self.assertEqual(tuple(seen["inputs_embeds"].shape[:2]), tuple(seen["attention_mask"].shape))


if __name__ == "__main__":
    unittest.main()
