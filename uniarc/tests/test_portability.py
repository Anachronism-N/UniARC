"""Resource-free configuration, data round-trip, and model-hook regressions."""

import ast
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
import wave

from uniarc.data import PackedAudioDataset
from uniarc.prepare_data import pack_manifest
from uniarc.run import ENCODERS, REPO_ROOT, load_config


def model_method(filename, name, namespace):
    """Execute the retained method without importing or downloading backbones."""
    source = (REPO_ROOT / "uniarc" / "model" / filename).read_text(encoding="utf-8")
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "IS")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    tree = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), filename, "exec"), namespace)
    return namespace[name]


class ConfigTests(unittest.TestCase):
    def test_all_encoder_configs_validate_without_loading_models(self):
        for encoder in ENCODERS:
            with self.subTest(encoder=encoder):
                config = load_config(REPO_ROOT / "uniarc" / "configs" / f"{encoder}_asr.json", "train")
                self.assertEqual(config["encoder"], encoder)
                self.assertTrue(Path(config["data"]["train"]["scp"]).is_absolute())

    def test_inference_requires_trained_adapter(self):
        with self.assertRaisesRegex(ValueError, "requires checkpoint"):
            load_config(REPO_ROOT / "uniarc" / "configs" / "hubert_asr.json", "infer")

    def test_misspelled_config_keys_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"encodre": "hubert"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown configuration keys"):
                load_config(path, "train")


class PackedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = struct.pack("<500h", *range(500))
        with wave.open(str(self.root / "example.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(self.payload)
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text(json.dumps({"id": "example", "audio": "example.wav", "text": "test reference"}) + "\n", encoding="utf-8")
        self.output = self.root / "packed"

    def test_pack_and_index_round_trip(self):
        self.assertEqual(pack_manifest(self.manifest, self.output), 1)
        self.assertEqual((self.output / "audio.seq").read_bytes(), self.payload)
        dataset = PackedAudioDataset(self.output / "audio.scp", self.output / "text.txt")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.records[0][0], "example")
        self.assertEqual(dataset.records[0][2:5], (0, 1000, "test reference"))

    def test_refuses_existing_output(self):
        pack_manifest(self.manifest, self.output)
        with self.assertRaises(FileExistsError):
            pack_manifest(self.manifest, self.output)

    def test_missing_label_fails_at_construction(self):
        pack_manifest(self.manifest, self.output)
        (self.output / "text.txt").write_text("different label\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing transcription"):
            PackedAudioDataset(self.output / "audio.scp", self.output / "text.txt")

    def test_invalid_mrk_count_fails(self):
        pack_manifest(self.manifest, self.output)
        (self.output / "audio.mrk").write_text("2\nexample 0 1000\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "record count mismatch"):
            PackedAudioDataset(self.output / "audio.scp", self.output / "text.txt")


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch is required for tensor regression checks")
class ModelRegressionTests(unittest.TestCase):
    def test_frozen_backbones_stay_in_eval_for_every_encoder(self):
        from torch import nn

        for filename, _ in ENCODERS.values():
            with self.subTest(model=filename):
                namespace = {}
                train = model_method(filename + ".py", "train", namespace)
                cls = type("IS", (nn.Module,), {"train": train})
                namespace["IS"] = cls
                model = cls()
                model.llama = nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.5))
                for name in ("audio_model", "dac", "speechtokenizer", "wavtokenizer"):
                    setattr(model, name, nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.5)))
                model.audio_embedding_last_Linear = nn.Linear(2, 3)
                model.train()
                self.assertTrue(model.audio_embedding_last_Linear.training)
                self.assertFalse(model.llama.training)
                frontend = "audio_model"
                for key, attribute in (("DAC", "dac"), ("speechtokenizer", "speechtokenizer"), ("wavtokenizer", "wavtokenizer")):
                    if key in filename:
                        frontend = attribute
                self.assertFalse(getattr(model, frontend).training)
                self.assertTrue(all(not child.training for child in getattr(model, frontend).modules()))
                model.eval()
                self.assertFalse(model.audio_embedding_last_Linear.training)

    def test_checkpoint_restores_trainable_tensors_for_every_encoder(self):
        import torch
        from torch import nn

        for filename, _ in ENCODERS.values():
            with self.subTest(model=filename):
                model = nn.Module()
                model.audio_embedding_last_Linear = nn.Linear(2, 3)
                model.frozen = nn.Linear(2, 3)
                model.frozen.requires_grad_(False)
                namespace = {}
                hook = model_method(filename + ".py", "on_load_checkpoint", namespace)
                desired = {"audio_embedding_last_Linear.weight": torch.full((3, 2), 0.75), "audio_embedding_last_Linear.bias": torch.full((3,), 0.25)}
                checkpoint = {"state_dict": desired.copy()}
                hook(model, checkpoint)
                self.assertEqual(set(checkpoint["state_dict"]), set(desired))
                model.load_state_dict(checkpoint["state_dict"], strict=False)
                torch.testing.assert_close(model.audio_embedding_last_Linear.weight, desired["audio_embedding_last_Linear.weight"])
                with self.assertRaisesRegex(ValueError, "missing trainable"):
                    hook(model, {"state_dict": {}})

    def test_adapter_dimension_follows_llama_for_every_encoder(self):
        import torch
        from torch import nn

        class FakeBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(hidden_size=16, vocab_size=8)
                self.embeddings = nn.Embedding(8, 16)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            load = from_pretrained
            load_from_checkpoint = from_pretrained
            from_pretrained0802 = from_pretrained

            def get_input_embeddings(self):
                return self.embeddings

        class FakeTokenizer:
            pad_token = "pad"
            bos_token_id = 0
            eos_token_id = 1

            def __len__(self):
                return 8

            @classmethod
            def from_pretrained(cls, *args):
                return cls()

        for filename, _ in ENCODERS.values():
            with self.subTest(model=filename), torch.device("meta"):
                namespace = {"torch": torch, "nn": nn, "AutoTokenizer": FakeTokenizer, "LlamaForCausalLM": FakeBackbone, "HubertModel": FakeBackbone, "WavLMModel": FakeBackbone, "Wav2Vec2Processor": FakeBackbone, "SpeechTokenizer": FakeBackbone, "WavTokenizer": FakeBackbone, "dac": types.SimpleNamespace(DAC=FakeBackbone), "CustomCrossEntropyLoss": lambda *args: nn.Identity()}
                init = model_method(filename + ".py", "__init__", namespace)
                cls = type("IS", (nn.Module,), {"__init__": init})
                namespace["IS"] = cls
                model = cls()
                self.assertEqual(model.audio_embedding_last_Linear[-1].out_features, 16)
                self.assertTrue(all(parameter.requires_grad for parameter in model.audio_embedding_last_Linear.parameters()))
                self.assertFalse(any(parameter.requires_grad for parameter in model.llama.parameters()))


if __name__ == "__main__":
    unittest.main()
