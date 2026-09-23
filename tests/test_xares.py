"""Configuration and CPU checks; no model or dataset downloads."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "xares-llm"
sys.path.insert(0, str(BACKEND / "src"))


class PresetTests(unittest.TestCase):
    def test_twenty_tasks_have_train_and_eval_configs(self):
        from xares_llm.config_paths import AVAILABLE_TRAINING_CONFIGS as train
        from xares_llm.config_paths import AVAILABLE_EVALUATION_CONFIGS as evaluate
        self.assertEqual(set(train), set(evaluate))
        self.assertEqual(len(set(train) - {"all", "task1", "task2"}), 20)
        for path in list(train.values()) + list(evaluate.values()):
            self.assertTrue(path.is_file(), str(path))

    def test_task_yaml_can_be_parsed(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("Optional PyYAML not installed")
        for path in (BACKEND / "src/xares_llm/tasks").rglob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertIsInstance(config, dict, path.name)
            self.assertTrue(config, path.name)


class CpuIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        needed = ("torch", "transformers", "peft", "loguru")
        if any(importlib.util.find_spec(name) is None for name in needed):
            raise unittest.SkipTest("Optional CPU ML dependencies not installed")

    def test_dummy_feature_mask_contract_and_file_import(self):
        import torch
        from xares_llm.utils import attr_from_py_path
        encoder = attr_from_py_path(str(BACKEND / "example/dummy/dummyencoder.py"), "Encoder")()
        features, mask = encoder(torch.zeros(2, 1600), torch.ones(2, 1600))
        self.assertEqual(tuple(features.shape), (2, 10, encoder.output_dim))
        self.assertTrue(mask is None or tuple(mask.shape) == (2, 10))

    def test_mlp_projection_backpropagates(self):
        import torch
        from xares_llm.modeling_audiollm.modeling_xaresllm import XaresLLMMLPProjector
        projector = XaresLLMMLPProjector(12, 16)
        inputs = torch.randn(2, 7, 12, requires_grad=True)
        output = projector(inputs)
        self.assertEqual(tuple(output.shape), (2, 7, 16))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(projector.fc1.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
