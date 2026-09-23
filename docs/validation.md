# Validation record

Validated on Windows, using Python 3.12.14 and CPU PyTorch 2.9.1. The integration
environment uses Transformers 4.57.3, Lightning / PyTorch Lightning 2.5.6,
TorchMetrics 1.8.2, NumPy 2.3.5, Hugging Face Hub 0.36.0 and JiWER 4.0.0.
XARES projector checks also used PEFT 0.18.0 and Accelerate 1.12.0.

## Checks performed

- Root test suite: **14 tests passed**. Includes launcher routing/argument
  preservation, JSONL validation, empty predictions, explicit label conventions,
  real JiWER calculation, all 20 paired task presets, YAML parsing, dummy encoder
  output shape/mask, and real CPU MLP forward/backpropagation.
- Probe test suite: **11 tests passed**. Includes all encoder configurations,
  PCM16 WAV packing/indexing, malformed data rejection, five encoders' checkpoint
  restoration and projector dimensions, frozen module modes, and a small real
  Llama forward/backward/optimizer step plus beam-search inference.
- The tiny-Llama test uses a randomly initialized one-layer language model, mock
  audio frontend/tokenizer, and a smaller projector hidden layer. It confirms
  execution and gradient flow; it does not test pretrained audio representations.
- XARES wheel build succeeded with `pip wheel --no-deps --no-build-isolation`;
  task YAML files and license were retained in the wheel. Installation was
  checked with `--no-deps` in a separate directory. This validates packaging,
  not the complete training dependency stack.
- Unified/backend CLI help and six probe configuration dry runs succeeded
  without downloading pretrained models. A real launch using missing example
  resources fails explicitly before training.
- Source release checks cover Python/JSON syntax, unexpected artifacts,
  case-colliding paths, developer-machine paths, and common literal credential
  patterns. This is a limited source hygiene check, not a security audit.

## Re-run

From the repository root:

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
python -m unittest discover -s uniarc/tests -v
```

Tests requiring optional ML packages are explicitly skipped when those packages
are absent. A passing source-only run is not equivalent to the CPU checks above.
The GitHub workflow performs source-only checks on Linux and Windows; its hosted
jobs have not been run as part of this local integration.

## Not validated

Full pretrained encoder loading, real-data training, multi-GPU execution,
TorchCodec/FFmpeg audio decoding in a complete XARES environment, FENSE model
downloads/scoring, codec-specific forward passes, and reproduction of manuscript
tables have not been run. No trained adapters or original data splits were
provided. Exact model revisions and K-means artifacts remain prerequisites for
matching historical results. See [paper_mapping.md](paper_mapping.md).

The integration fixes checkpoint restoration, frozen-component modes, feature
lengths, and padding positions. These changes can alter results compared with
the historical scripts; they are documented fixes, not a claim of bitwise or
metric-level equivalence with the original experiments.
