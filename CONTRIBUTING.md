# Contributing

Use a separate environment for each backend as described in the README.
Keep changes to experiment behavior separate from packaging and portability fixes.
When changing encoders, masks, projector shapes, prompts, or evaluation metrics,
describe how the change affects comparison with the manuscript.

Before opening a pull request, run from the repository root:

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
```

Report the operating system, Python/PyTorch/Transformers versions, GPU, command,
configuration, and a minimal traceback for training issues. Do not attach private
datasets, tokens, model weights, or complete user-specific logs. Explain whether
validation used synthetic inputs, a small real-data run, or a full experiment.

Contributions are accepted under Apache-2.0. Preserve third-party attribution.
