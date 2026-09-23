# UniARC

**Unified Audio Representation Comparison** for the manuscript
**Discrete vs. Continuous: A Comprehensive Study of Unified Audio Understanding in LALMs**.

[����˵��](README.zh-CN.md) �� [Paper-to-code mapping](docs/paper_mapping.md) ��
[Data and models](docs/data_and_models.md) �� [Integration history](docs/migration.md)

UniARC compares continuous audio features and discrete audio representations
across speech, sound, and music. This repository brings the two evaluation
strategies into one source distribution:

Both implementations are included in this repository: `uniarc/` comes from
the original `UniARC/code` tree, and `xares-llm/` comes from `UniARC_paper`.
Clone this repository once, then use the instructions for the chosen pipeline.
You do not need to clone the two original repositories to run this source tree.

| Strategy | Backend | Language model | Adaptation | Paper coverage |
| --- | --- | --- | --- | --- |
| Parameter-efficient fine-tuning | `xares-llm/` | SmolLM2 135M / 360M | Projector + LoRA | 20 tasks, nine encoder families; Figure 2 / Table 1 |
| Frozen-backbone probing | `uniarc/` | Llama 1B / 8B | Adapter with frozen language backbone | Seven task types across eight datasets; Table 2 |

The probe implementation includes HuBERT, WavLM, DAC, SpeechTokenizer, and an
additional WavTokenizer adapter. Table 2 reports four encoders and does not report
the WavTokenizer probe. The LoRA backend additionally includes Wav2Vec2, Whisper,
HuBERT-KMeans, and WavLM-KMeans.

This release provides source code, portable entry points, configuration examples,
and preparation instructions. It does **not** include datasets, pretrained weights,
trained adapters, or newly reproduced paper results. Important implementation and
protocol differences are documented in the [paper mapping](docs/paper_mapping.md).

## Layout

```text
UniARC/
  run.py                  # Unified launcher; Python standard library only
  uniarc/                 # Frozen-backbone probing
  xares-llm/              # XARES-LLM-based LoRA evaluation
  docs/                   # Paper mapping, resources, provenance, validation
  scripts/check_release.py
  tests/
  LICENSE                 # Apache-2.0
  NOTICE
  CITATION.cff
```

## Quick start

```bash
git clone https://github.com/Anachronism-N/UniARC.git
cd UniARC
```

Use separate virtual environments for the two backends: their original dependency
stacks differ. Linux with NVIDIA GPUs is the training target. Source-only checks
also run on Windows. Use the backend READMEs for dependency and resource details.

### Inspect the source without installing ML packages

Python 3.10 or newer is sufficient for these commands, run at the repository root:

```bash
python run.py --help
python scripts/check_release.py
python -m unittest discover -s tests -v
python run.py --show-command xares example/hubert-large/hubertlarge.py cremad cremad
python run.py probe --help
```

`--show-command` only prints the selected process and working directory. It does
not load models, download data, or start a training job.

### XARES-LLM / LoRA

From the repository root, create and activate its environment (Bash):

```bash
python3.11 -m venv .venv-xares
source .venv-xares/bin/activate
python -m pip install --upgrade pip
python -m pip install -c xares-llm/requirements-constraints.txt -e ./xares-llm
```

Use a PyTorch build appropriate for your CUDA driver and install FFmpeg for
TorchCodec; see [xares-llm/README.md](xares-llm/README.md). This command trains
and evaluates one task using the paper's MLP projector and a 135M backbone:

```bash
python run.py xares example/hubert-large/hubertlarge.py cremad cremad \
  --args '{"decoder_model_name":"HuggingFaceTB/SmolLM2-135M","projector_type":"mlp","lora_enabled":true}'
```

These multiline examples use Bash quoting. For Windows, use the backend guide
and pass JSON arguments using your shell's supported quoting. Use
`HuggingFaceTB/SmolLM2-360M` for the second model scale. This command is a starting
point for one experiment, not the entire paper evaluation. Supply the required
encoder resources before using codec or K-means adapters.

The launcher runs XARES in `xares-llm/`; encoder paths and custom task YAML paths
are relative to that directory. `--model_args` controls the audio encoder and
`--args` controls training/model configuration. Resource downloads can occur
during real execution; see [data preparation](docs/data_and_models.md).

### Frozen-backbone probing

In a separate shell at the repository root, install the probe environment (Bash):

```bash
python3.11 -m venv .venv-probe
source .venv-probe/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/probe.txt
```

Follow [uniarc/README.md](uniarc/README.md) to prepare model resources and data.
Copy and edit a configuration with your model identifiers, data manifests,
and output location:

```bash
python run.py probe train --config uniarc/configs/hubert_asr.json --dry-run
python run.py probe train --config uniarc/configs/hubert_asr.json
```

For inference, set `checkpoint` and `data.test` in the inference configuration
to the trained adapter and evaluation split, then run:

```bash
python run.py probe infer --config uniarc/configs/hubert_asr_infer.json
```

The first command validates the configuration without loading models. The second
requires prepared data and model resources. Probe execution uses the repository
root as its working directory. The backend guide explains path resolution,
manifest preparation, checkpoint loading, inference, and task prompts.

Score probe predictions with [scripts/evaluate.py](scripts/evaluate.py); see
[evaluation conventions](docs/evaluation.md) for Accuracy, WER/iWER, and FENSE.

For either backend, use a specific environment without activating it:

```bash
python run.py --python /path/to/environment/bin/python probe --help
```

## Reproducibility and validation

The original source snapshots are pinned in [migration.md](docs/migration.md).
Per-file provenance is recorded in [source_manifest.json](docs/source_manifest.json).
The [validation report](docs/validation.md) separates source/CPU checks from
full GPU experiments. The GitHub workflow runs source checks and launcher tests;
it does not train models or reproduce manuscript tables.

For an experiment report, record the source commit, backend, complete config,
model revision, dataset split, seed, environment, GPU, and output metrics. Preserve
both the unmodified reported protocol and any changes used in a new run.

## Citation and attribution

`CITATION.cff` currently describes the software. The supplied manuscript has an
anonymous author line; final author order, venue, and DOI have not been inferred
from older repository text. Add the verified paper citation when available.

The LoRA backend builds on [XARES-LLM](https://github.com/xiaomi-research/xares-llm).
Please acknowledge that work as well as UniARC. See [NOTICE](NOTICE) and
[third-party attribution](docs/third_party.md).

## License

Project code is licensed under [Apache-2.0](LICENSE). Original third-party notices
are retained. Dataset contents and model weights are obtained separately and
remain subject to their own terms.
