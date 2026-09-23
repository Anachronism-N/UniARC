# UniARC fine-tuning backend (XARES-LLM)

This directory contains the XARES-LLM backend used for UniARC's LoRA fine-tuning experiments. It preserves the upstream training/evaluation code, 20 single-dataset train/evaluation presets, the `all`/`task1`/`task2` presets, and the nine encoders in the paper. The unified repository's top-level documentation describes the separate frozen-probing backend.

This is derived from [Anachronism-N/UniARC_paper](https://github.com/Anachronism-N/UniARC_paper), which includes code from [Xiaomi XARES-LLM](https://github.com/xiaomi-research/xares-llm). The original Apache-2.0 license and source copyright notices are retained. Modified source files carry an explicit modification notice.

## Installation

Use a separate environment from the frozen-probing backend. From this directory, with Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements-constraints.txt -e .
python -m xares_llm.run --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. The paper experiments ran on NVIDIA A100 GPUs; Linux/CUDA is the intended reproduction environment. Install FFmpeg as required by [TorchCodec](https://github.com/meta-pytorch/torchcodec). The constraints record versions in the provided upstream `uv.lock` (PyTorch/Torchaudio 2.9.1, TorchCodec 0.9.0, Transformers 4.57.3); they are a source environment reference, not a claim that full experiments have been repeated. Select the matching PyTorch wheel index for your hardware when needed.

Extra encoders have separate dependencies:

```bash
python -m pip install -c requirements-constraints.txt -e '.[kmeans]'
python -m pip install -c requirements-constraints.txt -e '.[speechtokenizer]'
```

WavTokenizer requires its own [upstream implementation](https://github.com/jishengpeng/WavTokenizer) and dependencies. Set `WAVTOKENIZER_ROOT` to that checkout; the incomplete vendored copy from the input repository is not redistributed here. Model weights, centroid arrays, and datasets are not bundled or licensed by this code license.

## Run a paper configuration

Run one experiment on one GPU. The upstream paper scripts explicitly select the MLP projector and enable LoRA:

```bash
CUDA_VISIBLE_DEVICES=0 python -m xares_llm.run \
  example/whisper-large-v3/whisper_encoder.py esc-50 esc-50 \
  --args '{"decoder_model_name":"HuggingFaceTB/SmolLM2-135M","projector_type":"mlp","lora_enabled":true}'
```

Use `HuggingFaceTB/SmolLM2-360M` for the other main-table decoder. All 20 dataset keys appear in `--help`. Single-dataset YAMLs retain their upstream task-specific step budgets (for example, ESC-50 has 2,000 steps and LibriSpeech has 50,000); YAMLs without a step override use the 10,000-step default. The source defaults for learning rate, LoRA rank/alpha/dropout, gradient accumulation, cropping, and evaluation remain in `src/xares_llm/task.py` and the YAML files. `all` means joint training, and does not reproduce the separate single-dataset experiments.

The two positional configuration arguments may also be YAML paths. Use `--model_args` with a JSON object for encoder constructor settings and `--args` for fields of `XaresLLMTrainConfig`. The portable decoder default is now `HuggingFaceTB/SmolLM2-135M`; the original API's `linear` projector default is retained, so use the explicit `mlp` setting above for the paper scripts. `--no-benchmark` disables the CLI's deterministic-algorithm enforcement; seeded random-number initialization remains enabled.

## Encoders and resources

| Encoder | Python file under `example/` | Resource setting |
| --- | --- | --- |
| HuBERT | `hubert-large/hubertlarge.py` | `model_path`, default `facebook/hubert-large-ls960-ft` |
| WavLM | `wavlm_large/wavlm_large.py` | `model_path`, default `microsoft/wavlm-large` |
| Wav2Vec2 | `wav2vec-large/wav2vec_large.py` | `model_path`, default `facebook/wav2vec2-large-960h` |
| Whisper | `whisper-large-v3/whisper_encoder.py` | `model_path`, default `openai/whisper-large-v3` |
| HuBERT-KM | `hubert-kmeans/hubert_kmeans_encoder.py` | HuBERT `model_path` plus `kmeans_path` or `HUBERT_KMEANS_PATH` |
| WavLM-KM | `wavlm-kmeans/wavlm_kmeans_encoder.py` | WavLM `model_path` plus `kmeans_path` or `WAVLM_KMEANS_PATH` |
| DAC | `dac/dac_encoder.py` | `model_path`, default `descript/dac_16khz` |
| SpeechTokenizer | `speechtokenizer/st_encoder.py` | `model_path` or `SPEECHTOKENIZER_MODEL_PATH`, containing `config.json` and a `.pt` checkpoint |
| WavTokenizer | `wavtokenizer-large/wavtokenizer_encoder.py` | `config_path` and `model_path`, or `WAVTOKENIZER_CONFIG` and `WAVTOKENIZER_CHECKPOINT` |

All Hugging Face model identifiers also accept local directories. The public identifiers replace machine-specific paths to models with the same names; their exact revision identity with the authors' local files has not been established. For exact reproduction supply the original snapshots. KM requires the original centroid `.npy` files; retraining a codebook produces a different experimental artifact. SpeechTokenizer uses `forward_feature(layers=[0])` followed by its transform, as in the source. The WavTokenizer variant is large speech / 75-token with 24 kHz internal resampling.

Example with an explicitly supplied codebook:

```bash
python -m xares_llm.run example/hubert-kmeans/hubert_kmeans_encoder.py esc-50 esc-50 \
  --model_args '{"kmeans_path":"/path/to/hubert-large-ls960-ft_k1000_balanced.npy"}' \
  --args '{"decoder_model_name":"HuggingFaceTB/SmolLM2-135M","projector_type":"mlp","lora_enabled":true}'
```

## Data, output, and scope

`XARES_DATA_HOME` controls both downloaded shards and lookup of manually downloaded data. The default is `./xares_data` relative to the current working directory. Data URLs and prompts remain in the source YAML presets. After checking the source datasets' access terms, download with:

```bash
python -m xares_llm.download_data
```

The run writes checkpoints, predictions, score YAMLs, and `scores.tsv` under `experiments/<config>/<decoder>/<encoder>/`. Existing checkpoints resume training and existing score files are cached. Use a fresh `output_dir` via `--args` when changing weights or settings, so cached results cannot be mistaken for a new experiment.

The dummy encoder is an interface example and produces random features; it is not a paper baseline. The source's attention-mask conventions, SpeechTokenizer chunking, WavTokenizer all-valid output mask, and model-specific cropping are retained. These limitations should be considered when comparing new runs. The compact `qformer` implementation is the supplied query-attention projector, not a substituted pretrained Q-Former.

Release preparation removed private absolute paths, backups of `.git`, old logs/results, machine-specific launch scripts, debugging files, unrelated model examples, and the incomplete third-party checkout. Portability changes include deterministic YAML discovery, lazy CLI imports, relative/absolute file loading, a consistent data-home setting, missing-resource errors, the dummy output-mask fix, and normal error propagation from encoder validation. The encoder-freezing typo was corrected to `requires_grad_(False)`. DAC explicitly uses the quantized representation returned by the [pinned Transformers API](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/dac/modeling_dac.py), which was the effective branch in that version. No paper scores or new reproduction claims are added.
