# UniARC frozen-encoder probes

This directory integrates the five prompt-conditioned probe implementations from
`Anachronism-N/UniARC/code`. The paper's main comparison uses HuBERT, WavLM, DAC,
and SpeechTokenizer; WavTokenizer is an additional implementation retained from
the source repository. The encoder and Llama weights stay frozen. Ten adjacent
audio feature frames are concatenated and passed through the original
two-layer MLP (8192 hidden units, ReLU) into Llama's embedding dimension.

The release keeps the effective encoder feature computation and original
generation settings. It replaces experiment-specific scripts with a shared
configuration-driven launcher. It does not contain pretrained models, trained
adapters, audio datasets, cached tokens, logs, or reported predictions.

## Run

Run these commands from the unified repository root. A configuration dry run
requires only Python and checks configuration structure, **not model availability
or numerical reproducibility**:

```sh
python uniarc/run.py train --config uniarc/configs/hubert_asr.json --dry-run
python uniarc/run.py infer --config uniarc/configs/hubert_asr_infer.json --dry-run
```

Create a dedicated Python 3.11/3.12 environment (Bash example):

```sh
python3.11 -m venv .venv-probe
source .venv-probe/bin/activate
python -m pip install --upgrade pip
# Install the matching torch==2.9.1 / torchaudio==2.9.1 CUDA build for your driver.
python -m pip install -r requirements/probe.txt
```

On Windows, activate with `.venv-probe\Scripts\Activate.ps1`. The training target
is Linux/CUDA. `requirements/probe.txt` is an integration environment; the source
repository's historical export used torch 2.0.0 and transformers 4.36.1 and was
not a complete lock for these scripts. The new versions do not claim numerical
identity with the paper. CPU tests do not validate multi-GPU execution.

Install the optional encoder dependencies described below. Download the
pretrained resources under their respective terms, prepare data, then update
the example JSON paths:

```sh
python uniarc/run.py train --config uniarc/configs/hubert_asr.json
python uniarc/run.py infer --config uniarc/configs/hubert_asr_infer.json
```

Every local path in a run configuration resolves against the **repository root**,
regardless of the current directory or the configuration file's location. Use
`./models/...` for local model directories or supply an explicit Hugging Face
repository ID. Local file existence is checked before model imports in a real
run. Shell environment variables and `~` are expanded. All example model/data
paths are placeholders. WavLM retains the upstream HuBERT processor. The codec
probes no longer load an unused HuBERT processor.

Set `task` to `asr`, `emotion`, `music_genre`, `audio_caption`,
`sound_classification`, `intent`, or `music_caption`. An optional `prompt` overrides
the selected task prompt identically for training and inference. Templates for
each encoder are in `configs/`; these are runnable configuration examples, not
a claim that every encoder/task dataset was experimentally reproduced.

Training defaults to AdamW at `1e-5`, batch size 8, accumulation 4, seed 3407,
150 maximum epochs, and early-stopping patience 30. Change `trainer.devices`,
precision, batch size, and epochs for your hardware and experimental protocol.
These defaults derive from the original scripts and do not encode all paper
experiment settings. Full-size backbones can require substantial GPU memory.

`checkpoint` loads adapter parameters only. For training it is an initialization
checkpoint; optimizer, scheduler, epoch, and random state are **not resumed**.
Inference requires a trained adapter and a `data.test` split. Choose the exact
encoder and Llama resources used to train that adapter. Inference writes
`predictions.jsonl` with `id`, `prediction`, and `reference` and refuses to overwrite
an existing predictions file. The run's resolved configuration and training CSV
metrics are saved under `output_dir`.
See [`docs/evaluation.md`](../docs/evaluation.md) for the root prediction evaluator
and the normalization choices that affect WER and classification accuracy.

## Data

The original format is retained:

- SCP: one MRK path per line, resolving relative to the SCP file.
- MRK: first line is record count, subsequent lines are
  `utterance_id byte_offset byte_count`.
- SEQ: the same MRK stem with suffix `.seq`; concatenated raw little-endian
  PCM16, mono, 16-kHz audio.
- Text: `utterance_id reference text or class name`, one utterance per line.

Duplicate IDs, missing references, wrong record counts, invalid byte ranges,
and audio shorter than 400 samples are rejected. Binary files are opened
lazily per worker. The codec collator preserves PCM16 amplitude units; each
retained codec model performs its original normalization. Continuous probes
retain the original feature-extractor preprocessing.

To prepare your own WAV files, write a JSONL manifest:

```json
{"id":"utterance_001","audio":"audio/utterance_001.wav","text":"reference transcription or class label"}
```

Audio paths resolve relative to that manifest. The converter requires mono
16-kHz PCM16 WAV input and performs no silent resampling or relabeling:

```sh
python -m uniarc.prepare_data --manifest data/train.jsonl --output data/packed/train
```

Repeat for validation/test splits and point `data.<split>.scp` and `.text` at
the generated `audio.scp` and `text.txt`. The output split directory must be new.
Class labels, split assignments, dataset permissions, and metrics remain the
experimenter's responsibility; no research dataset is synthesized or bundled.

## Optional encoders

SpeechTokenizer must expose `speechtokenizer.SpeechTokenizer` with
`load_from_checkpoint`, `encode`, and `quantizer.decode`. WavTokenizer must expose
`decoder.pretrained.WavTokenizer.from_pretrained0802` and
`encoder.utils.convert_audio`. Use `external_source_dirs` for compatible local
source checkouts; example paths are `external/SpeechTokenizer` and
`external/WavTokenizer`. DAC uses `dac.DAC.load` and `audiotools.AudioSignal`.
These third-party repositories and their weights are not redistributed here.
Check their upstream licenses and API compatibility before running.

DAC's Python packages can be installed with `python -m pip install
descript-audio-codec descript-audiotools`. For SpeechTokenizer and WavTokenizer,
follow the linked source preparation instructions in
[`docs/data_and_models.md`](../docs/data_and_models.md), and use compatible
checkouts in `external_source_dirs`. Resolve their additional dependencies in
the probe environment and record the revisions used.

## Source changes and validation limits

- Retained five `model_llama2_*_prompt.py` implementations plus padding utilities.
  Removed unused legacy imports and constructor arguments, commented debugging
  code, stale machine paths, and unrelated experiment scripts.
- Parameterized task prompts and learning rate. Adapter output size is derived
  from `Llama.config.hidden_size`, supporting the source's 1B/8B variants without
  manual edits or silently mixing their dimensions.
- Fixed checkpoint filtering: trainability is checked against
  `named_parameters()`. The original code checked detached `state_dict()`
  tensors' `requires_grad`, which excluded all adapter parameters. Missing
  adapter parameters now raise an error; frozen backbone tensors are not saved.
- HuBERT/WavLM frame lengths now use the encoder's convolution length function
  rather than a rounded stride approximation, avoiding padding-mask errors for
  utterance lengths near convolution boundaries.
- Frozen encoder/codec and Llama modules stay in evaluation mode during adapter
  training. This prevents dropout, encoder masking, or codec codebook updates
  from being re-enabled by Lightning's recursive `train()` call. The adapter
  still enters training mode.
- Training explicitly assigns positions by cumulative valid-token count. Audio
  padding between the audio prefix and prompt therefore does not shift prompt
  positions; this matches the valid-token position convention used by generation.
- Added deterministic path resolution, data-format validation, worker-safe
  lazy binary reads, and a reproducible run configuration record.

Model-feature conventions remain unchanged, including SpeechTokenizer's original
use of all decoded codebooks. This probe and the XARES branch therefore should
not be assumed interchangeable. Generation settings also remain those of the
retained models. The fixes to training mode and token positions can change
numerical results from the original scripts; the release has not re-estimated
the paper's scores or established numerical equivalence.

Tests cover configuration failures, a synthetic PCM16 data-format round trip,
and, when CPU PyTorch is installed, actual adapter checkpoint restoration and
initializer output dimensions using mock backbones and PyTorch meta tensors.
With Transformers and Lightning installed, an additional CPU flow test runs
the HuBERT probe against a real, randomly initialized one-layer Llama. It checks
finite loss, backward gradients, an optimizer update limited to the adapter,
valid-token positions, and real beam-search inference. Its tokenizer/audio
frontend are test doubles and its projector hidden layer is reduced from 8192
to 32 units solely to keep this test small. These tests do not download models
or validate full-model numerical equivalence:

```sh
python -m unittest discover -s uniarc/tests -v
```
