# Preparing data and model resources

The repository contains source code, configurations, and preparation tools. Audio datasets, pretrained model weights, trained adapters, and K-means centroids must be obtained separately. Resource links and identifiers below come from the supplied repositories and their code; a model family name does not establish the revision used for the manuscript.

Use the installation instructions in [README.md](../README.md) first. The two backends use different data formats and can use separate Python environments.

## Where local resources live

A convenient layout at the repository root is:

```text
models/                  # locally downloaded pretrained weights
data/manifests/           # your train/validation/test manifests
data/packed/              # UniARC .scp/.mrk/.seq files
external/                # optional separately obtained codec source
runs/                    # new probing outputs and adapters
xares-llm/xares_data/     # default XARES dataset cache
xares-llm/experiments/    # new XARES results
```

The root launcher starts the probing backend from the repository root and XARES from `xares-llm/`. Probe configuration paths are resolved relative to the repository root. XARES relative encoder, data, and output paths are resolved from `xares-llm/`. Use absolute paths for external data caches and local model overrides when switching between the two backends.

Do not add downloaded corpora or weights to the source distribution. Their access conditions and licenses are separate from this repository's code license.

## Lightweight strategy: benchmark datasets

The included task YAML files refer to two Hugging Face dataset repositories:

- [mispeech/xares_llm_data](https://huggingface.co/datasets/mispeech/xares_llm_data): the benchmark tar shards for 19 of the 20 configured datasets.
- [mispeech/MECAT-Caption](https://huggingface.co/datasets/mispeech/MECAT-Caption): the MECAT captioning shards.

The original XARES README and `xares_llm/download_data.py` specify these identifiers. A task downloads the shards that it needs. To download the repositories in advance, run these commands from `xares-llm/` in its environment:

```bash
hf download mispeech/xares_llm_data --local-dir xares_data --repo-type dataset
hf download mispeech/MECAT-Caption --local-dir xares_data --repo-type dataset
```

Alternatively, `python -m xares_llm.download_data` fetches both repositories. This is a full dataset download, not a small smoke test. Set `XARES_DATA_HOME` to an absolute directory to change the cache location before invoking Python. For example, in Bash:

```bash
export XARES_DATA_HOME=/absolute/path/to/xares_data
```

In PowerShell:

```powershell
$env:XARES_DATA_HOME = 'D:\datasets\xares_data'
```

Inspect `xares-llm/src/xares_llm/tasks/single/train/` and `single/eval/` for exact shard names, label keys, prompts, and metrics. The dataset key list and paper mapping are in [paper_mapping.md](paper_mapping.md). The named train/test configurations do not by themselves provide the provenance of the separate frozen-probing splits.

For a custom XARES dataset, tar samples must pair an audio file with a JSON/JSONL record using the same sample key. The dataset configuration's `key` selects the reference text or label from that record. Use a training YAML and evaluation YAML with the schema shown in [the XARES README](../xares-llm/README.md). The audio loader converts input to 16 kHz; the training configuration can crop long audio.

## Frozen strategy: original dataset sources

These pointers were recorded in the source UniARC documentation. They identify acquisition sources, not the authors' prepared splits. Obtain the appropriate access and then create explicit, disjoint train/validation/test manifests.

| Dataset | Resource recorded by the source repository | Preparation needed for the paper task |
| --- | --- | --- |
| LibriSpeech | [OpenSLR 12](https://www.openslr.org/12/) | LS-100 versus LS-960 training; preserve test-clean/test-other as separate evaluations |
| IEMOCAP | Source documentation specifies academic access; no direct download URL was provided | Four-class emotion mapping and speaker/session split must be supplied |
| CREMA-D | [CREMA-D repository](https://github.com/CheyneyComputerScience/CREMA-D) | Reconstruct the intended emotion labels and split IDs |
| SLURP | [SLURP repository](https://github.com/pswietojanski/slurp) | Use the intended intent labels and train/development/test partitions |
| UrbanSound8K | [Dataset website](https://urbansounddataset.weebly.com/urbansound8k.html) | Record fold assignment and category-label text |
| Clotho | [Zenodo record in the source docs](https://zenodo.org/record/3490684) | Record dataset version, split, training-caption selection, and all evaluation references |
| GTZAN | [Download pointer in the source docs](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | Record genre labels and exact split IDs; this pointer alone does not define the study's partition |
| Song Describer Dataset | [Zenodo record in the source docs](https://zenodo.org/record/10072001) | Prepare the intended audio IDs, captions, and split files |

The original `data/` directory and corpus-to-split preprocessing scripts were not supplied as a complete reproducible dataset release. The new generic packer below standardizes a prepared manifest; it does not recover the authors' missing split decisions.

### Build the probing input files

Prepare each waveform as **mono, 16 kHz, uncompressed PCM16 WAV**. The packer validates that format and requires at least 400 samples; it does not resample, mix channels, normalize transcripts, choose folds, or assign emotion labels. The WavTokenizer model wrapper handles its own 16-to-24-kHz conversion.

Create a UTF-8 JSONL manifest with one object per utterance, for example `data/manifests/train.jsonl`:

```json
{"id":"train_0001","audio":"audio/train_0001.wav","text":"THE REFERENCE TRANSCRIPTION"}
{"id":"train_0002","audio":"audio/train_0002.wav","text":"THE SECOND REFERENCE"}
```

Here `audio` is relative to the manifest's directory, or an absolute path. IDs must be unique, nonempty, and contain no whitespace. `text` must be a nonempty single line. For a classification task, put the label text expected by the prompt in `text`.

From the repository root:

```bash
python -m uniarc.prepare_data --manifest data/manifests/train.jsonl --output data/packed/train
python -m uniarc.prepare_data --manifest data/manifests/val.jsonl --output data/packed/val
python -m uniarc.prepare_data --manifest data/manifests/test.jsonl --output data/packed/test
```

Each output directory must be new. It contains:

- `audio.seq`: concatenated raw PCM16 waveform bytes, without WAV headers.
- `audio.mrk`: utterance count followed by `utterance_id byte_offset byte_length` rows.
- `audio.scp`: the `.mrk` filename, resolved relative to the `.scp` file.
- `text.txt`: `utterance_id reference_text` rows.

Configure `data.train`, `data.val`, and `data.test` with their `scp` and `text` paths. Example configurations are in `uniarc/configs/`. Training requires train/validation data; inference requires test data and a trained adapter checkpoint. Select the appropriate task prompt instead of leaving `task: asr` for every dataset.

```bash
python run.py probe train --config uniarc/configs/hubert_asr.json --dry-run
```

This validates the configuration structure and displays resolved paths without loading models or checking whether the data and weights exist. The normal train/infer invocation checks required local paths before execution. Sample configurations use one GPU for accessibility; Section 3.3's frozen experiment used four A100 GPUs. The example is an execution template, not a claim of matched experimental conditions.

## Language model weights

| Strategy | Model resource | How it is selected |
| --- | --- | --- |
| Lightweight | [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | XARES `--args` field `decoder_model_name` |
| Lightweight | [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | Same field; use a distinct output directory for the new experiment |
| Frozen | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | Probe `model.llama_ckpt_path`, local directory or model identifier |
| Frozen | [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) | Same field; use matching backbone weights during training and inference |

Llama access may require accepting the provider's terms. Obtain authorization and authenticate through the model provider's normal tooling; no access tokens belong in configuration files committed to this repository. Source directories named `SmolLM2-135M` and `SmolLM2-360M` correspond to the base-model variants used by the supplied scripts, not an established instruction-tuned variant.

The original experiment snapshots did not consistently pin model revisions. For a reproducible new run, download a fixed revision into a local directory and record the revision and file checksums with the resolved experiment configuration.

## Audio encoder weights and optional code

| Encoder | Resource / acquisition method | Lightweight configuration | Frozen configuration |
| --- | --- | --- | --- |
| HuBERT | [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft) | `--model_args` field `model_path` | `model.hubert_ckpt_path` |
| WavLM | [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large) | `model_path` | `model.wavlm_ckpt_path`; the retained processor also uses `model.hubert_ckpt_path` |
| Wav2Vec2 | [facebook/wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h) | `model_path` | Not a retained Table 2 backend |
| Whisper | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | `model_path` | Not a retained Table 2 backend |
| DAC | [descript/dac_16khz](https://huggingface.co/descript/dac_16khz) for the Transformers wrapper; original DAC package download for probing | `model_path`, Transformers model format | `model.DAC_path`, a checkpoint accepted by `dac.DAC.load` |
| SpeechTokenizer | [SpeechTokenizer source repository](https://github.com/ZhangXInFD/SpeechTokenizer), model configuration and matching checkpoint | `model_path` or `SPEECHTOKENIZER_MODEL_PATH`, a directory with `config.json` and a `.pt` checkpoint | `model.speechtokenizer_config_path`, `model.speechtokenizer_ckpt_path`; install its package or configure `external_source_dirs` |
| WavTokenizer | [WavTokenizer large speech 75-token model](https://huggingface.co/novateur/WavTokenizer-large-speech-75token), with the matching source and YAML | `WAVTOKENIZER_ROOT`, plus `config_path` / `model_path` or `WAVTOKENIZER_CONFIG` / `WAVTOKENIZER_CHECKPOINT` | `model.wavtokenizer_config_path`, `model.wavtokenizer_ckpt_path`, and `external_source_dirs` |

The public model identifiers are also visible in the released wrapper defaults. They provide portable loading targets; they do not prove that an unpinned current download has the same bytes as the original research checkpoint.

For the original DAC format, the supplied research download script uses:

```python
import dac
checkpoint_path = dac.utils.download(model_type="16khz")
print(checkpoint_path)
```

Set the returned path as `model.DAC_path`. Do not pass a Hugging Face `safetensors` file to the native DAC loader merely because both resources are called DAC.

SpeechTokenizer needs its configuration and checkpoint from the same release. Keep one intended `.pt` checkpoint in the directory used by the lightweight wrapper, which selects a matching file from that directory. If its package is already installed, remove unused placeholder directories from the probe's `external_source_dirs`; a real run checks every configured directory. WavTokenizer needs the original module providing `decoder.pretrained.WavTokenizer`; the source repository's incomplete vendored copy is not included here. Use the source instructions linked from its model resource and point `WAVTOKENIZER_ROOT` or `external_source_dirs` at that checkout. Its configuration filename in the research tree is `wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml`, with checkpoint `wavtokenizer_large_speech_320_v2.ckpt`.

### K-means resources

The HuBERT/WavLM K-means wrappers additionally require `.npy` centroid matrices, expected for the paper to have **1,000 rows** and an encoder-matching feature dimension (1,024 for the selected large variants). Set `kmeans_path` through `--model_args`, or use `HUBERT_KMEANS_PATH` / `WAVLM_KMEANS_PATH`.

The original filenames are `hubert-large-ls960-ft_k1000_balanced.npy` and `wavlm-large_k1000_balanced.npy`. These arrays and their complete fit recipe are not bundled, and no public download was supplied. Supply the original centroids or fit a documented new clustering model on training data only; the latter constitutes a new experiment unless equivalence is established.

## Checks before a result can be compared with the paper

Record the dataset version and split IDs, label mapping, caption references, sample rate, model revision, centroid provenance, selected hidden layer/codebook, prompt, projector, seed, precision, GPU count, stopping rule, and decoding settings. Keep train, validation, and test IDs separate. The new packer only validates the supplied files; it cannot establish that the split is scientifically equivalent to the manuscript.

XARES produces per-task metric outputs using the selected YAML. Probe inference produces text predictions; assess them with the appropriate evaluation protocol. In particular, Table 2 uses WER percentages, while the lightweight suite uses clipped inverse WER/CER. Captioning needs the intended reference set and metric assets; a one-reference manifest is not automatically a full multi-reference evaluation.

No pretrained adapter or guaranteed reproduction score is supplied. See [paper_mapping.md](paper_mapping.md) for the exact experimental scope and known implementation differences.
