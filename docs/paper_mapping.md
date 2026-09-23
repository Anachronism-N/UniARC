# Paper-to-code mapping

This release brings the two research implementations into one repository under the **UniARC (Unified Audio Representation Comparison)** framework. It does not assert that the reported experiments have been rerun or that the released defaults reproduce every paper value.

The manuscript supplied for this release is **“Discrete vs. Continuous: A Comprehensive Study of Unified Audio Understanding in LALMs”**, labelled *Anonymous submission to Interspeech 2026*. Earlier repository descriptions use a different title. The supplied manuscript does not establish an accepted publication, a DOI, or a public author list; those details must be updated from the final publication when available.

## One framework, two evaluation strategies

The architecture in Section 2 and Figure 1 is an audio encoder, an embedding-space projector, and a language model that generates the task answer. Discrete representations enter the projector as codebook vectors; integer audio token IDs are not simply appended to the language model's text vocabulary.

| Paper component | Lightweight fine-tuning | Frozen-backbone probing |
| --- | --- | --- |
| Implementation directory | `xares-llm/` | `uniarc/` |
| Source repository | `Anachronism-N/UniARC_paper`, subdirectory `xares-llm` | `Anachronism-N/UniARC`, subdirectory `code` |
| Paper results | Figure 2 and Table 1 | Table 2; Figure 3 discusses convergence time |
| Backbones | SmolLM2-135M and SmolLM2-360M | Llama-3.2-1B and Llama-3.1-8B, called “Llama-3 1B/8B” in the manuscript |
| Updated parameters | Projector and LoRA adapters | Projector only; audio encoder and language model frozen |
| Task coverage | 20 dataset configurations | 7 task types across 8 datasets |
| Hardware reported in Section 3.3 | One NVIDIA A100 per experiment | Four NVIDIA A100 GPUs |
| Root launcher | `python run.py xares ...` | `python run.py probe train\|infer --config ...` |

XARES-LLM is the upstream implementation acknowledged in Section 2.1, not a separate paper contribution renamed by this release. Its copyright notices and Apache-2.0 license are retained. See [NOTICE](../NOTICE).

The two strategies deliberately have different model scales, trainable parameters, task coverage, and feature processing. Compare scaling within each strategy; a difference between their absolute scores is not an isolated causal effect of backbone size.

## Encoders and representation choices

Paths in the next table are relative to `xares-llm/example/`. The names of local research checkpoints support the variant mapping below; the manuscript itself often gives only the encoder family. The exact original checkpoint revisions were not recorded in a complete release manifest.

| Paper label | Family / source checkpoint variant | Lightweight wrapper | Frozen Table 2 |
| --- | --- | --- | --- |
| HuBERT | Continuous; `hubert-large-ls960-ft` | `hubert-large/hubertlarge.py` | Yes |
| Wav2Vec2 | Continuous; `wav2vec2-large-960h` | `wav2vec-large/wav2vec_large.py` | No |
| WavLM | Continuous; `wavlm-large` | `wavlm_large/wavlm_large.py` | Yes |
| Whisper | Continuous; `whisper-large-v3` | `whisper-large-v3/whisper_encoder.py` | No |
| HuBERT+Kmeans | Discrete clustering; nearest of 1,000 HuBERT centroids | `hubert-kmeans/hubert_kmeans_encoder.py` | No |
| WavLM+Kmeans | Discrete clustering; nearest of 1,000 WavLM centroids | `wavlm-kmeans/wavlm_kmeans_encoder.py` | No |
| DAC | Discrete codec; 16 kHz configuration | `dac/dac_encoder.py` | Yes |
| SpeechTokenizer | Discrete codec with semantic distillation | `speechtokenizer/st_encoder.py` | Yes |
| WavTokenizer | Discrete codec; large speech 75-token configuration | `wavtokenizer-large/wavtokenizer_encoder.py` | No; probing wrapper is an extension |

Section 3.1 specifies 16 kHz input for all encoder families except WavTokenizer, which uses 24 kHz. The XARES loader supplies 16 kHz audio, and the WavTokenizer wrapper resamples it to 24 kHz. HuBERT/WavLM use their last hidden state; clustering replaces those feature vectors with their nearest centroids.

The retained frozen model classes are `model_llama2_continus_prompt.py`, `model_llama2_wavlm_prompt.py`, `model_llama2_DAC_prompt.py`, `model_llama2_speechtokenizer_prompt.py`, and `model_llama2_wavtokenizer_prompt.py` under `uniarc/model/`. The legacy `llama2` filename is historical; it does not mean Table 2 used Llama 2.

## Parameters: reported protocol versus source defaults

“Default hyperparameters” in Section 3.3 is insufficient to identify all original runs. The table below distinguishes explicit paper settings from values recovered from source. A fresh experiment should save its resolved configuration, package versions, model revisions, seeds, and GPU information.

| Setting | Paper statement | Source evidence and release guidance |
| --- | --- | --- |
| Lightweight adaptation | LoRA-based instruction tuning | `xares_llm/task.py`: rank 8, alpha 32, dropout 0.1, target modules `all-linear`; these are source defaults |
| Lightweight projector | Audio-to-text embedding projector | The research experiment scripts explicitly selected `mlp`; the backend's generic default is `linear`, so select `mlp` in paper-oriented commands |
| Lightweight MLP | Not fully dimensioned in the manuscript | `modeling_xaresllm.py`: audio dimension -> decoder dimension -> decoder dimension, with GELU |
| Lightweight optimizer | Defaults | Source defaults: AdamW, learning rate `1e-4`, weight decay `0.01`, warmup 200 steps, gradient accumulation 4, batch size 4/device, seed 42 |
| Lightweight training length | Not specified per task | Base default is 10,000 steps; task YAML files may override it, including 50,000 for LibriSpeech. Inspect the selected YAML and overrides |
| Lightweight audio duration | Not specified | Source training default crops audio to 30 seconds; encoder-specific chunking and truncation also apply |
| Probing adaptation | Frozen LLM; only adapter optimized | Frozen encoders and LLM; optimizer receives trainable projector parameters |
| Probing projector | Two-layer MLP; concatenate 10 consecutive frames | Research classes use an intermediate width of 8,192 and ReLU; the output must match the selected Llama hidden size |
| Probing optimizer | AdamW, learning rate `1e-5` | Research defaults additionally use betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-6` |
| Probing trainer | Four A100 GPUs | Original HuBERT script: seed 3407, batch size 8/device, accumulation 4, maximum 150 epochs, validation-loss early stopping patience 30, bf16 mixed precision. Other scripts differ; these are not universal paper settings |
| Decoding | Task-dependent text generation | Original classes contain beam-search and repetition settings; save the actual generation settings used for each rerun |

## Figure 2 and Table 1: 20 datasets

Training and evaluation configurations are in `xares-llm/src/xares_llm/tasks/single/`; the same keys are accepted by the XARES launcher. The original single-task experiment scripts trained and evaluated the same dataset in each run. `all` trains a multi-dataset configuration and is a different experiment.

The manuscript calls this a classification and captioning suite, but its Figure 2 and the actual configurations also include speech recognition. The metric definitions below follow the released configuration files.

| Domain | Dataset key | Task | Metric in source |
| --- | --- | --- | --- |
| Speech | `librispeech` | English ASR, train-clean-100 / test-clean | iWER |
| Speech | `aishell-1` | Chinese ASR | iCER |
| Speech | `fluentspeechcommands` | Intent / command classification | Accuracy |
| Speech | `speechcommandsv1` | Keyword classification | Accuracy |
| Speech | `voxceleb1` | Speaker comparison | Accuracy |
| Speech | `cremad` | Emotion classification | Accuracy |
| Speech | `asvspoof2015` | Genuine / spoof detection | Accuracy |
| Speech | `voxlingua33` | Language classification | Accuracy |
| Speech | `libricount` | Speaker counting | Accuracy |
| Sound | `esc-50` | Environmental sound classification | Accuracy |
| Sound | `urbansound8k` | Urban sound classification | Accuracy |
| Sound | `fsd50k` | Multilabel sound classification | mAP, 200 classes |
| Sound | `fsdkaggle2018` | Multilabel sound classification | mAP, 41 classes |
| Sound | `clotho` | Audio captioning | FENSE |
| Sound | `mecat` | General audio captioning | DATE |
| Sound | `vocalsound` | Vocal sound classification | Accuracy |
| Music | `gtzan` | Genre classification | Accuracy |
| Music | `freemusicarchive` | Genre classification | Accuracy |
| Music | `nsynth` | Instrument-family classification | Accuracy |
| Music | `songdescriber` | Music captioning | FENSE |

`iWER = max(0, 1 - WER)` and `iCER = max(0, 1 - CER)` in `xares_llm/metrics.py`, with error rates expressed as fractions. Higher is better for these transformed scores. They must not be compared directly with Table 2's WER percentages. A zero iWER does not distinguish 100% WER from a larger error rate.

Table 1 groups nine speech, seven sound, and four music datasets. With unit task weights, Overall is the average across all 20 tasks, not the unweighted average of the three domain means. For example, the published Whisper/135M values are Speech 0.858, Sound 0.618, Music 0.666, Overall 0.735. These are manuscript reference values, not results measured during repository integration. Missing tasks or duplicate score rows change the aggregation and must be detected before comparison.

## Table 2: frozen probing tasks

Every reported row combines one of **HuBERT, WavLM, SpeechTokenizer, DAC** with a **1B or 8B** backbone. WavTokenizer, K-means, Wav2Vec2, and Whisper are not Table 2 rows.

| Domain | Paper task | Dataset / split scope | Metric |
| --- | --- | --- | --- |
| Speech | ASR | LibriSpeech LS-960, evaluated on test-clean and test-other | WER %, lower is better |
| Speech | ASR | LibriSpeech LS-100, evaluated on test-clean | WER %, lower is better |
| Speech | ER | IEMOCAP; source paths refer to a four-class preparation | Accuracy %, higher is better |
| Speech | ER | CREMA-D | Accuracy %, higher is better |
| Speech | IC | SLURP | Accuracy %, higher is better |
| Sound | USC | UrbanSound8K | Accuracy %, higher is better |
| Sound | SCap | Clotho | FENSE, higher is better |
| Music | GC | GTZAN | Accuracy %, higher is better |
| Music | MCap | Song Describer Dataset (SDD) | FENSE, higher is better |

Seven task types produce more than seven score columns because ASR includes three train/test combinations and ER includes two datasets. As one reference, the manuscript reports HuBERT/1B at 2.31/4.62 WER on LS-960 clean/other and 3.45 WER on LS-100 clean. WER can exceed 100%; the large DAC WER values in Table 2 must not be clipped to percentages between 0 and 100.

The paper does not fully specify all original train/dev/test IDs, speaker partitions, caption handling, checkpoint selections, or corpus preprocessing. Loading the same named dataset is insufficient to establish the same experimental split.

## Reproduction status and remaining differences

- **No end-to-end benchmark rerun is claimed.** Source inspection, dry runs, and small validation tests do not establish the reported WER, accuracy, FENSE, DATE, or convergence times. Weights, private prepared datasets, trained adapters, and original environment snapshots are not bundled.
- **Integration fixes change some inherited behavior.** Portable configuration, checkpoint handling, projector dimensions, and validation improve the release. The probing backend also keeps frozen modules in evaluation mode and corrects padding-related masks/positions: this changes the old training-time dropout/augmentation and padded-sequence behavior. Numerical equivalence to the original experiment scripts still requires paired runs with the same inputs and weights; the paper's results have not been rerun after these fixes.
- **SpeechTokenizer differs across strategies.** The lightweight wrapper takes `forward_feature(..., layers=[0])` and applies the model's transform. The frozen wrapper encodes the audio and decodes the complete returned code sequence through its quantizer. The manuscript does not explain this distinction; do not treat those two feature streams as identical.
- **K-means is not self-contained.** The 1,000-centroid arrays and a complete, versioned clustering fit recipe are absent. Corpus choice, sampling balance, seed, and checkpoint/layer must be recovered before claiming the original clustering experiment.
- **Padding and long audio need model-level checks.** Some inherited lightweight codec wrappers return no feature mask or an all-valid mask, and some wrappers chunk or truncate long waveforms. These choices can affect batched generation. The probe fixes do not establish equivalent feature processing across all encoders or between the two backends.
- **Results require provenance.** Historical predictions, score caches, training logs, and manuscript figures are not a substitute for a configuration-linked run record. This clean release does not use old result caches as evidence of successful reproduction.
- **Timing is unverified.** Figure 3 reports relative training duration with DAC as the baseline. Fresh timing comparisons require the same GPU count, precision, stopping criterion, data, and measured workload; no timing claim is made for this source release.

See [data_and_models.md](data_and_models.md) for resource preparation and [the root README](../README.md) for executable setup instructions.
