# UniARC: Unified Audio Representation Comparison

<p align="center">
  <b>Evaluating Discrete Tokens and Continuous Features for Unified Audio Understanding in AudioLLMs</b><br>
  <i>Submitted to Interspeech 2026</i>
</p>

## Overview

UniARC is a comprehensive evaluation framework designed to systematically benchmark and compare **discrete tokens** and **continuous features** for general audio understanding in Large Audio Language Models (AudioLLMs). 

Given the rapid development of AudioLLMs, the community has seen a shift from continuous features (e.g., HuBERT, WavLM, Whisper) to discrete tokens (e.g., DAC, SpeechTokenizer), and hybrid approaches. To address the lack of clear consensus on which paradigm is better for general audio understanding, this project conducts a systematic study across **Speech, Sound, and Music** domains using two complementary pipelines:
1. **XARES-LLM Framework**: Instruction-tuning based on SmolLM2-135M and SmolLM2-360M using Low-Rank Adaptation (LoRA).
2. **UniARC Pipeline**: A customized probing-based pipeline scaling up to Llama-3.2-1B and Llama-3.1-8B with frozen backbones.

## Key Findings

- **Semantic Compatibility is Key**: Explicit semantic alignment—rather than acoustic fidelity—is the decisive factor when designing representations for AudioLLMs.
- **The "Acoustic Redundancy" Trap**: Reconstruction-focused codecs (e.g., DAC, WavTokenizer) often struggle with high-level semantic understanding due to excessive retention of low-level acoustic variations which act as noise.
- **Power of Semantics in Tokens**: Semantically enriched tokenizers (e.g., SpeechTokenizer) can surprisingly surpass their continuous semantic teachers (e.g., HuBERT).
- **Multi-domain Pre-training**: Scaling the LLM capacity cannot compensate for an encoder's insufficient semantic density. Comprehensive multi-domain pre-training (like WavLM and Whisper) is critical for general-purpose audio intelligence.
- **Efficiency vs. Performance**: While continuous features may edge out in certain domains, discrete tokens significantly reduce sequence lengths and accelerate training convergence.

## Documentation

For comprehensive details about the project, framework architecture, and instructions on how to run the code, please refer to our dedicated documentation:

- 📖 [**Project Description**](docs/PROJECT.md): Detailed information on the research background, supported architectures, evaluated tasks, and full repository structure.
- 🚀 [**Usage Guide**](docs/USAGE.md): Step-by-step instructions for environment setup, data preparation, training, inference, and evaluation.

## Supported Tasks

Our evaluation spans 7 core tasks across 3 diverse domains:
- **Speech**: Automatic Speech Recognition (LibriSpeech), Emotion Recognition (IEMOCAP, CREMA-D), Intent Classification (SLURP)
- **Sound**: Urban Sound Classification (UrbanSound8K), Sound Captioning (Clotho)
- **Music**: Genre Classification (GTZAN), Music Captioning (Song Describer Dataset)

## Citation

If you find our work useful, please consider citing our paper (BibTeX will be updated upon acceptance):

```bibtex
@inproceedings{uniarc2026,
  title={Evaluating Discrete Tokens and Continuous Features for Unified Audio Understanding in AudioLLMs},
  author={Peng, Jing and Nie, Zichao and Zhang, Zhisheng and Wu, Zhiyong},
  booktitle={Interspeech 2026},
  year={2026}
}
```