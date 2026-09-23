# Third-party attribution

| Component | Included material | Attribution / terms |
| --- | --- | --- |
| XARES-LLM | Core package, task configurations, selected encoder adapters | [Xiaomi Research XARES-LLM](https://github.com/xiaomi-research/xares-llm), Apache-2.0. Original headers and `xares-llm/LICENSE` retained. The imported snapshot comes from the paper authors' modified repository. |
| PyTorch, Transformers, Lightning, PEFT and other dependencies | Dependency declarations only | Installed separately under their respective licenses. |
| DAC, SpeechTokenizer, WavTokenizer and K-means resources | Adapter code and preparation instructions | Third-party implementations and weights are obtained separately. Review the resource's own license and access conditions. |
| Llama, SmolLM2, HuBERT, WavLM, Wav2Vec2 and Whisper | Model identifiers / loading code | Weights are not bundled; obtain them from the relevant publisher. The repository license does not replace model terms. |
| Benchmark datasets | Configuration and loading code | Dataset audio and annotations are not bundled. Obtain the required data under each dataset's terms. |

Integration modifications are identified in modified source-file comments and
in `source_manifest.json`; new orchestration files are project code. This table
describes included material, not a claim that all external resources use the
same license.
