# Evaluating predictions

XARES-LLM evaluates its tasks during `xares_llm.run` and writes `scores.tsv`.
The frozen backend writes one `predictions.jsonl` record per input utterance:

```json
{"id":"utt-001","prediction":"happy","reference":"happy"}
```

The root evaluator keeps empty predictions and rejects missing fields, duplicate
IDs and blank records. It never truncates mismatched files or silently drops
failed generations. Examples below use an existing predictions file:

```bash
python scripts/evaluate.py runs/uniarc/hubert/emotion/predictions.jsonl --metric accuracy
python scripts/evaluate.py runs/uniarc/hubert/asr/predictions.jsonl --metric wer
python scripts/evaluate.py runs/uniarc/hubert/audio_caption/predictions.jsonl --metric fense
```

Accuracy defaults to stripped, case-sensitive exact label matching. The old
`ER_test.py` compared only the first whitespace-delimited token. Use
`--label-mode first-token` to reproduce that convention; it can conflate
multiword class names, so record the choice explicitly.

WER requires `jiwer` (`python -m pip install jiwer==4.0.0`). The default `xares`
normalization follows the old ASR evaluator's iWER path: NFKC, lowercase, replace
hyphens with spaces, delete ASCII punctuation, collapse whitespace. `--normalization
none` disables it. WER is a fraction (lower is better and it can exceed 1);
`iwer = 1 - min(WER, 1)` is also emitted. The old script had another normalization
that expands contractions; these values must not be mixed without documenting
the convention.

FENSE uses the original caption evaluator's `aac_metrics.Evaluate` interface
(`python -m pip install aac-metrics`). It accepts a reference string with `|`
separators or an explicit list of reference strings, and may download metric
models on first execution. `--device cuda` is optional. This integration has not
recomputed FENSE or verified its model downloads; pin and record the exact
`aac-metrics` version for a reproduced experiment.

`--output path.json` writes a new score file and refuses to overwrite one.
Metrics and preprocessing choices appear in that output. Scores from the two
backends are not automatically interchangeable; consult [paper_mapping.md](paper_mapping.md).
