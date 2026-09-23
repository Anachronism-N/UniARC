#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate aligned probe JSONL predictions with explicit metric conventions."""
import argparse
import json
from pathlib import Path
import re
import string
import unicodedata


def load_records(path):
    records, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank record at line {number}; refusing implicit filtering")
        item = json.loads(line)
        if not isinstance(item, dict) or not all(k in item for k in ("id", "prediction", "reference")):
            raise ValueError(f"Missing id/prediction/reference at line {number}")
        if not isinstance(item["id"], str) or not item["id"] or item["id"] in seen:
            raise ValueError(f"Invalid or duplicate ID at line {number}")
        if not isinstance(item["prediction"], str):
            raise ValueError(f"Prediction must be text at line {number}")
        refs = item["reference"]
        if not isinstance(refs, str) and not (
            isinstance(refs, list) and refs and all(isinstance(r, str) and r.strip() for r in refs)
        ):
            raise ValueError(f"Reference must be text or nonempty text list at line {number}")
        if isinstance(refs, str) and not refs.strip():
            raise ValueError(f"Empty reference at line {number}")
        seen.add(item["id"])
        records.append(item)
    if not records:
        raise ValueError("No prediction records")
    return records


def normalize(text, convention):
    if convention == "none":
        return text
    # Same normalization steps as the source ASR evaluator's iWER path.
    if convention == "xares":
        text = unicodedata.normalize("NFKC", text).lower().replace("-", " ")
        text = text.translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", text).strip()
    raise ValueError(f"Unknown normalization: {convention}")


def score(records, metric, label_mode="exact", normalization="xares", device="cpu"):
    predictions = [item["prediction"] for item in records]
    references = [item["reference"] for item in records]
    result = {"metric": metric, "samples": len(records)}
    if metric in {"accuracy", "wer"} and any(not isinstance(r, str) for r in references):
        raise ValueError(f"{metric} requires exactly one reference string per example")
    if metric == "accuracy":
        if label_mode not in {"exact", "first-token"}:
            raise ValueError("label_mode must be exact or first-token")
        def label(value):
            words = value.split()
            return (words[0] if words else None) if label_mode == "first-token" else value.strip()
        result.update(score=sum(label(p) == label(r) for p, r in zip(predictions, references)) / len(records),
                      label_mode=label_mode)
    elif metric == "wer":
        from jiwer import wer
        refs = [normalize(r, normalization) for r in references]
        if any(not r.strip() for r in refs):
            raise ValueError("Normalization produced an empty reference")
        value = float(wer(refs, [normalize(p, normalization) for p in predictions]))
        result.update(score=value, iwer=1.0 - min(value, 1.0), normalization=normalization)
    elif metric == "fense":
        from aac_metrics import Evaluate
        refs = [r if isinstance(r, list) else [x.strip() for x in r.split("|") if x.strip()]
                for r in references]
        if any(not r for r in refs):
            raise ValueError("FENSE requires at least one nonempty reference per example")
        corpus, _ = Evaluate(metrics=["fense"], device=device)(predictions, refs)
        result.update(score=float(corpus["fense"]), device=device)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--metric", choices=["accuracy", "wer", "fense"], required=True)
    parser.add_argument("--label-mode", choices=["exact", "first-token"], default="exact")
    parser.add_argument("--normalization", choices=["xares", "none"], default="xares")
    parser.add_argument("--device", default="cpu", help="FENSE device")
    parser.add_argument("--output", type=Path, help="Optional new JSON score file")
    args = parser.parse_args()
    try:
        result = score(load_records(args.predictions), args.metric, args.label_mode, args.normalization, args.device)
        payload = json.dumps(result, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        print(payload, end="")
    except (ValueError, OSError, ImportError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
