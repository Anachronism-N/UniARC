#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free source checks. Does not claim GPU or experiment validation."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist",
             "models", "checkpoints", "runs", "logs", "lightning_logs", "wandb",
             "xares_data", "third_party", "external", "data"}
BINARY_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors", ".pyc", ".wav", ".flac",
                   ".mp3", ".bin", ".zip", ".tar", ".pdf"}
SECRET_PATTERNS = [
    re.compile(r"\b" + "hf_" + r"[A-Za-z0-9]{25,}\b"),
    re.compile(r"\b" + "gh[pousr]_" + r"[A-Za-z0-9]{30,}\b"),
    re.compile("-----BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
]
MACHINE_PATH = re.compile("/" + r"(?:commondocument|home|mnt|scratch)/[\w.-]+/")


def source_files(root):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS or part.endswith(".egg-info") or part.startswith(".venv") or
               part.startswith("experiments") for part in parts[:-1]):
            continue
        yield path


def check(root=ROOT):
    errors = []
    counts = {"files": 0, "python": 0, "json": 0}
    folded = {}
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        counts["files"] += 1
        if relative.casefold() in folded:
            errors.append(f"case-colliding paths: {relative}, {folded[relative.casefold()]}")
        folded[relative.casefold()] = relative
        if path.suffix.lower() in BINARY_SUFFIXES or "tfevents" in path.name:
            errors.append(f"generated or binary artifact in source: {relative}")
        if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
            errors.append(f"local environment file: {relative}")
        if path.stat().st_size > 5_000_000:
            errors.append(f"unexpected file above 5 MB: {relative}")
            continue
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            errors.append(f"non-text source file: {relative}")
            continue
        if "\ufffd" in content:
            errors.append(f"Unicode replacement character; possible encoding damage: {relative}")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            errors.append(f"possible credential (value withheld): {relative}")
        if path.suffix in {".py", ".json", ".yaml", ".yml", ".toml", ".sh"}:
            if MACHINE_PATH.search(content):
                errors.append(f"developer-machine path: {relative}")
        try:
            if path.suffix == ".py":
                ast.parse(content, filename=relative)
                counts["python"] += 1
            elif path.suffix == ".json":
                json.loads(content)
                counts["json"] += 1
        except (SyntaxError, ValueError) as error:
            errors.append(f"invalid source syntax: {relative}: {error}")
    for required in ("LICENSE", "NOTICE", "README.md", "README.zh-CN.md", "CITATION.cff",
                     "uniarc/run.py", "xares-llm/pyproject.toml", "xares-llm/LICENSE",
                     "docs/paper_mapping.md", "docs/data_and_models.md",
                     "docs/source_manifest.json"):
        if not (root / required).is_file():
            errors.append(f"missing release file: {required}")
    return counts, errors


if __name__ == "__main__":
    counts, errors = check()
    print(json.dumps({"checked": counts, "errors": errors}, indent=2))
    sys.exit(bool(errors))
