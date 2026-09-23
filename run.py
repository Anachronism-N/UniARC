#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch either UniARC evaluation backend without importing its ML stack."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def build_command(backend, arguments, python=None):
    """Return an argv list, working directory and environment for a backend."""
    executable = python or sys.executable
    environment = os.environ.copy()
    if backend == "probe":
        directory = ROOT
        argv = [executable, str(ROOT / "uniarc" / "run.py"), *arguments]
        import_paths = [ROOT / "uniarc"]
    elif backend == "xares":
        directory = ROOT / "xares-llm"
        argv = [executable, "-m", "xares_llm.run", *arguments]
        import_paths = [directory / "src", directory]
    else:
        raise ValueError(f"Unknown backend: {backend}")
    old_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in import_paths] + ([old_path] if old_path else [])
    )
    return argv, directory, environment


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="UniARC: launch the frozen probe or XARES-LLM LoRA backend.",
        epilog="Backend options follow its name. Use 'probe --help' or 'xares --help'.",
    )
    parser.add_argument("--python", help="Python executable in the backend environment")
    parser.add_argument("--show-command", action="store_true",
                        help="Print command and working directory without running it")
    parser.add_argument("backend", choices=("probe", "xares"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    child_args = args.arguments
    if child_args[:1] == ["--"]:
        child_args = child_args[1:]
    command, directory, environment = build_command(args.backend, child_args, args.python)
    if args.show_command:
        print(json.dumps({"command": command, "cwd": str(directory)}, indent=2))
        return 0
    try:
        return subprocess.call(command, cwd=directory, env=environment)
    except FileNotFoundError as error:
        parser.exit(2, f"Cannot launch backend: {error}\nCheck --python and the source checkout.\n")


if __name__ == "__main__":
    raise SystemExit(main())
