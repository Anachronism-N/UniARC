"""Check routing and argument preservation without importing ML dependencies."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uniarc_launcher", ROOT / "run.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_xares_json_and_space_paths_are_preserved(self):
        args = ["example/custom encoder.py", "cremad", "cremad", "--args",
                '{"output_dir": "runs/my experiment"}']
        command, cwd, env = launcher.build_command("xares", args, "python-custom")
        self.assertEqual(command, ["python-custom", "-m", "xares_llm.run", *args])
        self.assertEqual(cwd, ROOT / "xares-llm")
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(cwd / "src"))

    def test_probe_uses_source_root(self):
        command, cwd, _ = launcher.build_command("probe", ["--help"])
        self.assertEqual(cwd, ROOT)
        self.assertEqual(command, [sys.executable, str(ROOT / "uniarc/run.py"), "--help"])

    def test_environment_is_not_mutated(self):
        with patch.dict(os.environ, {"PYTHONPATH": "existing-path"}):
            _, _, env = launcher.build_command("xares", [])
            self.assertTrue(env["PYTHONPATH"].endswith(os.pathsep + "existing-path"))
            self.assertEqual(os.environ["PYTHONPATH"], "existing-path")

    def test_child_exit_code_is_preserved(self):
        with patch.object(launcher.subprocess, "call", return_value=7) as child:
            self.assertEqual(launcher.main(["probe", "--help"]), 7)
        self.assertNotIn("shell", child.call_args.kwargs)

    def test_show_command_works_outside_checkout(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "--show-command", "xares", "--help"],
            cwd=ROOT.parent, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("xares_llm.run", result.stdout)


if __name__ == "__main__":
    unittest.main()
