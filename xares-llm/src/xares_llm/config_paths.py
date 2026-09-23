# SPDX-License-Identifier: Apache-2.0
"""Preset discovery without importing the training stack."""

from importlib.resources import files

_ROOT = files("xares_llm").joinpath("tasks")

AVAILABLE_TRAINING_CONFIGS = {
    "all": _ROOT.joinpath("all/train/all_train_config.yaml"),
    "task1": _ROOT.joinpath("task1/train/train_task1_config.yaml"),
    "task2": _ROOT.joinpath("task2/train/train_task2_config.yaml"),
}
AVAILABLE_EVALUATION_CONFIGS = {
    "all": _ROOT.joinpath("all/eval/evaluation_all.yaml"),
    "task1": _ROOT.joinpath("task1/eval/eval_task1_config.yaml"),
    "task2": _ROOT.joinpath("task2/eval/eval_task2_config.yaml"),
}
AVAILABLE_TRAINING_CONFIGS.update({
    item.name.removesuffix("_config.yaml"): item
    for item in _ROOT.joinpath("single/train").iterdir()
    if item.name.endswith("_config.yaml")
})
AVAILABLE_EVALUATION_CONFIGS.update({
    item.name.removesuffix("_test_config.yaml"): item
    for item in _ROOT.joinpath("single/eval").iterdir()
    if item.name.endswith("_test_config.yaml")
})
