"""Configurable UniARC training and inference with lazily imported ML packages."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENCODERS = {
    "hubert": ("model_llama2_continus_prompt", ["hubert_ckpt_path", "llama_ckpt_path"]),
    "wavlm": ("model_llama2_wavlm_prompt", ["hubert_ckpt_path", "wavlm_ckpt_path", "llama_ckpt_path"]),
    "dac": ("model_llama2_DAC_prompt", ["DAC_path", "llama_ckpt_path"]),
    "speechtokenizer": ("model_llama2_speechtokenizer_prompt", ["speechtokenizer_ckpt_path", "speechtokenizer_config_path", "llama_ckpt_path"]),
    "wavtokenizer": ("model_llama2_wavtokenizer_prompt", ["wavtokenizer_ckpt_path", "wavtokenizer_config_path", "llama_ckpt_path"]),
}
PROMPTS = {
    "asr": "Identify the text corresponding to the speech: ",
    "emotion": "Identify the emotion corresponding to the speech: ",
    "music_genre": "Identify the music genre corresponding to the audio: ",
    "audio_caption": "Identify the description corresponding to the audio:",
    "sound_classification": "Identify the urban sound category corresponding to the audio:",
    "intent": "Identify the intent corresponding to the speech: ",
    "music_caption": "Generate a caption for the music: ",
}
LOCAL_MODEL_FILES = {"DAC_path", "speechtokenizer_ckpt_path", "speechtokenizer_config_path", "wavtokenizer_ckpt_path", "wavtokenizer_config_path"}
TOP_LEVEL = {"encoder", "task", "prompt", "model", "data", "output_dir", "checkpoint", "trainer", "batch_size", "num_workers", "seed", "learning_rate", "external_source_dirs", "inference_device", "early_stopping_patience", "save_top_k"}


def resolve_path(value):
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return str(path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve())


def resolve_model(value, force_local=False):
    expanded = os.path.expandvars(os.path.expanduser(value))
    if force_local or Path(expanded).is_absolute() or expanded.startswith((".", "~")) or (REPO_ROOT / expanded).exists():
        return resolve_path(expanded)
    return expanded  # Explicit Hugging Face repository IDs are accepted.


def load_config(path, mode, check_files=False):
    """Validate without importing torch; relative local paths use repository root."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = config.keys() - TOP_LEVEL
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    encoder = config.get("encoder")
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {', '.join(ENCODERS)}")
    if mode not in ("train", "infer"):
        raise ValueError("mode must be train or infer")
    task = config.get("task")
    if task not in PROMPTS:
        raise ValueError(f"task must be one of {', '.join(PROMPTS)}")
    config["prompt"] = config.get("prompt", PROMPTS[task])
    if not isinstance(config["prompt"], str) or not config["prompt"].strip():
        raise ValueError("prompt must be a nonempty string")
    model = config.get("model", {})
    required = ENCODERS[encoder][1]
    if not isinstance(model, dict) or set(model) - set(required):
        raise ValueError(f"model accepts these keys for {encoder}: {required}")
    for key in required:
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ValueError(f"model.{key} is required")
        model[key] = resolve_model(model[key], key in LOCAL_MODEL_FILES)
        if check_files and (key in LOCAL_MODEL_FILES or Path(model[key]).is_absolute()) and not Path(model[key]).exists():
            raise ValueError(f"Model resource does not exist: {model[key]}")
    config["model"] = model
    data = config.get("data", {})
    if not isinstance(data, dict) or set(data) - {"train", "val", "test"}:
        raise ValueError("data accepts train, val and test split objects")
    needed_splits = ["train", "val"] if mode == "train" else ["test"]
    for split in needed_splits:
        if split not in data:
            raise ValueError(f"data.{split} is required for {mode}")
    for split, files in data.items():
        if not isinstance(files, dict) or set(files) != {"scp", "text"}:
            raise ValueError(f"data.{split} must contain scp and text")
        for key, value in files.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"data.{split}.{key} must be a path")
            files[key] = resolve_path(value)
            if check_files and split in needed_splits and not Path(files[key]).is_file():
                raise ValueError(f"Dataset file does not exist: {files[key]}")
    config["data"] = data
    for key, default in (("batch_size", 8), ("num_workers", 0), ("seed", 3407)):
        config.setdefault(key, default)
        if type(config[key]) is not int or config[key] < (1 if key == "batch_size" else 0):
            raise ValueError(f"{key} must be a valid {'positive' if key == 'batch_size' else 'nonnegative'} integer")
    config.setdefault("learning_rate", 1e-5)
    if not isinstance(config["learning_rate"], (int, float)) or not 0 < config["learning_rate"] < float("inf"):
        raise ValueError("learning_rate must be positive and finite")
    config["output_dir"] = resolve_path(config.get("output_dir", f"runs/uniarc/{encoder}/{task}"))
    if config.get("checkpoint"):
        config["checkpoint"] = resolve_path(config["checkpoint"])
        if check_files and not Path(config["checkpoint"]).is_file():
            raise ValueError(f"Checkpoint does not exist: {config['checkpoint']}")
    elif mode == "infer":
        raise ValueError("Inference requires checkpoint (a trained adapter)")
    dirs = config.get("external_source_dirs", [])
    if not isinstance(dirs, list) or any(not isinstance(item, str) for item in dirs):
        raise ValueError("external_source_dirs must be a list of local source directories")
    config["external_source_dirs"] = [resolve_path(item) for item in dirs]
    if check_files:
        for directory in config["external_source_dirs"]:
            if not Path(directory).is_dir():
                raise ValueError(f"External source directory does not exist: {directory}")
    trainer = {"max_epochs": 150, "accelerator": "auto", "devices": 1, "precision": "32-true", "accumulate_grad_batches": 4, "log_every_n_steps": 20}
    supplied = config.get("trainer", {})
    allowed = set(trainer) | {"strategy", "num_nodes", "gradient_clip_val", "limit_train_batches", "limit_val_batches", "deterministic"}
    if not isinstance(supplied, dict) or set(supplied) - allowed:
        raise ValueError(f"trainer accepts these keys: {sorted(allowed)}")
    trainer.update(supplied)
    config["trainer"] = trainer
    config.setdefault("early_stopping_patience", 30)
    config.setdefault("save_top_k", 2)
    config.setdefault("inference_device", "auto")
    return config


def load_adapter(model, checkpoint_path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected a Lightning adapter checkpoint containing state_dict")
    # The model hook verifies that every trainable adapter tensor exists.
    model.on_load_checkpoint(checkpoint)
    return model.load_state_dict(checkpoint["state_dict"], strict=False)


def execute(config, mode):
    for directory in reversed(config["external_source_dirs"]):
        sys.path.insert(0, directory)
    import lightning.pytorch as pl
    import torch
    from torch.utils.data import DataLoader
    from uniarc.data import PackedAudioDataset, collate_codec, collate_continuous

    pl.seed_everything(config["seed"], workers=True)
    module = importlib.import_module("uniarc.model." + ENCODERS[config["encoder"]][0])
    model = module.IS(**config["model"], task_prompt=config["prompt"], learning_rate=config["learning_rate"])
    if config.get("checkpoint"):
        load_adapter(model, config["checkpoint"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    collate = collate_continuous if config["encoder"] in {"hubert", "wavlm"} else collate_codec

    def loader(split, shuffle=False):
        dataset = PackedAudioDataset(**config["data"][split])
        return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle, num_workers=config["num_workers"], collate_fn=collate)

    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if mode == "train":
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger

        checkpoint = ModelCheckpoint(dirpath=output / "checkpoints", filename="{epoch:03d}-{val_loss:.4f}", monitor="val_loss", mode="min", save_top_k=config["save_top_k"])
        callbacks = [checkpoint, EarlyStopping(monitor="val_loss", mode="min", patience=config["early_stopping_patience"])]
        trainer = pl.Trainer(**config["trainer"], default_root_dir=output, logger=CSVLogger(output, name="metrics"), callbacks=callbacks)
        trainer.fit(model, loader("train", True), loader("val"))
        print(f"Best adapter checkpoint: {checkpoint.best_model_path}")
        return
    device = config["inference_device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    data_loader = loader("test")
    records = data_loader.dataset.records
    offset = 0
    # Exclusive creation protects previous inference results.
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        with torch.inference_mode():
            for batch in data_loader:
                batch = model.transfer_batch_to_device(batch, torch.device(device), 0)
                predictions = model.inference(batch)
                references = batch[1]
                if len(predictions) != len(references):
                    raise RuntimeError("Prediction count does not match the input batch")
                for index, (prediction, reference) in enumerate(zip(predictions, references)):
                    stream.write(json.dumps({"id": records[offset + index][0], "prediction": prediction, "reference": reference}, ensure_ascii=False) + "\n")
                offset += len(predictions)
    print(f"Wrote {offset} predictions to {output / 'predictions.jsonl'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "infer"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and show the plan without loading ML dependencies or resources")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.mode, check_files=not args.dry_run)
        if args.dry_run:
            print(json.dumps({"mode": args.mode, "config": config, "validation": "configuration only; model weights and data not checked"}, indent=2))
        else:
            execute(config, args.mode)
    except (ValueError, OSError, ImportError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
