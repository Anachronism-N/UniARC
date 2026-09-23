"""Pack user-owned mono 16-kHz PCM16 WAV files into the UniARC format."""

import argparse
import json
from pathlib import Path
import wave


def pack_manifest(manifest, output):
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    rows = []
    seen = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        utt, audio, text = row["id"], row["audio"], row["text"]
        if not isinstance(utt, str) or not utt or any(c.isspace() for c in utt) or utt in seen:
            raise ValueError(f"Invalid or duplicate ID on line {number}")
        if not isinstance(text, str) or not text.strip() or "\n" in text or "\r" in text:
            raise ValueError(f"Text must be a nonempty single line for {utt}")
        path = Path(audio)
        if not path.is_absolute():
            path = manifest.parent / path
        with wave.open(str(path), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError(f"Expected mono 16-kHz PCM16 WAV: {path}")
            if stream.getnframes() < 400:
                raise ValueError(f"Audio is too short for the encoder: {path}")
        seen.add(utt)
        rows.append((utt, path, text))
    if not rows:
        raise ValueError("Manifest contains no examples")
    # A new split directory avoids accidentally replacing an existing dataset.
    output.mkdir(parents=True, exist_ok=False)
    entries, texts = [], []
    with (output / "audio.seq").open("wb") as packed:
        for utt, path, text in rows:
            with wave.open(str(path), "rb") as stream:
                payload = stream.readframes(stream.getnframes())
            entries.append(f"{utt} {packed.tell()} {len(payload)}")
            packed.write(payload)
            texts.append(f"{utt} {text}")
    (output / "audio.mrk").write_text(str(len(entries)) + "\n" + "\n".join(entries) + "\n", encoding="utf-8")
    (output / "audio.scp").write_text("audio.mrk\n", encoding="utf-8")
    (output / "text.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = pack_manifest(args.manifest, args.output)
    except (ValueError, OSError, KeyError, wave.Error) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Packed {count} examples into {args.output}")


if __name__ == "__main__":
    main()
