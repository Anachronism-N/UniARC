"""Portable reader for the original UniARC packed PCM16 audio format."""

import math
import os
from pathlib import Path


class PackedAudioDataset:
    """Read SCP -> MRK/SEQ audio and utterance-keyed text files.

    SCP entries resolve relative to the SCP file. A MRK file starts with its
    record count, followed by ``utterance_id byte_offset byte_count`` rows.
    Its paired SEQ file contains raw mono little-endian PCM16 at 16 kHz.
    """

    def __init__(self, scp, text):
        self.records = []
        self._files = {}
        self._pid = os.getpid()
        scp = Path(scp).resolve()
        labels = {}
        for line_no, line in enumerate(Path(text).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2 or fields[0] in labels:
                raise ValueError(f"Invalid or duplicate text entry at {text}:{line_no}")
            labels[fields[0]] = fields[1]
        ids = set()
        for entry in scp.read_text(encoding="utf-8").splitlines():
            if not entry.strip():
                continue
            mrk = Path(entry.strip())
            if not mrk.is_absolute():
                mrk = scp.parent / mrk
            seq = mrk.with_suffix(".seq")
            lines = [line for line in mrk.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines or int(lines[0]) != len(lines) - 1:
                raise ValueError(f"MRK record count mismatch: {mrk}")
            size = seq.stat().st_size
            for line in lines[1:]:
                utt, offset, count = line.split()
                offset, count = int(offset), int(count)
                if utt in ids:
                    raise ValueError(f"Duplicate utterance ID: {utt}")
                if utt not in labels:
                    raise ValueError(f"Missing transcription for {utt}")
                if offset < 0 or offset % 2 or count < 800 or count % 2 or offset + count > size:
                    raise ValueError(f"Invalid PCM16 byte range for {utt}: {offset}, {count}")
                ids.add(utt)
                self.records.append((utt, str(seq.resolve()), offset, count, labels[utt]))
        if not self.records:
            raise ValueError(f"Dataset has no records: {scp}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import numpy as np

        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        _, seq, offset, count, text = self.records[index]
        if seq not in self._files:
            self._files[seq] = open(seq, "rb")
        stream = self._files[seq]
        stream.seek(offset)
        payload = stream.read(count)
        if len(payload) != count:
            raise IOError(f"Truncated SEQ file: {seq}")
        audio = np.frombuffer(payload, dtype="<i2").copy()
        return audio, text, math.ceil(len(audio) / 16000)

    def close(self):
        for stream in self._files.values():
            stream.close()
        self._files = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def __del__(self):
        if hasattr(self, "_files"):
            self.close()


def collate_continuous(items):
    return [item[0] for item in items], [item[1] for item in items]


def collate_codec(items):
    import numpy as np
    import torch
    from torch.nn.utils.rnn import pad_sequence

    audio = [torch.from_numpy(item[0].astype(np.float32)) for item in items]
    lengths = torch.tensor([len(item) for item in audio], dtype=torch.long)
    return pad_sequence(audio, batch_first=True), [item[1] for item in items], lengths
