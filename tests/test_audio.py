import math
import shutil
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transcriber  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def write_tone(path, sr=44100, seconds=0.5, channels=2):
    n = int(sr * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sr))) * channels for i in range(n)
        )
        w.writeframes(frames)


def test_decodes_to_mono_16k_float(tmp_path):
    src = tmp_path / "tone.wav"
    write_tone(src)                      # 44.1 kHz stereo in
    x = transcriber.load_audio(str(src))
    assert x.shape[0] == 1               # mono
    assert abs(x.shape[1] - 8000) <= 16  # 0.5 s at 16 kHz
    assert x.dtype.name == "float32"
    assert 0.1 < abs(x).max() <= 1.0


def test_undecodable_input_raises_value_error(tmp_path):
    bad = tmp_path / "not_audio.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(ValueError):
        transcriber.load_audio(str(bad))
