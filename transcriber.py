"""AI4Bharat IndicConformer-600M inference.

The model is loaded lazily and once per process: importing this module is
cheap, so the API and its tests can start without the 600M checkpoint.
"""
import os
import threading
import time
from contextlib import redirect_stderr, redirect_stdout

MODEL_ID = os.getenv("STT_MODEL_ID", "ai4bharat/indic-conformer-600m-multilingual")
DEFAULT_LANG = "te"
DECODE_TYPE = os.getenv("STT_DECODE", "rnnt")   # "rnnt" (more accurate) or "ctc" (faster)
TARGET_SR = 16000

# One joint network per language ships with the checkpoint
# (assets/joint_post_net_<code>.onnx): the 22 scheduled Indian languages.
SUPPORTED_LANGS = frozenset({
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
})

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import torch
                from transformers import AutoModel
                from transformers.utils import logging as hf_logging

                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                hf_logging.set_verbosity_error()
                torch.set_num_threads(int(os.getenv("STT_THREADS") or os.cpu_count() or 1))

                # The remote code prints a FRAME_DURATION_MS notice while loading.
                with open(os.devnull, "w") as null, redirect_stdout(null), redirect_stderr(null):
                    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
                model.eval()
                _model = model
    return _model


def load_audio(audio_path: str):
    """Decode any ffmpeg-readable file to a mono 16 kHz float tensor [1, T]."""
    import torchaudio

    wav, sr = torchaudio.load(audio_path, backend="ffmpeg")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav


def transcribe_audio(audio_path: str, lang: str = DEFAULT_LANG) -> dict:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language '{lang}'. Supported: {sorted(SUPPORTED_LANGS)}")

    import torch

    start = time.perf_counter()
    wav = load_audio(audio_path)
    audio_s = wav.shape[-1] / TARGET_SR

    model = get_model()
    with torch.inference_mode():
        # forward() returns the decoded string directly.
        text = model(wav, lang, DECODE_TYPE)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "filename": os.path.basename(audio_path),
        "text": str(text).strip(),
        "language": lang,
        "decoding": DECODE_TYPE,
        "audio_seconds": round(audio_s, 2),
        "latency_ms": latency_ms,
        # < 1.0 means faster than real time
        "rtf": round((latency_ms / 1000) / audio_s, 3) if audio_s else None,
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Transcribe one file locally (no server).")
    ap.add_argument("audio")
    ap.add_argument("--lang", default=DEFAULT_LANG)
    args = ap.parse_args()
    print(json.dumps(transcribe_audio(args.audio, args.lang), ensure_ascii=False, indent=2))
