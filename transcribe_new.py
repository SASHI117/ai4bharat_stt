import time
import os
import torch
import torchaudio
from contextlib import redirect_stdout, redirect_stderr
from transformers import AutoModel
from transformers.utils import logging as hf_logging

# =========================
# HARD DISABLE ALL HF LOGS
# =========================
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
hf_logging.set_verbosity_error()

# =========================
# CONFIG
# =========================
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
LANG = "te"
DECODE_TYPE = "rnnt"
TARGET_SR = 16000

# =========================
# LOAD MODEL (ONCE)
# =========================
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
model.eval()

# =========================
# TRANSCRIPTION FUNCTION
# =========================
def transcribe_audio(audio_path: str):
    start_time = time.time()

    # ALWAYS use ffmpeg
    wav, sr = torchaudio.load(audio_path, backend="ffmpeg")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

    # FULL stdout/stderr suppression
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            with torch.no_grad():
                result = model(wav, LANG, DECODE_TYPE)

    # 🔴 FORCE CLEAN OUTPUT
    if isinstance(result, (list, tuple)):
        text = result[0]
    elif isinstance(result, dict):
        text = result.get("text", "")
    else:
        text = str(result)

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # FINAL SANITY CLEAN
    text = text.strip()
    text = text.replace("Please check FRAME_DURATION_MS.", "")
    text = text.replace("Fetching", "").strip()

    return text, latency_ms
