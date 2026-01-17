import time
import torch
import torchaudio
import os
from contextlib import redirect_stdout, redirect_stderr
from transformers import AutoModel

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

    # Always use ffmpeg backend (VERY IMPORTANT)
    wav, sr = torchaudio.load(audio_path, backend="ffmpeg")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

    # 🔕 suppress model prints/logs
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            with torch.no_grad():
                result = model(wav, LANG, DECODE_TYPE)

    # Normalize output
    if isinstance(result, (list, tuple)):
        text = result[0]
    else:
        text = str(result)

    latency_ms = round((time.time() - start_time) * 1000, 2)
    return text.strip(), latency_ms
