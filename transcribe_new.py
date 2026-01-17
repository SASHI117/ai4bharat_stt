import time
import io
import torch
import torchaudio
from transformers import AutoModel
from contextlib import redirect_stdout, redirect_stderr

# =========================
# CONFIG
# =========================
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
LANG = "te"                 # Telugu (can be parameterized later)
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

    # Silence noisy logs
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        # Explicit backend avoids Linux backend issues
        wav, sr = torchaudio.load(audio_path, backend="ffmpeg")

        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != TARGET_SR:
            wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

        with torch.no_grad():
            text = model(wav, LANG, DECODE_TYPE)

    latency_ms = round((time.time() - start_time) * 1000, 2)
    return text, latency_ms
