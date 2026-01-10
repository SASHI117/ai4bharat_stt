import os
import time
import logging
import torch
import torchaudio
from transformers import AutoModel

# =========================
# SILENCE ALL LOGS
# =========================
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# =========================
# CONFIG
# =========================
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
LANG = "te"              # language code
DECODE_TYPE = "rnnt"     # or "ctc"
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

    wav, sr = torchaudio.load(audio_path)

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
