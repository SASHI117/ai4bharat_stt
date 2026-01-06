import time
import torch
import torchaudio
from transformers import AutoModel

# =========================
# CONFIG
# =========================
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
LANG = "te"            # Telugu
DECODE_TYPE = "rnnt"   # rnnt or ctc
TARGET_SR = 16000

# =========================
# LOAD MODEL (ONCE)
# =========================
print("🚀 Loading AI4Bharat model...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
model.eval()
print("✅ Model loaded successfully")

# =========================
# TRANSCRIPTION FUNCTION
# =========================
def transcribe_audio(audio_path: str):
    """
    Transcribe a single audio file.
    This function is API-safe (no input(), no prints).
    """

    start_time = time.time()

    # Load audio
    wav, sr = torchaudio.load(audio_path)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

    # Run inference
    with torch.no_grad():
        text = model(wav, LANG, DECODE_TYPE)

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return text, latency_ms
