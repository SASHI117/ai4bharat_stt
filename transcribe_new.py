import time
import torch
import torchaudio
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
LANG = "te"
DECODE_TYPE = "rnnt"

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
model.eval()
print("✅ Model loaded successfully")

def transcribe_audio(audio_path):
    start_time = time.time()

    wav, sr = torchaudio.load(audio_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    with torch.no_grad():
        text = model(wav, LANG, DECODE_TYPE)

    latency_ms = round((time.time() - start_time) * 1000, 2)
    return text, latency_ms
