import os
import time
import torch
import torchaudio
from transformers import AutoModel

# =========================
# CONFIG
# =========================
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
LANG = "te"   # Telugu
DECODE_TYPE = "rnnt"  # "rnnt" or "ctc"

# =========================
# LOAD MODEL (ONCE)
# =========================
print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
model.eval()
print("✅ Model loaded successfully")

# =========================
# TRANSCRIPTION FUNCTION
# =========================
def transcribe_audio(audio_path):
    print(f"\n🎧 Processing: {audio_path}")

    start_time = time.time()

    wav, sr = torchaudio.load(audio_path)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    with torch.no_grad():
        response = model(wav, LANG, DECODE_TYPE)

    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000, 2)

    print("\n--- TRANSCRIPTION ---")
    print(response)

    print("\n--- LATENCY ---")
    print(f"{latency_ms} ms")

    return response, latency_ms


# =========================
# MAIN PROGRAM
# =========================
print("\nChoose input type:")
print("1️⃣  Single audio file")
print("2️⃣  Folder with multiple audio files")

choice = input("\nEnter choice (1 or 2): ").strip()

# -------- SINGLE FILE --------
if choice == "1":
    file_path = input("\nEnter audio file path: ").strip()

    if not os.path.isfile(file_path):
        print("❌ File not found!")
    else:
        transcribe_audio(file_path)

# -------- MULTIPLE FILES --------
elif choice == "2":
    folder_path = input("\nEnter folder path containing audio files: ").strip()

    if not os.path.isdir(folder_path):
        print("❌ Folder not found!")
    else:
        audio_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg"))
        ]

        if not audio_files:
            print("❌ No audio files found in the folder.")
        else:
            print(f"\n📂 Found {len(audio_files)} audio files.\n")

            for file in audio_files:
                full_path = os.path.join(folder_path, file)
                transcribe_audio(full_path)

else:
    print("❌ Invalid choice. Please enter 1 or 2.")
