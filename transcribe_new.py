import os
import time
import requests

API_URL = "http://74.225.216.132:8000/stt"
API_KEY = "ai4bharat-secret-key-6262"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

SUPPORTED_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def send_audio(audio_path):
    with open(audio_path, "rb") as f:
        files = {"file": f}

        start = time.time()
        response = requests.post(
            API_URL,
            headers=HEADERS,
            files=files,
            timeout=120
        )
        end = time.time()

    latency_ms = round((end - start) * 1000, 2)

    if response.status_code != 200:
        print(f"❌ Error for {audio_path}")
        print(response.text)
        return

    result = response.json()

    print("\n📄 File:", os.path.basename(audio_path))
    print("📝 Transcription:")
    print(result.get("text") or result.get("transcription"))
    print("⏱ Latency:", latency_ms, "ms")


# =========================
# MAIN CLI
# =========================

print("\nChoose input type:")
print("1️⃣  Single audio file")
print("2️⃣  Folder with multiple audio files")

choice = input("\nEnter choice (1 or 2): ").strip()

# -------- SINGLE FILE --------
if choice == "1":
    file_path = input("\nEnter audio file path: ").strip('"')

    if not os.path.isfile(file_path):
        print("❌ File not found!")
    else:
        send_audio(file_path)

# -------- MULTIPLE FILES --------
elif choice == "2":
    folder_path = input("\nEnter folder path: ").strip('"')

    if not os.path.isdir(folder_path):
        print("❌ Folder not found!")
    else:
        audio_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(SUPPORTED_EXT)
        ]

        if not audio_files:
            print("❌ No audio files found.")
        else:
            print(f"\n📂 Found {len(audio_files)} audio files")

            for audio in audio_files:
                send_audio(audio)

else:
    print("❌ Invalid choice")
