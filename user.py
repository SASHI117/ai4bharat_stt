import os
import requests

# ---------------- CONFIG ----------------
API_URL = "YOUR_API_URL"
API_KEY = "YOUR_API_KEY"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

SUPPORTED_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


# ---------------- FUNCTION ----------------
def send_audio_to_vm(audio_path):
    try:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, headers=HEADERS, files=files, timeout=120)

        if response.status_code == 200:
            data = response.json()
            print("\n✅ Transcription received")
            print(f"📄 File      : {data.get('filename')}")
            print(f"📝 Text      : {data.get('text')}")
            print(f"⏱ Latency ms: {data.get('latency_ms')}")
        else:
            print(f"\n❌ Error {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"\n❌ Failed for {audio_path}")
        print(e)


# ---------------- MENU ----------------
print("\nChoose input type:")
print("1️⃣  Single audio file")
print("2️⃣  Folder with multiple audio files")

choice = input("\nEnter choice (1 or 2): ").strip()

# -------- SINGLE FILE --------
if choice == "1":
    path = input("\nEnter audio file path: ").strip()

    if os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXT):
        send_audio_to_vm(path)
    else:
        print("❌ File not found or unsupported format")

# -------- FOLDER --------
elif choice == "2":
    folder = input("\nEnter folder path: ").strip()

    if os.path.isdir(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED_EXT)]

        if not files:
            print("❌ No supported audio files found")
        else:
            for f in files:
                full_path = os.path.join(folder, f)
                print(f"\n📤 Sending: {f}")
                send_audio_to_vm(full_path)
    else:
        print("❌ Folder not found")

else:
    print("❌ Invalid choice")
