import os
import uuid
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from transcribe_new import transcribe_audio

API_KEY = os.getenv("STT_API_KEY")
if API_KEY is None:
    raise RuntimeError("STT_API_KEY environment variable not set")

app = FastAPI(
    title="AI4Bharat STT API",
    description="Speech-to-Text API using AI4Bharat IndicConformer",
    version="1.0"
)

@app.post("/stt")
async def stt(
    file: UploadFile = File(...),
    authorization: str = Header(None),
    x_language: str = Header(None)   # ✅ NEW (optional)
):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ✅ Backward-compatible default
    lang = x_language.lower() if x_language else "te"

    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # ✅ Pass language safely
        result = transcribe_audio(temp_path, lang)

        # ✅ LOG output for debugging in Uvicorn
        print("🧠 Transcription result:")
        print(result)

        return {
            "filename": result["filename"],
            "text": result.get("text") or result.get("transcription"),
            "latency_ms": result["latency_ms"]
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
