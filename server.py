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
    authorization: str = Header(None)
):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        result = transcribe_audio(temp_path)

        # ✅ LOG output for debugging in Uvicorn
        print("🧠 Transcription result:")
        print(result)

        return {
            "filename": result["filename"],
            "text": result["text"],
            "latency_ms": result["latency_ms"]
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
