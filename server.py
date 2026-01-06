import os
import uuid
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from transcribe_new import transcribe_audio

API_KEY = os.getenv("STT_API_KEY")

app = FastAPI()

@app.post("/stt")
async def stt(
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    if API_KEY is None:
        raise HTTPException(500, "STT_API_KEY not set")

    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "Invalid API key")

    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    text, latency_ms = transcribe_audio(temp_path)

    os.remove(temp_path)

    return {
        "filename": file.filename,
        "text": text,
        "latency_ms": latency_ms
    }
