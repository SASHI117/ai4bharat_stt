import os
import uuid
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from transcribe_new import transcribe_audio

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("STT_API_KEY")

if API_KEY is None:
    raise RuntimeError("STT_API_KEY environment variable not set")

app = FastAPI(
    title="AI4Bharat STT API",
    description="Speech-to-Text API using AI4Bharat IndicConformer",
    version="1.0"
)

# =========================
# STT ENDPOINT
# =========================
@app.post("/stt")
async def stt(
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    # --- Auth check ---
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

    # --- Save uploaded audio temporarily ---
    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # ✅ transcribe_audio RETURNS A DICT
        result = transcribe_audio(temp_path)

        return result

    finally:
        # --- Cleanup ---
        if os.path.exists(temp_path):
            os.remove(temp_path)
