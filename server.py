import hmac
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

import transcriber

logger = logging.getLogger("stt")
logging.basicConfig(level=logging.INFO)

MAX_UPLOAD_BYTES = int(os.getenv("STT_MAX_UPLOAD_MB") or 50) * 1024 * 1024


def _api_key() -> str:
    key = os.getenv("STT_API_KEY")
    if not key:
        raise RuntimeError("STT_API_KEY environment variable not set")
    return key


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _api_key()  # fail fast on a misconfigured deployment
    if os.getenv("STT_PRELOAD", "1") == "1":
        # Load the checkpoint at startup, not on the first user request.
        await run_in_threadpool(transcriber.get_model)
    yield


app = FastAPI(
    title="AI4Bharat STT API",
    description="Speech-to-text for 22 Indian languages using AI4Bharat IndicConformer-600M",
    version="1.1",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model": transcriber.MODEL_ID, "decoding": transcriber.DECODE_TYPE}


@app.get("/languages")
def languages():
    return sorted(transcriber.SUPPORTED_LANGS)


@app.post("/stt")
async def stt(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    x_language: Optional[str] = Header(None),
):
    # Constant-time comparison so the key cannot be recovered by timing.
    expected = f"Bearer {_api_key()}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")

    lang = (x_language or transcriber.DEFAULT_LANG).strip().lower()
    if lang not in transcriber.SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large")

    # Never build a path from the client's filename; keep only its extension.
    suffix = Path(file.filename or "").suffix.lower()[:10]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name

    try:
        # Inference is CPU-bound; keep the event loop free for other requests.
        result = await run_in_threadpool(transcriber.transcribe_audio, path, lang)
    except Exception as e:
        logger.exception("transcription failed")
        raise HTTPException(status_code=422, detail=f"Could not transcribe audio: {e}") from e
    finally:
        os.remove(path)

    result["filename"] = file.filename
    logger.info("lang=%s audio=%.1fs latency=%.0fms rtf=%s",
                lang, result["audio_seconds"], result["latency_ms"], result["rtf"])
    return result
