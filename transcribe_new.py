import time
import os
import traceback

import soundfile as sf
import torchaudio
import numpy as np

from fastapi import HTTPException

# ============================
# Force stable audio backend
# ============================
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass


def load_audio_safe(audio_path: str):
    """
    Robust audio loader for cloud environments.
    Tries soundfile first, then torchaudio fallback.
    Returns: waveform (numpy), sample_rate
    """
    # ---- Try soundfile (best for cloud) ----
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # convert to mono
        return audio, sr
    except Exception as sf_err:
        sf_error = str(sf_err)

    # ---- Fallback to torchaudio ----
    try:
        wav, sr = torchaudio.load(audio_path)
        wav = wav.mean(dim=0).numpy()
        return wav, sr
    except Exception as ta_err:
        raise RuntimeError(
            f"Audio decode failed.\n"
            f"soundfile error: {sf_error}\n"
            f"torchaudio error: {ta_err}"
        )


def transcribe_audio(audio_path: str):
    """
    Main transcription function.
    Returns:
    {
        "text": "...",
        "latency_ms": 1234.56
    }
    """
    start_time = time.time()

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail="Audio file not found")

    try:
        # ============================
        # Load audio safely
        # ============================
        audio, sr = load_audio_safe(audio_path)

        # ============================
        # TODO: YOUR ASR MODEL LOGIC
        # ============================
        # Replace this block with AI4Bharat / Whisper / any model
        #
        # Example placeholder:
        #
        # text = model.transcribe(audio, sr)
        #
        # --------------------------------
        text = "DUMMY_TRANSCRIPTION_REPLACE_ME"
        # --------------------------------

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "text": text,
            "latency_ms": latency_ms
        }

    except Exception as e:
        # ALWAYS return JSON-safe error
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"STT processing failed: {str(e)}"
        )
