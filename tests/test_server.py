import io
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["STT_API_KEY"] = "test-key"
os.environ["STT_PRELOAD"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
import transcriber  # noqa: E402

AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture
def client(monkeypatch):
    calls = []

    def fake_transcribe(path, lang):
        calls.append((path, lang))
        assert os.path.exists(path)
        return {"filename": os.path.basename(path), "text": "నమస్కారం", "language": lang,
                "decoding": "rnnt", "audio_seconds": 2.0, "latency_ms": 500.0, "rtf": 0.25}

    monkeypatch.setattr(transcriber, "transcribe_audio", fake_transcribe)
    with TestClient(server.app) as c:
        c.calls = calls
        yield c


def upload(client, headers=None, name="clip.wav", body=b"RIFF0000WAVE"):
    return client.post("/stt", files={"file": (name, io.BytesIO(body), "audio/wav")},
                       headers=headers or {})


def test_rejects_missing_or_wrong_key(client):
    assert upload(client).status_code == 401
    assert upload(client, {"Authorization": "Bearer nope"}).status_code == 401


def test_defaults_to_telugu(client):
    r = upload(client, AUTH)
    assert r.status_code == 200
    assert r.json()["language"] == "te"
    assert r.json()["filename"] == "clip.wav"


def test_language_header_is_normalised(client):
    assert upload(client, {**AUTH, "X-Language": " HI "}).json()["language"] == "hi"


def test_unsupported_language_is_400(client):
    assert upload(client, {**AUTH, "X-Language": "fr"}).status_code == 400


def test_client_filename_never_becomes_a_path(client):
    upload(client, AUTH, name="../../etc/passwd.mp3")
    path, _ = client.calls[-1]
    assert ".." not in Path(path).name and path.endswith(".mp3")
    assert not os.path.exists(path), "temp file must be cleaned up"


def test_empty_upload_is_400(client):
    assert upload(client, AUTH, body=b"").status_code == 400


def test_health_and_languages(client):
    assert client.get("/health").json()["status"] == "ok"
    langs = client.get("/languages").json()
    assert len(langs) == 22 and "te" in langs and "sat" in langs


def test_transcriber_rejects_unknown_language_before_loading_model():
    with pytest.raises(ValueError):
        transcriber.transcribe_audio("missing.wav", "xx")
    assert transcriber._model is None
