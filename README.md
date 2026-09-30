# AI4Bharat STT Server

[![CI](https://github.com/SASHI117/ai4bharat_stt/actions/workflows/ci.yml/badge.svg)](https://github.com/SASHI117/ai4bharat_stt/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10--3.12-3776AB)
![Model](https://img.shields.io/badge/model-IndicConformer--600M-orange)

A self-hosted speech-to-text API for the **22 scheduled Indian languages**,
serving AI4Bharat's open [IndicConformer-600M multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
model behind FastAPI. I deployed it on an Azure VM so an open, self-hostable
model could be compared against commercial APIs in
[stt-benchmark-backend](https://github.com/SASHI117/stt-benchmark-backend).
A command-line client for batch transcription lives in
[users_ai4bharat_stt](https://github.com/SASHI117/users_ai4bharat_stt).

```mermaid
flowchart LR
    C[client / benchmark] -- "POST /stt<br/>Bearer key, X-Language" --> S[FastAPI]
    S -- tempfile --> T["transcriber.py<br/>ffmpeg CLI → mono 16 kHz float32"]
    T --> M["IndicConformer (ONNX)<br/>shared encoder → RNNT or CTC decoding<br/>restricted to the chosen language"]
    M --> S
    S -- "text, latency, RTF" --> C
```

## How it works

- **Model.** One Conformer encoder is shared by all 22 languages. RNNT
  decoding uses a separate joint network per language
  (`joint_post_net_<lang>.onnx`), and CTC decoding masks a shared decoder's
  output to that language's vocabulary. It is distributed as ONNX graphs and
  loaded through `trust_remote_code`. The language is therefore an *input*,
  not something the model detects: audio sent with the wrong `X-Language`
  is decoded into the wrong language's vocabulary.
- **Decoding.** `rnnt` is the default. `ctc` is a single argmax pass over
  the encoder output, cheaper than RNNT's step-by-step decoding loop.
  Switch with `STT_DECODE`.
- **Loading.** The ~2.5 GB checkpoint is loaded once, lazily and behind a
  lock. The server preloads it at startup (`STT_PRELOAD=1`) so the first
  request doesn't pay the load time.
- **Serving.** Inference runs in FastAPI's threadpool so a long clip doesn't
  block health checks or other requests. Each response carries latency and
  **real-time factor** (processing time ÷ audio duration).

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/stt` | multipart `file`. Headers: `Authorization: Bearer <STT_API_KEY>` and optional `X-Language` (default `te`) |
| `GET` | `/health` | model id and decoder |
| `GET` | `/languages` | `as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur` |

```bash
curl -X POST http://localhost:8000/stt \
  -H "Authorization: Bearer $STT_API_KEY" -H "X-Language: hi" \
  -F file=@sample.wav
```

Response (values are illustrative):

```json
{"filename": "sample.wav", "text": "…", "language": "hi", "decoding": "rnnt",
 "audio_seconds": 6.4, "latency_ms": 2210.5, "rtf": 0.345}
```

Errors: `401` bad key, `400` unsupported language or empty file, `413` file
over `STT_MAX_UPLOAD_MB`, `422` audio that ffmpeg can't decode.

## Setup

The model is **gated**. Request access on its Hugging Face page, then
authenticate with `huggingface-cli login` or `HF_TOKEN`.

```bash
sudo apt install ffmpeg                  # Windows: install ffmpeg and add it to PATH
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
cp .env.example .env                     # set STT_API_KEY
export $(grep -v '^#' .env | xargs)
uvicorn server:app --host 0.0.0.0 --port 8000
```

Transcribe one file without the server:

```bash
python transcriber.py sample.wav --lang te
```

Docker (CPU):

```bash
docker build -t ai4bharat-stt .
docker run --env-file .env -v hf-models:/models -p 8000:8000 ai4bharat-stt
```

For a VM deployment, put the service behind a reverse proxy with TLS
(Nginx + Let's Encrypt) and keep it off a public port. The bearer key only
works if it can't be sniffed.

### Dependency notes

- `onnxruntime` and `huggingface_hub` are imported by the model's remote code,
  so they must be installed even though this repo never imports them directly.
- Audio is decoded by calling the `ffmpeg` CLI, so any FFmpeg version on `PATH` works
  and there is no torchaudio dependency.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests replace the model with a fake. They cover authentication,
language defaults and validation, upload limits, temp-file cleanup, and a
client filename like `../../etc/passwd.mp3` never becoming a path. CI runs
them without installing torch.

## Verified end to end

On my laptop CPU, using the real checkpoint, I ran the server and called it with the
[client](https://github.com/SASHI117/users_ai4bharat_stt). The inputs were Google-TTS
sentences, so the correct transcript was known:

| Input | Output (RNNT and CTC) |
|---|---|
| రైతులు ఈ సంవత్సరం వరి పంటను ఎక్కువగా సాగు చేశారు (Telugu, 4.8 s) | identical |
| किसान भाई इस साल गेहूं की फसल अच्छी हुई है (Hindi, 3.8 s) | identical words (with standard nukta/chandrabindu spelling) |

- Startup, meaning model load plus warm-up, took about 22 s.
- Steady-state RTF was about 0.5–0.9 with RNNT on these short clips. On a 60 s clip, RTF was
  0.86 with RNNT and 0.51 with CTC.
- A startup warm-up pre-initializes ONNX Runtime, so the **first request is ~4× faster**
  (3.0 s instead of 11.7 s) and runs at steady-state speed.
- Wrong language → `400`, wrong key → `401`.

## Roadmap

- Batched and GPU inference (`onnxruntime-gpu`) for higher throughput.
- Automatic language identification in front of the model.
- Word-level timestamps from the CTC decoder, exposed through the API.
