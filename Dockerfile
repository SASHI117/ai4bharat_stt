FROM python:3.11-slim

# ffmpeg CLI decodes every input format (see transcriber.load_audio)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY server.py transcriber.py ./

# Mount a volume here to keep the ~2.5 GB checkpoint across restarts.
ENV HF_HOME=/models
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
