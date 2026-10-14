FROM python:3.12-slim

# ffmpeg: video transcription; libgl/libglib: OpenCV (used by the dental model)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
# fetch the dental model at build time if it wasn't copied in, so the first X-ray isn't slow
RUN python -c "from dental import get_model; get_model()"

RUN useradd --create-home app && chown -R app /app
USER app

ENV PORT=8000 COOKIE_SECURE=1 YOLO_CONFIG_DIR=/tmp
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
# one worker: sessions, rate limits and the model live in process memory (see ponytail notes in app.py)
CMD gunicorn app:app --workers 1 --threads 8 --timeout 300 --bind 0.0.0.0:$PORT
