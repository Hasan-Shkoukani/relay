from dotenv import load_dotenv
load_dotenv()  # Load env vars BEFORE importing other modules

from collections import OrderedDict, defaultdict, deque
import logging
import os
import secrets
import threading
import time

from flask import Flask, jsonify, render_template, request, session
from werkzeug.middleware.proxy_fix import ProxyFix

from main import answer, handleDocument, handleImage, handleVideo, knowledge_source

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)  # real client IP/scheme behind a load balancer
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)
if not os.getenv("SECRET_KEY"):
    app.logger.warning("SECRET_KEY not set: sessions reset on every restart.")

MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
app.config.update(
    MAX_CONTENT_LENGTH=MAX_DOCUMENT_BYTES + 1024 * 1024,  # reject oversized uploads before reading them
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "0") == "1",
)

# ponytail: single-process in-memory store, one loaded source per (session, tab), capped so it can't
# grow forever. Swap for redis/a db if this ever runs behind multiple workers.
MAX_SOURCES = 150
SOURCES = OrderedDict()
MODES = ("files", "image", "video")

# ponytail: in-memory per-IP sliding window, same single-process limit as SOURCES
RATE_LIMITS = {"load": int(os.getenv("LOADS_PER_MINUTE", 10)), "ask": int(os.getenv("ASKS_PER_MINUTE", 30))}
_hits = defaultdict(deque)
_hits_lock = threading.Lock()


def _rate_limited(kind):
    now, key = time.monotonic(), (kind, request.remote_addr)
    with _hits_lock:
        hits = _hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= RATE_LIMITS[kind]:
            return True
        hits.append(now)
        return False


def _remember_source(key, source):
    SOURCES[key] = source
    SOURCES.move_to_end(key)
    while len(SOURCES) > MAX_SOURCES:
        SOURCES.popitem(last=False)


def _session_id():
    session_id = session.get("sid")
    if not session_id:
        session_id = secrets.token_hex(16)
        session["sid"] = session_id
    return session_id


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="That file is too large."), 413


@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")


@app.route("/app", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify(ok=True)


@app.route("/api/load", methods=["POST"])
def load_source():
    if _rate_limited("load"):
        return jsonify(error="Too many uploads. Wait a minute and try again."), 429
    mode = request.form.get("mode", "").strip()

    try:
        if mode == "files":
            doc_file = request.files.get("document_file")
            if not doc_file or not doc_file.filename:
                return jsonify(error="Choose a document to upload."), 400
            data = doc_file.read()
            if len(data) > MAX_DOCUMENT_BYTES:
                return jsonify(error="That file is over the 50 MB limit."), 400
            source = handleDocument(data, doc_file.filename)
            source["name"] = doc_file.filename

        elif mode == "image":
            image_file = request.files.get("image_file")
            if not image_file or not image_file.filename:
                return jsonify(error="Choose or capture an image first."), 400
            data = image_file.read()
            if len(data) > MAX_IMAGE_BYTES:
                return jsonify(error="That image is over the 20 MB limit."), 400
            source = handleImage(data)
            source["name"] = image_file.filename

        elif mode == "video":
            video_url = request.form.get("video_url", "").strip()
            if not video_url:
                return jsonify(error="Paste a YouTube link first."), 400
            source = handleVideo(video_url)
            source["name"] = source.pop("title")

        else:
            return jsonify(error="Unknown tab."), 400

    except RuntimeError as exc:  # raised with a message meant for the user
        app.logger.warning("load failed (mode=%s): %s", mode, exc)
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("load failed (mode=%s)", mode)
        return jsonify(error="Couldn't read that. Try again, or try a different file."), 500

    _remember_source((_session_id(), mode), source)
    return jsonify(ok=True, name=source["name"], kind=source["kind"],
                   annotated=source.get("annotated"), findings=source.get("findings"))


@app.route("/api/ask", methods=["POST"])
def ask():
    if _rate_limited("ask"):
        return jsonify(error="Too many questions. Wait a minute and try again."), 429
    payload = request.get_json(silent=True) or {}
    mode = (payload.get("mode") or "").strip()
    question = (payload.get("question") or "").strip()[:2000]

    if not question:
        return jsonify(error="Type a question first."), 400

    if mode == "text":
        source = None
    elif mode in MODES:
        source = SOURCES.get((session.get("sid"), mode))
        if not source:
            return jsonify(error="That file is no longer loaded. Load it again to keep asking.", reload=True), 409
    else:
        return jsonify(error="Unknown tab."), 400

    try:
        result = answer(source or knowledge_source(), question)
    except Exception:
        app.logger.exception("ask failed (mode=%s)", mode)
        return jsonify(error="Couldn't answer that right now. Try again in a moment."), 500

    return jsonify(result)


@app.route("/api/clear", methods=["POST"])
def clear_source():
    mode = (request.get_json(silent=True) or {}).get("mode")
    SOURCES.pop((session.get("sid"), mode), None)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", 5000)), threaded=True)
