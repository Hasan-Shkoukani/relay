"""Turn documents, videos, images and the knowledge base into location-tagged chunks, and answer
questions from them with citations.

Every answer says where it came from. Fallback order: the source itself (for X-rays: the dental
model's findings) -> GPT vision looking at the image (images only), web search (knowledge base) -> general GPT knowledge.
"""
import base64
import glob
import json
import os
import re
import tempfile
import uuid
from io import BytesIO

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5.4-mini")
EMBED_MODEL = "text-embedding-3-small"
NOT_FOUND = "NOT_FOUND"


# ---------- chunking with locations ----------

def split_text(text, chunk_size=1500, overlap=200):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]


def pack(units, label, href=lambda key: None, size=1500):
    """Merge small (key, text) units (pages, paragraphs, timed captions) into ~size-char chunks.
    Each chunk remembers where it came from: label(first_key, last_key)."""
    chunks, cur, n = [], [], 0

    def flush():
        chunks.append({"loc": label(cur[0][0], cur[-1][0]), "href": href(cur[0][0]), "text": "\n".join(t for _, t in cur)})

    for key, text in units:
        text = text.strip()
        for piece in (split_text(text, size) if len(text) > size else [text] if text else []):
            if cur and n + len(piece) > size:
                flush()
                cur, n = [], 0
            cur.append((key, piece))
            n += len(piece)
    if cur:
        flush()
    return chunks


def span(singular, plural):
    return lambda a, b: f"{singular} {a}" if a == b else f"{plural} {a}-{b}"


def timestamp(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


# ---------- documents ----------

def handleDocument(file_bytes, filename):
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        import pypdfium2 as pdfium
        units = [(i, page.get_textpage().get_text_range()) for i, page in enumerate(pdfium.PdfDocument(file_bytes), 1)]
        chunks = pack(units, span("Page", "Pages"))
    elif ext == "docx":
        from docx import Document
        try:
            doc = Document(BytesIO(file_bytes))
        except Exception:
            raise RuntimeError("That Word file couldn't be opened. Check it isn't password-protected or damaged.")
        units = [(i, p.text) for i, p in enumerate((p for p in doc.paragraphs if p.text.strip()), 1)]
        chunks = pack(units, span("Paragraph", "Paragraphs"))
    elif ext in ("txt", "text", "md"):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")
        chunks = pack(list(enumerate(text.splitlines(), 1)), span("Line", "Lines"))
    elif ext == "doc":
        raise RuntimeError("Old .doc files aren't supported. Save it as .docx or PDF and upload that.")
    else:
        raise RuntimeError(f"Relay can't read .{ext} files. Upload a PDF, DOCX or TXT file.")

    if not chunks:
        raise RuntimeError("No text found in that file. If it's a scanned PDF, upload the pages as images instead.")
    return {"kind": "document", "chunks": embed_chunks(chunks)}


# ---------- video ----------

def youtube_id(url):
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path.startswith(("/shorts/", "/live/", "/embed/")):
            return parsed.path.split("/")[2]
        return parse_qs(parsed.query).get("v", [""])[0]
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    return ""


def _transcript(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except Exception:
        fetched = next(iter(api.list(video_id))).fetch()  # any language; the answer model can read it
    return [(s.start, s.text) for s in fetched]


def _parse_vtt(path):
    units, start, last = [], None, None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = re.sub(r"<[^>]+>", "", line).strip()
            if "-->" in line:
                h, m, s = line.split("-->")[0].strip().replace(",", ".").rjust(12, "0").split(":")[-3:]
                start = int(h) * 3600 + int(m) * 60 + float(s)
            elif line and start is not None and not line.isdigit() and line != last:
                units.append((start, line))
                last = line  # auto-captions repeat each line as it scrolls
    return units


def _subtitles(url):
    import yt_dlp

    base = os.path.join(tempfile.gettempdir(), f"subtitle_{uuid.uuid4().hex}")
    opts = {"skip_download": True, "writesubtitles": True, "writeautomaticsub": True, "subtitleslangs": ["en"],
            "subtitlesformat": "vtt", "outtmpl": f"{base}.%(ext)s", "quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        matches = glob.glob(f"{base}*.vtt")
        if not matches:
            raise RuntimeError("No subtitles were downloaded.")
        return _parse_vtt(matches[0])
    finally:
        for path in glob.glob(f"{base}*"):
            os.remove(path)


def _transcribe(url):
    import yt_dlp

    base = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}")
    opts = {"format": "bestaudio/best", "outtmpl": f"{base}.%(ext)s", "quiet": True, "no_warnings": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "64"}]}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        with open(f"{base}.mp3", "rb") as audio:
            result = client.audio.transcriptions.create(file=audio, model="whisper-1", response_format="verbose_json")
        return [(seg.start, seg.text) for seg in result.segments]
    except Exception as exc:
        raise RuntimeError("Couldn't get a transcript or the audio for that video. Check the link is public.") from exc
    finally:
        for path in glob.glob(f"{base}.*"):  # the mp3 and any unconverted source (.webm)
            os.remove(path)


def handleVideo(url):
    import yt_dlp

    try:
        info = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}).extract_info(url, download=False)
    except Exception:
        raise RuntimeError("That link didn't open as a video. Paste a public YouTube URL.")
    if (info.get("duration") or 0) > 3600:
        raise RuntimeError("That video is over an hour long. Use one under an hour.")

    video_id = youtube_id(url) or info.get("id", "")
    # cheapest first: YouTube's transcript, then subtitle files, then paid Whisper transcription
    attempts = ([lambda: _transcript(video_id)] if video_id else []) + [lambda: _subtitles(url), lambda: _transcribe(url)]
    units = None
    for attempt in attempts:
        try:
            units = attempt()
        except Exception:
            continue
        if units:
            break
    if not units:
        raise RuntimeError("Couldn't get a transcript or the audio for that video. Check the link is public.")

    watch = f"https://www.youtube.com/watch?v={video_id}" if video_id else url
    chunks = pack(units, lambda a, b: f"{timestamp(a)}-{timestamp(b)}", href=lambda a: f"{watch}&t={int(a)}s", size=1000)
    return {"kind": "video", "chunks": embed_chunks(chunks), "title": info.get("title") or url}


# ---------- images ----------

def _jpeg(img, max_side=1280):
    import cv2

    scale = min(1.0, max_side / max(img.shape[:2]))
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def _vision(jpeg_bytes, prompt, max_tokens=500):
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()
    response = client.chat.completions.create(
        model=VISION_MODEL,
        max_completion_tokens=max_tokens,
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}],
    )
    return response.choices[0].message.content.strip()


DENTAL_FLAG = "DENTAL_XRAY"
MODEL_LOC = "Dental detection model (dental-radiography-yolo26)"
VISION_LOC = f"GPT vision ({VISION_MODEL}), describing your image"


def handleImage(image_bytes):
    """The dental model reads the image first. GPT vision only describes it when the model can't."""
    import dental

    img = dental.decode(image_bytes)
    jpeg = _jpeg(img)
    dets = dental.detect(img)
    description = None

    if not (dets and dental.is_grayscale(img)):
        # not obviously an X-ray: let GPT vision look. It catches X-rays the model found nothing on, and photos of
        # X-rays taken off a screen, and hands them back to the model's findings.
        try:
            description = _vision(jpeg, "Extract any text in this image and describe what you see in detail. "
                                        f"If it is a dental X-ray or dental radiograph, start your reply with {DENTAL_FLAG}.")
        except Exception as exc:
            raise RuntimeError("Couldn't analyse that image right now. Try again in a moment.") from exc
        if DENTAL_FLAG not in description:
            chunks = [{"loc": VISION_LOC, "href": None, "text": description}]
            return {"kind": "photo", "chunks": chunks, "image": jpeg, "annotated": None, "findings": None}
        description = description.replace(DENTAL_FLAG, "").strip()

    # X-ray: the model's findings come first (even when it found nothing); GPT's description is only backup
    text, findings = dental.summarize(dets)
    chunks = [{"loc": MODEL_LOC, "href": None, "text": text}]
    if description:
        chunks.append({"loc": VISION_LOC, "href": None, "text": description})
    return {"kind": "xray", "chunks": chunks, "image": jpeg, "annotated": dental.draw(img, dets), "findings": findings}


# ---------- knowledge base ----------

_kb = {"mtime": None, "source": None}


def knowledge_source(path="knowledge.json"):
    """knowledge.json as a source, embedded once and re-embedded only when the file changes.
    Entries: {"id", "text", optional "source": URL shown with the citation}."""
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    if _kb["mtime"] != mtime or _kb["source"] is None:
        items = json.load(open(path, encoding="utf-8")) if mtime else []
        chunks = [{"loc": f"Knowledge base, entry {item.get('id', i)}", "href": item.get("source"),
                   "text": item.get("text") or item.get("content") or ""} for i, item in enumerate(items, 1)]
        _kb.update(mtime=mtime, source={"kind": "knowledge", "chunks": embed_chunks([c for c in chunks if c["text"]])})
    return _kb["source"]


# ---------- retrieval + answers ----------

def embed(texts):
    vectors = []
    for i in range(0, len(texts), 100):  # stay well under the per-request token limit
        vectors += [d.embedding for d in client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 100]).data]
    return vectors


def embed_chunks(chunks):
    """Embed once at load time; small sources skip embedding since every chunk fits in the prompt."""
    if len(chunks) > 4:
        for chunk, vector in zip(chunks, embed([c["text"] for c in chunks])):
            chunk["vec"] = np.asarray(vector, np.float32)  # ~6 KB each instead of ~50 KB as a float list
    return chunks


def retrieve(chunks, question, top_k=4):
    if len(chunks) <= top_k:
        return chunks
    q = np.asarray(embed([question])[0], np.float32)
    vecs = np.stack([c["vec"] for c in chunks])
    scores = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q) + 1e-9)
    return [chunks[i] for i in np.argsort(-scores)[:top_k]]


def llm(system, user, max_tokens=500):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        max_completion_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return response.choices[0].message.content.strip()


NOUN = {"document": "document", "video": "video", "photo": "image", "xray": "X-ray", "knowledge": "knowledge base"}

SOURCE_PROMPT = f"""You answer questions using only the numbered excerpts provided. Each excerpt says where it comes from.
Partial answers and reasonable inferences from the excerpts are fine. Talk to the user directly and keep it concise.
Never say "excerpt" or "context"; refer to the {{noun}} instead.
When excerpts come from the dental detection model, rely on them first; they are automated findings, not a diagnosis.
The model only reports fillings, implants, cavities, impacted teeth and the few other marks it lists. It does not count
teeth, measure anything or judge overall health, so for questions like that don't infer from its counts.
End with one line listing the excerpt numbers you used, exactly like: SOURCES: 1, 3
If the excerpts don't contain what's needed to actually answer (for example a count, a detail or a fact they never
mention), don't hedge with "I can't tell": reply with exactly {NOT_FOUND} so a better source can answer."""

VISION_PROMPT = f"""Look at this image and answer the user's question directly and concisely, from what is visible.
Count, describe and point to locations (upper/lower, the patient's left/right) when asked. If you're unsure, give your
best estimate and say how confident you are. Never reply only that you can't determine something: for questions that
need a dentist's judgement (a root canal, bone loss, whether a tooth is healthy), describe the visible signs that matter,
for example dark areas at root tips, decay reaching the nerve, existing root canal fillings, or the bone level around
the teeth, and then say a dentist should confirm. End dental answers with one short line that this isn't a diagnosis.
If the question has nothing to do with this image, reply with exactly: {NOT_FOUND}
"""

WEB_PROMPT = """Search the web and answer the question directly and concisely (under 150 words). Prefer official and
reputable sources."""

GENERAL_PROMPT = """You are a helpful, knowledgeable assistant. Answer the question directly and concisely (under 150 words)
from your general knowledge. For medical or dental questions, give useful general information and suggest seeing a
dentist or doctor for a diagnosis."""

# A source answer that hedges ("I can't tell", "it doesn't mention") means the source lacks the answer: fall through.
HEDGE = re.compile(
    r"\b(?:i|you|we)\s*(?:can['’]?t|cannot|am unable to|['’]m unable to|am not able to|['’]m not able to)\s+"
    r"(?:tell|determine|say|count|see|identify|assess|confirm|know)"
    r"|\b(?:does not|doesn['’]?t|do not|don['’]?t|did not|didn['’]?t)\s+(?:mention|say|show|include|provide|specify|contain|give|list|state)",
    re.IGNORECASE)
# Vision and the last step may hedge about certainty (that's honest), but a flat refusal still falls through.
REFUSAL = re.compile(r"\bi['’]?m sorry\b|\bi\s*(?:can['’]?t|cannot|am unable to|won['’]?t)\s+(?:help|assist|provide|analy[sz]e|do that)", re.IGNORECASE)


def answered(raw, strict=True):
    if not raw or NOT_FOUND in raw or REFUSAL.search(raw):
        return False
    return not (strict and HEDGE.search(raw))


def parse_cited(raw, n):
    """Split 'answer ... SOURCES: 1, 3' into the answer and the cited excerpt indices (0-based)."""
    match = re.search(r"\n?\s*SOURCES?:\s*([\d,\s]*)\s*$", raw, re.IGNORECASE)
    if not match:
        return raw.strip(), []
    cited = sorted({int(x) - 1 for x in re.findall(r"\d+", match.group(1)) if 0 < int(x) <= n})
    return raw[:match.start()].strip(), cited


def web_answer(question, context=""):
    """Answer from a live web search. `context` (the closest knowledge entries) resolves vague questions like
    "who is the current rector?" to the knowledge base's subject. -> (text, [{"label", "href"}]) or (None, [])."""
    instructions = WEB_PROMPT + (f"\nThe user is asking in the context of this knowledge base; read the question in "
                                 f"that light:\n{context}" if context else "")
    try:
        r = client.responses.create(model=CHAT_MODEL, tools=[{"type": "web_search"}], instructions=instructions, input=question)
    except Exception:
        return None, []
    sources = {}
    for item in r.output:
        for part in getattr(item, "content", None) or []:
            for a in getattr(part, "annotations", None) or []:
                if a.type == "url_citation":
                    url = re.sub(r"[?&]utm_source=openai$", "", a.url)
                    sources[url] = {"label": a.title or url, "href": url}
    # the model also inlines "([site](url))" citations; the page shows them as chips instead
    text = re.sub(r"\s*\(\[[^\]]*\]\([^)]*\)\)", "", r.output_text or "").strip()
    return text or None, list(sources.values())


def answer(source, question):
    """-> {"answer", "origin", "sources": [{"label", "href"}]}
    Order: the source itself (X-rays: the dental model) -> GPT vision (images) / web search (knowledge) -> general GPT."""
    kind = source["kind"]
    noun = NOUN[kind]
    top = retrieve(source["chunks"], question)
    if top:
        excerpts = "\n\n".join(f"[{i}] ({c['loc']})\n{c['text']}" for i, c in enumerate(top, 1))
        raw = llm(SOURCE_PROMPT.format(noun=noun), f"Question:\n{question}\n\nExcerpts:\n{excerpts}")
        if answered(raw):
            text, cited = parse_cited(raw, len(top))
            used = [top[i] for i in cited] or top[:1]
            origin = "model" if any(c["loc"] == MODEL_LOC for c in used) else kind
            sources = list({c["loc"]: {"label": c["loc"], "href": c["href"]} for c in used}.values())
            return {"answer": text, "origin": origin, "sources": sources}

    if source.get("image"):
        findings = next((c["text"] for c in source["chunks"] if c["loc"] == MODEL_LOC), "")
        prompt = VISION_PROMPT + (f"\nThe dental detection model already found:\n{findings}\n" if findings else "")
        raw = _vision(source["image"], f"{prompt}\nQuestion: {question}")
        if answered(raw, strict=False):
            sources = [{"label": f"GPT vision ({VISION_MODEL}), looking at your image", "href": None}]
            if findings:
                sources.append({"label": MODEL_LOC, "href": None})
            return {"answer": raw, "origin": "vision", "sources": sources}

    if kind == "knowledge":
        text, sources = web_answer(question, "\n".join(c["text"] for c in top[:2]))
        if text:
            return {"answer": text, "origin": "web", "sources": sources or [{"label": "Web search", "href": None}]}

    return {"answer": llm(GENERAL_PROMPT, question, max_tokens=600), "origin": "general",
            "sources": [{"label": f"General knowledge from GPT, not from your {noun}", "href": None}]}


if __name__ == "__main__":
    chunks = pack([(1, "a" * 900), (2, "b" * 900), (3, "c" * 100)], span("Page", "Pages"))
    assert [c["loc"] for c in chunks] == ["Page 1", "Pages 2-3"], [c["loc"] for c in chunks]
    long = pack([(1, "x" * 4000)], span("Page", "Pages"))
    assert len(long) == 3 and all(c["loc"] == "Page 1" for c in long)
    vids = pack([(5.2, "hi"), (65, "there")], lambda a, b: f"{timestamp(a)}-{timestamp(b)}", href=lambda a: f"v&t={int(a)}s")
    assert vids == [{"loc": "0:05-1:05", "href": "v&t=5s", "text": "hi\nthere"}]
    assert timestamp(3725) == "1:02:05"
    assert parse_cited("Five fillings.\nSOURCES: 1, 3, 9", 4) == ("Five fillings.", [0, 2])
    assert parse_cited("No marker here", 2) == ("No marker here", [])
    assert youtube_id("https://youtu.be/abc") == "abc" and youtube_id("https://www.youtube.com/shorts/xyz") == "xyz"
    assert youtube_id("https://www.youtube.com/watch?v=q1&t=3") == "q1"
    assert not answered("I can’t tell from this image alone whether you need a root canal.")
    assert not answered("I'm sorry, I can't help with that.") and not answered("The X-ray doesn't mention bone loss.")
    assert not answered("You can see the full set, but the X-ray does not give an exact tooth count.")
    assert answered("It's difficult to assess bone loss here, but bone levels look even.", strict=False)
    assert not answered("I'm sorry, but I can't analyze medical images.", strict=False)
    assert answered("You have 5 fillings, mostly in the upper jaw.\nSOURCES: 1")
    print("ok")
