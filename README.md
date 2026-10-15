# Relay

**Talk to your dental X-rays, documents, videos and a knowledge base, by typing or out loud.**

Relay is a Flask web app. Upload a panoramic dental X-ray and a dental detection model marks
fillings, implants, cavities and impacted teeth; then ask questions about it in plain language.
The same conversation works for PDFs, Word files, YouTube videos and a built-in knowledge base.
Every answer shows exactly where it came from.

![Relay workspace: an X-ray with impacted teeth marked, and answers with their sources](docs/workspace.png)

---

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Try it: sample X-rays](#try-it-sample-x-rays)
- [Where answers come from](#where-answers-come-from)
- [The dental model](#the-dental-model)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [HTTP API](#http-api)
- [Testing](#testing)
- [Deployment](#deployment)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Credits and licenses](#credits-and-licenses)

---

## Features

| Tab | What you give it | What answers cite |
|---|---|---|
| **X-ray or photo** | JPG/PNG up to 20 MB, or a camera photo | the dental model's findings, or GPT vision looking at the image |
| **Document** | PDF, DOCX or TXT up to 50 MB | page (PDF), paragraph (DOCX) or line (TXT) |
| **Video** | a public YouTube link under an hour | timestamp range, linked to that moment in the video |
| **Knowledge** | nothing, it's built in (`knowledge.json`) | the knowledge base entry, or the web pages it searched |

- **Model first, GPT as fallback.** On images the dental model always runs first; GPT is only used when the model can't answer.
- **Fewer "I don't know"s.** Answers that dodge ("I can't tell", "it doesn't mention") are not shown; the question moves on to the next source instead.
- **Voice.** Press the microphone to ask out loud; press *Listen* on any answer to hear it.
- **Annotated X-rays.** Findings are drawn on the image in colour and listed with their position (upper/lower jaw, the patient's left/right/front).
- **Each tab keeps its own source**, so you can switch between an X-ray and a PDF without reloading either.

## Quick start

Requirements: **Python 3.12**, **ffmpeg** on your `PATH` (only for videos without captions) and an **OpenAI API key**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env               # then set OPENAI_API_KEY and SECRET_KEY
python app.py
```

Open <http://127.0.0.1:5000> for the landing page, or <http://127.0.0.1:5000/app> for the workspace.
The model weights (`models/dental-yolo26.pt`, 44 MB) are included and are downloaded automatically if missing.

## Try it: sample X-rays

The **X-ray or photo** tab has three one-click samples. You can also download them or use the originals:

| Sample | What the model finds | Bundled copy | Original |
|---|---|---|---|
| Impacted wisdom teeth | 4 impacted teeth | [`/static/samples/impacted-wisdom-teeth.jpg`](http://127.0.0.1:5000/static/samples/impacted-wisdom-teeth.jpg) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Orthopantomogram_showing_mesio_angular_impaction_of_third_molars.jpg) |
| Fillings and bone loss | 9 fillings, bone loss | [`/static/samples/fillings-and-bone-loss.jpg`](http://127.0.0.1:5000/static/samples/fillings-and-bone-loss.jpg) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Orthopantomogram_of_an_abscessed_tooth.jpg) |
| Implant and fillings | 5 fillings, 1 implant, 1 impacted tooth | [`/static/samples/implant-and-fillings.jpg`](http://127.0.0.1:5000/static/samples/implant-and-fillings.jpg) | [Kaggle Dental Radiography](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography) |

A good fourth test is [this X-ray of historic implants](https://upload.wikimedia.org/wikipedia/commons/b/b3/Panoramic_radiograph_of_historic_dental_implants.jpg):
it's slightly sepia-tinted, and the model finds 8 implants. Any ordinary photo (a selfie, a street) is a good check that
non-X-rays are *not* treated as dental.

Questions worth trying: *What did the model find? · Are there any cavities? · How many teeth can you see? · Do I need a root canal?*
In the Knowledge tab: *When was Near East University founded?* (knowledge base) · *Who is the current rector?* (web search).

## Where answers come from

Each answer ends with a **From** line: a coloured dot for the kind of source, then each location it used.
Relay tries sources in this order and stops at the first real answer:

```
X-ray        dental model findings  ->  GPT vision looks at the image  ->  general GPT knowledge
Photo        GPT vision description ->  GPT vision looks again          ->  general GPT knowledge
Document     the pages/paragraphs   ->  general GPT knowledge
Video        the transcript         ->  general GPT knowledge
Knowledge    knowledge.json entry   ->  live web search (linked pages)  ->  general GPT knowledge
```

How each step works (`main.py`):

1. **The source.** The question is matched against the source's location-tagged chunks (embeddings for large
   sources) and answered from the best four, which the model must cite by number. If the answer is `NOT_FOUND` or
   hedges ("I can't tell", "it doesn't mention"), the next step runs.
2. **GPT vision** (images). Sees the image plus the dental model's findings and must give its best direct answer
   (counts, positions, visible signs) instead of declining. Only an outright refusal moves on.
3. **Web search** (knowledge base). OpenAI's `web_search` tool, given the closest knowledge entries as context so a
   vague question like "who is the rector?" is read as being about the knowledge base's subject. The pages it cites
   are shown as links.
4. **General knowledge** is the last resort and is always labelled *"General knowledge from GPT, not from your …"*.

To link knowledge base entries to where their facts came from, add a `source` URL:

```json
{ "id": "1", "text": "Near East University ... founded in 1988 ...", "source": "https://neu.edu.tr/" }
```

## The dental model

**[`lukasiktar11/dental-radiography-yolo26`](https://huggingface.co/lukasiktar11/dental-radiography-yolo26)**, a
pretrained detector used as-is (no training). Of 11 pretrained dental models downloaded from Hugging Face, the 7 that
detect at least one of the four target classes were scored, and this one was best, on
194 panoramic X-rays from the [Kaggle Dental Radiography](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography)
test and validation sets, none of which they had seen.

Image-level F1 (does it correctly say whether a finding is in the X-ray; 1.00 is perfect), confidence 0.4:

| Class | F1 | How to read it |
|---|---|---|
| Fillings | **0.93** | dependable |
| Cavity | 0.56 | a second look, not a verdict |
| Impacted tooth | 0.50 | a second look, not a verdict |
| Implant | 0.39 | a second look, not a verdict |

The model also marks other things it knows (crowns, root canal treatment, bone loss, …). These are listed as
"other" findings and drawn in grey.

**Is it an X-ray?** Every image goes through the model first. On its own the model "finds" implants in colourful
abstract pictures, so an image is only treated as an X-ray straight away when the model finds something **and** the
image is grayscale (mean saturation ≤ 35; all 1,269 Kaggle X-rays measure 0, a sepia scan 28, ordinary photos 42 and
up). Anything else goes to GPT vision first. If GPT recognises a dental X-ray (say, a phone photo of one, or an X-ray
where the model found nothing), the model's findings are still used, showing "none found" where it found nothing.

## Configuration

Set these in `.env` (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Chat, vision, embeddings, web search and Whisper. |
| `SECRET_KEY` | random per start | **Set it**, or loaded sources are lost on every restart. |
| `COOKIE_SECURE` | `0` | `1` when served over HTTPS. |
| `PORT` | `5000` | Port for `python app.py`. |
| `CHAT_MODEL` | `gpt-5.4-mini` | Answers and web search (must support the `web_search` tool). |
| `VISION_MODEL` | `gpt-5.4-mini` | Looks at images (must accept image input). |
| `DENTAL_CONF` | `0.4` | Minimum detection confidence. Lower finds more, with more false alarms. |
| `XRAY_MAX_SATURATION` | `35` | Grayscale cut-off for treating an image as an X-ray. |
| `DENTAL_MODEL_PATH` | `models/dental-yolo26.pt` | Where the weights live. |
| `ASKS_PER_MINUTE` / `LOADS_PER_MINUTE` | `30` / `10` | Per-IP rate limits. |
| `FLASK_DEBUG` | off | `1` for the Flask debugger (never in production). |

## Project structure

```
app.py              Flask routes, per-tab source store, rate limiting, upload limits
main.py             extraction with locations, retrieval, cited answers, fallback chain, web search
dental.py           dental model: X-ray check, detection, positions, findings text, annotated image
knowledge.json      the knowledge base (id, text, optional source URL)
models/             dental model weights
templates/          landing.html (/) and index.html (/app)
static/css, js      the site's styles and the workspace script
static/samples/     sample X-rays for testing
test_app.py         end-to-end checks with OpenAI stubbed
docs/               README images
Dockerfile          production image (gunicorn)
```

## HTTP API

All responses are JSON. `mode` is the tab: `image`, `files`, `video` or `text` (knowledge).

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/api/load` | form: `mode` + `image_file` / `document_file` / `video_url` | `{name, kind, annotated, findings}` |
| `POST` | `/api/ask` | JSON: `{mode, question}` | `{answer, origin, sources: [{label, href}]}` |
| `POST` | `/api/clear` | JSON: `{mode}` | `{ok}` |
| `GET` | `/healthz` | — | `{ok: true}` |

`kind` is `xray`, `photo`, `document` or `video`. `origin` is `model`, `xray`, `photo`, `vision`, `document`,
`video`, `knowledge`, `web` or `general`. `findings` (X-rays only) is a list of `{cls, count, where}` for the four
classes. If the server no longer has a tab's source (for example after a restart), `/api/ask` returns `409` with
`reload: true` and the page asks you to load it again.

## Testing

```bash
python test_app.py                       # routing, citations and fallbacks, OpenAI stubbed (free)
TEST_XRAY=static/samples/implant-and-fillings.jpg python test_app.py   # also runs the real dental model
python main.py                           # self-checks: chunk locations, citations, dodge filter
python dental.py                         # self-checks: findings text, positions, X-ray check
```

## Deployment

```bash
docker build -t relay .
docker run -p 8000:8000 --env-file .env relay
```

The image runs gunicorn (1 worker, 8 threads) as a non-root user, with `COOKIE_SECURE=1` and a health check, so serve
it over HTTPS (or pass `-e COOKIE_SECURE=0`). It builds successfully but hasn't been run-tested. Loaded sources and rate
limits live in process memory, which is why it runs one worker; to scale out, move them to Redis.

## Evaluation

A small evaluation of the dental detector and the answer pipeline, run on 2026-10-15 with `gpt-5.4-mini` and
`text-embedding-3-small` (40 OpenAI calls). Scripts, raw outputs and per-question tables are in `report_eval/`
(`results.md`, `failures.md`, `details.md`) in the main repository.

**Detector** (194 Kaggle test + valid X-rays, image level, confidence 0.4; reproduces the F1 values above exactly):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Fillings | 0.93 | 0.94 | 0.93 |
| Cavity | 0.51 | 0.63 | 0.56 |
| Impacted tooth | 0.53 | 0.48 | 0.50 |
| Implant | 0.92 | 0.24 | 0.39 |

Implants are the weak spot: 68 of the 90 images that contain one got no detection. Seven failures were inspected by
hand (false positives next to large restorations or on upper wisdom teeth, false negatives on metal-heavy
post/root-filled teeth). The causes given in `failures.md` are hypotheses, not verified.

**Question answering** (20 questions: 5 X-ray, 5 knowledge base, 5 PDF, 5 video):

| Source | Cited location supports the answer | Evasive, first source only → after full chain | Mean usefulness (1-5) |
|---|---|---|---|
| X-ray | 3/5 | 1/5 → 0/5 | 4.0 |
| Knowledge base | 4/5 | 1/5 → 0/5 | 4.8 |
| PDF | 5/5 | 0/5 → 0/5 | 5.0 |
| Video | 5/5 | 0/5 → 0/5 | 4.8 |
| **Total** | **17/20 (85%)** | **2/20 (10%) → 0/20** | **4.65** |

- The fallback chain rescued both questions the first source could not answer (a tooth count via GPT vision, the
  current rector via web search). The `HEDGE` filter itself never fired; both rejections were `NOT_FOUND` replies.
- One clear failure: the detector found 5 fillings where the Kaggle label says 7, and the answer stated 5 without a
  caveat. The chain only moves on after a hedge, refusal or `NOT_FOUND`, never after a confident wrong answer.
- Caveats: only 20 questions; the PDF and video questions were written after reading their sources, so they are
  easy; citation judgements and usefulness ratings were made by Claude and are awaiting human review; X-ray reference
  answers for three images are a visual read, not dentist-verified.

## Limitations

- **Not a diagnosis.** The model is reliable for fillings only; treat cavities, impacted teeth and implants as hints.
  GPT vision is a fallback, not a diagnostician: in testing it missed the abscess on an X-ray labelled as showing one.
- Positions are approximate (upper/lower half, left/right/front third of a panoramic image).
- Scanned PDFs have no text to read; upload the pages as images instead.
- No user accounts: uploads are tied to a browser session and kept in memory.
- Everything you upload or ask is sent to OpenAI (except the dental model, which runs locally).

## Credits and licenses

- Dental model: [dental-radiography-yolo26](https://huggingface.co/lukasiktar11/dental-radiography-yolo26) by
  lukasiktar11, **AGPL-3.0**. If you offer Relay as a public service, AGPL may require you to publish your source.
- Sample X-rays: [Kaggle Dental Radiography](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography)
  (CC BY-SA 4.0); [Challiyan](https://commons.wikimedia.org/wiki/File:Orthopantomogram_showing_mesio_angular_impaction_of_third_molars.jpg)
  (CC BY-SA 4.0) and [O'Dea](https://commons.wikimedia.org/wiki/File:Orthopantomogram_of_an_abscessed_tooth.jpg)
  (CC BY-SA 3.0) on Wikimedia Commons, resized.
- Built with Flask, Ultralytics, OpenAI, pypdfium2, python-docx, yt-dlp and youtube-transcript-api.
# relay
