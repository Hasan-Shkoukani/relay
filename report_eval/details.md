# Relay implementation details (read-only extraction)

All code blocks below are sliced programmatically from the files (not retyped). Line numbers are for the committed versions of `main.py` / `dental.py` / `app.py` in this repo (`git` HEAD 9e9c771).

## a) Models

`main.py:22-26`
```python
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5.4-mini")
EMBED_MODEL = "text-embedding-3-small"
NOT_FOUND = "NOT_FOUND"
```

- Chat model: `CHAT_MODEL`, default **`gpt-5.4-mini`** (main.py:23). Used by `llm()` (main.py:302-309, `client.chat.completions.create`, `temperature=0`) and by the web-search step (main.py:371, `client.responses.create(model=CHAT_MODEL, tools=[{"type": "web_search"}])`).
- Vision model: `VISION_MODEL`, default **`gpt-5.4-mini`** (main.py:24). Used by `_vision()` (main.py:214-221).
- Embedding model: **`text-embedding-3-small`** (main.py:25), used by `embed()` (main.py:278-282).
- Also: speech-to-text fallback for videos with no transcript uses **`whisper-1`** (main.py:165).
- Local detector: `lukasiktar11/dental-radiography-yolo26`, weights `models/dental-yolo26.pt` (dental.py:3, 17-19), confidence threshold `DENTAL_CONF` default 0.4 (dental.py:20).
- Env overrides: `.env` in this repo contains only the keys `OPENAI_API_KEY` and `SECRET_KEY`, so no model override applies and the defaults above are what runs here. `.env.example` lists `CHAT_MODEL=gpt-5.4-mini` and `VISION_MODEL=gpt-5.4-mini`.

## b) Hedge and refusal regular expressions

`main.py:340-353`
```python
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
```

Reading of the code (not quoted from a comment):
- `HEDGE` has **two alternatives** (the second starts at the `|` on line 344): (1) first-person inability: `i|you|we` + `can't/cannot/am unable to/'m unable to/am not able to/'m not able to` + `tell|determine|say|count|see|identify|assess|confirm|know`; (2) negated reporting verb: `does not|doesn't|do not|don't|did not|didn't` + `mention|say|show|include|provide|specify|contain|give|list|state`. Case-insensitive; matches both `'` and `’`.
- `REFUSAL` (line 347) is the single flat-refusal pattern: `i'm sorry` OR `i can't/cannot/am unable to/won't` + `help|assist|provide|analyse|analyze|do that`.
- `answered(raw, strict)` (350-353) returns False if `raw` is empty, contains the substring `NOT_FOUND` anywhere, or matches `REFUSAL`; with `strict=True` it additionally returns False if `HEDGE` matches. The source step uses `strict=True` (main.py:395), the vision step `strict=False` (main.py:406). The web and general steps are not filtered.

## c) Prompt templates

### c1) Generation prompt with numbered excerpts (source step)

`main.py:314-322`
```python
SOURCE_PROMPT = f"""You answer questions using only the numbered excerpts provided. Each excerpt says where it comes from.
Partial answers and reasonable inferences from the excerpts are fine. Talk to the user directly and keep it concise.
Never say "excerpt" or "context"; refer to the {{noun}} instead.
When excerpts come from the dental detection model, rely on them first; they are automated findings, not a diagnosis.
The model only reports fillings, implants, cavities, impacted teeth and the few other marks it lists. It does not count
teeth, measure anything or judge overall health, so for questions like that don't infer from its counts.
End with one line listing the excerpt numbers you used, exactly like: SOURCES: 1, 3
If the excerpts don't contain what's needed to actually answer (for example a count, a detail or a fact they never
mention), don't hedge with "I can't tell": reply with exactly {NOT_FOUND} so a better source can answer."""
```

Resolution notes: it is an f-string, so `{NOT_FOUND}` is replaced by `NOT_FOUND` at import time, and `{{noun}}` becomes `{noun}`, filled at call time by `SOURCE_PROMPT.format(noun=noun)` (main.py:394) with `NOUN[kind]` (main.py:312):

`main.py:312-312`
```python
NOUN = {"document": "document", "video": "video", "photo": "image", "xray": "X-ray", "knowledge": "knowledge base"}
```

The user message and the numbered-excerpt format (main.py:391-394):

`main.py:391-394`
```python
    top = retrieve(source["chunks"], question)
    if top:
        excerpts = "\n\n".join(f"[{i}] ({c['loc']})\n{c['text']}" for i, c in enumerate(top, 1))
        raw = llm(SOURCE_PROMPT.format(noun=noun), f"Question:\n{question}\n\nExcerpts:\n{excerpts}")
```

The excerpt-number citation is parsed from the final `SOURCES:` line by `parse_cited` (main.py:356-362).

### c2) X-ray instruction

The X-ray-specific instruction to the generator is lines 317-319 of `SOURCE_PROMPT` (already shown in c1, repeated here):

`main.py:317-319`
```python
When excerpts come from the dental detection model, rely on them first; they are automated findings, not a diagnosis.
The model only reports fillings, implants, cavities, impacted teeth and the few other marks it lists. It does not count
teeth, measure anything or judge overall health, so for questions like that don't infer from its counts.
```

The X-ray excerpt it reads is built by `dental.summarize` (dental.py:80-99). Its first excerpt text format:

`dental.py:85-99`
```python
    findings, lines = [], [f"Dental X-ray findings from the {MODEL_NAME} detection model (automated, not a diagnosis):"]
    for cls in FOCUS.values():
        found = groups.get(cls, [])
        places = ", ".join(sorted({d["where"] for d in found}))
        confs = [d["conf"] for d in found]
        if found:
            lines.append(f"- {cls}: {len(found)} detected ({places}; confidence {min(confs):.0%}-{max(confs):.0%})")
        else:
            lines.append(f"- {cls}: none detected")
        findings.append({"cls": cls, "count": len(found), "where": places})
    other = {cls: g for cls, g in groups.items() if not g[0]["focus"]}
    if other:
        lines.append("Other things the model marked: " + "; ".join(
            f"{cls} x{len(g)} ({', '.join(sorted({d['where'] for d in g}))})" for cls, g in other.items()))
    return "\n".join(lines), findings
```

The X-ray tab also has a *classification* prompt, sent to GPT vision at upload time when the detector found nothing or the image is not grayscale (main.py:238-249):

`main.py:224-224`
```python
DENTAL_FLAG = "DENTAL_XRAY"
```

`main.py:238-249`
```python
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
```

### c3) GPT vision fallback

`main.py:324-331`
```python
VISION_PROMPT = f"""Look at this image and answer the user's question directly and concisely, from what is visible.
Count, describe and point to locations (upper/lower, the patient's left/right) when asked. If you're unsure, give your
best estimate and say how confident you are. Never reply only that you can't determine something: for questions that
need a dentist's judgement (a root canal, bone loss, whether a tooth is healthy), describe the visible signs that matter,
for example dark areas at root tips, decay reaching the nerve, existing root canal fillings, or the bone level around
the teeth, and then say a dentist should confirm. End dental answers with one short line that this isn't a diagnosis.
If the question has nothing to do with this image, reply with exactly: {NOT_FOUND}
"""
```

How it is called (main.py:402-410): the prompt is extended with the detector findings (line 404) and the question, and sent with the image:

`main.py:402-410`
```python
    if source.get("image"):
        findings = next((c["text"] for c in source["chunks"] if c["loc"] == MODEL_LOC), "")
        prompt = VISION_PROMPT + (f"\nThe dental detection model already found:\n{findings}\n" if findings else "")
        raw = _vision(source["image"], f"{prompt}\nQuestion: {question}")
        if answered(raw, strict=False):
            sources = [{"label": f"GPT vision ({VISION_MODEL}), looking at your image", "href": None}]
            if findings:
                sources.append({"label": MODEL_LOC, "href": None})
            return {"answer": raw, "origin": "vision", "sources": sources}
```

### c4) Web search step

`main.py:333-334`
```python
WEB_PROMPT = """Search the web and answer the question directly and concisely (under 150 words). Prefer official and
reputable sources."""
```

Context suffix and call (main.py:365-383):

`main.py:365-383`
```python
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
```

`context` is `"\n".join(c["text"] for c in top[:2])` (main.py:413): the text of the two best-ranked knowledge entries.

### c5) General-knowledge step

`main.py:336-338`
```python
GENERAL_PROMPT = """You are a helpful, knowledgeable assistant. Answer the question directly and concisely (under 150 words)
from your general knowledge. For medical or dental questions, give useful general information and suggest seeing a
dentist or doctor for a diagnosis."""
```

Called as `llm(GENERAL_PROMPT, question, max_tokens=600)` (main.py:417), with only the question (no excerpts).

## d) Fallback chain per tab

Tab to mode mapping (app.py:37 `MODES = ("files", "image", "video")`; `mode == "text"` is the Knowledge tab, app.py:152-153, which calls `knowledge_source()`). All tabs go through the same function, `answer(source, question)` (main.py:386-418), called at app.py:162.

The full function:

`main.py:386-418`
```python
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
```

Retrieval (main.py:293-299): `retrieve(chunks, question, top_k=4)` returns **all chunks if there are 4 or fewer**, otherwise embeds the question (1 embedding call) and returns the 4 chunks with highest cosine similarity. Chunks are embedded at load time only when a source has more than 4 chunks (main.py:285-290).

### X-ray tab (mode `image`, source kind `xray`; kind `photo` for non-X-ray images)
1. **Source = detector findings.** `top` = all chunks (X-ray source has 1 chunk, or 2 if GPT vision described it). One `llm()` call with `SOURCE_PROMPT`. Moves on to step 2 unless `answered(raw)` (strict): i.e. it moves on if the reply is empty, contains `NOT_FOUND`, matches `REFUSAL`, or matches `HEDGE` (main.py:395). For a `photo`, the "source" chunk is GPT vision's upload-time description (main.py:247).
2. **GPT vision on the image** (runs if `source.get("image")`, true for both xray and photo, main.py:402). Moves on only if `answered(raw, strict=False)` is False, i.e. empty, contains `NOT_FOUND`, or matches `REFUSAL`. Hedges such as "I can't tell" are **accepted** here (main.py:406).
3. **General GPT knowledge.** Unconditional last resort; no filter; labelled "General knowledge from GPT, not from your X-ray" (main.py:417-418). The web step is skipped (`kind == "knowledge"` only).

### Document tab (mode `files`, kind `document`)
1. **Source = best chunks of the document** (page / paragraph / line chunks, main.py:67-94): retrieve top 4 by embedding similarity (all chunks if <=4), one `llm()` call with `SOURCE_PROMPT`; accepted if `answered(raw)` strict (same condition as above).
2. Otherwise **straight to general GPT knowledge** (main.py:417). There is no vision step (no `image` key, main.py:402) and no web step. **After retrieval nothing else touches the document**: there is no re-retrieval with a larger top_k, no query rewriting, no second pass; the general step receives only the question, not the document text, and is labelled "General knowledge from GPT, not from your document".

### Video tab (mode `video`, kind `video`)
Identical to Document: transcript chunks (timestamped, ~1000 chars each, main.py:198-200), retrieve top 4, `SOURCE_PROMPT` with noun "video", strict `answered`; on failure straight to general GPT knowledge, labelled "...not from your video". No vision, no web, no re-retrieval.

### Knowledge tab (mode `text`, kind `knowledge`)
1. **Knowledge entry**: `knowledge.json` entries as chunks (main.py:264-273; 5 entries here, so they are embedded (5 > 4) and the top 4 are retrieved). `SOURCE_PROMPT` with noun "knowledge base"; accepted if `answered(raw)` strict.
2. **Web search** (main.py:412-415): `web_answer(question, context=top 2 entries)`. Accepted if it returns non-empty text; any exception returns `(None, [])` (main.py:372-373) and moves on. No hedge/refusal filter on this step.
3. **General GPT knowledge**: unconditional.

## e) Source index

| Item | Where |
|---|---|
| Chat / vision / embedding model names | main.py:23, 24, 25 (whisper-1: main.py:165) |
| `NOT_FOUND` sentinel | main.py:26 |
| `HEDGE` regex | main.py:341-345 |
| `REFUSAL` regex | main.py:347 |
| `answered()` | main.py:350-353 |
| `parse_cited()` | main.py:356-362 |
| `SOURCE_PROMPT` (generation with numbered excerpts; contains the X-ray instruction at 317-319) | main.py:314-322 |
| Excerpt formatting and user message | main.py:393-394 |
| `NOUN` map | main.py:312 |
| X-ray findings text | dental.py:80-99 (`summarize`) |
| X-ray classification prompt (`DENTAL_FLAG`) | main.py:224, 238-249 (`handleImage`) |
| `VISION_PROMPT` | main.py:324-331; call at 402-410 |
| `WEB_PROMPT` | main.py:333-334; function `web_answer` 365-383 |
| `GENERAL_PROMPT` | main.py:336-338; call at 417-418 |
| Retrieval (`retrieve`, `embed_chunks`, `embed`) | main.py:293-299, 285-290, 278-282 |
| Fallback chain (`answer`) | main.py:386-418 |
| Tab-to-mode routing | app.py:37, 152-162 |
