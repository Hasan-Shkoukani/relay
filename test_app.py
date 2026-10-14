"""End-to-end checks through the Flask app with OpenAI stubbed out (no API cost). Run: python test_app.py
The dental model is real; set TEST_XRAY to a panoramic X-ray to include it (skipped otherwise)."""
import io
import os

import cv2
import numpy as np

import main
from app import app

calls = []


def fake_llm(system, user, max_tokens=500):
    calls.append("llm")
    if "general knowledge" in system:
        return "General answer."
    if "bread" in user or "offline" in user:
        return main.NOT_FOUND
    if "hedge" in user:
        return "The document doesn't mention that.\nSOURCES: 1"
    first_loc = user.split("[1] (", 1)[1].split(")", 1)[0]
    return f"Answer from {first_loc}.\nSOURCES: 1"


def fake_vision(jpeg, prompt, max_tokens=500):
    calls.append("vision")
    return main.NOT_FOUND if "Question:" in prompt else "A red square on white."


def fake_web(question, context=""):
    calls.append("web")
    if "offline" in question:
        return None, []
    return "Found online.", [{"label": "Example page", "href": "https://example.org/page"}]


main.llm = fake_llm
main.web_answer = fake_web
main._vision = fake_vision
main.embed = lambda texts: [[float(len(t) % 7), 1.0, 0.5] for t in texts]


def ask(c, mode, q):
    return c.post("/api/ask", json={"mode": mode, "question": q})


def png(img):
    return io.BytesIO(cv2.imencode(".png", img)[1].tobytes())


with app.test_client() as c:
    assert c.get("/").status_code == 200 and c.get("/app").status_code == 200 and c.get("/healthz").json == {"ok": True}

    # knowledge: cited by entry -> web search with linked pages -> labelled general knowledge
    r = ask(c, "text", "When was it founded?").json
    assert r["origin"] == "knowledge" and r["sources"][0]["label"].startswith("Knowledge base, entry"), r
    r = ask(c, "text", "How do I bake bread?").json
    assert r["origin"] == "web" and r["sources"][0]["href"] == "https://example.org/page", r
    r = ask(c, "text", "offline question").json
    assert r["origin"] == "general" and "not from your knowledge base" in r["sources"][0]["label"], r

    # asking a tab with nothing loaded tells the page to show the upload again
    r = ask(c, "files", "hi")
    assert r.status_code == 409 and r.json["reload"] is True

    # documents are cited by page
    r = c.post("/api/load", data={"mode": "files", "document_file": (open("sample.pdf", "rb"), "sample.pdf")})
    assert r.status_code == 200 and r.json["kind"] == "document", r.json
    r = ask(c, "files", "What is this?").json
    assert r["origin"] == "document" and r["sources"][0]["label"].startswith("Page"), r
    r = ask(c, "files", "hedge please").json  # a dodge ("doesn't mention") never reaches the user as the answer
    assert r["origin"] == "general", r
    r = c.post("/api/load", data={"mode": "files", "document_file": (io.BytesIO(b"x"), "old.doc")})
    assert r.status_code == 400 and ".docx" in r.json["error"]

    # a colourful non-X-ray never counts as dental, even if the model fires: GPT-4o vision describes it
    photo = np.full((300, 300, 3), 255, np.uint8)
    photo[100:200, 100:200] = (0, 0, 255)
    r = c.post("/api/load", data={"mode": "image", "image_file": (png(photo), "square.png")}).json
    assert r["kind"] == "photo" and r["annotated"] is None, r
    calls.clear()
    r = ask(c, "image", "How do I bake bread?").json  # not in the description -> vision -> general
    assert r["origin"] == "general" and calls == ["llm", "vision", "llm"], (r, calls)

    # X-rays: the dental model answers first
    if os.getenv("TEST_XRAY"):
        r = c.post("/api/load", data={"mode": "image", "image_file": (open(os.environ["TEST_XRAY"], "rb"), "x.jpg")}).json
        assert r["kind"] == "xray" and r["annotated"].startswith("data:image/jpeg") and len(r["findings"]) == 4, r
        r = ask(c, "image", "What did the model find?").json
        assert r["origin"] == "model" and "Dental detection model" in r["sources"][0]["label"], r
        # the document loaded earlier is still there in its own tab
        assert ask(c, "files", "What is this?").json["origin"] == "document"

    r = c.post("/api/load", data={"mode": "image", "image_file": (io.BytesIO(b"not an image"), "x.jpg")})
    assert r.status_code == 400 and "isn't an image" in r.json["error"]

print("ok")
