"""Dental X-ray findings: the model runs first on every image, GPT is only the fallback.

Model: lukasiktar11/dental-radiography-yolo26 (Hugging Face, AGPL-3.0), picked over 9 other pretrained
models on the Kaggle dental-radiography test+valid splits (image-level F1 at conf 0.4: Fillings 0.93,
Cavity 0.56, Impacted 0.50, Implant 0.39). Reliable for fillings, a hint for the rest.
"""
import base64
import os
import threading
import urllib.request
from collections import defaultdict
from functools import cache

import cv2
import numpy as np

MODEL_NAME = "dental-radiography-yolo26"
MODEL_PATH = os.getenv("DENTAL_MODEL_PATH", "models/dental-yolo26.pt")
MODEL_URL = "https://huggingface.co/lukasiktar11/dental-radiography-yolo26/resolve/main/best.pt"
CONF = float(os.getenv("DENTAL_CONF", 0.4))
# X-rays are grayscale (saturation 0 on all 1,269 Kaggle images, 28 on a sepia-scanned one; photos measured 42-245). The model alone
# hallucinates "implants" on colourful abstract images, so it only counts as an X-ray when it's grey too.
XRAY_MAX_SATURATION = float(os.getenv("XRAY_MAX_SATURATION", 35))

# model class -> the 4 classes this project is about, with the colours used on the site (BGR for OpenCV)
FOCUS = {"Filling": "Fillings", "Implant": "Implant", "Caries": "Cavity", "impacted tooth": "Impacted Tooth"}
COLORS = {"Fillings": (240, 195, 108), "Implant": (78, 193, 242), "Cavity": (92, 106, 240), "Impacted Tooth": (245, 151, 181)}
OTHER_COLOR = (171, 165, 154)

# ponytail: one model shared by all request threads behind a lock; run more workers if inference queues up
_lock = threading.Lock()


@cache
def get_model():
    from ultralytics import YOLO  # heavy import, only when an image actually shows up

    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return YOLO(MODEL_PATH)


def decode(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("That file isn't an image Relay can read. Try a JPG or PNG.")
    return img


def is_grayscale(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()) <= XRAY_MAX_SATURATION


def where(cx, cy):
    """Rough position on a panoramic X-ray. The patient's right is on the image's left."""
    jaw = "upper" if cy < 0.5 else "lower"
    side = "front" if 0.38 <= cx <= 0.62 else ("right" if cx < 0.5 else "left")
    return f"{jaw} {side}"


def detect(img):
    model = get_model()
    with _lock:
        result = model.predict(img, imgsz=640, conf=CONF, device="cpu", verbose=False)[0]
    h, w = img.shape[:2]
    dets = []
    for c, conf, (x1, y1, x2, y2) in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
        raw = model.names[int(c)]
        dets.append({
            "cls": FOCUS.get(raw, raw),
            "focus": raw in FOCUS,
            "conf": round(conf, 2),
            "box": [x1, y1, x2, y2],
            "where": where((x1 + x2) / 2 / w, (y1 + y2) / 2 / h),
        })
    return dets


def summarize(dets):
    """Findings text the chat answers from, and the per-class list the UI shows."""
    groups = defaultdict(list)
    for d in dets:
        groups[d["cls"]].append(d)
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


def draw(img, dets):
    """Annotated JPEG as a data URL, in the site's colours. Small exports are upscaled so labels stay legible."""
    scale = max(1.0, 1280 / img.shape[1])
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else img.copy()
    t = max(2, round(out.shape[1] / 640))
    font = out.shape[1] / 1300  # labels stay readable when the image is shown at ~500px wide
    for d in sorted(dets, key=lambda d: d["focus"]):  # focus classes drawn on top
        x1, y1, x2, y2 = (int(v * scale) for v in d["box"])
        color = COLORS.get(d["cls"], OTHER_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t if d["focus"] else 1)
        if d["focus"]:
            label = f"{d['cls']} {d['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, 2)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(out, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, font, (28, 22, 12), 2, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


if __name__ == "__main__":
    dets = [
        {"cls": "Fillings", "focus": True, "conf": 0.8, "box": [0, 0, 1, 1], "where": "upper right"},
        {"cls": "Fillings", "focus": True, "conf": 0.7, "box": [0, 0, 1, 1], "where": "lower left"},
        {"cls": "Crown", "focus": False, "conf": 0.6, "box": [0, 0, 1, 1], "where": "lower left"},
    ]
    text, findings = summarize(dets)
    assert "- Fillings: 2 detected (lower left, upper right; confidence 70%-80%)" in text, text
    assert "- Cavity: none detected" in text and "Crown x1 (lower left)" in text
    assert findings[0] == {"cls": "Fillings", "count": 2, "where": "lower left, upper right"}
    assert where(0.2, 0.3) == "upper right" and where(0.5, 0.8) == "lower front" and where(0.9, 0.6) == "lower left"
    assert is_grayscale(np.full((10, 10, 3), 120, np.uint8)) and not is_grayscale(np.dstack([np.zeros((10, 10)), np.zeros((10, 10)), np.full((10, 10), 255)]).astype(np.uint8))
    print("ok")
