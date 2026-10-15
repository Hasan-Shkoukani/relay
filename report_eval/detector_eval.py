"""Run the app's dental detector (dental.detect, conf 0.4) on the Kaggle valid+test images and compare with the CSV labels.
Read-only use of dental.py. Run from the repo root: python report_eval/detector_eval.py"""
import csv, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
import cv2
import dental

DS = os.path.expanduser("~/.cache/dental-eval/ds")
CLASSES = list(dental.FOCUS.values())  # Fillings, Implant, Cavity, Impacted Tooth
rows = []
for split in ("test", "valid"):
    gt = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f"{DS}/{split}/_annotations.csv")):
        gt[r["filename"]][r["class"]].append([int(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")])
    for fn in sorted(f for f in os.listdir(f"{DS}/{split}") if f.endswith(".jpg")):
        img = cv2.imread(f"{DS}/{split}/{fn}")
        dets = dental.detect(img)
        low = dental.get_model().predict(img, imgsz=640, conf=0.05, device="cpu", verbose=False)[0]
        lowdets = [(dental.FOCUS.get(dental.get_model().names[int(c)], "other"), round(p, 2))
                   for c, p in zip(low.boxes.cls.tolist(), low.boxes.conf.tolist())]
        rows.append({"split": split, "file": fn, "size": img.shape[:2],
                     "gt": {c: len(gt[fn].get(c, [])) for c in CLASSES},
                     "gt_boxes": {c: gt[fn].get(c, []) for c in CLASSES},
                     "det": {c: [d["conf"] for d in dets if d["cls"] == c] for c in CLASSES},
                     "det_boxes": {c: [[round(v) for v in d["box"]] for d in dets if d["cls"] == c] for c in CLASSES},
                     "det_where": {c: [d["where"] for d in dets if d["cls"] == c] for c in CLASSES},
                     "low_conf": {c: sorted([p for k, p in lowdets if k == c], reverse=True) for c in CLASSES}})
json.dump(rows, open("report_eval/detector_results.json", "w"))
print(len(rows), "images")
for c in CLASSES:
    tp = sum(1 for r in rows if r["gt"][c] and r["det"][c]); fp = sum(1 for r in rows if not r["gt"][c] and r["det"][c])
    fn = sum(1 for r in rows if r["gt"][c] and not r["det"][c]); tn = len(rows) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
    print(f"{c}: TP={tp} FP={fp} FN={fn} TN={tn} P={p:.2f} R={rc:.2f} F1={2*p*rc/(p+rc) if p+rc else 0:.2f}")
