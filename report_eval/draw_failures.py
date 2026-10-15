"""Overlay ground truth (green) and detector output at conf 0.4 (red) for the failure cases. python report_eval/draw_failures.py"""
import json, os, cv2
DS = os.path.expanduser("~/.cache/dental-eval/ds")
CASES = [("valid","0886","Cavity"),("test","0564","Cavity"),("test","0038","Cavity"),("valid","0680","Impacted Tooth"),("test","0213","Impacted Tooth"),
         ("valid","0150","Impacted Tooth"),("valid","0878","Implant"),("test","0794","Implant"),("test","0976","Implant")]
R = {(r["split"], r["file"][:4]): r for r in json.load(open("report_eval/detector_results.json"))}
for split, n, cls in CASES:
    r = R[(split, n)]; img = cv2.imread(f"{DS}/{split}/{r['file']}"); s = 3
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    for b in r["gt_boxes"][cls]: cv2.rectangle(img, tuple(v*s for v in b[:2]), tuple(v*s for v in b[2:]), (0,200,0), 2)
    for b, c in zip(r["det_boxes"][cls], r["det"][cls]):
        cv2.rectangle(img, tuple(v*s for v in b[:2]), tuple(v*s for v in b[2:]), (0,0,255), 2); cv2.putText(img, f"{c:.2f}", (b[0]*s, b[1]*s-4), 0, 0.7, (0,0,255), 2)
    cv2.putText(img, f"{split}/{n} {cls}: green=ground truth red=detector", (8, 22), 0, 0.7, (0,255,255), 2)
    cv2.imwrite(f"report_eval/failure_images/{split}_{n}_{cls.split()[0]}.jpg", img)
