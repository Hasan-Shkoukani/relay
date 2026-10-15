"""Run report_eval/questions.json through the real pipeline (main.answer) and log every step.
Only reads the app code; wraps its OpenAI-calling functions in this process to record their outputs.
Run from the repo root with the project's Python:  python report_eval/run_eval.py --approved
Costs OpenAI API calls (chat, vision, embeddings, web search), hence the flag."""
import json, os, sys, time

sys.path.insert(0, os.getcwd())
import main

if "--approved" not in sys.argv:
    sys.exit("This spends OpenAI API calls. Re-run with --approved.")

LOG = []


def wrap(name):
    fn = getattr(main, name)

    def inner(*a, **k):
        entry = {"fn": name, "in": str(a[1] if name == "llm" else a[-1] if name == "_vision" else a[0])[:300]}
        LOG.append(entry)
        try:
            out = fn(*a, **k)
        except Exception as exc:
            entry["error"] = repr(exc)
            raise
        entry["out"] = out
        return out
    setattr(main, name, inner)


for n in ("llm", "_vision", "web_answer", "embed"):
    wrap(n)

# which filter rejected a raw answer (None = it passed)
def flag(raw):
    if not raw: return "EMPTY"
    if main.NOT_FOUND in raw: return "NOT_FOUND"
    if main.REFUSAL.search(raw): return "REFUSAL"
    if main.HEDGE.search(raw): return "HEDGE"
    return None


def load(group, q):
    if group == "xray":
        path = q["source"].split(" ")[0]
        return main.handleImage(open(path, "rb").read())
    if group == "knowledge":
        return main.knowledge_source()
    if group == "pdf":
        return main.handleDocument(open(q["source"], "rb").read(), q["source"])
    if group == "video":
        return main.handleVideo(q["source"])


questions = json.load(open("report_eval/questions.json"))
results, loaded = [], {}
for group in ("xray", "knowledge", "pdf", "video"):
    items = questions[group]
    if not isinstance(items, list):
        continue  # NOT AVAILABLE
    for q in items:
        del LOG[:]
        t0 = time.time()
        rec = {"id": q["id"], "group": group, "question": q["question"]}
        try:
            key = (group, q["source"])
            if key not in loaded:
                loaded[key] = load(group, q)
            src = loaded[key]
            n_load = len(LOG)  # calls made while loading (embedding / vision classification)
            del LOG[:]
            res = main.answer(src, q["question"])
            first = next((e for e in LOG if e["fn"] == "llm"), None)
            first_raw = first.get("out") if first else None
            labels = {s["label"] for s in res["sources"]}
            rec.update(answer=res["answer"], origin=res["origin"], sources=res["sources"],
                       cited_text=[c["text"] for c in src["chunks"] if c["loc"] in labels],
                       first_source_raw=first_raw, first_source_flag=flag(first_raw) if first else "NO_FIRST_STEP",
                       final_flag=flag(res["answer"]), steps=[e["fn"] for e in LOG],
                       step_outputs=[{k: v for k, v in e.items() if k != "in"} for e in LOG],
                       api_calls=len(LOG) + n_load, seconds=round(time.time() - t0, 1))
        except Exception as exc:
            rec["error"] = repr(exc)
        results.append(rec)
        print(rec["id"], rec.get("origin"), rec.get("first_source_flag"), rec.get("error", ""), flush=True)

json.dump(results, open("report_eval/results_raw.json", "w"), indent=1, ensure_ascii=False)
print("calls", sum(r.get("api_calls", 0) for r in results))
