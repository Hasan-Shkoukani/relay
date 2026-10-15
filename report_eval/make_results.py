"""Builds results.md from results_raw.json plus the hand judgements below (made by Claude; to be reviewed by the user)."""
import json
R = {r["id"]: r for r in json.load(open("report_eval/results_raw.json"))}
Q = json.load(open("report_eval/questions.json"))
REF = {q["id"]: q for g in ("xray", "knowledge", "pdf", "video") for q in Q[g]}
# id: (citation supported Yes/No/NOT VERIFIED, reason, usefulness 1-5, reason)
J = {
 "X1": ("Yes", "Cited detector text says 'Impacted Tooth: 4 detected'; ref = 4", 5, "Correct and direct."),
 "X2": ("Yes", "Cited text: 'Fillings: 9 detected' in lower front/left/right and upper right; ref = fillings on upper and lower back teeth, both sides", 4, "Correct and located; the count of 9 is not independently verified."),
 "X3": ("No", "Cited text says 'Fillings: 5'; ground truth (Kaggle CSV) is 7 fillings. Implant 1 matches.", 2, "States 5 fillings flatly; real answer is 7, no caveat about detector reliability."),
 "X4": ("Yes", "Cited text: 'Impacted Tooth: 1 detected (lower right ...)'; GT box is lower jaw, patient's right", 5, "Correct and located."),
 "X5": ("No", "Cited sources are 'GPT vision' (not a checkable location) and the detector chunk, which contains no tooth count; the 28 came from the image itself", 4, "Gives an estimate, says confidence is moderate, adds not-a-diagnosis line; 28 is at the low end of my 28-32 reference."),
 "K1": ("Yes", "Entry 1 says 'founded in 1988'", 5, "Correct, one sentence."),
 "K2": ("Yes", "Entry 2 says 'more than 26,000 students'", 5, "Correct."),
 "K3": ("Yes", "Entry 3 says '20 faculties comprising around 220 departments'", 5, "Correct, terse."),
 "K4": ("Yes", "Entry 5 says 'opened in 2008'", 5, "Correct."),
 "K5": ("NOT VERIFIED", "Cited page 'Management Board - Near East University' was not opened by me. A separate web search I ran found neu.edu.tr naming Prof. Dr. Tamer Sanlidag as rector (newest item found: July 2024)", 4, "Gives a specific name from a cited page, but I could not confirm it is current as of today."),
 "P1": ("Yes", "Page 2 text (inside cited 'Pages 1-2'): 'Best initial test -> Hepatitis C antibody. If positive>>>Most accurate>>> PCR'", 5, "Correct and complete."),
 "P2": ("Yes", "Page 6 (inside cited 'Pages 5-6'): paracentesis, >250 neutrophils, IV cefotaxime and albumin", 5, "Correct and complete."),
 "P3": ("Yes", "Page 5 (inside cited 'Pages 5-6'): 'Esophageal varices >>>Tx: propranolol'", 5, "Correct."),
 "P4": ("Yes", "Page 7 (cited 'Page 7'): Penicillamine; slit lamp and ceruloplasmin", 5, "Correct and complete."),
 "P5": ("Yes", "Page 9 (cited 'Page 9'): >45% transferrin saturation, ferritin >250 ug/L on more than one occasion", 5, "Correct and complete."),
 "V1": ("Yes", "Cited 0:54-1:37 and 1:39-2:26: 'images as the recipe', 'container is like the actual meal', 'one image ... multiple containers'", 5, "Correct and complete."),
 "V2": ("Yes", "Cited 4:45-5:28 covers dependencies first then code and layer caching", 5, "Correct and explains why."),
 "V3": ("Yes", "Cited 5:29-6:12 and 6:14-6:56: 'Run keyword happens when we're building the image while CMD is what Docker uses to start the container'", 4, "Correct; slightly wordy."),
 "V4": ("Yes", "Cited 7:41-8:23: looks inside the image, report of packages, vulnerabilities, suggestions", 5, "Correct and complete."),
 "V5": ("Yes", "Cited 10:41-11:27 contains 'up to 39 times faster'", 5, "Correct."),
}
GROUPS = [("X-ray", "X"), ("PDF", "P"), ("Video", "V"), ("Knowledge base", "K")]
def ids(p): return [i for i in J if i.startswith(p)]
o = ["# QA evaluation results\n",
 "Run: `python report_eval/run_eval.py --approved` (raw log in `results_raw.json`), through the real `main.answer()` with `gpt-5.4-mini` / `text-embedding-3-small`. **40 OpenAI API calls** were made (counted by the script: loads, embeddings, chat, vision, web search).\n",
 "**Citation judgements and usefulness ratings below are by Claude, not by a human or an independent model; the user will review them.** Caveats: only 20 questions; I wrote the PDF/video questions after reading the PDF text and transcript, so they are answerable from the source (easy for retrieval); the PDF is the user's own medical study notes (`~/Documents/Hepatology.pdf`, 11 pages) and the video is `https://youtu.be/DQdB7wFEygo` (Docker tutorial, 713 s).\n",
 "## Headline\n"]
def stats(L):
    n = len(L); sup = sum(J[i][0] == "Yes" for i in L); bef = sum(R[i]["first_source_flag"] is not None for i in L)
    aft = sum(R[i]["final_flag"] is not None for i in L); use = sum(J[i][2] for i in L) / n
    return n, sup, bef, aft, use
o += ["| Source type | Questions | Citation support (Yes / n) | Evasive, first source only | Evasive after full chain | Mean usefulness (1-5) |", "|---|---|---|---|---|---|"]
for name, p in GROUPS:
    n, sup, bef, aft, use = stats(ids(p)); o.append(f"| {name} | {n} | {sup}/{n} ({sup/n:.0%}) | {bef}/{n} ({bef/n:.0%}) | {aft}/{n} ({aft/n:.0%}) | {use:.1f} |")
n, sup, bef, aft, use = stats(list(J)); o.append(f"| **Total** | {n} | **{sup}/{n} ({sup/n:.0%})** | **{bef}/{n} ({bef/n:.0%})** | **{aft}/{n} ({aft/n:.0%})** | **{use:.2f}** |\n")
o += ["Definitions:",
 "- **Citation support**: 'Yes' only if the text at the location the app cited (page, timestamp range, entry, or detector findings) contains the information in my reference answer. 'No' and 'NOT VERIFIED' both count as not supported in the rate above. Excluding the one NOT VERIFIED case (K5) the rate is 17/19 (89%).",
 "- **Evasive, first source only**: the first answer from the question's own source (detector findings / best document, transcript or knowledge entries) was rejected by the app's own filter `answered()` (NOT_FOUND, `REFUSAL`, or `HEDGE` match), so the app would have moved on. 'After full chain' applies the same regexes to the final answer. Note the regexes are the app's, not an independent judge; I also read every final answer and none is evasive.",
 "- The `HEDGE` regex itself fired on **0/20** first answers; both rejections (X5, K5) were the model replying exactly `NOT_FOUND`, as the prompt tells it to. Without the chain these two questions would have gone unanswered, so the chain rescued 2/2.",
 "- Fallback step that produced the final answer: model/source 18, vision 1 (X5), web 1 (K5), general 0.\n",
 "## Per-question table\n",
 "| ID | Question | Answer (abridged) | Step that produced it | First-source filter | Cited location(s) | Reference | Cited location supports it? | Why | Usefulness (Claude) | Reason |", "|---|---|---|---|---|---|---|---|---|---|---|"]
STEP = {"model": "source (detector)", "knowledge": "source (KB entry)", "document": "source (PDF)", "video": "source (transcript)", "vision": "GPT vision", "web": "web search", "general": "general GPT"}
def cell(s): return str(s).replace("\n", " ").replace("|", "/")
for i in J:
    r = R[i]; c = J[i]
    o.append(f"| {i} | {cell(r['question'])} | {cell(r['answer'])[:220]} | {STEP[r['origin']]} | {r['first_source_flag'] or 'passed'} | {cell('; '.join(s['label'] for s in r['sources']))} | {cell(REF[i]['reference_answer'])[:160]} (expected: {cell(REF[i]['expected_location'])}) | {c[0]} | {cell(c[1])} | {c[2]} | {cell(c[3])} |")
o.append("\nNOT AVAILABLE: none. Video chunk timestamps from the app run about 7 s later than the transcript I used to write the reference times (the app's `handleVideo` evidently used a different caption source); every cited range still contains the fact.")
open("report_eval/results.md", "w").write("\n".join(o) + "\n")
print("\n".join(o[7:15]))
