# Failure examples

## a) Detector failures on the Kaggle valid/test images

Setup: `dental.detect()` from the app, confidence 0.4 (the app default), run on the 194 Kaggle Dental Radiography images in `~/.cache/dental-eval/ds/{test,valid}` (the same local copy used for the README's model comparison; 73 test + 121 valid). Ground truth = the `_annotations.csv` next to each split. Comparison is **image-level** (is the class present or not), as in the README. Reproduced by `report_eval/detector_eval.py` (raw: `detector_results.json`); the overlays are in `report_eval/failure_images/` (green = ground truth, red = detector output at 0.4).

Image-level results I got on these 194 images (they reproduce the README F1 values exactly):

| Class | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Fillings | 144 | 11 | 10 | 29 | 0.93 | 0.94 | 0.93 |
| Cavity | 22 | 21 | 13 | 138 | 0.51 | 0.63 | 0.56 |
| Impacted Tooth | 20 | 18 | 22 | 134 | 0.53 | 0.48 | 0.50 |
| Implant | 22 | 2 | 68 | 102 | 0.92 | 0.24 | 0.39 |

So there are plenty of failures of each kind (Cavity 21 FP / 13 FN, Impacted 18 FP / 22 FN, Implant 2 FP / 68 FN); the cases below are a hand-picked sample that I looked at in the overlay images. The most striking pattern is the **Implant recall of 0.24**: most implants are simply not detected.

### False positive: Cavity, `0886_jpg.rf.392b31859a90f6265fe9ae541485d7d5.jpg` (valid)

- Ground truth (CSV): Fillings: 1, Implant: 2, Cavity: 0, Impacted Tooth: 0
- Detector at 0.4: Fillings: 3 (conf 0.81, 0.77, 0.74), Implant: 0, Cavity: 1 (conf 0.70), Impacted Tooth: 0
- Focus class `Cavity`: detector gave [0.7] at ['upper right'], ground truth has none.
- Overlay: `report_eval/failure_images/valid_0886_Cavity.jpg`
- Hypothesis: the box sits in the dark gap beside a large bright filling on the upper-right molars; a narrow dark band next to a restoration looks like a cavity (or recurrent decay) to the model, but the annotators did not label it.

### False negative: Cavity, `0038_jpg.rf.e13603a0be37e61c53a3b781040ce9f7.jpg` (test)

- Ground truth (CSV): Fillings: 1, Implant: 1, Cavity: 1, Impacted Tooth: 1
- Detector at 0.4: Fillings: 5 (conf 0.71, 0.68, 0.57, 0.54, 0.44), Implant: 0, Cavity: 0, Impacted Tooth: 0
- Focus class `Cavity`: detector gave nothing at 0.4; highest scores the model produced below the threshold: [0.38, 0.34, 0.21]; ground-truth boxes: 1.
- Overlay: `report_eval/failure_images/test_0038_Cavity.jpg`
- Hypothesis: low-contrast, thin upper-left tooth overlapping a restoration, in a dark image with little tooth detail; the model's highest below-threshold Cavity score on this image was 0.38 (I did not check whether it sits on the ground-truth box), just under the 0.4 cut-off.

### False positive: Impacted Tooth, `0680_jpg.rf.5596d1bc316cbf0d84897706ef97ed3e.jpg` (valid)

- Ground truth (CSV): Fillings: 1, Implant: 1, Cavity: 0, Impacted Tooth: 0
- Detector at 0.4: Fillings: 8 (conf 0.76, 0.69, 0.68, 0.62, 0.59, 0.56, 0.52, 0.49), Implant: 0, Cavity: 0, Impacted Tooth: 2 (conf 0.80, 0.55)
- Focus class `Impacted Tooth`: detector gave [0.8, 0.55] at ['upper right', 'upper left'], ground truth has none.
- Overlay: `report_eval/failure_images/valid_0680_Impacted.jpg`
- Hypothesis: both boxes are upper third molars at the very edge of the arch, which resemble partly hidden wisdom teeth, the typical 'impacted' appearance; the annotators judged them not impacted (or did not label them). May also be a label-consistency issue in the dataset.

### False negative: Impacted Tooth, `0213_jpg.rf.6304d5ebe5f8c87c3a8ed5495cc2bd58.jpg` (test)

- Ground truth (CSV): Fillings: 0, Implant: 0, Cavity: 0, Impacted Tooth: 4
- Detector at 0.4: Fillings: 0, Implant: 0, Cavity: 0, Impacted Tooth: 0
- Focus class `Impacted Tooth`: detector gave nothing at 0.4; highest scores the model produced below the threshold: [0.34, 0.25, 0.2]; ground-truth boxes: 4.
- Overlay: `report_eval/failure_images/test_0213_Impacted.jpg`
- Hypothesis: all four wisdom teeth are labelled impacted in a clearly tilted position, but this image has low contrast with overlapping jaw structures around the lower molars and the sinus floor above the upper ones; scores were only 0.20-0.34.

### False negative: Implant, `0878_jpg.rf.78ef2a6b6ad228db74e583701c39cd39.jpg` (valid)

- Ground truth (CSV): Fillings: 11, Implant: 7, Cavity: 0, Impacted Tooth: 0
- Detector at 0.4: Fillings: 2 (conf 0.53, 0.41), Implant: 0, Cavity: 0, Impacted Tooth: 0
- Focus class `Implant`: detector gave nothing at 0.4; highest scores the model produced below the threshold: [0.31, 0.08]; ground-truth boxes: 7.
- Overlay: `report_eval/failure_images/valid_0878_Implant.jpg`
- Hypothesis: the labelled 'implants' are metal-heavy teeth with posts/root fillings, with large overlapping bright restorations next to them; the model produced no box at 0.4 (best 0.31). The label definition in this dataset may also be broader than 'screw-type implant'.

### False negative: Implant, `0794_jpg.rf.cc0c4ed7311e57db7eb6aa95d0dcbe47.jpg` (test)

- Ground truth (CSV): Fillings: 10, Implant: 9, Cavity: 0, Impacted Tooth: 0
- Detector at 0.4: Fillings: 2 (conf 0.58, 0.52), Implant: 0, Cavity: 0, Impacted Tooth: 0
- Focus class `Implant`: detector gave nothing at 0.4; highest scores the model produced below the threshold: [0.15, 0.11]; ground-truth boxes: 9.
- Overlay: `report_eval/failure_images/test_0794_Implant.jpg`
- Hypothesis: same pattern: nine labelled implants are posts/root-filled teeth under bright crowns or restorations; the model's best score was 0.15. Overlapping restorations that saturate the image hide the shape the model needs.

### False positive: Implant, `0976_jpg.rf.83d1e5e77aacc0d1c648682aa54ecd06.jpg` (test)

- Ground truth (CSV): Fillings: 2, Implant: 0, Cavity: 0, Impacted Tooth: 0
- Detector at 0.4: Fillings: 5 (conf 0.66, 0.65, 0.64, 0.63, 0.48), Implant: 1 (conf 0.57), Cavity: 0, Impacted Tooth: 0
- Focus class `Implant`: detector gave [0.57] at ['lower front'], ground truth has none.
- Overlay: `report_eval/failure_images/test_0976_Implant.jpg`
- Hypothesis: a narrow vertical tooth/root in the upper front region with a dark background is read as an implant fixture; only two small fillings are labelled in the ground truth. Also note the app's position text calls this box 'lower front' because `dental.where()` splits at half the image height, although the box is in the upper jaw.


## b) QA cases where the fallback chain still gave an unsatisfactory answer (from the Task 2 run)

Only **one** clearly unsatisfactory answer and two weaker ones were found in 20 questions (the other 17 were correct and cited correctly). Details and raw outputs: `results.md`, `results_raw.json`.

### 1. X3: wrong count, stated confidently (clear failure)
- **Question:** "How many fillings and how many implants are there?" on `implant-and-fillings.jpg` (= Kaggle test/0239).
- **Answer:** "There are 5 fillings and 1 implant."
- **Step that produced it:** source step (detector findings). The first answer passed the filter, so the chain never moved on.
- **Why unsatisfactory:** ground truth in the Kaggle CSV is 7 fillings and 1 implant. The detector found 5 (conf 0.69-0.81), and the answer repeats that with no caveat. The fallback chain only triggers on `NOT_FOUND`/hedges/refusals, not on wrong-but-confident answers, so GPT vision (which might have counted more) was never consulted.

### 2. X5: answer fine, citation does not support it (weak)
- **Question:** "How many teeth can you see?" on `impacted-wisdom-teeth.jpg`.
- **Answer:** "I can see about 28 teeth ... Confidence: moderate. This isn't a diagnosis."
- **Step that produced it:** GPT vision (the source step returned `NOT_FOUND`, as the prompt instructs for counting questions).
- **Why unsatisfactory:** the estimate is plausible (my reference was 28-32, a visual read, not verified) but the listed sources are "GPT vision" plus the detector findings, and the detector text contains no tooth count. A user clicking the detector citation finds nothing that backs the number. 28 is also the low end of my range.

### 3. K5: specific answer from a page I could not verify (weak)
- **Question:** "Who is the current rector?" (not in `knowledge.json`).
- **Answer:** "The current rector of Near East University is Prof. Dr. Tamer Şanlıdağ."
- **Step that produced it:** web search (source step returned `NOT_FOUND`).
- **Why unsatisfactory / uncertain:** cited page "Management Board – Near East University". I did not open it. A separate web search I ran found neu.edu.tr naming the same person, but the newest item was from July 2024, so I cannot confirm it is current in October 2026. The answer states "current" with no date.
