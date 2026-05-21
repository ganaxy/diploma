# CONSISTENCY TABLE — every number/claim that appears in >1 place

**Purpose:** side-by-side view of every metric, count, and methodological claim, with the
**canonical (ground-truth) value** and its source in the actual code/data. This is the
backbone for the fix pass. English (study aid).

**Ground-truth sources (authoritative):**
- Canonical dataset = `sample scores/relabeled_v7_corrected.csv` (the file
  `retrain_sota_corrected.py` line 56 trains on; reproduces the headline 0.8326/0.8062).
- Training script: `sample scores/retrain_sota_corrected.py`
- Training log: `sample scores/run_retrain_sota_corrected.log`
- `SESSION_REPORT.md`, `_defense_numbers.json`

---

## 0. CANONICAL REFERENCE BLOCK (the correct values)

| Quantity | Canonical value | Source |
|---|---|---|
| Labeled corpus size | **10,000** | csv rows |
| Class dist — Бүтээлч шүүмжлэл (CONSTRUCTIVE) | **4,356 (43.56%)** | csv `label` |
| Class dist — Саармаг (NEUTRAL) | **3,026 (30.26%)** | csv |
| Class dist — Хортой сөрөг (TOXIC) | **1,963 (19.63%)** | csv |
| Class dist — Эерэг (POSITIVE) | **655 (6.55%)** | csv |
| Split | **train 7,043 (70.43%) / val 1,428 (14.28%) / test 1,529 (15.29%)** | csv `split`, log line 2 |
| Test per-class | **POS 87 / NEU 466 / CON 682 / TOX 294** | csv; log lines 44-47 |
| Train per-class | POS 475 / NEU 2,109 / CON 3,090 / TOX 1,369 | log line 4 |
| Val per-class | POS 93 / NEU 451 / CON 584 / TOX 300 | csv |
| **Final flat — Accuracy** | **0.8326** | log line 37 |
| **Final flat — Macro-F1** | **0.8062** | log line 38 |
| Final flat — Weighted-F1 | 0.8323 | log line 39 |
| Per-class F1 | POS 0.7416 / NEU 0.8030 / CON 0.8760 / TOX 0.8041 | log 44-47 |
| Per-class Precision | POS 0.7253 / NEU 0.8146 / CON 0.8666 / TOX 0.8125 | log 44-47 |
| Per-class Recall | POS 0.7586 / NEU 0.7918 / CON 0.8856 / TOX 0.7959 | log 44-47 |
| Confusion (true→pred P/N/C/T) | POS[66,12,7,2] NEU[17,369,50,30] CON[6,50,604,22] TOX[2,22,36,234] | log 54-58 |
| Original (pre-correction) flat | Acc 0.8143 / Macro-F1 0.7760 | log line 64; SESSION_REPORT §0 |
| Loss function (final flat) | **Focal Loss γ=2.0 + class-weighted α** | retrain script lines 70,126-147,194 |
| α (focal class weights) | POS 3.707 / NEU 0.835 / CON 0.57 / TOX 1.286 | log line 5 |
| Extra imbalance handling | **WeightedRandomSampler** (replacement=True) | retrain script line 181 |
| Learning rate | **3e-5** | retrain script line 67 |
| Batch size | 16 | line 67 |
| Max sequence length | **256** | line 69; `_defense_numbers.json` "faithful_256" |
| Dropout (classifier) | **0.15** | line 68 |
| Weight decay | 0.01 | line 67 |
| Optimizer | AdamW | line 188 |
| Scheduler | cosine + linear warmup, ratio 0.10 (warmup 441 / total 4410) | lines 67,191; log 12 |
| Seed | 42 | line 72 |
| Epochs | max 10, patience 3, **best epoch = 8** | lines 69; log 25,30 |
| Base model | `tugstugi/bert-base-mongolian-cased` | line 55 |
| Tokenizer | SentencePiece (Albert-type), vocab 32,001, cased | HF model config |
| Input format | `[{source}] {text_normalized}` (source tag prepended) | retrain script lines 85-86 |
| Dedup before split | **disabled** (`near_dedup_enabled: False`) | data/preprocessing/preprocess.py |
| 47 test relabels | all 47 in **test split**, both directions; NEU→CON 19 (largest) | csv `was_user_relabeled` |
| Two-stage Stage-1 (3-cls) MN-BERT | Acc 0.8306 / Macro-F1 0.7778 | thesis 6.1 (no code artifact seen — VERIFY) |
| Two-stage Stage-2 (binary) MN-BERT | Acc 0.9144 / Macro-F1 0.9023 | thesis 6.2; SESSION_REPORT §6 ("0.9144") |
| Two-stage cumulative end-to-end | **0.7861 (78.61%)** ← canonical | SESSION_REPORT §6 ("0.7861"); researchoverview |
| Prev diploma 1 (jargalmaa2025) | Acc 0.8270 / F1 0.80 (weighted 0.81 in src table) | SESSION_REPORT §5; Ch2 Tbl 2.2 |
| Prev diploma 2 | Acc 0.6265 / F1 0.6309 | SESSION_REPORT §5 |
| Sources (count) | **8** (NOT "3 / news.mn+Facebook+gogo.mn") | csv `source` |
| Source dist | news.mn 4230(42.3%) · gogo.mn 2811(28.1%) · IKON.mn 919(9.2%) · E-Mongolia 644(6.4%) · "Мэдэхгүй зүйлээ асуу" 632(6.3%) · Medee.mn 342(3.4%) · Zaluusinfo 234(2.3%) · Eagle News 188(1.9%) | csv |
| Per-class char length (median) | NEU 59 · TOX 93 · POS 90 · CON 137 (means 90/129/138/184) | csv `text_normalized` |

---

## 1. CLASS DISTRIBUTION — three thesis tables, none correct, mutually inconsistent

| Class | **Canonical (data)** | Tbl 4.1 `tab:collection` (Ch4:55-58) | Tbl 4.2 `tab:data_labeled` (Ch4:74-77) | Tbl 4.4 `tab:data-stats` (Ch4:443-446) |
|---|---|---|---|---|
| CONSTRUCTIVE | **4,356 / 43.56%** | 4,387 / 43.9% | 4,582 / 45.8% | 4,387 (total col) |
| NEUTRAL | **3,026 / 30.26%** | 2,995 / 30.0% | 2,898 / 29.0% | 2,995 |
| TOXIC | **1,963 / 19.63%** | 1,944 / 19.4% | 1,870 / 18.7% | 1,944 |
| POSITIVE | **655 / 6.55%** | 674 / 6.7% | 650 / 6.5% | 674 |

- Tbl 4.2 (650/2898/1870/4582) = **stale `10k_fully_labeled.csv`** (oldest, pre-relabel).
- Tbl 4.1 / Tbl 4.4 (674/2995/1944/4387) = a different stale version; agree with each other, disagree with 4.2.
- **None** matches the dataset the reported model was trained/tested on.
- Same numbers re-cited in prose: Ch4:111-119 (45.8%, 6.5%, "7:1"), Ch4:313-315 (43.9%/30.0%/6.7%), Ch5:259-260 (43.9%/30.0%/6.7%). All must move to canonical.
- **Canonical "max:min ratio"** = 4,356 : 655 ≈ **6.65 : 1** (thesis says "7:1" at Ch4:119).

## 2. TRAIN/VAL/TEST SPLIT

| Location | Stated split | Test n | Verdict |
|---|---|---|---|
| **Canonical (data + log)** | **70.43 / 14.28 / 15.29** | **1,529** | — |
| Ch4:428 prose | "70% / 15% / 15%" | (implies 1,500) | approx; restate exact |
| Ch4:443-448 Tbl 4.4 | 70/15/15 | **1,480** (val 1,477) | WRONG |
| Ch6:101 | "70/15/15" | — | approx |
| Ch6:244 text | — | **1,530** | WRONG (off-by-one vs 1,529) |
| Ch6:263 Tbl 6.4 caption | — | **1,529** ✓ | CORRECT |
| Ch6:321 fig:confusion | — | **1,529** ✓ | CORRECT |
| Ch8:33 | "70/15/15" | — | approx |
- Canonical: 70.43/14.28/15.29; test=1,529 (POS 87/NEU 466/CON 682/TOX 294). Tbl 4.4 test composition (100/443/288/649=1,480) contradicts Ch6 Tbl 6.4 (87/466/294/682=1,529, correct).

## 3. LOSS FUNCTION (the most pervasive contradiction)

| Location | Says | Verdict |
|---|---|---|
| **Canonical (retrain script)** | **Focal Loss γ=2.0 + class-weighted α + WeightedRandomSampler** | — |
| Ch1:76-77 (SMART) | "SMOTE + weighted loss" for MN-BERT | WRONG |
| Ch1:133-134 (H2) | "weighted loss MN-BERT-д" | incomplete |
| Ch1:167 (novelty) | "FocalLoss + cosine scheduler" | ✓ CORRECT |
| Ch3:356-364 Tbl 3.1 | Loss = Cross-entropy / Binary CE | WRONG (no Focal) |
| Ch3:384-387, 428-429 | "class-weighted loss ... inverse frequency" | incomplete (no Focal, no formula) |
| Ch4:120, 301-302 | "class-weighted loss + SMOTE" | WRONG |
| Ch5:109 fig (arch) | "Linear → C (Focal Loss)" | ✓ CORRECT |
| Ch5:161 Tbl 5.1 row5 | "cross-entropy loss тооцох" | WRONG |
| Ch5:193-201 | Focal Loss for flat (pivot section) | ✓ CORRECT |
| Ch5:269-277 | "MN-BERT-д ... cross-entropy loss-ыг ашиглав; w_c=N/(K·n_c)" | WRONG (it's Focal α, not CE weight) — formula itself is right |
| Ch5:458-459 (summary) | "жин оногдуулсан cross-entropy BERT-д" | WRONG |
| Ch6:263 Tbl 6.4 caption | "Focal Loss + Cosine Warmup" | ✓ CORRECT |
| Ch7:99 | "жин оногдуулсан cross-entropy (MN-BERT-д)" | WRONG |
| Ch7:218-220 (AI disc.) | "FocalLoss-ыг torch.gather ашиглан" | ✓ CORRECT |
| Ch8:38 | "SMOTE + жин оногдуулсан алдааны функц" | WRONG |
| Abstract:26 / researchoverview:21 | "Focal Loss + Cosine Warmup" | ✓ CORRECT |
- **Canonical story to enforce everywhere:** classical baselines (LR/NB/SVM) use SMOTE on
  TF-IDF; **the final flat MN-BERT uses Focal Loss (γ=2.0) with class-weighted α
  (αc = N/(K·nc)) plus a WeightedRandomSampler.** The two-stage models used weighted CE
  (historical) — keep that scoped to the two-stage description only.

## 4. MAX SEQUENCE LENGTH

| Location | Says | Verdict |
|---|---|---|
| **Canonical** | **256** | — |
| Ch3:281, 359; Ch3:483 fig | 128 | WRONG (two-stage used 128; final flat = 256) |
| Ch5:19-21 prose | "128 token ... ихэнх энэ хязгаарт багтдаг" | WRONG |
| Ch5:158 Tbl 5.1 | "max_length=128" | WRONG |
| Ch5:226 Tbl 5.2 | 128 / 128 | WRONG (flat = 256) |
| Ch5:298 Gradio prose | "max_length = 256" (flat) | ✓ CORRECT |
| Ch5:419-421 fig caption | flat 256 / two-stage 128 | ✓ CORRECT |
- `_defense_numbers.json`: max_len 256 = "faithful_256" → 0.8326; max_len 128 = "app_skew_128" → only 0.8136. The 128 is a known app-side skew/bug, not the reported model.

## 5. HYPERPARAMETERS — three conflicting tables

| Param | **Canonical (flat SOTA)** | Tbl 3.1 (Ch3:356-364) | Tbl 5.2 S1/S2 (Ch5:220-229) |
|---|---|---|---|
| Learning rate | **3e-5** | 2e-5 | 3e-5 / 2e-5 |
| Batch size | 16 | 16 | 16 / 16 |
| Max seq length | **256** | 128 | 128 / 128 |
| Dropout | **0.15** | — | 0.1 / 0.1 |
| Epochs (best) | **8** (max 10) | 3–5 | 6 / 5 |
| Loss | **Focal γ=2.0** | CE / BCE | (not listed) |
| Seed | **42** | — | — |
| Sampler | **WeightedRandomSampler** | — | — |
| Warmup ratio | 0.10 | 10% | 0.1 |
| Weight decay | 0.01 | 0.01 | 0.01 |
- Neither table describes the **final flat model**; both describe the abandoned two-stage.
  Fix: add/replace with a flat-model hyperparameter table matching canonical.

## 6. TWO-STAGE CUMULATIVE (END-TO-END) METRIC — internal 3-way contradiction

| Location | Value | Verdict |
|---|---|---|
| **Canonical (SESSION_REPORT §6)** | **0.7861 / 78.61%** | — |
| researchoverview:24-25 | 78.61% | ✓ |
| Abstract:26 | (Stage-2 91.44% only; cumulative omitted) | MISSING caveat |
| Ch5:299, 354, 420, 459 | 78.61% | ✓ |
| Ch6:258-259 text | "нэгдсэн ... Macro F1 0.7598 ... 4.64 нэгжээр доогуур" | WRONG (0.7598) |
| Ch6:416 Tbl 6.6 | End-to-End F1 **0.7760** | WRONG (0.7760 = also the *original flat* macro-F1 — likely mis-copied) |
| Ch6:451 fig:comparison-bar | End-to-End 0.7760 | WRONG |
- Pick **0.7861** everywhere (or the value the author confirms from a two-stage pipeline
  run). Remove 0.7598 and 0.7760-as-end-to-end.

## 7. STAGE-2 91.44% PRESENTATION

| Location | Has cumulative (78.61%) caveat next to 91.44%? |
|---|---|
| researchoverview:23-25 | ✓ yes |
| Ch5:299 / Ch6:257-258 | ✓ yes (context present) |
| **Abstract:26** | ✗ **NO** — "Stage 2 ... 91.44% ... системийн практик ач холбогдлыг батлав" with no cumulative caveat → **misleading**; two-stage was rejected |

## 8. "BEATS PREVIOUS WORK 82.70%" / "16% vs TF-IDF"

| Location | Claim | Verdict |
|---|---|---|
| Canonical | flat **acc 0.8326 > prev-dip-1 acc 0.8270** (+0.56 pt); best baseline acc 0.6691 → +16.35 pt | valid (accuracy vs accuracy) |
| Ch6:503-505, Tbl 6.7 | 0.8326 vs 0.8270 (accuracy) | ✓ CORRECT framing |
| Abstract:26 | "0.8062 Macro-F1 ... TF-IDF-аас дунджаар 16% илүү, өмнөх ... (82.70%) давсан" | MISLEADING — juxtaposes F1 (0.8062) with accuracies (82.70%, "16%"); "дунджаар" wrong (it's vs *best* baseline acc) |
| Ch8:47-48,180 | "0.8062 Macro-F1 ... (82.70%) давсан" | same F1-vs-acc juxtaposition |
| Ch8:91 | "+16.35%" vs best baseline 66.91% | ✓ correct (accuracy) |
- Note: with the *uncorrected* numbers (0.8143) the "beats 82.70%" claim fails — this is
  why the label-correction disclosure (§ AUDIT A1) is mandatory.

## 9. IAA / ANNOTATION PROTOCOL — Ch4 vs Ch7 materially different

| Aspect | Ch4 (167-202) | Ch7 (49-76) | Ch2:94-95 / Ch1:168 |
|---|---|---|---|
| Who labels what | 3 annotators label **ALL 10,000** independently | first 200 joint; **≥20% double**-annotated by 2 | "2+ annotators" |
| κ metric named | **"Cohen's κ = 0.72"** | "Cohen's (2) / Fleiss' (>2); κ=0.72" | "Cohen's Kappa" |
| Disagreement resolved | discussion among the **3** | by a **third** experienced annotator | — |
- With 3 annotators on all data the correct metric is **Fleiss' κ** (or Krippendorff's α),
  not Cohen's. Pick ONE protocol+metric and make Ch1/Ch2/Ch4/Ch7 agree. κ value 0.72 is
  consistent everywhere (only the metric name and protocol differ).

## 10. DATA SOURCES — prose (3) vs Table 4.3 (8, correct)

| Location | Says | Verdict |
|---|---|---|
| **Canonical / Ch4 Tbl 4.3** | **8 sources** (news.mn, gogo.mn, IKON.mn, E-Mongolia, "Мэдэхгүй зүйлээ асуу", Medee.mn, Zaluusinfo, Eagle News) | ✓ Tbl 4.3 CORRECT |
| Ch4:10-13, Tbl 4.1 "Source" col | "news.mn, Facebook, gogo.mn" (3) | WRONG/misleading |
| Ch1:91-93,151-152; Ch3:136,168; Ch7:16-18; Ch8:31,104; Abstract; researchoverview; diagrams | "news.mn, Facebook, gogo.mn" | INCONSISTENT with Tbl 4.3 + data |
| Ch7:22 | "Хоёр эх сурвалж" (says **two**, lists three) | numbering slip |
- "Facebook" is not a `source` value; only one FB group ("Мэдэхгүй зүйлээ асуу", 632 = 6.3%);
  ~91% is news-site comment sections. Align all prose to the 8-source reality.

## 11. SAMPLE-LENGTH ANALYSIS (Ch4 fig:len-dist)

| Class | Thesis fig (Ch4:354-359, "avg") | Canonical mean / median |
|---|---|---|
| Хортой сөрөг | 54 | 129 / 93 |
| Саармаг | 62 | 90 / 59 |
| Эерэг | 82 | 138 / 90 |
| Бүтээлч шүүмжлэл | 150 | 184 / 137 |
- Numbers and ordering both wrong; only "Constructive longest" survives. Professor also
  flagged this figure as unclear. Replace with real per-class median (or mean).

## 12. MISC CROSS-REFERENCES

| Item | Canonical | Thesis | Verdict |
|---|---|---|---|
| Prev-dip-1 F1 | 0.80 (SESSION_REPORT) / 0.81 weighted (Ch2 Tbl 2.2) | Ch6 Tbl 6.7 "0.8000"; Ch2 Tbl 2.2 "0.81" | minor inconsistency — align |
| Stage-1 acc 0.8306 vs flat 0.8326 | distinct (3-cls vs 4-cls) | Ch6 6.1 (0.8306) vs 6.4 (0.8326) | OK but conflation risk — keep labels explicit |
| Baseline Toxic F1 0.7487 | not in any seen artifact | Ch6:311 | UNVERIFIED — provenance needed |
| Stage-1 FN=87 / FP=145 | not in any seen artifact | Ch6:247-252 | UNVERIFIED — provenance needed |
| Inference "2× faster" | not measured in artifacts | Ch5:198 | UNSUPPORTED — soften |
| Inference 20–35 ms GPU / 200–300 ms CPU | not measured in artifacts | Ch5:318 | UNSUPPORTED — soften or measure |
| STILTs expansion | "Supplementary Training on Intermediate Labeled-data Tasks" (Phang 2018) | Ch6:176 "Sequential Transfer of Information and Language Skills" | WRONG expansion |
| Datareportal year | bib `datareportal2024` (2024) | Ch1:16 "DataReportal (2025)" plain text | mismatch + not \cited |
