# CITATION GAPS

Every claim that needs a citation but lacks one, every broken `\cite`, and every
bibliography hygiene issue. English (study aid). Bib file: `references.bib` (biblatex,
backend=bibtex).

---

## A. BROKEN `\cite` — render as bold `?` (CRITICAL, from `main.log:2101-2112`)

| `\cite{key}` in text | File:line | Problem | Fix |
|---|---|---|---|
| `fan2021` | Ch2:366 | bib key is `fan2021toxic` | rename cite → `fan2021toxic` (or bib key → `fan2021`) |
| `ranasinghe2021` | Ch2:378 | **no bib entry at all** | add new entry (below) |
| `nabiilah2023` | Ch2:383 | bib key is `nabiilah2023bert` | rename cite → `nabiilah2023bert` |
| `akhter2022` | Ch2:392 | bib key is `akhter2022multi` | rename cite → `akhter2022multi` |

Recommended: change the **`\cite` calls** in Ch2 to the existing bib keys
(`fan2021toxic`, `nabiilah2023bert`, `akhter2022multi`) — least disruptive — and add a
new entry for `ranasinghe2021`:

```bibtex
@inproceedings{ranasinghe2021,
  author    = {Tharindu Ranasinghe and Marcos Zampieri},
  title     = {Multilingual Offensive Language Identification for Low-resource Languages},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2021},
  pages     = {3105--3111},
  year      = {2021}
}
```
(Verify the exact venue/pages against the real paper before final submission.)

---

## B. PLAIN-TEXT "citations" not using `\cite` (claim unsupported as written)

| Author-year in prose | File:line | Needs |
|---|---|---|
| "DataReportal (2025)" — Mongolia 83% internet penetration stat | Ch1:16-17 | `\cite{datareportal2024}` (entry exists, **orphan**) — and fix year (bib says 2024; text says 2025 — make consistent) |
| "Phang et al., 2018" (STILTs) | Ch6:176 | add `@article{phang2018,…}` + `\cite`; **also fix the wrong acronym expansion** → "Supplementary Training on Intermediate Labeled-data Tasks" |
| "GloVe (Pennington et al., 2014)" | Ch2:195 | add `@inproceedings{pennington2014,…}` + `\cite` |
| "FastText (Bojanowski et al., 2017)" | Ch2:195 | add `@article{bojanowski2017,…}` + `\cite` |
| "Davidson et al. (2017) … 90.8%" | Ch2:323-326 | `\cite{davidson2017}` is present nearby — verify the 90.8% figure matches the cited paper (Davidson 2017 reports ~0.90 weighted F1; check before defense) |
| Mongolian laws: "Хувь хүний нууцын тухай хууль (2021)", "Цахим гарын үсэг…хууль" | Ch7:26-28 | acceptable as legal references in prose; optionally add to bib as `@misc`. LOW |
| "Perspective API (Jigsaw/Google, 2016)", "Kaggle Toxic Comment Challenge (2018)" | Ch2:76-80,469-478 | well-known; optionally cite. LOW |

Suggested entries:
```bibtex
@inproceedings{pennington2014,
  author={Jeffrey Pennington and Richard Socher and Christopher D. Manning},
  title={{GloVe}: Global Vectors for Word Representation},
  booktitle={Proceedings of EMNLP 2014}, pages={1532--1543}, year={2014}}

@article{bojanowski2017,
  author={Piotr Bojanowski and Edouard Grave and Armand Joulin and Tomas Mikolov},
  title={Enriching Word Vectors with Subword Information},
  journal={Transactions of the ACL}, volume={5}, pages={135--146}, year={2017}}

@article{phang2018,
  author={Jason Phang and Thibault F\'evry and Samuel R. Bowman},
  title={Sentence Encoders on {STILTs}: Supplementary Training on Intermediate
         Labeled-data Tasks},
  journal={arXiv preprint arXiv:1811.01088}, year={2018}}
```

---

## C. CLAIMS THAT SHOULD CARRY A CITATION OR EVIDENCE (rhetorical, currently bare)

| Claim | File:line | Why it needs support | Suggested action |
|---|---|---|---|
| "MN-BERT outperforms TF-IDF because of Mongolian pre-training + SentencePiece" | Ch6:459-475 | stated as explanation, not result | frame as "хүлээж байна / тайлбарлаж болно" (already hedged with "гэж хүлээж байна" at :459-460 — keep that hedge; optionally cite caselli2021/nabiilah2023bert which you already use for this exact argument) |
| "Pre-training corpora handle agglutination better" | Ch2:38-42, Ch6:463-466 | linguistic claim | cite nabiilah2023bert (IndoBERT, agglutinative, already in text) + devlin2019 |
| "First / одоогоор боловсруулагдаагүй" (no Mongolian Toxic-vs-Constructive system) | researchoverview:10-12, Ch1:46-52,162, Ch8:22 | absolute novelty claim | tie explicitly to Ch2 §2.8.4 / §2.9.1 survey + soften to "бидний мэдэхээр"; cite jargalmaa2025, mnunlp2021 as the closest prior |
| "GDPR principles followed" | Ch7:33-38 | jurisdiction nuance | reword "inspired by" not "compliance"; no citation strictly needed |
| Cyberbullying psychological harm | Ch1:27-29 | already cited `\cite{kowalski2014,vogels2022}` ✓ | OK |
| TAM model | Ch2:137 | already cited `\cite{davis1989}` ✓ | OK |

---

## D. BIBLIOGRAPHY HYGIENE

| Entry | Status | Action |
|---|---|---|
| `datareportal2024` | defined, **never `\cite`d** (orphan); text refers to it as plain "DataReportal (2025)" with wrong year | `\cite` it in Ch1:16; reconcile year (2024 vs 2025) |
| `fan2021toxic` | defined, **never `\cite`d** (text uses wrong key `fan2021`) | fix the cite key (A) |
| `nabiilah2023bert` | defined, never `\cite`d (text uses `nabiilah2023`) | fix the cite key (A) |
| `akhter2022multi` | defined, never `\cite`d (text uses `akhter2022`) | fix the cite key (A) |
| `mnunlp2021` | cited Ch2:453; institutional report, no URL | OK; optionally add note/URL |
| `jargalmaa2025` | cited (Ch2:263,269,287,461) ✓ — the 82.70% source | OK; ensure F1 figure (0.80 vs 0.81) consistent with Ch6 Tbl 6.7 |
| Header comment in `references.bib:4-10` | TODO comment listing the 7 "missing" refs — but `caselli/mathew/davidson` ARE defined; `fan/nabiilah/akhter` defined under *different keys*; only `ranasinghe` truly missing | clean the comment after fixing keys |
| arXiv vs published | `phang2018` (to add) is arXiv-only — mark as arXiv; check `fan2021toxic` (a CS224N report, not peer-reviewed) — acceptable but note its nature | keep but be ready to defend `fan2021toxic`'s status (course report) |
| Style consistency | mixed `@inproceedings/@article/@thesis/@techreport/@misc`, names mostly "First Last" | acceptable for biblatex; verify final rendered list is uniform after rebuild |

---

## E. POST-FIX VERIFICATION

After applying A–D, rebuild and confirm `main.log` shows **zero**
`Citation '...' undefined` and **no** "There were undefined references."

```
pdflatex main  →  bibtex main  →  pdflatex main  →  pdflatex main
```
Then grep the log: `Warning.*undefined` must return nothing.
