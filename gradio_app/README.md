# MN-BERT Gradio Demo — Mongolian Comment Classifier

Interactive web UI for the Mongolian social-media comment classifier built in
the bachelor diploma project. Loads the already-trained MN-BERT models from
`..\sample scores\` relative to the copied project folder and exposes both the **Flat 4-class**
model (Chapter 5 "optimal solution") and the **Two-stage** pipeline (Chapter 3
original architecture) with a live toggle.

## Features

- Architecture toggle — Flat (one pass, 4 classes) vs. Two-stage (Stage 1 → conditional Stage 2).
- **Single-text tab** — Mongolian Cyrillic textarea, predicted label in Mongolian,
  confidence bar chart, Input × Gradient token-importance highlights, and a full
  decision-trace explanation panel that shows the Stage 1 / Stage 2 numbers and
  the joint confidence multiplication.
- **Batch tab** — upload a CSV with a `text` or `comment` column, get back a
  table, class-distribution chart, and a downloadable processed CSV.
- Four pre-filled example buttons (one per class).
- Sidebar with model paths, active device, and reported metrics from the thesis.

## Install & run

From the project root:

```bash
pip install -r gradio_app/requirements.txt
python -m gradio_app.app
```

Or from inside this folder:

```bash
cd gradio_app
python app.py
```

On startup, Gradio prints two URLs:

```
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://<random>.gradio.live
```

The public URL is a 72-hour tunnel — share it with anyone who needs access.
Close the terminal to take the server down.

### GPU acceleration (optional)

If you have an AMD GPU on Windows, install the optional DirectML backend:

```bash
pip install torch-directml
```

The app auto-detects it on startup. If absent, it falls back to CUDA, then CPU.

## Files

| File             | Purpose                                                         |
| ---------------- | --------------------------------------------------------------- |
| `app.py`         | Gradio Blocks UI, tabs, event wiring                            |
| `models.py`      | `FlatClassifier` and `TwoStageClassifier` wrappers + device pick |
| `preprocess.py`  | Text cleaning + Cyrillic-ratio diagnostic                       |
| `explain.py`     | Input × Gradient token-importance attributions                  |
| `config.py`      | Absolute model paths, label maps (EN ↔ MN), example texts       |
| `requirements.txt` | Pinned-floor dependency list                                  |

## Notes

- Models are loaded **once at startup** and cached in memory. Inference is
  ~10 ms on DirectML and ~200–300 ms on CPU (per the thesis §5).
- Safetensors / pytorch_model.bin are **referenced by absolute path**, not copied.
- Token importance uses a minimal Input × Gradient approach (multiply the input
  embedding by its gradient w.r.t. the predicted-class logit, sum over the
  hidden dim, take the absolute value, normalise) — produces useful highlights
  for the defense without pulling in Captum.
- This is **inference only**. No training code lives here.
