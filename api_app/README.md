# MN-BERT REST API prototype

This is the deployment/API prototype for the diploma classifier. It exposes the
trained MN-BERT inference pipeline as a reusable service, separate from the
Gradio demo UI.

The default serving path is intentionally the same as the Gradio app's final
Flat MN-BERT model:

- checkpoint: `sample scores/best_mnbert_sota_corrected_model`
- preprocessing: `clean_text` -> `normalize_flat` -> prepend `[news.mn] `
- tokenizer max length: `256`
- reported test metrics: accuracy `0.8326`, macro-F1 `0.8062`, `n=1529`

## Install

From the project root:

```powershell
pip install -r api_app/requirements.txt
```

For a reliable defense demo on CPU:

```powershell
$env:API_FORCE_CPU = "1"
```

On an NVIDIA laptop, such as an RTX 3060 machine, use the separate CUDA setup
script from the project root instead:

```powershell
setup_api_demo_nvidia.bat
run_api_demo_nvidia.bat
```

These scripts create a local `.venv_api_nvidia` environment and install the
CUDA PyTorch wheel. The original `setup_api_demo.bat` / `run_api_demo.bat`
remain the CPU-safe fallback.

## Run

```powershell
uvicorn api_app.main:app --host 127.0.0.1 --port 8000
```

Open the interactive API docs at:

```text
http://127.0.0.1:8000/docs
```

For presenting or quick testing, use the simple demo page instead:

```text
http://127.0.0.1:8000/
```

## Endpoints

- `GET /health`: check model paths and whether models are loaded.
- `POST /predict`: classify one comment.
- `POST /batch`: classify many comments from a JSON list.
- `POST /audit`: classify many comments and return a moderation-style summary.

Supported `architecture` values:

- `flat`: final one-stage MN-BERT classifier. This is the default and the model
  to present as the deployed API path.
- `two_stage`: original hard Stage 1 -> Stage 2 cascade.
- `soft_two_stage`: no-retraining architecture change that always runs the
  existing Stage 1 and Stage 2 models, then combines their probabilities.
- `strict_two_stage`: no-retraining architecture change that only routes to
  Stage 2 when Stage 1 negative probability is at least `0.90`.

`GET /info` and `GET /health` also return a `final_model` block with the exact
checkpoint, max length, preprocessing description, label order, and reported
metrics. Use that as quick evidence that the API and Gradio app are aligned.

## Example request

```powershell
$body = @'
{
  "text": "\u042d\u043d\u044d \u04af\u0439\u043b\u0447\u0438\u043b\u0433\u044d\u044d \u043c\u0430\u0448 \u0443\u0434\u0430\u0430\u043d \u0431\u0430\u0439\u043d\u0430, \u0437\u0430\u0441\u0430\u0445 \u0445\u044d\u0440\u044d\u0433\u0442\u044d\u0439.",
  "architecture": "flat",
  "review_threshold": 0.70
}
'@

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

The example uses JSON Unicode escapes and explicit UTF-8 bytes so Mongolian
Cyrillic text stays intact in Windows PowerShell.

The response contains the predicted label, confidence, four-class probability
distribution, latency, and a moderation action:

- `allow`: non-toxic prediction above the review threshold.
- `flag`: toxic prediction above the review threshold.
- `review`: confidence is below the review threshold, so a human should check it.

## Thesis framing

Use this as the answer to the audit feedback about API deployment:

> The previous Gradio interface was a human-facing demo. The added FastAPI
> prototype exposes the trained MN-BERT classifier as a machine-to-machine REST
> service for batch audit and real-time moderation workflows.

In plain language: yes, this is the "API thing" if the professor meant a
backend/service that other programs can call over HTTP instead of a human-only
Gradio screen. Gradio is the demo UI; FastAPI is the integration/service layer.

## Architecture evaluation

Use this to test whether the no-retraining architecture change actually improves
accuracy on the saved test split:

```powershell
python -m api_app.evaluate_architectures --limit 50
```

Remove `--limit 50` for the full test split:

```powershell
python -m api_app.evaluate_architectures
```

The script saves per-architecture predictions and metrics under:

```text
api_app/eval_outputs/
```

This is the evidence to use in the defense. Do not claim the soft two-stage
architecture is more accurate unless the saved metrics show it.

To search many two-stage decision rules without retraining BERT:

```powershell
python -m api_app.search_two_stage_rules
```

This caches Stage 1 and Stage 2 probabilities once, tunes lightweight decision
rules on the validation split, and reports the selected rules on the test split.

For the small decision-model experiment, train the lightweight classifier on the
training split probabilities and evaluate on the test split:

```powershell
python -m api_app.search_two_stage_rules --tiny-train-split train
```

Useful output files:

- `api_app/eval_outputs/two_stage_rule_search_test_top.csv`
- `api_app/eval_outputs/two_stage_tiny_model_results.csv`
- `api_app/eval_outputs/two_stage_probability_cache.csv`
