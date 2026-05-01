# Mongolian Comment Classifier (Batch API)

Classifies Mongolian social media comments into four categories using the Anthropic Batch API with Claude Opus 4.6.

**Labels:** POSITIVE, TOXIC, CONSTRUCTIVE, NEUTRAL

## Setup

```bash
pip install anthropic pandas
```

Set your API key:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

On Windows (PowerShell):
```powershell
$env:ANTHROPIC_API_KEY = "your-key-here"
```

## Usage

### Step 1: Submit batch job

```bash
python submit_batch.py --input sampled_batch_2.csv --comment-col text_light_clean
```

Arguments:
- `--input` (required): Path to input CSV file
- `--comment-col` (default: `comment`): Name of the column containing comments

This creates `batch_tracker.json` with the batch ID and metadata.

### Step 2: Collect results

```bash
python collect_results.py --output labeled_output.csv
```

Arguments:
- `--output` (default: `labeled_output.csv`): Path for the labeled output CSV

This script:
1. Polls the batch job every 30 seconds until complete
2. Parses labels and reasoning from model responses
3. Merges results into the original CSV with `label` and `reasoning` columns
4. Prints token usage, cost estimate, and label distribution
5. Saves unlabeled rows to `.retry.csv` and errors to `.errors.json` if any

## Processing multiple CSV files

Run submit + collect for each file separately:

```bash
python submit_batch.py --input sampled_batch_2.csv --comment-col text_light_clean
python collect_results.py --output labeled_batch_2.csv

python submit_batch.py --input sampled_batch_3.csv --comment-col text_light_clean
python collect_results.py --output labeled_batch_3.csv

python submit_batch.py --input sampled_batch_4.csv --comment-col text_light_clean
python collect_results.py --output labeled_batch_4.csv
```

## Output

- `labeled_output.csv` — Original CSV with added `label` and `reasoning` columns
- `*.retry.csv` — Rows that were not labeled (for reprocessing)
- `*.errors.json` — Any batch request errors
