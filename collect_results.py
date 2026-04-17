import argparse
import json
import re
import sys
import time

import anthropic
import pandas as pd

LABEL_RE = re.compile(r"(\d+)\.\s*(POSITIVE|TOXIC|CONSTRUCTIVE|NEUTRAL)\s*\|\s*(.+)")

def poll_until_done(client: anthropic.Anthropic, batch_id: str) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"Status: {batch.processing_status} | "
            f"succeeded: {counts.succeeded}, errored: {counts.errored}, "
            f"expired: {counts.expired}, canceled: {counts.canceled}, "
            f"processing: {counts.processing}"
        )
        if batch.processing_status == "ended":
            return
        time.sleep(30)

def parse_labels(text: str) -> list[tuple[int, str, str]]:
    results = []
    for line in text.strip().split("\n"):
        m = LABEL_RE.match(line.strip())
        if m:
            idx = int(m.group(1))
            label = m.group(2)
            reasoning = m.group(3).strip()
            results.append((idx, label, reasoning))
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Collect batch classification results and merge into CSV"
    )
    parser.add_argument(
        "--output",
        default="labeled_output.csv",
        help="Path for labeled output CSV (default: labeled_output.csv)",
    )
    args = parser.parse_args()

    try:
        with open("batch_tracker.json", "r", encoding="utf-8") as f:
            tracker = json.load(f)
    except FileNotFoundError:
        print("Error: batch_tracker.json not found. Run submit_batch.py first.", file=sys.stderr)
        sys.exit(1)

    batch_id = tracker["batch_id"]
    input_file = tracker["input_file"]
    comment_col = tracker["comment_col"]
    total_comments = tracker["total_comments"]

    print(f"Batch ID: {batch_id}")
    print(f"Input file: {input_file}")
    print(f"Total comments: {total_comments}")
    print(f"Polling for completion...\n")

    client = anthropic.Anthropic()

    poll_until_done(client, batch_id)
    print("\nBatch completed. Collecting results...\n")

    errors = []
    total_input_tokens = 0
    total_output_tokens = 0

    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            msg = result.result.message
            total_input_tokens += msg.usage.input_tokens
            total_output_tokens += msg.usage.output_tokens

            text_parts = []
            for block in msg.content:
                if block.type == "text":
                    text_parts.append(block.text)

            full_text = "\n".join(text_parts)
            parsed = parse_labels(full_text)
            for idx, label, reasoning in parsed:
                labels[idx] = (label, reasoning)
        else:
            errors.append({
                "custom_id": result.custom_id,
                "type": result.result.type,
            })

    print(f"Token usage:")
    print(f"  Input tokens:  {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")

    input_cost = (total_input_tokens / 1_000_000) * BATCH_INPUT_PRICE_PER_M
    output_cost = (total_output_tokens / 1_000_000) * BATCH_OUTPUT_PRICE_PER_M
    total_cost = input_cost + output_cost
    print(f"\nEstimated cost (Batch pricing):")
    print(f"  Input:  ${input_cost:.4f}")
    print(f"  Output: ${output_cost:.4f}")
    print(f"  Total:  ${total_cost:.4f}")

    df = pd.read_csv(input_file)
    df["label"] = None
    df["reasoning"] = None

    labeled_count = 0
    for idx, (label, reasoning) in labels.items():
        if 0 <= idx < len(df):
            df.at[idx, "label"] = label
            df.at[idx, "reasoning"] = reasoning
            labeled_count += 1

    label_series = df["label"].dropna()
    print(f"\nLabeled: {labeled_count} / {total_comments}")
    print(f"\nLabel distribution:")
    for lbl, count in label_series.value_counts().items():
        pct = count / labeled_count * 100 if labeled_count > 0 else 0
        print(f"  {lbl}: {count} ({pct:.1f}%)")

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nSaved labeled output to {args.output}")

    unlabeled = df[df["label"].isna()]
    if len(unlabeled) > 0:
        retry_file = args.output.replace(".csv", ".retry.csv")
        unlabeled.to_csv(retry_file, index=False, encoding="utf-8-sig")
        print(f"Saved {len(unlabeled)} unlabeled rows to {retry_file}")

    if errors:
        errors_file = args.output.replace(".csv", ".errors.json")
        with open(errors_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(errors)} errors to {errors_file}")

    if not errors and len(unlabeled) == 0:
        print("\nAll comments labeled successfully!")

if __name__ == "__main__":
    main()
