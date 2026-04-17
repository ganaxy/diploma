import argparse
import json
import math
import sys

import anthropic
import pandas as pd

SYSTEM_PROMPT = """You are a senior Mongolian NLP researcher and expert data annotator classifying social media comments.

You use deep semantic reasoning — not keyword matching, not sentiment polarity, not profanity detection.

<labels>
POSITIVE — Main force is support, gratitude, encouragement, praise, hope, patriotic pride, or morally uplifting sentiment.

TOXIC — Main force is harm: insults, humiliation, dehumanization, profanity-as-attack, hate speech, xenophobia, hostile mockery, wishing harm, baseless accusations fueling hostility, or identity/dignity attacks.

CONSTRUCTIVE — Contains criticism/complaint BUT identifies a real problem (system failure, policy issue, service defect, governance problem) a decision-maker could act on. Can be angry or harsh — if a real issue is identifiable, it is CONSTRUCTIVE.

NEUTRAL — Factual, descriptive, emotionally flat, casual filler, simple observation. No meaningful encouragement, harm, or actionable criticism.
</labels>

<decision_logic>
1. INTENT — Supporting → POSITIVE | Attacking → TOXIC | Criticizing meaningfully → CONSTRUCTIVE | Stating → NEUTRAL
2. TARGET — Attacking person/group dignity → TOXIC. Criticizing system/service/policy → CONSTRUCTIVE.
3. SIGNAL — Only emotional aggression, no issue → TOXIC. Frustration + real problem → CONSTRUCTIVE.
4. ACTIONABILITY — Could a policymaker learn something useful? Yes → CONSTRUCTIVE. No, mainly degrades → TOXIC.
5. RESIDUAL — Not clearly positive, toxic, or constructive → NEUTRAL.
</decision_logic>

<boundary_rules>
- Angry ≠ TOXIC. Harsh criticism with a real issue = CONSTRUCTIVE.
- Insults aimed at person's identity/dignity = TOXIC, even if framed as "criticism."
- Sarcasm: label by function. Humiliates person → TOXIC. Criticizes real issue → CONSTRUCTIVE.
- Short comments: infer communicative function. Supportive → POSITIVE. Dismissive mockery → TOXIC. Filler → NEUTRAL.
- If BOTH toxic attack AND constructive feedback exist, label by the DOMINANT force.
</boundary_rules>

<output_format>
For EACH numbered comment, return EXACTLY one line in this format:
[number]. [LABEL] | [1-sentence reasoning in English]

NEVER skip a comment. NEVER output anything except the numbered label lines. No preamble, no summary.
</output_format>"""

GROUP_SIZE = 10

def build_requests(df: pd.DataFrame, comment_col: str) -> list[dict]:
    comments = df[comment_col].astype(str).tolist()
    total = len(comments)
    num_groups = math.ceil(total / GROUP_SIZE)
    requests = []

    for g in range(num_groups):
        start = g * GROUP_SIZE

        lines = []
        for i in range(start, end + 1):
            lines.append(f'{i}. "{comments[i]}"')
        user_text = "Classify the following comments:\n\n" + "\n".join(lines)

        custom_id = f"group-{g:05d}-rows-{start}-{end}"

        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": "claude-opus-4-6",
                "max_tokens": 16000,
                "thinking": {"type": "adaptive"},
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_text}
                ],
            },
        })

    return requests

def main():
    parser = argparse.ArgumentParser(
        description="Submit Mongolian comment classification batch job"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--comment-col",
        default="comment",
        help="Name of the column containing comments (default: comment)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.comment_col not in df.columns:
        print(f"Error: column '{args.comment_col}' not found in CSV.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    total_comments = len(df)
    requests = build_requests(df, args.comment_col)

    print(f"Total comments: {total_comments}")
    print(f"Number of requests: {len(requests)} (groups of {GROUP_SIZE})")
    print(f"Model: claude-opus-4-6")
    print(f"Submitting batch...")

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)

    tracker = {
        "batch_id": batch.id,
        "input_file": args.input,
        "comment_col": args.comment_col,
        "total_comments": total_comments,
        "num_requests": len(requests),
        "model": "claude-opus-4-6",
    }

    with open("batch_tracker.json", "w", encoding="utf-8") as f:
        json.dump(tracker, f, indent=2, ensure_ascii=False)

    print(f"\nBatch submitted successfully!")
    print(f"Batch ID: {batch.id}")
    print(f"Tracking info saved to batch_tracker.json")

if __name__ == "__main__":
    main()
