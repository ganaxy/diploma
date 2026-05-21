"""End-to-end smoke test for the Gradio app modules.

Run as:    python -m gradio_app._smoke_test    (from the diplom folder)

Checks:
  1. Both classifiers load.
  2. Each of the 4 example texts predicts a label and produces a 4-class projection.
  3. Token attributions return at least one (token, score) tuple.
  4. The bar chart renders to PNG bytes without erroring.
  5. The batch handler accepts a CSV with a `text` column.

This is a developer aid; not loaded by the Gradio runtime.
"""

import io
import os
import sys
from pathlib import Path

# Ensure Cyrillic prints don't crash the Windows console
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Make `gradio_app` importable when run as `python -m gradio_app._smoke_test`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from gradio_app import config
from gradio_app.app import (
    ARCH_FLAT,
    ARCH_TWO_STAGE,
    _bar_chart,
    _four_class_view,
    predict_batch,
    predict_single,
)


class _FakeFile:
    """Minimal stand-in for the gr.File object (only `.name` is used)."""
    def __init__(self, name: str):
        self.name = name


def _check_single(text: str, arch: str) -> None:
    headline, fig, highlights, explanation = predict_single(text, arch)
    assert headline and headline.strip(), "empty headline"
    assert fig is not None, "no chart returned"
    assert isinstance(highlights, list) and len(highlights) > 0, "no highlights"
    assert explanation and "Эцсийн ангилал" in explanation, "explanation missing label header"
    print(f"  [{arch}] '{text[:40]}...' -> {headline.splitlines()[0].strip('# ')}")


def main() -> None:
    print("=" * 60)
    print(" SMOKE TEST")
    print("=" * 60)

    # --- 1. Single-text path on both architectures ---
    print("\n[1] Single-text predictions:")
    for label_mn, sample in config.EXAMPLES:
        for arch in (ARCH_FLAT, ARCH_TWO_STAGE):
            _check_single(sample, arch)

    # --- 2. Bar chart render ---
    print("\n[2] Bar chart renders to PNG:")
    fig = _bar_chart({"POSITIVE": 0.7, "NEUTRAL": 0.1, "TOXIC": 0.05, "CONSTRUCTIVE": 0.15},
                     "Тест диаграм")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    print(f"  ok — {len(buf.getvalue())} bytes")

    # --- 3. Batch CSV handler ---
    print("\n[3] Batch handler on a 4-row CSV:")
    csv_path = Path(__file__).parent / "_smoke_batch.csv"
    pd.DataFrame({"text": [s for _, s in config.EXAMPLES]}).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    status, table, dist_fig, download_path = predict_batch(_FakeFile(str(csv_path)), ARCH_FLAT)
    print(f"  status: {status}")
    assert isinstance(table, pd.DataFrame) and len(table) == 4
    assert dist_fig is not None
    assert download_path and Path(download_path).exists()
    print(f"  table rows: {len(table)}, download: {download_path}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
