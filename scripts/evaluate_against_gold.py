#!/usr/bin/env python
"""Evaluate CLOSE Capture outputs against gold/reference labels.

Computes accuracy, Cohen's kappa, sensitivity, specificity, PPV, NPV, and F1 for:
  - report-level in-scope finding presence
  - report-level actionable finding presence
  - per-organ presence (kidney/liver/adrenal/pancreas/adnexal)

Example:
  PYTHONPATH=src python scripts/evaluate_against_gold.py \\
    --gold data/gold_labels/gold_labels.jsonl \\
    --predictions outputs/model_comparison_9500762/qwen/close_results.jsonl \\
    --model-name Qwen3.6-35B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.evaluation.gold_eval import (  # noqa: E402
    evaluate_predictions_against_gold,
    load_jsonl_by_report_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    model_name = args.model_name or args.predictions.parent.name
    metrics = evaluate_predictions_against_gold(
        load_jsonl_by_report_id(args.gold),
        load_jsonl_by_report_id(args.predictions),
        model_name,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
