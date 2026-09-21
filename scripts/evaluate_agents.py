#!/usr/bin/env python
"""Evaluate every CLOSE agent's output against the rule-based reference pipeline.

Capture is scored against `data/gold_labels/`; Link, Orchestrate, and the
end-to-end status are scored against a full rule-based run of the same cohort.

Example:
  PYTHONPATH=src python scripts/evaluate_agents.py \\
    --reference outputs/reference_rule_based/close_results.jsonl \\
    --predictions outputs/model_comparison_9500762/qwen/close_results.jsonl \\
    --gold data/gold_labels/gold_labels.jsonl \\
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

from close_framework.evaluation.agent_eval import evaluate_agents  # noqa: E402
from close_framework.evaluation.gold_eval import (  # noqa: E402
    evaluate_predictions_against_gold,
    load_jsonl_by_report_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_name = args.model_name or args.predictions.parent.name
    predictions = load_jsonl_by_report_id(args.predictions)
    payload = evaluate_agents(load_jsonl_by_report_id(args.reference), predictions, model_name)
    if args.gold:
        payload["capture"] = evaluate_predictions_against_gold(
            load_jsonl_by_report_id(args.gold), predictions, model_name
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
