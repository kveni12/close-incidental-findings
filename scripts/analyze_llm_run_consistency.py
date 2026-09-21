#!/usr/bin/env python
"""Placeholder / harness for multi-run LLM Capture consistency (expensive).

Does NOT run models by default. When given multiple JSONL generations for the
same reports, computes run-to-run agreement on presence and actionability.

Example (after producing run_a/run_b offline):
  PYTHONPATH=src python scripts/analyze_llm_run_consistency.py \\
    --runs run_a.jsonl run_b.jsonl run_c.jsonl
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.evaluation.gold_eval import (  # noqa: E402
    load_jsonl_by_report_id,
    prediction_findings,
    report_actionability_category,
    has_actionable_prediction,
)
from close_framework.evaluation.metrics import binary_metrics, cohen_kappa  # noqa: E402
from close_framework.evaluation.gold_eval import ACTIONABILITY_SCALE  # noqa: E402


def pairwise_agreement(a: dict, b: dict) -> dict:
    ids = sorted(set(a) & set(b))
    presence_a = [bool(prediction_findings(a[i])) for i in ids]
    presence_b = [bool(prediction_findings(b[i])) for i in ids]
    action_a = [has_actionable_prediction(a[i]) for i in ids]
    action_b = [has_actionable_prediction(b[i]) for i in ids]
    cat_a = [report_actionability_category(prediction_findings(a[i])) for i in ids]
    cat_b = [report_actionability_category(prediction_findings(b[i])) for i in ids]
    return {
        "n": len(ids),
        "presence": binary_metrics(presence_a, presence_b),
        "actionable": binary_metrics(action_a, action_b),
        "actionability_linear_kappa": cohen_kappa(
            cat_a, cat_b, labels=list(ACTIONABILITY_SCALE), weights="linear"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    loaded = {str(path): load_jsonl_by_report_id(path) for path in args.runs}
    pairs = []
    for (name_a, data_a), (name_b, data_b) in combinations(loaded.items(), 2):
        pairs.append({"run_a": name_a, "run_b": name_b, **pairwise_agreement(data_a, data_b)})
    print(json.dumps({"pairwise": pairs, "note": "Requires independently generated Capture JSONLs."}, indent=2))


if __name__ == "__main__":
    main()
