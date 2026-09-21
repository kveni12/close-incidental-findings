"""Evaluation runner scaffolding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from close_framework.data.schemas import WorkflowResult
from close_framework.evaluation.gold_eval import evaluate_predictions_against_gold, load_jsonl_by_report_id
from close_framework.evaluation.metrics import descriptive_summary_metrics


def run_descriptive_evaluation(results: Iterable[WorkflowResult]) -> Dict[str, Any]:
    """Return descriptive metrics (always available)."""

    return descriptive_summary_metrics(results)


def run_gold_evaluation(
    gold_path: str | Path,
    predictions_path: str | Path,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate Capture predictions against gold/reference labels."""

    name = model_name or Path(predictions_path).parent.name
    return evaluate_predictions_against_gold(
        load_jsonl_by_report_id(Path(gold_path)),
        load_jsonl_by_report_id(Path(predictions_path)),
        name,
    )
