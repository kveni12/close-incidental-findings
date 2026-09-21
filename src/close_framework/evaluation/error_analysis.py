"""Error analysis scaffolding for manual review."""

from __future__ import annotations

from typing import List

from close_framework.data.schemas import WorkflowResult


def collect_human_review_cases(results: List[WorkflowResult]) -> List[WorkflowResult]:
    """Return reports that require human review for future qualitative analysis."""

    return [result for result in results if result.final_status == "human_review_required"]
