"""Workflow status helpers."""

from __future__ import annotations

from close_framework.data.schemas import FinalStatus


def choose_final_status(has_findings: bool, has_recommendations: bool, human_review_required: bool) -> FinalStatus:
    """Choose a report-level workflow status from pipeline outcomes."""

    if human_review_required:
        return "human_review_required"
    if has_recommendations:
        return "recommendation_generated"
    if not has_findings:
        return "no_actionable_finding"
    return "no_actionable_finding"
