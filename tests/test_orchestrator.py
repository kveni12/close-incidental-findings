"""Tests for Central Orchestrator."""

from __future__ import annotations

from close_framework.data.schemas import RadiologyReport
from close_framework.orchestrator.central_orchestrator import CentralOrchestrator


def test_orchestrator_runs_actionable_finding_to_human_review():
    report = RadiologyReport(
        report_id="r1",
        patient_id="p1",
        report_text="There is an indeterminate 2 cm right adrenal nodule.",
        clinical_setting="ED",
    )

    result = CentralOrchestrator().run_report(report)

    assert result.final_status == "human_review_required"
    assert len(result.capture_result.findings) == 1
    assert len(result.provider_assignments) == 1
    assert len(result.follow_up_recommendations) == 1


def test_orchestrator_no_actionable_finding():
    report = RadiologyReport(
        report_id="r2",
        report_text="No suspicious focal liver lesion. No adrenal nodules.",
    )

    result = CentralOrchestrator().run_report(report)

    assert result.final_status == "no_actionable_finding"
    assert result.provider_assignments == {}
