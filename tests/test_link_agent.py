"""Tests for Link Agent."""

from __future__ import annotations

from close_framework.agents.link_agent import LinkAgent
from close_framework.data.schemas import IncidentalFinding, PatientContext


def _finding() -> IncidentalFinding:
    return IncidentalFinding(
        finding_id="f1",
        organ_system="kidney",
        finding_type="lesion",
        description="Indeterminate renal lesion.",
        actionability="actionable",
        urgency_tier="routine",
        evidence_span="Indeterminate renal lesion.",
        confidence=0.7,
    )


def test_ed_patient_with_no_pcp_routes_to_safety_pool():
    context = PatientContext(patient_id="p1", clinical_setting="ED")
    assignment = LinkAgent().run(_finding(), context)

    assert assignment.provider_type == "safety_pool"
    assert assignment.failsafe_triggered is True
    assert assignment.human_review_required is True


def test_active_pcp_preferred():
    context = PatientContext(patient_id="p1", active_pcp_id="pcp1", active_pcp_name="Dr PCP")
    assignment = LinkAgent().run(_finding(), context)

    assert assignment.provider_type == "PCP"
    assert assignment.responsible_provider_id == "pcp1"
