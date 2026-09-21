"""Tests for Orchestrate Agent."""

from __future__ import annotations

from close_framework.agents.orchestrate_agent import OrchestrateAgent
from close_framework.data.schemas import IncidentalFinding


def test_missing_exact_guideline_routes_to_human_review():
    finding = IncidentalFinding(
        finding_id="f1",
        organ_system="adrenal",
        finding_type="nodule",
        description="A 1.2 cm adrenal nodule is indeterminate.",
        actionability="actionable",
        urgency_tier="routine",
        evidence_span="A 1.2 cm adrenal nodule is indeterminate.",
        confidence=0.7,
    )

    recommendation = OrchestrateAgent().run(finding)

    assert recommendation.human_review_required is True
    assert recommendation.guideline_name == "Management of Incidental Adrenal Masses"
    assert recommendation.rule_id.startswith("ADRENAL")
    assert recommendation.order_set_stub == "CLOSE_ADRENAL_FOLLOW_UP_REVIEW"
