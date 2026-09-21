"""Tests for Capture Agent."""

from __future__ import annotations

import pytest

from close_framework.agents.capture_agent import CaptureAgent
from close_framework.data.schemas import RadiologyReport


@pytest.fixture()
def agent() -> CaptureAgent:
    return CaptureAgent()


def _report(text: str) -> RadiologyReport:
    return RadiologyReport(report_id="r-test", report_text=text)


def test_no_incidental_finding(agent: CaptureAgent):
    result = agent.run(_report("CT abdomen and pelvis. Liver, kidneys, adrenals and pancreas are unremarkable."))
    assert result.findings == []


def test_renal_lesion_requires_review(agent: CaptureAgent):
    result = agent.run(_report("There is an indeterminate 1.5 cm left renal lesion. MRI follow-up is recommended."))
    finding = result.findings[0]
    assert finding.organ_system == "kidney"
    assert finding.actionability == "actionable"
    assert finding.size_mm == 15


def test_adrenal_nodule(agent: CaptureAgent):
    result = agent.run(_report("A 1.8 cm right adrenal nodule is indeterminate on this CT."))
    assert result.findings[0].organ_system == "adrenal"
    assert result.findings[0].actionability == "actionable"


def test_pancreatic_cyst(agent: CaptureAgent):
    result = agent.run(_report("Incidental 8 mm pancreatic cystic lesion is present."))
    assert result.findings[0].organ_system == "pancreas"
    assert result.findings[0].human_review_required is False


def test_liver_lesion(agent: CaptureAgent):
    result = agent.run(_report("There is a suspicious 2 cm hepatic lesion in segment 7. MRI is recommended."))
    assert result.findings[0].organ_system == "liver"
    assert result.findings[0].actionability == "actionable"


def test_stable_previously_known_finding(agent: CaptureAgent):
    result = agent.run(_report("Stable left renal cyst unchanged since 2018, benign."))
    finding = result.findings[0]
    assert finding.previously_known == "true"
    assert finding.actionability == "non_actionable"


def test_ambiguous_finding_requires_human_review(agent: CaptureAgent):
    result = agent.run(_report("A tiny hypodense renal lesion is noted and incompletely characterized."))
    finding = result.findings[0]
    assert finding.actionability == "uncertain"
    assert finding.human_review_required is True
