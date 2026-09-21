"""Tests for ACR-derived guideline engines."""

from __future__ import annotations

from close_framework.agents.capture_agent import CaptureAgent
from close_framework.agents.orchestrate_agent import OrchestrateAgent
from close_framework.data.schemas import RadiologyReport


def _recommend(text: str):
    finding = CaptureAgent().run(RadiologyReport(report_id="r", report_text=text)).findings[0]
    return OrchestrateAgent().run(finding)


def test_renal_simple_hu_no_followup():
    rec = _recommend("Homogeneous right renal cyst measures 12 HU on CT. No follow-up needed.")
    assert rec.rule_id == "RENAL_HOMOGENEOUS_MINUS10_TO_20_HU_NO_FOLLOWUP"


def test_renal_complex_cyst_characterization():
    rec = _recommend("Complex cystic left renal lesion with thick septations and mural nodule measures 2.4 cm.")
    assert rec.rule_id == "RENAL_BOSNIAK_IV_REFER"


def test_liver_high_risk_subcentimeter_mri():
    rec = _recommend("Patient with colon cancer has an indeterminate 8 mm hepatic lesion.")
    assert rec.rule_id == "LIVER_HIGH_RISK_LT1CM_FOLLOWUP_6_MONTHS"


def test_pancreatic_cyst_interval():
    rec = _recommend("Incidental 1.7 cm pancreatic cyst without mural nodule or main duct dilation.")
    assert rec.rule_id == "PANC_1_5_TO_1_9CM_SURVEILLANCE"
