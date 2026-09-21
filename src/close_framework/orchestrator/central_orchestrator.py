"""Central orchestration for the CLOSE pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from close_framework.agents.capture_agent import CaptureAgent
from close_framework.agents.link_agent import LinkAgent
from close_framework.agents.orchestrate_agent import OrchestrateAgent
from close_framework.config.settings import CloseSettings, load_settings
from close_framework.data.loaders import load_reports
from close_framework.data.schemas import (
    CaptureResult,
    FollowUpRecommendation,
    IncidentalFinding,
    PatientContext,
    ProviderAssignment,
    RadiologyReport,
    WorkflowResult,
)
from close_framework.ehr.ehr_context import context_from_report
from close_framework.orchestrator.workflow_state import choose_final_status

LOGGER = logging.getLogger(__name__)


class CentralOrchestrator:
    """Run Capture, Link, and Orchestrate agents over local reports."""

    def __init__(
        self,
        settings: Optional[CloseSettings] = None,
        capture_agent: Optional[CaptureAgent] = None,
        link_agent: Optional[LinkAgent] = None,
        orchestrate_agent: Optional[OrchestrateAgent] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.capture_agent = capture_agent or CaptureAgent(self.settings)
        self.link_agent = link_agent or LinkAgent(self.settings)
        self.orchestrate_agent = orchestrate_agent or OrchestrateAgent()

    def run_from_directory(self, data_dir: str | Path | None = None) -> List[WorkflowResult]:
        """Load reports from a directory and run the CLOSE pipeline."""

        reports = load_reports(data_dir or self.settings.data_dir)
        return self.run_reports(reports)

    def run_reports(self, reports: Iterable[RadiologyReport]) -> List[WorkflowResult]:
        """Run the pipeline for each report, preserving per-report errors."""

        results: List[WorkflowResult] = []
        for report in reports:
            try:
                results.append(self.run_report(report))
            except Exception as exc:  # noqa: BLE001 - batch processing must continue
                LOGGER.exception("Pipeline failed for report %s: %s", report.report_id, exc)
                capture_result = CaptureResult(
                    report_id=report.report_id,
                    patient_id=report.patient_id,
                    errors=[str(exc)],
                )
                results.append(
                    WorkflowResult(
                        report_id=report.report_id,
                        patient_id=report.patient_id,
                        capture_result=capture_result,
                        final_status="error",
                        errors=[str(exc)],
                    )
                )
        return results

    def run_report(self, report: RadiologyReport, context: Optional[PatientContext] = None) -> WorkflowResult:
        """Run the complete pipeline for one report."""

        capture_result = self.capture_agent.run(report)
        context = context or context_from_report(report)
        candidate_findings = self._deduplicate_findings(
            finding
            for finding in capture_result.findings
            if finding.actionability in {"actionable", "uncertain"}
        )

        provider_assignments: Dict[str, ProviderAssignment] = {}
        recommendations: Dict[str, FollowUpRecommendation] = {}
        for finding in candidate_findings:
            provider_assignments[finding.finding_id] = self.link_agent.run(finding, context)
            recommendations[finding.finding_id] = self.orchestrate_agent.run(finding)

        human_review_required = any(
            item.human_review_required
            for item in list(candidate_findings) + list(provider_assignments.values()) + list(recommendations.values())
        )
        final_status = choose_final_status(
            has_findings=bool(candidate_findings),
            has_recommendations=bool(recommendations),
            human_review_required=human_review_required,
        )
        return WorkflowResult(
            report_id=report.report_id,
            patient_id=report.patient_id,
            capture_result=capture_result,
            provider_assignments=provider_assignments,
            follow_up_recommendations=recommendations,
            final_status=final_status,
            warnings=capture_result.warnings,
            errors=capture_result.errors,
        )

    def run_from_capture(
        self,
        report: RadiologyReport,
        capture_result: CaptureResult,
        context: Optional[PatientContext] = None,
    ) -> WorkflowResult:
        """Run Link and Orchestrate from an existing Capture result."""

        context = context or context_from_report(report)
        candidate_findings = self._deduplicate_findings(
            finding
            for finding in capture_result.findings
            if finding.actionability in {"actionable", "uncertain"}
        )

        provider_assignments: Dict[str, ProviderAssignment] = {}
        recommendations: Dict[str, FollowUpRecommendation] = {}
        for finding in candidate_findings:
            provider_assignments[finding.finding_id] = self.link_agent.run(finding, context)
            recommendations[finding.finding_id] = self.orchestrate_agent.run(finding)

        human_review_required = any(
            item.human_review_required
            for item in list(candidate_findings) + list(provider_assignments.values()) + list(recommendations.values())
        )
        final_status = choose_final_status(
            has_findings=bool(candidate_findings),
            has_recommendations=bool(recommendations),
            human_review_required=human_review_required,
        )
        return WorkflowResult(
            report_id=report.report_id,
            patient_id=report.patient_id,
            capture_result=capture_result,
            provider_assignments=provider_assignments,
            follow_up_recommendations=recommendations,
            final_status=final_status,
            warnings=capture_result.warnings,
            errors=capture_result.errors,
        )

    def _deduplicate_findings(self, findings: Iterable[IncidentalFinding]) -> List[IncidentalFinding]:
        seen = set()
        deduped: List[IncidentalFinding] = []
        for finding in findings:
            key = (
                finding.organ_system,
                finding.finding_type.lower(),
                finding.laterality,
                round(finding.size_mm or -1, 1),
                finding.evidence_span.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(finding)
        return deduped
