"""Helpers for building patient context from report metadata."""

from __future__ import annotations

from close_framework.data.schemas import PatientContext, RadiologyReport


def context_from_report(report: RadiologyReport) -> PatientContext:
    """Create a best-effort patient context from local report metadata."""

    metadata = report.metadata
    return PatientContext(
        patient_id=report.patient_id,
        clinical_setting=report.clinical_setting,
        ordering_provider_id=metadata.get("Ordering_Provider_ID") or metadata.get("ordering_provider_id"),
        ordering_provider_name=metadata.get("Ordering_Provider") or metadata.get("ordering_provider_name"),
        active_pcp_id=metadata.get("PCP_ID") or metadata.get("active_pcp_id"),
        active_pcp_name=metadata.get("PCP") or metadata.get("active_pcp_name"),
        metadata=metadata,
    )
