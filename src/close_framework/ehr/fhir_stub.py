"""FHIR integration stubs for future Epic connectivity."""

from __future__ import annotations

from close_framework.data.schemas import PatientContext


class FHIRClientStub:
    """Placeholder for future Epic FHIR lookups."""

    def get_patient_context(self, patient_id: str | None) -> PatientContext:
        """Return empty context until a real FHIR client is configured."""

        return PatientContext(patient_id=patient_id, clinical_setting="unknown")
