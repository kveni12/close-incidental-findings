"""Provider routing utilities for future EHR integration."""

from __future__ import annotations

from close_framework.agents.link_agent import LinkAgent
from close_framework.data.schemas import IncidentalFinding, PatientContext, ProviderAssignment


class ProviderRouter:
    """Thin wrapper around LinkAgent for Epic/FHIR expansion later."""

    def __init__(self, link_agent: LinkAgent | None = None) -> None:
        self.link_agent = link_agent or LinkAgent()

    def route(self, finding: IncidentalFinding, context: PatientContext) -> ProviderAssignment:
        """Route a finding to a responsible follow-up owner."""

        return self.link_agent.run(finding, context)
