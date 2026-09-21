"""Link Agent for assigning follow-up responsibility."""

from __future__ import annotations

from typing import Optional

from close_framework.agents.base_agent import BaseAgent
from close_framework.config.settings import CloseSettings, load_settings
from close_framework.data.schemas import IncidentalFinding, PatientContext, ProviderAssignment


class LinkAgent(BaseAgent):
    """Route actionable findings to the most appropriate available provider."""

    agent_name = "LinkAgent"

    def __init__(self, settings: Optional[CloseSettings] = None) -> None:
        super().__init__()
        self.settings = settings or load_settings()

    def run(self, finding: IncidentalFinding, context: PatientContext) -> ProviderAssignment:
        """Return a provider assignment using PCP, specialist, ordering provider, then failsafe pool."""

        if context.active_pcp_id or context.active_pcp_name:
            return ProviderAssignment(
                finding_id=finding.finding_id,
                responsible_provider_id=context.active_pcp_id,
                responsible_provider_name=context.active_pcp_name,
                provider_type="PCP",
                routing_reason="Active PCP available for longitudinal follow-up.",
                confidence=0.9,
                failsafe_triggered=False,
                human_review_required=finding.human_review_required,
            )

        specialist = context.specialists.get(finding.organ_system)
        if specialist and (specialist.get("provider_id") or specialist.get("provider_name")):
            return ProviderAssignment(
                finding_id=finding.finding_id,
                responsible_provider_id=specialist.get("provider_id"),
                responsible_provider_name=specialist.get("provider_name"),
                provider_type="specialist",
                routing_reason=f"Existing {finding.organ_system} specialist appears to be managing related care.",
                confidence=0.82,
                failsafe_triggered=False,
                human_review_required=finding.human_review_required,
            )

        if context.clinical_setting != "ED" and (context.ordering_provider_id or context.ordering_provider_name):
            return ProviderAssignment(
                finding_id=finding.finding_id,
                responsible_provider_id=context.ordering_provider_id,
                responsible_provider_name=context.ordering_provider_name,
                provider_type="ordering_provider",
                routing_reason="No longitudinal provider found; routing to ordering provider.",
                confidence=0.68,
                failsafe_triggered=False,
                human_review_required=True,
            )

        reason = "ED-only or missing provider context; routing to diagnostic safety failsafe queue."
        if context.ordering_provider_id or context.ordering_provider_name:
            reason = "ED setting without active PCP; safety pool review required before assigning follow-up."
        return ProviderAssignment(
            finding_id=finding.finding_id,
            responsible_provider_id=self.settings.safety_pool_id,
            responsible_provider_name=self.settings.safety_pool_name,
            provider_type="safety_pool",
            routing_reason=reason,
            confidence=0.55,
            failsafe_triggered=True,
            human_review_required=True,
        )
