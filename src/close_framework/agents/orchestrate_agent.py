"""Orchestrate Agent for guideline-aware follow-up recommendations."""

from __future__ import annotations

from close_framework.agents.base_agent import BaseAgent
from close_framework.data.schemas import FollowUpRecommendation, IncidentalFinding
from close_framework.guidelines.guideline_retriever import GuidelineRetriever


class OrchestrateAgent(BaseAgent):
    """Map captured findings to deterministic ACR-derived pathways."""

    agent_name = "OrchestrateAgent"

    def __init__(self, retriever: GuidelineRetriever | None = None) -> None:
        super().__init__()
        self.retriever = retriever or GuidelineRetriever()

    def run(self, finding: IncidentalFinding) -> FollowUpRecommendation:
        """Generate a structured recommendation or require human review."""

        if finding.actionability == "non_actionable":
            return FollowUpRecommendation(
                finding_id=finding.finding_id,
                recommended_action="No downstream CLOSE follow-up generated.",
                recommended_modality=None,
                recommended_timeframe=None,
                rationale="Capture Agent classified the finding as non-actionable based on report language.",
                guideline_source="Capture Agent non-actionable classification",
                urgency_tier=finding.urgency_tier,
                order_set_stub=None,
                human_review_required=False,
                confidence=finding.confidence,
                guideline_name="Capture Agent non-actionable classification",
                publication_year=None,
                rule_id="CAPTURE_NON_ACTIONABLE",
                recommendation_text="No downstream CLOSE follow-up generated.",
                evidence_used={"evidence_span": finding.evidence_span},
                matched_rule="CAPTURE_NON_ACTIONABLE",
                triggering_evidence=finding.evidence_span,
                decision_path=["capture_non_actionable"],
                guideline_reasoning_trace=[
                    "Capture Agent classified the finding as non-actionable.",
                    finding.evidence_span,
                ],
            )

        match = self.retriever.retrieve(finding)
        return FollowUpRecommendation(
            finding_id=finding.finding_id,
            recommended_action=match.recommended_action,
            recommended_modality=match.recommended_modality,
            recommended_timeframe=match.recommended_timeframe,
            rationale=match.rationale,
            guideline_source=match.guideline_source,
            urgency_tier=finding.urgency_tier if finding.urgency_tier != "unknown" else match.urgency_tier,
            order_set_stub=self._order_set_stub(finding),
            human_review_required=match.human_review_required or finding.human_review_required,
            confidence=min(match.confidence, finding.confidence),
            guideline_name=match.guideline_name,
            publication_year=match.publication_year,
            rule_id=match.rule_id,
            recommendation_text=match.recommendation_text,
            evidence_used=match.evidence_used,
            matched_rule=match.matched_rule,
            triggering_evidence=match.triggering_evidence,
            decision_path=match.decision_path,
            guideline_reasoning_trace=match.guideline_reasoning_trace,
        )

    def _order_set_stub(self, finding: IncidentalFinding) -> str:
        """Return a non-executable order set identifier for downstream experiments."""

        return f"CLOSE_{finding.organ_system.upper()}_FOLLOW_UP_REVIEW"
