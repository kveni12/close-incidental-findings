"""Shared types and helpers for ACR white-paper guideline engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from close_framework.data.schemas import GuidelineMatch, IncidentalFinding
from close_framework.guidelines.white_paper_catalog import doc_for_organ


def source_url_for_organ(organ: str, page: Optional[int] = None) -> str:
    """Relative citation into incidental_findings_white_papers/."""

    doc = doc_for_organ(organ)
    if not doc:
        return "incidental_findings_white_papers/"
    suffix = f"#page={page}" if page else ""
    return f"incidental_findings_white_papers/{doc.filename}{suffix}"


@dataclass(frozen=True)
class ACRGuidelineMetadata:
    """Source metadata for a guideline engine."""

    guideline_name: str
    publication_year: int
    source_url: str


@dataclass
class RuleEvidence:
    """Normalized evidence consumed by guideline rules."""

    organ_system: str
    finding_type: str
    size_mm: Optional[float]
    attenuation_hu: Optional[float]
    enhancement: Optional[str]
    cystic: Optional[bool]
    solid: Optional[bool]
    septations: Optional[bool]
    nodularity: Optional[bool]
    wall_thickening: Optional[bool]
    main_duct_mm: Optional[float]
    previously_known: str
    prior_comparison_date: Optional[str]
    no_followup_needed: bool
    patient_age: Optional[int]
    patient_risk: Optional[str]
    cancer_history: Optional[bool]
    evidence_span: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_finding(cls, finding: IncidentalFinding) -> "RuleEvidence":
        """Build rule evidence from capture output."""

        return cls(
            organ_system=finding.organ_system,
            finding_type=finding.finding_type,
            size_mm=finding.size_mm,
            attenuation_hu=finding.attenuation_hu,
            enhancement=finding.enhancement,
            cystic=finding.cystic,
            solid=finding.solid,
            septations=finding.septations,
            nodularity=finding.nodularity,
            wall_thickening=finding.wall_thickening,
            main_duct_mm=finding.main_duct_mm,
            previously_known=finding.previously_known,
            prior_comparison_date=finding.prior_comparison_date,
            no_followup_needed=finding.no_followup_needed,
            patient_age=finding.patient_age,
            patient_risk=finding.patient_risk,
            cancer_history=finding.cancer_history,
            evidence_span=finding.evidence_span,
            metadata=finding.metadata,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Serialize evidence for recommendation traceability."""

        return dict(self.__dict__)


class GuidelineEngine:
    """Base class for deterministic organ-specific guideline engines."""

    organ_system: str
    metadata: ACRGuidelineMetadata

    def recommend(self, finding: IncidentalFinding) -> GuidelineMatch:
        """Return an ACR-derived recommendation for one finding."""

        raise NotImplementedError

    def _match(
        self,
        finding: IncidentalFinding,
        rule_id: str,
        action: str,
        rationale: str,
        decision_path: List[str],
        modality: Optional[str] = None,
        timeframe: Optional[str] = None,
        urgency: str = "routine",
        confidence: float = 0.86,
        human_review: bool = False,
    ) -> GuidelineMatch:
        evidence = RuleEvidence.from_finding(finding)
        recommendation_text = action if not timeframe else f"{action} Timeframe: {timeframe}."
        return GuidelineMatch(
            finding_id=finding.finding_id,
            guideline_name=self.metadata.guideline_name,
            publication_year=self.metadata.publication_year,
            rule_id=rule_id,
            guideline_source=self.metadata.source_url,
            recommended_action=action,
            recommended_modality=modality,
            recommended_timeframe=timeframe,
            recommendation_text=recommendation_text,
            rationale=rationale,
            evidence_used=evidence.as_dict(),
            matched_rule=rule_id,
            triggering_evidence=finding.evidence_span,
            urgency_tier=urgency,  # type: ignore[arg-type]
            confidence=confidence,
            human_review_required=human_review,
            decision_path=decision_path,
            guideline_reasoning_trace=[
                f"Guideline={self.metadata.guideline_name} ({self.metadata.publication_year})",
                f"Source={self.metadata.source_url}",
                f"Rule={rule_id}",
                f"Evidence={finding.evidence_span}",
                rationale,
            ],
        )

    def _manual_review(
        self,
        finding: IncidentalFinding,
        rule_id: str,
        rationale: str,
        decision_path: List[str],
    ) -> GuidelineMatch:
        return self._match(
            finding=finding,
            rule_id=rule_id,
            action="Manual radiology review for ACR pathway assignment.",
            rationale=rationale,
            decision_path=decision_path,
            urgency="unknown",
            confidence=0.45,
            human_review=True,
        )
