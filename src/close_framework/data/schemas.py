"""Shared data models for the CLOSE incidental findings workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - exercised only when pydantic is absent
    import json

    _MISSING = object()

    class _FieldInfo:
        """Small Field replacement used when Pydantic is unavailable."""

        def __init__(self, default: Any = _MISSING, default_factory: Any = None) -> None:
            self.default = default
            self.default_factory = default_factory

        def get_default(self) -> Any:
            if self.default_factory is not None:
                return self.default_factory()
            if self.default is not _MISSING:
                return self.default
            return None

    def Field(default: Any = _MISSING, default_factory: Any = None, **_: Any) -> _FieldInfo:
        return _FieldInfo(default=default, default_factory=default_factory)

    class BaseModel:  # type: ignore[no-redef]
        """Minimal Pydantic-like model for local environments without Pydantic."""

        def __init__(self, **kwargs: Any) -> None:
            fields: Dict[str, Any] = {}
            for cls in reversed(self.__class__.mro()):
                fields.update(getattr(cls, "__annotations__", {}))
            for name in fields:
                if name in kwargs:
                    value = kwargs.pop(name)
                else:
                    default = getattr(self.__class__, name, None)
                    value = default.get_default() if isinstance(default, _FieldInfo) else default
                setattr(self, name, value)
            for name, value in kwargs.items():
                setattr(self, name, value)

        def model_dump(self, mode: str = "python") -> Dict[str, Any]:
            return {key: _dump_value(value, mode=mode) for key, value in self.__dict__.items()}

        def model_dump_json(self, indent: int | None = None) -> str:
            return json.dumps(self.model_dump(mode="json"), indent=indent)

    def _dump_value(value: Any, mode: str = "python") -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode=mode)
        if isinstance(value, list):
            return [_dump_value(item, mode=mode) for item in value]
        if isinstance(value, dict):
            return {key: _dump_value(item, mode=mode) for key, item in value.items()}
        if mode == "json" and isinstance(value, datetime):
            return value.isoformat()
        return value


ClinicalSetting = Literal["ED", "inpatient", "outpatient", "unknown"]
OrganSystem = Literal["kidney", "liver", "adrenal", "pancreas", "adnexal", "other"]
Actionability = Literal["actionable", "non_actionable", "uncertain"]
UrgencyTier = Literal["routine", "short_interval", "urgent", "unknown"]
KnownStatus = Literal["true", "false", "unknown"]
ProviderType = Literal["PCP", "specialist", "ordering_provider", "safety_pool", "unknown"]
FinalStatus = Literal[
    "no_actionable_finding",
    "recommendation_generated",
    "human_review_required",
    "error",
]


class RadiologyReport(BaseModel):
    """Normalized representation of one radiology report."""

    report_id: str
    patient_id: Optional[str] = None
    accession_number: Optional[str] = None
    report_text: str
    exam_type: Optional[str] = None
    report_date: Optional[datetime] = None
    clinical_setting: ClinicalSetting = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IncidentalFinding(BaseModel):
    """Structured incidental abdominal finding extracted from a report."""

    finding_id: str
    organ_system: OrganSystem
    finding_type: str
    size_mm: Optional[float] = None
    laterality: Optional[str] = None
    description: str
    actionability: Actionability
    urgency_tier: UrgencyTier = "unknown"
    previously_known: KnownStatus = "unknown"
    evidence_span: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    human_review_required: bool = False
    enhancement: Optional[str] = None
    attenuation_hu: Optional[float] = None
    cystic: Optional[bool] = None
    solid: Optional[bool] = None
    septations: Optional[bool] = None
    nodularity: Optional[bool] = None
    wall_thickening: Optional[bool] = None
    main_duct_mm: Optional[float] = None
    prior_comparison_date: Optional[str] = None
    already_recommended_followup: Optional[str] = None
    no_followup_needed: bool = False
    incidental: Optional[bool] = None
    patient_age: Optional[int] = None
    patient_risk: Optional[str] = None
    cancer_history: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CaptureResult(BaseModel):
    """Output from the Capture Agent for one report."""

    report_id: str
    patient_id: Optional[str] = None
    findings: List[IncidentalFinding] = Field(default_factory=list)
    agent_decisions: List["AgentDecision"] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PatientContext(BaseModel):
    """Flexible EHR-like context used by the Link Agent."""

    patient_id: Optional[str] = None
    clinical_setting: ClinicalSetting = "unknown"
    active_pcp_id: Optional[str] = None
    active_pcp_name: Optional[str] = None
    ordering_provider_id: Optional[str] = None
    ordering_provider_name: Optional[str] = None
    specialists: Dict[str, Dict[str, Optional[str]]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProviderAssignment(BaseModel):
    """Responsible provider routing decision for a finding."""

    finding_id: str
    responsible_provider_id: Optional[str] = None
    responsible_provider_name: Optional[str] = None
    provider_type: ProviderType = "unknown"
    routing_reason: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    failsafe_triggered: bool = False
    human_review_required: bool = False


class FollowUpRecommendation(BaseModel):
    """Structured follow-up recommendation from the Orchestrate Agent."""

    finding_id: str
    recommended_action: str
    recommended_modality: Optional[str] = None
    recommended_timeframe: Optional[str] = None
    rationale: str
    guideline_source: str
    urgency_tier: UrgencyTier = "unknown"
    order_set_stub: Optional[str] = None
    human_review_required: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    guideline_name: Optional[str] = None
    publication_year: Optional[int] = None
    rule_id: Optional[str] = None
    recommendation_text: Optional[str] = None
    evidence_used: Dict[str, Any] = Field(default_factory=dict)
    matched_rule: Optional[str] = None
    triggering_evidence: Optional[str] = None
    decision_path: List[str] = Field(default_factory=list)
    guideline_reasoning_trace: List[str] = Field(default_factory=list)


class AgentDecision(BaseModel):
    """Traceable decision made by an agent."""

    agent_name: str
    decision_type: str
    decision: str
    rationale: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    human_review_required: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    """End-to-end CLOSE result for one report."""

    report_id: str
    patient_id: Optional[str] = None
    capture_result: CaptureResult
    provider_assignments: Dict[str, ProviderAssignment] = Field(default_factory=dict)
    follow_up_recommendations: Dict[str, FollowUpRecommendation] = Field(default_factory=dict)
    final_status: FinalStatus
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class GuidelineMatch(BaseModel):
    """Retrieved local guideline pathway for a finding."""

    finding_id: str
    guideline_name: str
    publication_year: int
    rule_id: str
    guideline_source: str
    recommended_action: str
    recommended_modality: Optional[str] = None
    recommended_timeframe: Optional[str] = None
    recommendation_text: str
    rationale: str
    evidence_used: Dict[str, Any] = Field(default_factory=dict)
    matched_rule: str
    triggering_evidence: str
    urgency_tier: UrgencyTier = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    human_review_required: bool = False
    decision_path: List[str] = Field(default_factory=list)
    guideline_reasoning_trace: List[str] = Field(default_factory=list)


def to_plain_dict(model: Any) -> Dict[str, Any]:
    """Return a JSON-serializable dict for Pydantic v2, v1, or dataclass-like models."""

    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model.__dict__)
