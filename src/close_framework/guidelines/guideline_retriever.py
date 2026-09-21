"""Guideline retrieval: organ rule engines + white-paper RAG evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from close_framework.data.schemas import GuidelineMatch, IncidentalFinding
from close_framework.guidelines.adrenal_guidelines import AdrenalGuidelineEngine
from close_framework.guidelines.adnexal_guidelines import AdnexalGuidelineEngine
from close_framework.guidelines.guideline_types import GuidelineEngine
from close_framework.guidelines.kidney_guidelines import KidneyGuidelineEngine
from close_framework.guidelines.liver_guidelines import LiverGuidelineEngine
from close_framework.guidelines.pancreas_guidelines import PancreasGuidelineEngine
from close_framework.guidelines.rag_index import WhitePaperRAG, finding_query_text


class GuidelineRetriever:
    """Dispatch findings to ACR engines and attach retrieved white-paper evidence."""

    def __init__(
        self,
        engines: Optional[Dict[str, GuidelineEngine]] = None,
        rag: Optional[WhitePaperRAG] = None,
        rag_path: Optional[Path] = None,
        rag_top_k: int = 3,
    ) -> None:
        self.engines = engines or {
            "kidney": KidneyGuidelineEngine(),
            "adrenal": AdrenalGuidelineEngine(),
            "liver": LiverGuidelineEngine(),
            "pancreas": PancreasGuidelineEngine(),
            "adnexal": AdnexalGuidelineEngine(),
        }
        self.rag_top_k = rag_top_k
        self._rag = rag
        self._rag_path = rag_path
        self._rag_load_attempted = rag is not None

    @property
    def rag(self) -> Optional[WhitePaperRAG]:
        if self._rag is not None:
            return self._rag
        if self._rag_load_attempted:
            return None
        self._rag_load_attempted = True
        try:
            self._rag = WhitePaperRAG.from_disk(self._rag_path)
        except FileNotFoundError:
            self._rag = None
        return self._rag

    def retrieve(self, finding: IncidentalFinding) -> GuidelineMatch:
        """Return the best local guideline match for a finding, with RAG context."""

        engine = self.engines.get(finding.organ_system)
        if not engine:
            match = self._human_review(finding, "No local guideline pathway encoded for this organ system.")
        else:
            match = engine.recommend(finding)
        return self._augment_with_rag(finding, match)

    def _augment_with_rag(self, finding: IncidentalFinding, match: GuidelineMatch) -> GuidelineMatch:
        rag = self.rag
        if rag is None:
            match.guideline_reasoning_trace = list(match.guideline_reasoning_trace or []) + [
                "RAG index unavailable; recommendation used deterministic engine only."
            ]
            return match

        query = finding_query_text(
            organ=finding.organ_system,
            finding_type=finding.finding_type,
            description=finding.description,
            evidence_span=finding.evidence_span,
            extra=[match.rule_id or "", match.rationale or ""],
        )
        hits = rag.retrieve(query, organ=finding.organ_system, top_k=self.rag_top_k)
        if not hits:
            return match

        evidence = dict(match.evidence_used or {})
        evidence["rag_chunks"] = [
            {
                "chunk_id": hit.chunk_id,
                "doc": hit.doc_filename,
                "page": hit.page,
                "score": round(hit.score, 4),
                "title": hit.title,
                "excerpt": hit.text[:500],
            }
            for hit in hits
        ]
        match.evidence_used = evidence
        top = hits[0]
        if not match.guideline_source or "white_papers" not in str(match.guideline_source):
            match.guideline_source = f"incidental_findings_white_papers/{top.doc_filename}#page={top.page}"
        trace = list(match.guideline_reasoning_trace or [])
        trace.append(
            f"RAG top hit: {top.doc_filename} p{top.page} (score={top.score:.3f}) — {top.text[:240]}"
        )
        for hit in hits[1:]:
            trace.append(f"RAG support: {hit.doc_filename} p{hit.page} (score={hit.score:.3f})")
        match.guideline_reasoning_trace = trace
        return match

    def _human_review(self, finding: IncidentalFinding, rationale: str) -> GuidelineMatch:
        return GuidelineMatch(
            finding_id=finding.finding_id,
            guideline_name="No matching ACR incidental finding guideline",
            publication_year=0,
            rule_id="NO_ENGINE",
            guideline_source="No validated local guideline match",
            recommended_action="Human review required before issuing follow-up recommendation.",
            recommended_modality=None,
            recommended_timeframe=None,
            recommendation_text="Human review required before issuing follow-up recommendation.",
            rationale=rationale,
            evidence_used={"evidence_span": finding.evidence_span},
            matched_rule="NO_ENGINE",
            triggering_evidence=finding.evidence_span,
            urgency_tier="unknown",
            confidence=0.0,
            human_review_required=True,
            decision_path=["no_engine"],
            guideline_reasoning_trace=[rationale],
        )
