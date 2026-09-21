"""Capture Agent for extracting actionable abdominal incidental findings."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

from close_framework.agents.base_agent import BaseAgent
from close_framework.config.settings import CloseSettings, InferenceMode, load_settings
from close_framework.data.preprocessing import split_sentences
from close_framework.data.schemas import AgentDecision, CaptureResult, IncidentalFinding, RadiologyReport
from close_framework.inference.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)

ORGAN_TERMS: Dict[str, Tuple[str, ...]] = {
    "kidney": ("kidney", "kidneys", "renal", "nephro"),
    "liver": ("liver", "hepatic", "hepat"),
    "adrenal": ("adrenal", "adrenal gland"),
    "pancreas": ("pancreas", "pancreatic"),
    "adnexal": ("adnexal", "ovary", "ovarian", "adnexa", "paraovarian"),
}

FINDING_TERMS = (
    "lesion",
    "mass",
    "nodule",
    "cyst",
    "cystic",
    "solid",
    "indeterminate",
    "hyperenhancing",
    "hypodense",
    "attenuating",
    "duct dilation",
    "duct dilatation",
    "main pancreatic duct",
)

BENIGN_TERMS = (
    "simple cyst",
    "cysts",
    "benign",
    "hemangioma",
    "myelolipoma",
    "adenoma",
    "no suspicious",
    "no solid masses",
    "no nodules",
    "too small to characterize",
)

STABLE_TERMS = (
    "stable",
    "unchanged",
    "previously known",
    "redemonstrated",
    "re-demonstrated",
    "since",
)

NO_FOLLOW_UP_TERMS = (
    "no follow-up",
    "no further follow up",
    "no further follow-up",
    "does not require follow",
)

ACTIONABLE_TERMS = (
    "recommend",
    "recommendation",
    "follow-up",
    "follow up",
    "mri",
    "ct renal mass",
    "adrenal protocol",
    "endoscopic ultrasound",
    "eus",
    "mr abdomen",
    "mrcp",
    "occult mass",
    "suspicious",
    "indeterminate",
    "complex",
    "enhancing",
    "solid",
)

URGENT_TERMS = ("urgent", "emergent", "obstruct", "jaundice", "occult mass")


class CaptureAgent(BaseAgent):
    """Detect and structure candidate incidental findings from one report."""

    agent_name = "CaptureAgent"

    def __init__(self, settings: Optional[CloseSettings] = None) -> None:
        super().__init__()
        self.settings = settings or load_settings()
        self.llm_client = LLMClient(self.settings) if self.settings.inference_mode != InferenceMode.RULE_BASED_ONLY else None
        self._last_llm_failed = False
        self._last_llm_error: Optional[str] = None

    def run(self, report: RadiologyReport) -> CaptureResult:
        """Extract incidental findings from a normalized report.

        When an LLM backend is configured (LOCAL_OLLAMA / LOCAL_VLLM / API / hybrid), Capture is
        LLM-primary. Keyword heuristics are used only in RULE_BASED_ONLY mode, or
        when CLOSE_ALLOW_RULE_FALLBACK is explicitly enabled after LLM failure.
        """

        findings: List[IncidentalFinding] = []
        decisions: List[AgentDecision] = []
        report_context = self._report_context(report)
        warnings: List[str] = []
        errors: List[str] = []
        used_llm = False
        used_rules_fallback = False

        if self.llm_client is not None:
            llm_findings = self._run_llm_extraction(report, report_context, start_ordinal=1)
            if not self._last_llm_failed:
                findings = list(llm_findings or [])
                used_llm = True
                decisions.append(
                    AgentDecision(
                        agent_name=self.agent_name,
                        decision_type="llm_primary_capture",
                        decision=f"{len(findings)} LLM findings extracted",
                        rationale="LLM-primary Capture for model comparison research.",
                        confidence=0.8 if findings else 0.6,
                        human_review_required=any(f.human_review_required for f in findings),
                        metadata={
                            "inference_mode": self.settings.inference_mode.value,
                            "llm_model": self.settings.llm_model,
                            "capture_mode": "llm_primary",
                        },
                    )
                )
            elif self.settings.allow_rule_fallback:
                used_rules_fallback = True
                findings = self._run_rule_extraction(report, report_context)
                warnings.append(
                    f"LLM capture failed ({self._last_llm_error or 'unknown'}); used rule fallback."
                )
                decisions.append(
                    AgentDecision(
                        agent_name=self.agent_name,
                        decision_type="rule_fallback_after_llm_failure",
                        decision=f"{len(findings)} rule findings after LLM failure",
                        rationale="LLM extraction failed; CLOSE_ALLOW_RULE_FALLBACK enabled.",
                        confidence=0.4,
                        human_review_required=True,
                        metadata={
                            "inference_mode": self.settings.inference_mode.value,
                            "llm_model": self.settings.llm_model,
                            "capture_mode": "rule_fallback",
                            "llm_error": self._last_llm_error,
                        },
                    )
                )
            else:
                errors.append(f"LLM capture failed: {self._last_llm_error or 'unknown'}")
                decisions.append(
                    AgentDecision(
                        agent_name=self.agent_name,
                        decision_type="llm_capture_failed",
                        decision="0 findings; LLM failure recorded",
                        rationale="LLM-primary mode with rule fallback disabled to keep model comparison clean.",
                        confidence=0.0,
                        human_review_required=True,
                        metadata={
                            "inference_mode": self.settings.inference_mode.value,
                            "llm_model": self.settings.llm_model,
                            "capture_mode": "llm_failed",
                            "llm_error": self._last_llm_error,
                        },
                    )
                )
        else:
            findings = self._run_rule_extraction(report, report_context)
            decisions.append(
                AgentDecision(
                    agent_name=self.agent_name,
                    decision_type="rule_only_capture",
                    decision=f"{len(findings)} rule findings detected",
                    rationale="RULE_BASED_ONLY mode: deterministic keyword/regex extraction (no LLM).",
                    confidence=0.75 if findings else 0.9,
                    human_review_required=any(f.human_review_required for f in findings),
                    metadata={"capture_mode": "rule_only"},
                )
            )

        decision = AgentDecision(
            agent_name=self.agent_name,
            decision_type="capture_summary",
            decision=f"{len(findings)} candidate findings detected",
            rationale=(
                "LLM-primary capture"
                if used_llm
                else ("Rule fallback after LLM failure" if used_rules_fallback else "Rule-only or failed LLM capture")
            ),
            confidence=0.75 if findings else 0.5,
            human_review_required=any(f.human_review_required for f in findings) or bool(errors),
        )
        decisions.append(decision)
        return CaptureResult(
            report_id=report.report_id,
            patient_id=report.patient_id,
            findings=findings,
            agent_decisions=decisions,
            warnings=warnings,
            errors=errors,
        )

    def _run_rule_extraction(
        self,
        report: RadiologyReport,
        report_context: Dict[str, object],
    ) -> List[IncidentalFinding]:
        """Keyword/regex sentence scanner (not ACR guideline logic)."""

        findings: List[IncidentalFinding] = []
        for sentence in split_sentences(report.report_text):
            organ = self._classify_organ(sentence)
            if organ is None and self._is_out_of_scope_incidental_section(sentence):
                findings.append(self._build_out_of_scope_incidental(report, sentence, len(findings) + 1))
                continue
            if organ is None or not self._has_finding_language(sentence):
                continue
            finding = self._build_finding(report, sentence, organ, len(findings) + 1, report_context)
            if not self._duplicate(finding, findings):
                findings.append(finding)
        return findings

    def _classify_organ(self, sentence: str) -> Optional[str]:
        lower = sentence.lower()
        for organ, terms in ORGAN_TERMS.items():
            if any(re.search(rf"\b{re.escape(term)}\b", lower) for term in terms):
                return organ
        return None

    def _is_out_of_scope_incidental_section(self, sentence: str) -> bool:
        lower = sentence.lower()
        if "incidental findings:" not in lower:
            return False
        return not any(term in lower for terms in ORGAN_TERMS.values() for term in terms)

    def _build_out_of_scope_incidental(
        self,
        report: RadiologyReport,
        sentence: str,
        ordinal: int,
    ) -> IncidentalFinding:
        finding_id = self._finding_id(report.report_id, ordinal, sentence)
        return IncidentalFinding(
            finding_id=finding_id,
            organ_system="other",
            finding_type="out_of_scope_incidental_section",
            description=sentence,
            actionability="non_actionable",
            urgency_tier="routine",
            previously_known="unknown",
            evidence_span=sentence,
            confidence=0.9,
            human_review_required=False,
            incidental=True,
            metadata={
                "out_of_scope": True,
                "reason": "Incidental finding section does not mention kidney, liver, adrenal, pancreas, or adnexal findings.",
            },
        )

    def _run_llm_extraction(
        self,
        report: RadiologyReport,
        report_context: Dict[str, object],
        start_ordinal: int,
    ) -> List[IncidentalFinding]:
        self._last_llm_failed = False
        self._last_llm_error = None
        if self.llm_client is None:
            self._last_llm_failed = True
            self._last_llm_error = "llm_client_missing"
            return []

        system = (
            "You extract structured incidental abdominal findings for research. "
            "Return valid JSON only. Be thorough: list every in-scope lesion mentioned."
        )
        response = self.llm_client.complete(self._llm_prompt(report), system=system)
        if response is None or not response.text:
            self._last_llm_failed = True
            self._last_llm_error = getattr(self.llm_client, "last_error", None) or "empty_or_null_response"
            return []

        payload = None
        raw_text = response.text
        try:
            payload = self._extract_json_payload(raw_text)
        except (ValueError, TypeError, json.JSONDecodeError):
            repair = self.llm_client.complete(
                self._json_repair_prompt(raw_text),
                system="Return only valid JSON. No markdown.",
                max_tokens=min(2048, int(self.settings.llm_max_tokens or 1536)),
            )
            if repair and repair.text:
                try:
                    payload = self._extract_json_payload(repair.text)
                    response = repair
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._last_llm_failed = True
                    self._last_llm_error = f"json_parse_failed: {exc}"
                    return []
            else:
                self._last_llm_failed = True
                self._last_llm_error = "json_parse_failed_and_repair_failed"
                return []

        findings_payload = payload if isinstance(payload, list) else (payload or {}).get("findings", [])
        if not isinstance(findings_payload, list):
            self._last_llm_failed = True
            self._last_llm_error = "findings_not_a_list"
            return []

        findings: List[IncidentalFinding] = []
        skipped = 0
        organ_aliases = {
            "renal": "kidney",
            "hepatic": "liver",
            "ovary": "adnexal",
            "ovarian": "adnexal",
            "adnexa": "adnexal",
        }
        for offset, item in enumerate(findings_payload):
            if not isinstance(item, dict):
                skipped += 1
                continue
            evidence = str(item.get("evidence_span") or item.get("description") or "").strip()
            organ = str(item.get("organ_system") or "").lower().strip()
            organ = organ_aliases.get(organ, organ)
            if organ not in ORGAN_TERMS or not evidence:
                skipped += 1
                continue
            actionability = str(item.get("actionability") or "uncertain").lower()
            if actionability not in {"actionable", "non_actionable", "uncertain"}:
                actionability = "uncertain"
            urgency = str(item.get("urgency_tier") or "unknown").lower()
            if urgency not in {"routine", "short_interval", "urgent", "unknown"}:
                urgency = "unknown"
            known = str(item.get("previously_known") or "unknown").lower()
            if known not in {"true", "false", "unknown"}:
                known = "unknown"
            finding_id = self._finding_id(report.report_id, start_ordinal + offset, evidence)
            try:
                confidence = float(item.get("confidence") or 0.6)
            except (TypeError, ValueError):
                confidence = 0.6
            try:
                findings.append(
                    IncidentalFinding(
                        finding_id=finding_id,
                        organ_system=organ,  # type: ignore[arg-type]
                        finding_type=str(item.get("finding_type") or "incidental finding"),
                        size_mm=self._coerce_optional_float(item.get("size_mm")) or self._extract_size_mm(evidence),
                        laterality=self._coerce_optional_str(item.get("laterality")),
                        description=str(item.get("description") or evidence),
                        actionability=actionability,  # type: ignore[arg-type]
                        urgency_tier=urgency,  # type: ignore[arg-type]
                        previously_known=known,  # type: ignore[arg-type]
                        evidence_span=evidence,
                        confidence=max(0.0, min(confidence, 1.0)),
                        human_review_required=self._coerce_bool(
                            item.get("human_review_required"), default=(actionability == "uncertain")
                        ),
                        enhancement=self._coerce_optional_str(item.get("enhancement")),
                        attenuation_hu=self._coerce_optional_float(item.get("attenuation_hu"))
                        or self._extract_hu(evidence),
                        cystic=self._coerce_optional_bool(item.get("cystic")),
                        solid=self._coerce_optional_bool(item.get("solid")),
                        septations=self._coerce_optional_bool(item.get("septations")),
                        nodularity=self._coerce_optional_bool(item.get("nodularity")),
                        wall_thickening=self._coerce_optional_bool(item.get("wall_thickening")),
                        main_duct_mm=self._coerce_optional_float(item.get("main_duct_mm"))
                        or self._extract_main_duct_mm(evidence),
                        prior_comparison_date=self._coerce_optional_str(item.get("prior_comparison_date")),
                        already_recommended_followup=self._coerce_followup_text(
                            item.get("already_recommended_followup")
                        ),
                        no_followup_needed=self._coerce_bool(item.get("no_followup_needed"), default=False),
                        incidental=self._coerce_optional_bool(item.get("incidental")),
                        patient_age=report_context.get("patient_age"),  # type: ignore[arg-type]
                        patient_risk=report_context.get("patient_risk"),  # type: ignore[arg-type]
                        cancer_history=report_context.get("cancer_history"),  # type: ignore[arg-type]
                        metadata={
                            "llm_extracted": True,
                            "llm_model": response.model,
                            "llm_backend": response.backend,
                            "capture_mode": "llm_primary",
                            "llm_attempts": response.attempts,
                        },
                    )
                )
            except Exception as exc:
                skipped += 1
                LOGGER.warning(
                    "Skipping unusable LLM finding item report_id=%s organ=%s err=%s",
                    report.report_id,
                    organ,
                    exc,
                )
                continue

        if findings_payload and not findings and skipped == len(findings_payload):
            self._last_llm_failed = True
            self._last_llm_error = "all_finding_items_unusable"
            return []

        if self.settings.recalibrate_llm_capture:
            findings = [self._recalibrate_llm_finding(finding, report_context) for finding in findings]

        if self.settings.supplement_rule_findings:
            findings = self._supplement_with_rule_findings(report, report_context, findings, start_ordinal)

        return findings

    def _llm_prompt(self, report: RadiologyReport) -> str:
        n_chars = int(self.settings.llm_report_chars or 12000)
        body = report.report_text[:n_chars]
        return f"""You are the Capture agent in the CLOSE incidental-findings research pipeline.

Task: exhaustively extract EVERY incidental abdominal finding in scope from the report.

In scope organs ONLY:
- kidney / renal
- liver / hepatic
- adrenal
- pancreas / pancreatic
- adnexal / ovary / ovarian

Rules:
1. Ignore clearly negated normals (e.g. "no renal mass", "adrenals unremarkable").
2. Include lesions even if labeled benign, simple cyst, or stable — set actionability accordingly.
3. Prefer one finding object per distinct lesion (not per sentence).
4. Copy evidence_span as a short exact quote from the report.
5. size_mm must be millimeters (convert cm × 10).
6. already_recommended_followup must be null or a short string, NEVER a boolean.
7. If nothing in scope, return {{"findings": []}}.

Actionability calibration (match ACR incidental-finding practice — do NOT default everything to non_actionable):
- actionable: indeterminate/suspicious/complex/enhancing lesion; size ≥10 mm kidney/liver/adrenal mass; pancreatic cyst ≥5 mm;
  adnexal lesion ≥30 mm; explicit follow-up imaging recommended; occult/solid/complex features
- uncertain: too small to characterize, incompletely evaluated, or likely benign without definitive benign features
- non_actionable: clearly benign/simple/stable with no follow-up language, or explicit "no follow-up needed"

Return ONLY JSON (no markdown, no commentary):
{{"findings": [
  {{"organ_system": "kidney|liver|adrenal|pancreas|adnexal", "finding_type": "...", "size_mm": null,
    "laterality": null, "description": "...", "actionability": "actionable|non_actionable|uncertain",
    "urgency_tier": "routine|short_interval|urgent|unknown", "previously_known": "true|false|unknown",
    "evidence_span": "exact quote", "confidence": 0.0, "human_review_required": false,
    "enhancement": null, "attenuation_hu": null, "cystic": null, "solid": null,
    "septations": null, "nodularity": null, "wall_thickening": null, "main_duct_mm": null,
    "prior_comparison_date": null, "already_recommended_followup": null,
    "no_followup_needed": false, "incidental": true}}
]}}

Report:
{body}
"""

    def _json_repair_prompt(self, bad_text: str) -> str:
        clipped = bad_text[:8000]
        return f"""The following model output was supposed to be JSON for incidental findings but is invalid.
Convert it into valid JSON with shape {{"findings":[...]}} only.
If you cannot recover findings, return {{"findings": []}}.

Broken output:
{clipped}
"""

    @staticmethod
    def _coerce_optional_float(value: object) -> Optional[float]:
        if value in {None, "", "null"}:
            return None
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_str(value: object) -> Optional[str]:
        if value in {None, "", "null"}:
            return None
        if isinstance(value, bool):
            return None
        return str(value)

    @staticmethod
    def _coerce_optional_bool(value: object) -> Optional[bool]:
        if value in {None, "", "null"}:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"true", "yes", "1"}:
                return True
            if lower in {"false", "no", "0"}:
                return False
        return None

    @staticmethod
    def _coerce_bool(value: object, default: bool = False) -> bool:
        coerced = CaptureAgent._coerce_optional_bool(value)
        return default if coerced is None else coerced

    @staticmethod
    def _coerce_followup_text(value: object) -> Optional[str]:
        if value in {None, "", "null", False}:
            return None
        if value is True:
            return "follow-up recommended in report"
        return str(value)

    def _extract_json_payload(self, text: str):
        stripped = (text or "").strip()
        stripped = re.sub(r"<think>[\s\S]*?</think>", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"<\|[^>]+\|>", "", stripped)
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
            stripped = re.sub(r"```$", "", stripped).strip()
        candidates = []
        for match in re.finditer(r"(\{[\s\S]*\}|\[[\s\S]*\])", stripped):
            candidates.append(match.group(1))
        if not candidates:
            candidates = [stripped]
        candidates.sort(key=len, reverse=True)
        errors: List[Exception] = []
        for candidate in candidates[:5]:
            cleaned = candidate.strip()
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                errors.append(exc)
                repaired = self._repair_truncated_json(cleaned)
                if repaired is not None:
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError as exc2:
                        errors.append(exc2)
        raise ValueError(f"No JSON payload found in LLM response ({errors[-1] if errors else 'unknown'})")

    @staticmethod
    def _repair_truncated_json(text: str) -> Optional[str]:
        if not text or text[0] not in "[{":
            return None
        last_brace = max(text.rfind("}"), text.rfind("]"))
        snippet = text if last_brace < 0 else text[: last_brace + 1]
        # Walk structure outside of strings to recover missing closers.
        stack: List[str] = []
        in_string = False
        escape = False
        for ch in snippet:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
            elif ch in "}]":
                return None
        if in_string:
            snippet += '"'
        closers = "".join("}" if opened == "{" else "]" for opened in reversed(stack))
        return snippet + closers

    def _actionability_from_sentence(
        self,
        *,
        sentence: str,
        organ: str,
        size_mm: Optional[float],
        finding_type: str,
        report_context: Dict[str, object],
        no_follow_up: bool,
        negated_finding: bool,
        benign: bool,
        likely_benign: bool,
        actionable_language: bool,
        existing_followup: Optional[str],
        urgent_language: bool,
        previously_known: str,
        incidental: Optional[bool],
        attenuation_hu: Optional[float],
    ) -> Tuple[str, float, bool, str]:
        lower = sentence.lower()
        if no_follow_up or negated_finding or (benign and not actionable_language):
            actionability = "non_actionable"
            confidence = 0.9 if no_follow_up or negated_finding else 0.82
            human_review = False
            urgency = "routine"
        elif likely_benign and not actionable_language:
            actionability = "uncertain"
            confidence = 0.62
            human_review = True
            urgency = "routine"
        elif (
            actionable_language
            or self._size_suggests_review(organ, size_mm, finding_type)
            or self._feature_suggests_review(organ, lower)
        ):
            actionability = "actionable"
            confidence = self._calibrate_confidence(
                base=0.72 if actionable_language else 0.68,
                size_mm=size_mm,
                has_hu=attenuation_hu is not None,
                has_followup=existing_followup is not None,
                incidental=incidental,
            )
            human_review = confidence < self.settings.capture_confidence_threshold
            urgency = self._infer_urgency(lower, existing_followup, urgent_language)
        else:
            actionability = "uncertain"
            confidence = 0.5
            human_review = True
            urgency = "unknown"

        if previously_known == "true" and not actionable_language:
            actionability = "non_actionable"
            confidence = max(confidence, 0.7)
            human_review = False

        return actionability, confidence, human_review, urgency

    def _recalibrate_llm_finding(
        self, finding: IncidentalFinding, report_context: Dict[str, object]
    ) -> IncidentalFinding:
        """Re-apply ACR-aligned rule heuristics to LLM labels (hybrid Capture)."""

        sentence = finding.evidence_span or finding.description
        lower = sentence.lower()
        size_mm = finding.size_mm or self._extract_size_mm(sentence)
        finding_type = finding.finding_type or self._extract_finding_type(sentence, finding.organ_system)
        benign = any(term in lower for term in BENIGN_TERMS)
        likely_benign = "likely benign" in lower or "probably benign" in lower
        negated_finding = bool(
            re.search(
                r"\bno\b.{0,60}\b(lesions?|masses?|nodules?|cysts?|duct(?:al)?\s+dilat(?:ation|ion)|duct(?:al)?\s+dilation|solid\s+masses?)\b",
                lower,
            )
        )
        no_follow_up = finding.no_followup_needed or any(term in lower for term in NO_FOLLOW_UP_TERMS)
        actionable_language = any(term in lower for term in ACTIONABLE_TERMS)
        existing_followup = finding.already_recommended_followup or self._extract_followup_language(sentence)
        urgent_language = any(term in lower for term in URGENT_TERMS)
        actionability, confidence, human_review, urgency = self._actionability_from_sentence(
            sentence=sentence,
            organ=finding.organ_system,
            size_mm=size_mm,
            finding_type=finding_type,
            report_context=report_context,
            no_follow_up=no_follow_up,
            negated_finding=negated_finding,
            benign=benign,
            likely_benign=likely_benign,
            actionable_language=actionable_language,
            existing_followup=existing_followup,
            urgent_language=urgent_language,
            previously_known=finding.previously_known,
            incidental=finding.incidental,
            attenuation_hu=finding.attenuation_hu or self._extract_hu(sentence),
        )
        metadata = dict(finding.metadata or {})
        metadata.update(
            {
                "recalibrated_by_rules": True,
                "llm_actionability_before_recalibration": finding.actionability,
            }
        )
        return finding.model_copy(
            update={
                "size_mm": size_mm,
                "finding_type": finding_type,
                "actionability": actionability,  # type: ignore[arg-type]
                "urgency_tier": urgency,  # type: ignore[arg-type]
                "confidence": confidence,
                "human_review_required": human_review,
                "already_recommended_followup": existing_followup,
                "no_followup_needed": no_follow_up,
                "metadata": metadata,
            }
        )

    def recalibrate_finding(
        self, finding: IncidentalFinding, report_context: Optional[Dict[str, object]] = None
    ) -> IncidentalFinding:
        """Public hook for offline JSONL recalibration scripts."""

        return self._recalibrate_llm_finding(finding, report_context or {})

    @staticmethod
    def _findings_overlap(a: IncidentalFinding, b: IncidentalFinding) -> bool:
        a_text = (a.evidence_span or a.description or "").lower().strip()
        b_text = (b.evidence_span or b.description or "").lower().strip()
        if not a_text or not b_text:
            return False
        if a_text in b_text or b_text in a_text:
            return True
        if a.organ_system != b.organ_system:
            return False
        if a.size_mm is not None and b.size_mm is not None and abs(a.size_mm - b.size_mm) <= 2:
            return True
        return False

    def _supplement_with_rule_findings(
        self,
        report: RadiologyReport,
        report_context: Dict[str, object],
        llm_findings: List[IncidentalFinding],
        start_ordinal: int,
    ) -> List[IncidentalFinding]:
        """Add rule-detected in-scope findings the LLM missed."""

        rule_findings = self._run_rule_extraction(report, report_context)
        merged = list(llm_findings)
        next_ordinal = start_ordinal + len(llm_findings)
        for rule_finding in rule_findings:
            if any(self._findings_overlap(rule_finding, existing) for existing in merged):
                continue
            metadata = dict(rule_finding.metadata or {})
            metadata["supplemented_from_rules"] = True
            merged.append(rule_finding.model_copy(update={"metadata": metadata}))
            next_ordinal += 1
        return merged

    def _has_finding_language(self, sentence: str) -> bool:
        lower = sentence.lower()
        return any(term in lower for term in FINDING_TERMS + ACTIONABLE_TERMS)

    def _build_finding(
        self,
        report: RadiologyReport,
        sentence: str,
        organ: str,
        ordinal: int,
        report_context: Dict[str, object],
    ) -> IncidentalFinding:
        lower = sentence.lower()
        size_mm = self._extract_size_mm(sentence)
        previously_known = "true" if any(term in lower for term in STABLE_TERMS) else "unknown"
        laterality = self._extract_laterality(sentence)
        finding_type = self._extract_finding_type(sentence, organ)
        attenuation_hu = self._extract_hu(sentence)
        main_duct_mm = self._extract_main_duct_mm(sentence)
        enhancement = self._extract_enhancement(sentence)
        cystic = "cyst" in lower or "cystic" in lower
        solid = "solid" in lower or "mass" in lower and "cyst" not in lower
        septations = any(term in lower for term in ("septation", "septated", "septal", "thick septa"))
        nodularity = any(term in lower for term in ("mural nodule", "nodular", "nodule"))
        wall_thickening = any(term in lower for term in ("thick wall", "wall thickening", "thickened wall"))
        benign = any(term in lower for term in BENIGN_TERMS)
        likely_benign = "likely benign" in lower or "probably benign" in lower
        negated_finding = bool(
            re.search(
                r"\bno\b.{0,60}\b(lesions?|masses?|nodules?|cysts?|duct(?:al)?\s+dilat(?:ation|ion)|duct(?:al)?\s+dilation|solid\s+masses?)\b",
                lower,
            )
        )
        benign = benign or negated_finding
        no_follow_up = any(term in lower for term in NO_FOLLOW_UP_TERMS)
        actionable_language = any(term in lower for term in ACTIONABLE_TERMS)
        existing_followup = self._extract_followup_language(sentence)
        urgent_language = any(term in lower for term in URGENT_TERMS)
        prior_date = self._extract_prior_comparison_date(sentence) or report_context.get("comparison_date")
        incidental = self._infer_incidental(report_context, organ, sentence)

        actionability, confidence, human_review, urgency = self._actionability_from_sentence(
            sentence=sentence,
            organ=organ,
            size_mm=size_mm,
            finding_type=finding_type,
            report_context=report_context,
            no_follow_up=no_follow_up,
            negated_finding=negated_finding,
            benign=benign,
            likely_benign=likely_benign,
            actionable_language=actionable_language,
            existing_followup=existing_followup,
            urgent_language=urgent_language,
            previously_known=previously_known,
            incidental=incidental,
            attenuation_hu=attenuation_hu,
        )

        finding_id = self._finding_id(report.report_id, ordinal, sentence)
        return IncidentalFinding(
            finding_id=finding_id,
            organ_system=organ,  # type: ignore[arg-type]
            finding_type=finding_type,
            size_mm=size_mm,
            laterality=laterality,
            description=sentence,
            actionability=actionability,  # type: ignore[arg-type]
            urgency_tier=urgency,  # type: ignore[arg-type]
            previously_known=previously_known,  # type: ignore[arg-type]
            evidence_span=sentence,
            confidence=confidence,
            human_review_required=human_review,
            enhancement=enhancement,
            attenuation_hu=attenuation_hu,
            cystic=cystic,
            solid=solid,
            septations=septations,
            nodularity=nodularity,
            wall_thickening=wall_thickening,
            main_duct_mm=main_duct_mm,
            prior_comparison_date=str(prior_date) if prior_date else None,
            already_recommended_followup=existing_followup,
            no_followup_needed=no_follow_up,
            incidental=incidental,
            patient_age=report_context.get("patient_age"),  # type: ignore[arg-type]
            patient_risk=report_context.get("patient_risk"),  # type: ignore[arg-type]
            cancer_history=report_context.get("cancer_history"),  # type: ignore[arg-type]
            metadata={
                "rule_based": True,
                "primary_indication_organs": report_context.get("primary_indication_organs", []),
                "extraction_features": {
                    "benign_language": benign,
                    "likely_benign_language": likely_benign,
                    "actionable_language": actionable_language,
                    "existing_followup_language": existing_followup,
                },
            },
        )

    def _extract_size_mm(self, sentence: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:x\s*\d+(?:\.\d+)?\s*)?(cm|mm)", sentence, flags=re.IGNORECASE)
        if not match:
            return None
        value = float(match.group(1))
        return value * 10 if match.group(2).lower() == "cm" else value

    def _extract_hu(self, sentence: str) -> Optional[float]:
        patterns = [
            r"(-?\d+(?:\.\d+)?)\s*(?:hounsfield units|hu)\b",
            r"(?:attenuation|density)\s*(?:of|measuring|measures|is|=)?\s*(-?\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_main_duct_mm(self, sentence: str) -> Optional[float]:
        lower = sentence.lower()
        if "duct" not in lower or "without main duct" in lower:
            return None
        match = re.search(r"(?:main pancreatic duct|main duct|duct)[^\d]{0,30}(\d+(?:\.\d+)?)\s*(cm|mm)", sentence, flags=re.IGNORECASE)
        if not match:
            return None
        value = float(match.group(1))
        return value * 10 if match.group(2).lower() == "cm" else value

    def _extract_laterality(self, sentence: str) -> Optional[str]:
        lower = sentence.lower()
        if "bilateral" in lower:
            return "bilateral"
        if "left" in lower:
            return "left"
        if "right" in lower:
            return "right"
        return None

    def _extract_finding_type(self, sentence: str, organ: str) -> str:
        lower = sentence.lower()
        for term in (
            "cystic lesion",
            "solid mass",
            "adrenal adenoma",
            "myelolipoma",
            "hemangioma",
            "mass",
            "nodule",
            "cyst",
            "lesion",
            "duct dilation",
            "duct dilatation",
        ):
            if term in lower:
                return term
        if organ == "pancreas" and "duct" in lower:
            return "pancreatic duct abnormality"
        return "incidental finding"

    def _extract_enhancement(self, sentence: str) -> Optional[str]:
        lower = sentence.lower()
        if "nonenhancing" in lower or "non-enhancing" in lower or "no enhancement" in lower:
            return "nonenhancing"
        if "hyperenhancing" in lower or "arterial enhancing" in lower:
            return "hyperenhancing"
        if "enhancing" in lower or "enhancement" in lower:
            return "enhancing"
        return None

    def _size_suggests_review(self, organ: str, size_mm: Optional[float], finding_type: str) -> bool:
        if size_mm is None:
            return False
        if organ == "adrenal" and size_mm >= 10:
            return True
        if organ == "pancreas" and ("cyst" in finding_type or "lesion" in finding_type) and size_mm >= 5:
            return True
        if organ == "adnexal" and size_mm >= 30:
            return True
        if organ in {"kidney", "liver"} and size_mm >= 10 and "simple cyst" not in finding_type:
            return True
        return False

    def _feature_suggests_review(self, organ: str, lower: str) -> bool:
        if re.search(r"\bno\b.{0,60}\b(mass|masses|nodule|nodules|lesion|lesions|duct(?:al)?\s+dilat(?:ation|ion)|duct(?:al)?\s+dilation)", lower):
            return False
        if organ == "kidney":
            return any(term in lower for term in ("enhancing", "solid", "complex", "septation", "nodular"))
        if organ == "adrenal":
            return any(term in lower for term in ("indeterminate", ">10 hu", "washout", "heterogeneous"))
        if organ == "liver":
            return any(term in lower for term in ("indeterminate", "suspicious", "enhancing", "ill-defined"))
        if organ == "pancreas":
            return any(term in lower for term in ("cyst", "duct", "mural nodule", "solid component"))
        if organ == "adnexal":
            return any(term in lower for term in ("complex", "solid", "septation", "nodule", "suspicious"))
        return False

    def _extract_followup_language(self, sentence: str) -> Optional[str]:
        match = re.search(
            r"((?:recommend|consider|advise)[^.]{0,120}(?:mri|ct|mrcp|eus|endoscopic ultrasound|follow-up|follow up)[^.]*\.)?",
            sentence,
            flags=re.IGNORECASE,
        )
        if match and match.group(1):
            return match.group(1).strip()
        if re.search(r"(mri|ct|mrcp|eus).{0,40}(recommended|advised|follow)", sentence, flags=re.IGNORECASE):
            return sentence.strip()
        return None

    def _extract_prior_comparison_date(self, sentence: str) -> Optional[str]:
        match = re.search(r"(?:since|comparison:|compared (?:with|to))\s*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{4}[-/][A-Za-z]{3,9}[-/]\d{1,2}|\d{4}[-/]\d{1,2}[-/]\d{1,2})", sentence, flags=re.IGNORECASE)
        return match.group(1) if match else None

    def _infer_urgency(self, lower: str, followup: Optional[str], urgent_language: bool) -> str:
        if urgent_language or "prompt" in lower:
            return "urgent" if "obstruct" in lower or "jaundice" in lower else "short_interval"
        if followup or any(term in lower for term in ("3-6 months", "6 months", "short interval")):
            return "short_interval"
        return "routine"

    def _calibrate_confidence(self, base: float, size_mm: Optional[float], has_hu: bool, has_followup: bool, incidental: Optional[bool]) -> float:
        confidence = base
        if size_mm is not None:
            confidence += 0.06
        if has_hu:
            confidence += 0.05
        if has_followup:
            confidence += 0.08
        if incidental is False:
            confidence -= 0.12
        return max(0.0, min(confidence, 0.97))

    def _report_context(self, report: RadiologyReport) -> Dict[str, object]:
        text = report.report_text.lower()
        metadata_text = " ".join(str(value).lower() for value in report.metadata.values())
        indication_match = re.search(r"(?:indication|provided indication).*?(?:\n|$)(.*)", report.report_text, flags=re.IGNORECASE)
        indication = indication_match.group(0).lower() if indication_match else metadata_text[:500]
        primary_organs = [organ for organ, terms in ORGAN_TERMS.items() if any(term in indication for term in terms)]
        age_match = re.search(r"\b(\d{1,3})\s*(?:yo|year-old|year old|m|f)\b", text)
        cancer_history = any(term in text or term in metadata_text for term in ("history of cancer", "known malignancy", "metastatic", "colon cancer", "breast cancer", "pancreatic cancer", "cholangiocarcinoma"))
        age = int(age_match.group(1)) if age_match else None
        patient_risk = (
            "high"
            if cancer_history or any(term in text for term in ("cirrhosis", "hepatitis", "nash", "sclerosing cholangitis"))
            else "average"
            if age is not None and age > 40
            else "low"
        )
        comparison_match = re.search(r"comparison:\s*([^\n.]+)", report.report_text, flags=re.IGNORECASE)
        return {
            "primary_indication_organs": primary_organs,
            "patient_age": age,
            "cancer_history": cancer_history,
            "patient_risk": patient_risk,
            "comparison_date": comparison_match.group(1).strip() if comparison_match else None,
        }

    def _infer_incidental(self, report_context: Dict[str, object], organ: str, sentence: str) -> Optional[bool]:
        primary_organs = report_context.get("primary_indication_organs", [])
        if organ in primary_organs and any(term in sentence.lower() for term in ("known", "follow up", "follow-up", "history")):
            return False
        if primary_organs and organ not in primary_organs:
            return True
        return None

    def _duplicate(self, finding: IncidentalFinding, existing: Iterable[IncidentalFinding]) -> bool:
        finding_key = (finding.organ_system, finding.finding_type, finding.evidence_span.lower())
        return any((item.organ_system, item.finding_type, item.evidence_span.lower()) == finding_key for item in existing)

    def _finding_id(self, report_id: str, ordinal: int, evidence: str) -> str:
        digest = hashlib.sha1(f"{report_id}:{ordinal}:{evidence}".encode("utf-8")).hexdigest()[:10]
        return f"{report_id}-finding-{digest}"
