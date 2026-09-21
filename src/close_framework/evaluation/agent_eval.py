"""Per-agent evaluation of CLOSE runs against a rule-based reference pipeline.

Capture is evaluated in :mod:`close_framework.evaluation.gold_eval`. This module
covers the downstream agents by comparing a model run's workflow output against
the same pipeline executed in rule-based mode.

Important interpretation note: Link and Orchestrate are deterministic engines
shared by both runs, so differences here are propagated Capture differences
rather than independent agent quality. These metrics quantify how much upstream
extraction error survives to the point of care.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from close_framework.evaluation.metrics import binary_metrics, multiclass_agreement_metrics

IN_SCOPE = ("kidney", "liver", "adrenal", "pancreas", "adnexal")

GUIDELINE_LABELS = (
    "none",
    "Management of the Incidental Renal Mass on CT",
    "Management of Incidental Adrenal Masses",
    "Management of Incidental Liver Lesions on CT",
    "Management of Incidental Pancreatic Cysts",
    "Management of Incidental Adnexal Findings on CT and MRI",
)
MODALITY_FAMILIES = (
    "none",
    "renal_protocol",
    "adrenal_protocol",
    "liver_mri",
    "pancreas_mri_mrcp",
    "eus_fna",
    "pelvic_us_mri",
    "other",
)
TIMEFRAME_CLASSES = ("none", "prompt", "interval_followup", "long_term_surveillance", "other")
PROVIDER_TYPES = ("none", "safety_pool", "pcp", "specialist", "care_team", "other")
FINAL_STATUSES = ("no_actionable_finding", "human_review_required", "completed", "error")


def modality_family(value: Optional[str]) -> str:
    """Collapse engine modality strings into comparable imaging families."""

    if not value:
        return "none"
    text = value.lower()
    if "eus" in text:
        return "eus_fna"
    if "renal" in text:
        return "renal_protocol"
    if "adrenal" in text:
        return "adrenal_protocol"
    if "pelvis" in text:
        return "pelvic_us_mri"
    if "mrcp" in text or "pancreas" in text:
        return "pancreas_mri_mrcp"
    if "liver" in text or "mri abdomen" in text:
        return "liver_mri"
    return "other"


def timeframe_class(value: Optional[str]) -> str:
    """Collapse engine timeframe strings into comparable follow-up horizons."""

    if not value:
        return "none"
    text = value.lower()
    if "prompt" in text:
        return "prompt"
    if any(token in text for token in ("yearly for", "every 2 years", "every 6 months for", "annual imaging")):
        return "long_term_surveillance"
    if any(token in text for token in ("months", "2 years")):
        return "interval_followup"
    return "other"


def provider_type_class(value: Optional[str]) -> str:
    if not value:
        return "none"
    return value if value in PROVIDER_TYPES else "other"


def _findings(result: dict) -> List[dict]:
    findings = (result.get("capture_result") or {}).get("findings") or []
    return [f for f in findings if str(f.get("organ_system")) in IN_SCOPE]


def _finding_organ(result: dict) -> Dict[str, str]:
    return {str(f.get("finding_id")): str(f.get("organ_system")) for f in _findings(result)}


def _organ_entries(result: dict, section: str) -> Dict[str, List[dict]]:
    """Group Link assignments or Orchestrate recommendations by organ."""

    organ_of = _finding_organ(result)
    grouped: Dict[str, List[dict]] = {organ: [] for organ in IN_SCOPE}
    for finding_id, entry in (result.get(section) or {}).items():
        organ = organ_of.get(str(finding_id))
        if organ in grouped:
            grouped[organ].append(entry)
    return grouped


def _first(entries: Sequence[dict], key: str):
    for entry in entries:
        value = entry.get(key)
        if value is not None:
            return value
    return None


def evaluate_link_agent(reference: Dict[str, dict], candidate: Dict[str, dict], ids: List[str]) -> dict:
    ref_assign = {i: reference[i].get("provider_assignments") or {} for i in ids}
    cand_assign = {i: candidate[i].get("provider_assignments") or {} for i in ids}

    any_true = [bool(ref_assign[i]) for i in ids]
    any_pred = [bool(cand_assign[i]) for i in ids]

    review_true = [any(a.get("human_review_required") for a in ref_assign[i].values()) for i in ids]
    review_pred = [any(a.get("human_review_required") for a in cand_assign[i].values()) for i in ids]

    failsafe_true = [any(a.get("failsafe_triggered") for a in ref_assign[i].values()) for i in ids]
    failsafe_pred = [any(a.get("failsafe_triggered") for a in cand_assign[i].values()) for i in ids]

    provider_true = [
        provider_type_class(_first(list(ref_assign[i].values()), "provider_type")) for i in ids
    ]
    provider_pred = [
        provider_type_class(_first(list(cand_assign[i].values()), "provider_type")) for i in ids
    ]
    provider_labels = [
        label for label in PROVIDER_TYPES if label in set(provider_true) | set(provider_pred)
    ]

    organ_true: List[bool] = []
    organ_pred: List[bool] = []
    per_organ = {}
    for organ in IN_SCOPE:
        yt = [bool(_organ_entries(reference[i], "provider_assignments")[organ]) for i in ids]
        yp = [bool(_organ_entries(candidate[i], "provider_assignments")[organ]) for i in ids]
        per_organ[organ] = binary_metrics(yt, yp)
        organ_true.extend(yt)
        organ_pred.extend(yp)

    count_exact = sum(len(ref_assign[i]) == len(cand_assign[i]) for i in ids)
    return {
        "any_assignment": binary_metrics(any_true, any_pred),
        "human_review_flag": binary_metrics(review_true, review_pred),
        "failsafe_triggered": binary_metrics(failsafe_true, failsafe_pred),
        "provider_type_category": multiclass_agreement_metrics(
            provider_true, provider_pred, labels=provider_labels
        ),
        "organ_assignment_by_organ": per_organ,
        "organ_assignment_micro": binary_metrics(organ_true, organ_pred),
        "assignment_count_exact_match": count_exact / len(ids) if ids else None,
        "reference_assignments": float(sum(len(ref_assign[i]) for i in ids)),
        "candidate_assignments": float(sum(len(cand_assign[i]) for i in ids)),
    }


def evaluate_orchestrate_agent(
    reference: Dict[str, dict], candidate: Dict[str, dict], ids: List[str]
) -> dict:
    ref_rec = {i: reference[i].get("follow_up_recommendations") or {} for i in ids}
    cand_rec = {i: candidate[i].get("follow_up_recommendations") or {} for i in ids}

    any_true = [bool(ref_rec[i]) for i in ids]
    any_pred = [bool(cand_rec[i]) for i in ids]

    imaging_true = [any(r.get("recommended_modality") for r in ref_rec[i].values()) for i in ids]
    imaging_pred = [any(r.get("recommended_modality") for r in cand_rec[i].values()) for i in ids]

    review_true = [any(r.get("human_review_required") for r in ref_rec[i].values()) for i in ids]
    review_pred = [any(r.get("human_review_required") for r in cand_rec[i].values()) for i in ids]

    guideline_true: List[str] = []
    guideline_pred: List[str] = []
    modality_true: List[str] = []
    modality_pred: List[str] = []
    timeframe_true: List[str] = []
    timeframe_pred: List[str] = []
    organ_rec_true: List[bool] = []
    organ_rec_pred: List[bool] = []
    per_organ = {}
    rule_matches = 0
    rule_comparable = 0

    for organ in IN_SCOPE:
        yt: List[bool] = []
        yp: List[bool] = []
        for report_id in ids:
            ref_entries = _organ_entries(reference[report_id], "follow_up_recommendations")[organ]
            cand_entries = _organ_entries(candidate[report_id], "follow_up_recommendations")[organ]
            yt.append(bool(ref_entries))
            yp.append(bool(cand_entries))
            guideline_true.append(str(_first(ref_entries, "guideline_name") or "none"))
            guideline_pred.append(str(_first(cand_entries, "guideline_name") or "none"))
            modality_true.append(modality_family(_first(ref_entries, "recommended_modality")))
            modality_pred.append(modality_family(_first(cand_entries, "recommended_modality")))
            timeframe_true.append(timeframe_class(_first(ref_entries, "recommended_timeframe")))
            timeframe_pred.append(timeframe_class(_first(cand_entries, "recommended_timeframe")))
            ref_rule = _first(ref_entries, "rule_id")
            cand_rule = _first(cand_entries, "rule_id")
            if ref_rule and cand_rule:
                rule_comparable += 1
                rule_matches += int(str(ref_rule) == str(cand_rule))
        per_organ[organ] = binary_metrics(yt, yp)
        organ_rec_true.extend(yt)
        organ_rec_pred.extend(yp)

    guideline_labels = [
        label for label in GUIDELINE_LABELS if label in set(guideline_true) | set(guideline_pred)
    ]
    modality_labels = [
        label for label in MODALITY_FAMILIES if label in set(modality_true) | set(modality_pred)
    ]
    timeframe_labels = [
        label for label in TIMEFRAME_CLASSES if label in set(timeframe_true) | set(timeframe_pred)
    ]

    return {
        "any_recommendation": binary_metrics(any_true, any_pred),
        "imaging_followup_recommended": binary_metrics(imaging_true, imaging_pred),
        "human_review_flag": binary_metrics(review_true, review_pred),
        "organ_recommendation_by_organ": per_organ,
        "organ_recommendation_micro": binary_metrics(organ_rec_true, organ_rec_pred),
        "guideline_category": multiclass_agreement_metrics(
            guideline_true, guideline_pred, labels=guideline_labels
        ),
        "modality_family_category": multiclass_agreement_metrics(
            modality_true, modality_pred, labels=modality_labels
        ),
        "timeframe_category": multiclass_agreement_metrics(
            timeframe_true, timeframe_pred, labels=timeframe_labels, ordinal=True
        ),
        "rule_id_exact_match_when_both_recommend": (
            rule_matches / rule_comparable if rule_comparable else None
        ),
        "rule_id_comparable_units": float(rule_comparable),
        "reference_recommendations": float(sum(len(ref_rec[i]) for i in ids)),
        "candidate_recommendations": float(sum(len(cand_rec[i]) for i in ids)),
    }


def evaluate_end_to_end(reference: Dict[str, dict], candidate: Dict[str, dict], ids: List[str]) -> dict:
    status_true = [str(reference[i].get("final_status")) for i in ids]
    status_pred = [str(candidate[i].get("final_status")) for i in ids]
    labels = [label for label in FINAL_STATUSES if label in set(status_true) | set(status_pred)]
    return {
        "final_status_category": multiclass_agreement_metrics(status_true, status_pred, labels=labels),
        "human_review_flag": binary_metrics(
            [s == "human_review_required" for s in status_true],
            [s == "human_review_required" for s in status_pred],
        ),
    }


def evaluate_agents(reference: Dict[str, dict], candidate: Dict[str, dict], model_name: str) -> dict:
    ids = sorted(set(reference) & set(candidate))
    return {
        "model": model_name,
        "n_shared_reports": len(ids),
        "link": evaluate_link_agent(reference, candidate, ids),
        "orchestrate": evaluate_orchestrate_agent(reference, candidate, ids),
        "end_to_end": evaluate_end_to_end(reference, candidate, ids),
    }
