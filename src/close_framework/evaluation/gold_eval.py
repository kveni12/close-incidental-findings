"""Gold / reference-standard Capture evaluation helpers.

Metrics fall into two families:

* **Detection metrics** (sensitivity, specificity, PPV, NPV) treat the reference
  labels as truth for a binary event.
* **Agreement metrics** (Cohen's kappa, weighted kappa) are chance-corrected
  agreement between two raters on a categorical scale. When the reference
  standard is itself an automated rule engine rather than expert adjudication,
  kappa should be read as model-vs-reference agreement, not clinical accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from close_framework.evaluation.metrics import binary_metrics, multiclass_agreement_metrics

IN_SCOPE = ("kidney", "liver", "adrenal", "pancreas", "adnexal")

# Ordinal severity scale; "none" means no finding was reported for that unit.
ACTIONABILITY_SCALE = ("none", "non_actionable", "uncertain", "actionable")
# Urgency is nominal because "unknown" does not sit on the severity ordering.
URGENCY_SCALE = ("none", "unknown", "routine", "short_interval", "urgent")
# Ordinal subset used when neither rater emitted "unknown".
URGENCY_ORDINAL_SCALE = ("none", "routine", "short_interval", "urgent")


def load_jsonl_by_report_id(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows[str(payload["report_id"])] = payload
    return rows


def gold_findings(label: dict) -> List[dict]:
    return [f for f in (label.get("findings") or []) if str(f.get("organ_system")) in IN_SCOPE]


def prediction_findings(result: dict) -> List[dict]:
    findings = (result.get("capture_result") or {}).get("findings") or []
    return [f for f in findings if str(f.get("organ_system")) in IN_SCOPE]


def _max_on_scale(values: Sequence[Optional[str]], scale: Sequence[str]) -> str:
    """Return the most severe value present, or the scale's zero category."""

    ranked = [scale.index(v) for v in values if v in scale]
    return scale[max(ranked)] if ranked else scale[0]


def report_actionability_category(findings: Sequence[dict]) -> str:
    return _max_on_scale([f.get("actionability") for f in findings], ACTIONABILITY_SCALE)


def organ_actionability_categories(findings: Sequence[dict]) -> Dict[str, str]:
    categories = {organ: ACTIONABILITY_SCALE[0] for organ in IN_SCOPE}
    for organ in IN_SCOPE:
        organ_findings = [f for f in findings if str(f.get("organ_system")) == organ]
        categories[organ] = _max_on_scale(
            [f.get("actionability") for f in organ_findings], ACTIONABILITY_SCALE
        )
    return categories


def report_urgency_category(findings: Sequence[dict]) -> str:
    values = [f.get("urgency_tier") for f in findings]
    ranked = [URGENCY_SCALE.index(v) for v in values if v in URGENCY_SCALE and v != "unknown"]
    if ranked:
        return URGENCY_SCALE[max(ranked)]
    if any(v == "unknown" for v in values):
        return "unknown"
    return "none"


def organs_from_prediction(result: dict) -> Set[str]:
    return {str(f.get("organ_system")) for f in prediction_findings(result)}


def has_actionable_prediction(result: dict) -> bool:
    return any(f.get("actionability") == "actionable" for f in prediction_findings(result))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def evaluate_predictions_against_gold(
    gold: Dict[str, dict],
    predictions: Dict[str, dict],
    model_name: str,
) -> dict:
    ids = sorted(set(gold) & set(predictions))
    gold_finds = {i: gold_findings(gold[i]) for i in ids}
    pred_finds = {i: prediction_findings(predictions[i]) for i in ids}

    y_presence_true = [bool(gold[i]["has_in_scope_finding"]) for i in ids]
    y_presence_pred = [bool(pred_finds[i]) for i in ids]
    y_action_true = [bool(gold[i]["has_actionable_finding"]) for i in ids]
    y_action_pred = [has_actionable_prediction(predictions[i]) for i in ids]

    organ_metrics = {}
    organ_micro_true: List[bool] = []
    organ_micro_pred: List[bool] = []
    for organ in IN_SCOPE:
        yt = [organ in (gold[i].get("organs_present") or []) for i in ids]
        yp = [organ in organs_from_prediction(predictions[i]) for i in ids]
        organ_metrics[organ] = binary_metrics(yt, yp)
        organ_micro_true.extend(yt)
        organ_micro_pred.extend(yp)

    # --- Categorical (multi-class) agreement ---
    report_action_true = [report_actionability_category(gold_finds[i]) for i in ids]
    report_action_pred = [report_actionability_category(pred_finds[i]) for i in ids]

    organ_action_true: List[str] = []
    organ_action_pred: List[str] = []
    organ_level_categorical = {}
    for organ in IN_SCOPE:
        yt = [organ_actionability_categories(gold_finds[i])[organ] for i in ids]
        yp = [organ_actionability_categories(pred_finds[i])[organ] for i in ids]
        organ_level_categorical[organ] = multiclass_agreement_metrics(
            yt, yp, labels=list(ACTIONABILITY_SCALE), ordinal=True
        )
        organ_action_true.extend(yt)
        organ_action_pred.extend(yp)

    urgency_true = [report_urgency_category(gold_finds[i]) for i in ids]
    urgency_pred = [report_urgency_category(pred_finds[i]) for i in ids]
    ordinal_pairs = [
        (t, p)
        for t, p in zip(urgency_true, urgency_pred)
        if t in URGENCY_ORDINAL_SCALE and p in URGENCY_ORDINAL_SCALE
    ]

    jaccards = [
        _jaccard(set(gold[i].get("organs_present") or []), organs_from_prediction(predictions[i]))
        for i in ids
    ]
    exact_organ_match = [
        set(gold[i].get("organs_present") or []) == organs_from_prediction(predictions[i])
        for i in ids
    ]

    return {
        "model": model_name,
        "n_shared_reports": len(ids),
        "gold_only_reports": len(set(gold) - set(predictions)),
        "prediction_only_reports": len(set(predictions) - set(gold)),
        "presence": binary_metrics(y_presence_true, y_presence_pred),
        "actionable": binary_metrics(y_action_true, y_action_pred),
        "organ_presence": organ_metrics,
        "organ_presence_micro": binary_metrics(organ_micro_true, organ_micro_pred),
        "categorical": {
            "report_actionability": multiclass_agreement_metrics(
                report_action_true, report_action_pred, labels=list(ACTIONABILITY_SCALE), ordinal=True
            ),
            "organ_actionability_pooled": multiclass_agreement_metrics(
                organ_action_true, organ_action_pred, labels=list(ACTIONABILITY_SCALE), ordinal=True
            ),
            "organ_actionability_by_organ": organ_level_categorical,
            "report_urgency_nominal": multiclass_agreement_metrics(
                urgency_true, urgency_pred, labels=list(URGENCY_SCALE), ordinal=False
            ),
            "report_urgency_ordinal_known_only": multiclass_agreement_metrics(
                [t for t, _ in ordinal_pairs],
                [p for _, p in ordinal_pairs],
                labels=list(URGENCY_ORDINAL_SCALE),
                ordinal=True,
            )
            if ordinal_pairs
            else None,
        },
        "mean_organ_jaccard": sum(jaccards) / len(jaccards) if jaccards else None,
        "exact_organ_set_accuracy": (
            sum(exact_organ_match) / len(exact_organ_match) if exact_organ_match else None
        ),
        "gold_prevalence": {
            "has_in_scope_finding": sum(y_presence_true) / len(ids) if ids else None,
            "has_actionable_finding": sum(y_action_true) / len(ids) if ids else None,
        },
    }
