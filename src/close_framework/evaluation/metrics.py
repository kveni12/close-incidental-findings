"""Evaluation metrics and descriptive summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from close_framework.data.schemas import WorkflowResult


def descriptive_summary_metrics(results: Iterable[WorkflowResult]) -> Dict[str, Any]:
    """Produce descriptive metrics when gold labels are unavailable."""

    result_list = list(results)
    findings = [finding for result in result_list for finding in result.capture_result.findings]
    recommendations = [
        recommendation
        for result in result_list
        for recommendation in result.follow_up_recommendations.values()
    ]
    assignments = [
        assignment
        for result in result_list
        for assignment in result.provider_assignments.values()
    ]
    capture_modes = Counter()
    llm_errors = Counter()
    llm_findings = 0
    rule_findings = 0
    for result in result_list:
        for decision in result.capture_result.agent_decisions:
            mode = (decision.metadata or {}).get("capture_mode")
            if mode:
                capture_modes[str(mode)] += 1
            err = (decision.metadata or {}).get("llm_error")
            if err:
                llm_errors[str(err)[:160]] += 1
        for finding in result.capture_result.findings:
            if finding.metadata.get("llm_extracted"):
                llm_findings += 1
            else:
                rule_findings += 1

    return {
        "reports_processed": len(result_list),
        "findings_detected": len(findings),
        "llm_extracted_findings": llm_findings,
        "rule_extracted_findings": rule_findings,
        "capture_mode_counts": dict(capture_modes),
        "llm_error_counts": dict(llm_errors),
        "capture_error_reports": sum(bool(result.capture_result.errors) for result in result_list),
        "out_of_scope_incidental_findings": sum(bool(f.metadata.get("out_of_scope")) for f in findings),
        "in_scope_findings": sum(
            f.organ_system in {"kidney", "liver", "adrenal", "pancreas", "adnexal"} for f in findings
        ),
        "actionable_findings": sum(f.actionability == "actionable" for f in findings),
        "uncertain_findings": sum(f.actionability == "uncertain" for f in findings),
        "non_actionable_findings": sum(f.actionability == "non_actionable" for f in findings),
        "human_review_required": sum(result.final_status == "human_review_required" for result in result_list),
        "recommendations_generated": len(recommendations),
        "failsafe_rate": _rate(sum(a.failsafe_triggered for a in assignments), len(assignments)),
        "human_review_rate": _rate(
            sum(
                result.final_status == "human_review_required"
                for result in result_list
            ),
            len(result_list),
        ),
        "error_rate": _rate(sum(result.final_status == "error" for result in result_list), len(result_list)),
        "organ_counts": dict(Counter(f.organ_system for f in findings)),
        "urgency_tier_counts": dict(Counter(f.urgency_tier for f in findings)),
        "final_status_counts": dict(Counter(result.final_status for result in result_list)),
    }


def classification_metrics(y_true: List[str], y_pred: List[str], positive_label: str) -> Dict[str, Optional[float]]:
    """Basic sensitivity, PPV, and accuracy for labeled evaluations."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(t == positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fp = sum(t != positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fn = sum(t == positive_label and p != positive_label for t, p in zip(y_true, y_pred))
    tn = sum(t != positive_label and p != positive_label for t, p in zip(y_true, y_pred))
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "positive_predictive_value": _safe_div(tp, tp + fp),
        "negative_predictive_value": _safe_div(tn, tn + fn),
        "accuracy": _safe_div(correct, len(y_true)),
        "balanced_accuracy": (
            (sensitivity + specificity) / 2.0
            if sensitivity is not None and specificity is not None
            else None
        ),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "cohen_kappa": cohen_kappa(y_true, y_pred),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_negatives": float(tn),
        "n": float(len(y_true)),
        "prevalence": _safe_div(tp + fn, len(y_true)),
    }


def binary_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Optional[float]]:
    """Binary classification metrics with Cohen's kappa."""

    return classification_metrics(
        ["pos" if value else "neg" for value in y_true],
        ["pos" if value else "neg" for value in y_pred],
        positive_label="pos",
    )


def cohen_kappa(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    weights: Optional[str] = None,
) -> Optional[float]:
    """Cohen's kappa for categorical labels.

    ``weights`` may be ``None`` (nominal), ``"linear"``, or ``"quadratic"``.
    Weighted variants require ``labels`` in ordinal order.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return None
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:  # pragma: no cover - sklearn is expected in the eval env
        if weights is not None:
            return None
        return _cohen_kappa_fallback(y_true, y_pred)
    value = cohen_kappa_score(y_true, y_pred, labels=labels, weights=weights)
    if value != value:  # NaN when there is no expected disagreement
        return None
    return float(value)


def multiclass_agreement_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    ordinal: bool = False,
) -> Dict[str, Any]:
    """Chance-corrected agreement for a multi-category label.

    ``labels`` fixes the category order, which matters for the weighted kappa
    variants reported when ``ordinal`` is True (adjacent-category disagreements
    are penalized less than distant ones).
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    n = len(y_true)
    matrix = confusion_matrix_with_labels(y_true, y_pred, labels)
    per_class: Dict[str, Dict[str, Optional[float]]] = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        per_class[label] = {
            "support": float(sum(matrix[label][other] for other in labels)),
            "predicted": float(sum(matrix[other][label] for other in labels)),
            "recall": _safe_div(tp, tp + fn),
            "precision": _safe_div(tp, tp + fp),
            "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        }
    observed = _safe_div(sum(matrix[label][label] for label in labels), n)
    # Macro averages span only classes the reference actually uses, so absent
    # categories cannot drag the mean toward zero.
    present = [label for label in labels if per_class[label]["support"] > 0]
    nominal_kappa = cohen_kappa(y_true, y_pred, labels=labels)
    output: Dict[str, Any] = {
        "n": float(n),
        "labels": list(labels),
        "accuracy": observed,
        "cohen_kappa_nominal": nominal_kappa,
        "cohen_kappa": nominal_kappa,
        "macro_precision": _macro_average(per_class, present, "precision"),
        "macro_recall": _macro_average(per_class, present, "recall"),
        "macro_f1": _macro_average(per_class, present, "f1"),
        "weighted_f1": _support_weighted_average(per_class, present, "f1"),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "true_distribution": {label: float(sum(matrix[label][o] for o in labels)) for label in labels},
        "pred_distribution": {label: float(sum(matrix[o][label] for o in labels)) for label in labels},
    }
    if ordinal:
        linear_kappa = cohen_kappa(y_true, y_pred, labels=labels, weights="linear")
        quadratic_kappa = cohen_kappa(y_true, y_pred, labels=labels, weights="quadratic")
        output["cohen_kappa_linear_weighted"] = linear_kappa
        output["cohen_kappa_quadratic_weighted"] = quadratic_kappa
        # Linear weighted κ is the primary agreement statistic for ordinal scales.
        output["cohen_kappa"] = linear_kappa
    return output


def _macro_average(
    per_class: Dict[str, Dict[str, Optional[float]]], labels: List[str], key: str
) -> Optional[float]:
    values = [per_class[label][key] for label in labels if per_class[label][key] is not None]
    return sum(values) / len(values) if values else None


def _support_weighted_average(
    per_class: Dict[str, Dict[str, Optional[float]]], labels: List[str], key: str
) -> Optional[float]:
    pairs = [
        (per_class[label][key], per_class[label]["support"])
        for label in labels
        if per_class[label][key] is not None
    ]
    total = sum(support for _, support in pairs)
    if not pairs or total == 0:
        return None
    return sum(value * support for value, support in pairs) / total


def confusion_matrix_with_labels(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, Dict[str, int]]:
    """Confusion matrix over a fixed label set (rows = reference, cols = prediction)."""

    matrix = {truth: {pred: 0 for pred in labels} for truth in labels}
    for truth, pred in zip(y_true, y_pred):
        if truth not in matrix or pred not in matrix[truth]:
            raise ValueError(f"Unexpected label pair ({truth}, {pred}) outside {labels}")
        matrix[truth][pred] += 1
    return matrix


def _cohen_kappa_fallback(y_true: List[str], y_pred: List[str]) -> Optional[float]:
    labels = sorted(set(y_true) | set(y_pred))
    if len(labels) < 2:
        return 1.0 if y_true == y_pred else 0.0
    n = len(y_true)
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    po = sum(matrix[label][label] for label in labels) / n
    truth_marginal = {label: sum(matrix[label][pred] for pred in labels) / n for label in labels}
    pred_marginal = {label: sum(matrix[truth][label] for truth in labels) / n for label in labels}
    pe = sum(truth_marginal[label] * pred_marginal[label] for label in labels)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def confusion_matrix(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
    """Return a label-by-label confusion matrix."""

    labels = sorted(set(y_true) | set(y_pred))
    matrix = {truth: {pred: 0 for pred in labels} for truth in labels}
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    return matrix


def organ_specific_metrics(
    y_true: List[str],
    y_pred: List[str],
    organs: List[str],
    positive_label: str = "actionable",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute capture metrics per organ."""

    output: Dict[str, Dict[str, Optional[float]]] = {}
    for organ in sorted(set(organs)):
        indices = [idx for idx, value in enumerate(organs) if value == organ]
        output[organ] = classification_metrics(
            [y_true[idx] for idx in indices],
            [y_pred[idx] for idx in indices],
            positive_label=positive_label,
        )
    return output


def false_positive_analysis(y_true: List[str], y_pred: List[str], records: List[dict], positive_label: str = "actionable") -> List[dict]:
    """Return records predicted positive when the label is negative."""

    return [records[idx] for idx, (truth, pred) in enumerate(zip(y_true, y_pred)) if truth != positive_label and pred == positive_label]


def false_negative_analysis(y_true: List[str], y_pred: List[str], records: List[dict], positive_label: str = "actionable") -> List[dict]:
    """Return records predicted negative when the label is positive."""

    return [records[idx] for idx, (truth, pred) in enumerate(zip(y_true, y_pred)) if truth == positive_label and pred != positive_label]


def recommendation_concordance(expected: List[str], observed: List[str]) -> Dict[str, Optional[float]]:
    """Measure exact recommendation concordance for labeled eval sets."""

    if len(expected) != len(observed):
        raise ValueError("expected and observed must have the same length")
    return {"recommendation_concordance": _safe_div(sum(e == o for e, o in zip(expected, observed)), len(expected))}


def human_review_burden(results: Iterable[WorkflowResult]) -> Dict[str, float]:
    """Compute report and finding-level human review burden."""

    result_list = list(results)
    findings = [finding for result in result_list for finding in result.capture_result.findings]
    review_findings = [
        finding
        for result in result_list
        for finding in result.capture_result.findings
        if finding.human_review_required
        or finding.finding_id in result.provider_assignments
        and result.provider_assignments[finding.finding_id].human_review_required
        or finding.finding_id in result.follow_up_recommendations
        and result.follow_up_recommendations[finding.finding_id].human_review_required
    ]
    return {
        "report_human_review_rate": _rate(sum(result.final_status == "human_review_required" for result in result_list), len(result_list)),
        "finding_human_review_rate": _rate(len(review_findings), len(findings)),
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(_safe_div(numerator, denominator) or 0.0)


def _safe_div(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator
