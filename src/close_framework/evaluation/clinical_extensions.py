"""Clinically oriented paper metrics beyond aggregate κ.

Focuses on per-class sensitivity (esp. highest-urgency tier), directional
severity errors, stage-wise oracle ablation, recovery audit, abstention curves,
and run-to-run stability helpers.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from close_framework.evaluation.agent_eval import (
    evaluate_agents,
    modality_family,
    timeframe_class,
)
from close_framework.evaluation.gold_eval import (
    ACTIONABILITY_SCALE,
    IN_SCOPE,
    URGENCY_ORDINAL_SCALE,
    gold_findings,
    has_actionable_prediction,
    organ_actionability_categories,
    organs_from_prediction,
    prediction_findings,
    report_actionability_category,
    report_urgency_category,
)
from close_framework.evaluation.metrics import (
    binary_metrics,
    classification_metrics,
    cohen_kappa,
    multiclass_agreement_metrics,
)
from close_framework.evaluation.paper_analyses import ACTION_RANK, end_to_end_clinical_errors

NEGATION_SPAN = re.compile(
    r"\b(no|without|absent|unremarkable|normal|negative for)\b",
    re.I,
)
NEGATIVE_FINDING = re.compile(
    r"\b(no focal lesions?|no .* (cyst|mass|lesion|nodule)|unremarkable|within normal limits)\b",
    re.I,
)


def _ids(a: Dict[str, dict], b: Dict[str, dict]) -> List[str]:
    return sorted(set(a) & set(b))


def _safe(v: Optional[float]) -> Optional[float]:
    return None if v is None else float(v)


def per_class_actionability_metrics(
    gold: Dict[str, dict],
    predictions: Dict[str, dict],
    model_name: str,
    configuration: str,
) -> dict:
    """One-vs-rest sens/spec + ordinal κ disclosure for the 4-level scale."""

    ids = _ids(gold, predictions)
    y_true = [report_actionability_category(gold_findings(gold[i])) for i in ids]
    y_pred = [report_actionability_category(prediction_findings(predictions[i])) for i in ids]
    multi = multiclass_agreement_metrics(
        y_true, y_pred, labels=list(ACTIONABILITY_SCALE), ordinal=True
    )

    per_class = {}
    for label in ACTIONABILITY_SCALE:
        cm = classification_metrics(y_true, y_pred, positive_label=label)
        per_class[label] = {
            "support": multi["per_class"][label]["support"],
            "sensitivity": cm["sensitivity"],
            "specificity": cm["specificity"],
            "ppv": cm["positive_predictive_value"],
            "npv": cm["negative_predictive_value"],
            "f1": cm["f1"],
            "prevalence": cm["prevalence"],
        }

    # Highest-urgency actionability tier = "actionable"
    actionable = per_class["actionable"]
    # Ordinal urgency among reports (urgent / short_interval)
    urg_true = [report_urgency_category(gold_findings(gold[i])) for i in ids]
    urg_pred = [report_urgency_category(prediction_findings(predictions[i])) for i in ids]
    high_true = [u in {"urgent", "short_interval"} for u in urg_true]
    high_pred = [u in {"urgent", "short_interval"} for u in urg_pred]
    high_urg = binary_metrics(high_true, high_pred)
    urgent_true = [u == "urgent" for u in urg_true]
    urgent_pred = [u == "urgent" for u in urg_pred]
    urgent_only = binary_metrics(urgent_true, urgent_pred)

    return {
        "model": model_name,
        "configuration": configuration,
        "n": len(ids),
        "true_distribution": multi["true_distribution"],
        "pred_distribution": multi["pred_distribution"],
        "kappa_nominal": multi["cohen_kappa_nominal"],
        "kappa_linear_weighted": multi["cohen_kappa_linear_weighted"],
        "kappa_quadratic_weighted": multi["cohen_kappa_quadratic_weighted"],
        "primary_kappa": multi["cohen_kappa"],  # linear when ordinal=True
        "kappa_weighting_note": (
            "Primary reported κ is linear-weighted Cohen's κ on the ordinal scale "
            "none < non_actionable < uncertain < actionable. Quadratic κ also reported. "
            "Unweighted (nominal) κ is shown for completeness but is not appropriate as "
            "the primary ordinal endpoint."
        ),
        "per_class": per_class,
        "actionable_sensitivity": actionable["sensitivity"],
        "actionable_specificity": actionable["specificity"],
        "high_urgency_binary": high_urg,
        "urgent_only_binary": urgent_only,
        "n_true_actionable": int(multi["true_distribution"].get("actionable", 0)),
        "n_true_high_urgency": int(sum(high_true)),
        "n_true_urgent": int(sum(urgent_true)),
    }


def per_organ_clinical_metrics(
    gold: Dict[str, dict],
    predictions: Dict[str, dict],
    model_name: str,
    configuration: str,
) -> List[dict]:
    """Per-organ presence + 4-level actionability clinical metrics."""

    ids = _ids(gold, predictions)
    rows: List[dict] = []
    for organ in IN_SCOPE:
        yt_pres = [organ in (gold[i].get("organs_present") or []) for i in ids]
        yp_pres = [organ in organs_from_prediction(predictions[i]) for i in ids]
        presence = binary_metrics(yt_pres, yp_pres)

        yt = [organ_actionability_categories(gold_findings(gold[i]))[organ] for i in ids]
        yp = [organ_actionability_categories(prediction_findings(predictions[i]))[organ] for i in ids]
        multi = multiclass_agreement_metrics(yt, yp, labels=list(ACTIONABILITY_SCALE), ordinal=True)
        act = classification_metrics(yt, yp, positive_label="actionable")
        rows.append(
            {
                "model": model_name,
                "configuration": configuration,
                "organ": organ,
                "n_true_present": int(sum(yt_pres)),
                "n_true_actionable": int(sum(t == "actionable" for t in yt)),
                "presence_sensitivity": presence["sensitivity"],
                "presence_specificity": presence["specificity"],
                "presence_kappa": presence["cohen_kappa"],
                "actionability_linear_kappa": multi["cohen_kappa_linear_weighted"],
                "actionability_quadratic_kappa": multi["cohen_kappa_quadratic_weighted"],
                "actionable_sensitivity": act["sensitivity"],
                "actionable_specificity": act["specificity"],
                "actionable_ppv": act["positive_predictive_value"],
                "true_distribution": multi["true_distribution"],
            }
        )
    return rows


def directional_severity_errors(
    gold: Dict[str, dict],
    predictions: Dict[str, dict],
    model_name: str,
    configuration: str,
) -> dict:
    """Separate undercalls vs overcalls with ordinal distance matrix."""

    ids = _ids(gold, predictions)
    y_true = [report_actionability_category(gold_findings(gold[i])) for i in ids]
    y_pred = [report_actionability_category(prediction_findings(predictions[i])) for i in ids]
    multi = multiclass_agreement_metrics(
        y_true, y_pred, labels=list(ACTIONABILITY_SCALE), ordinal=True
    )

    under = over = exact = 0
    under_ge2 = over_ge2 = 0
    signed = []
    # Distance matrix: rows=true, cols=pred, values=signed rank delta counts
    delta_counts = Counter()
    for t, p in zip(y_true, y_pred):
        d = ACTION_RANK[p] - ACTION_RANK[t]
        signed.append(d)
        delta_counts[d] += 1
        if d == 0:
            exact += 1
        elif d < 0:
            under += 1
            if d <= -2:
                under_ge2 += 1
        else:
            over += 1
            if d >= 2:
                over_ge2 += 1

    n = len(ids)
    # Safety-critical: true actionable missed or downgraded
    true_actionable = [(t, p) for t, p in zip(y_true, y_pred) if t == "actionable"]
    actionable_missed = sum(1 for t, p in true_actionable if p != "actionable")
    actionable_to_none = sum(1 for t, p in true_actionable if p == "none")

    return {
        "model": model_name,
        "configuration": configuration,
        "n": n,
        "exact_agreement": exact,
        "exact_rate": exact / n if n else None,
        "undercalls": under,
        "overcalls": over,
        "undercall_rate": under / n if n else None,
        "overcall_rate": over / n if n else None,
        "undercalls_off_by_ge2": under_ge2,
        "overcalls_off_by_ge2": over_ge2,
        "mean_signed_severity_bias": (sum(signed) / n) if n else None,
        "delta_count_by_signed_rank": {str(k): v for k, v in sorted(delta_counts.items())},
        "confusion_matrix": multi["confusion_matrix"],
        "n_true_actionable": len(true_actionable),
        "actionable_not_called_actionable": actionable_missed,
        "actionable_sensitivity": (
            1.0 - actionable_missed / len(true_actionable) if true_actionable else None
        ),
        "actionable_collapsed_to_none": actionable_to_none,
        "operating_preference": (
            "Prefer overcalls to undercalls for safety-critical incidental findings: "
            "underrating severity is a potential missed cancer; overrating drives cost "
            "and anxiety. Deterministic ACR recalibration is included specifically to "
            "reduce LLM severity undercalls."
        ),
    }


def oracle_stage_ablation(
    reference: Dict[str, dict],
    llm_only: Dict[str, dict],
    hybrid: Dict[str, dict],
    model_name: str,
) -> dict:
    """Stage-wise attribution: LLM / hybrid / oracle-Capture ceilings vs reference.

    Oracle Capture = reference Capture findings (hand-corrected / ACR engine).
    Because Orchestrate is the shared deterministic mapper, oracle E2E concordance
    with the reference pipeline should approach 1.0; residual gap is Orchestrate
    non-identity (should be ~0). Errors that disappear when Capture is replaced
    by oracle are Capture-attributable.
    """

    ids = _ids(reference, llm_only)
    ids = [i for i in ids if i in hybrid]

    def _followup_needed(result: dict) -> bool:
        return bool(result.get("follow_up_recommendations"))

    def _organs_actionable(result: dict) -> Set[str]:
        return {
            str(f.get("organ_system"))
            for f in prediction_findings(result)
            if f.get("actionability") in {"actionable", "uncertain"}
        }

    arms = {
        "llm_only": llm_only,
        "full_hybrid": hybrid,
        "oracle_capture": reference,  # perfect Capture by construction
    }
    arm_metrics = {}
    for arm_name, arm in arms.items():
        y_true = [_followup_needed(reference[i]) for i in ids]
        y_pred = [_followup_needed(arm[i]) for i in ids]
        fu = binary_metrics(y_true, y_pred)
        agents = evaluate_agents(reference, arm, model_name=f"{model_name}/{arm_name}")
        orch = agents.get("orchestrate") or {}
        e2e = end_to_end_clinical_errors(reference, arm, f"{model_name}/{arm_name}")
        arm_metrics[arm_name] = {
            "followup_needed": fu,
            "orchestrate_guideline_kappa": (orch.get("guideline_category") or {}).get(
                "cohen_kappa"
            ),
            "orchestrate_modality_kappa": (orch.get("modality_family_category") or {}).get(
                "cohen_kappa"
            ),
            "end_to_end_errors": e2e.get("error_counts"),
            "capture_driven_errors": e2e.get("errors_with_capture_mismatch"),
            "despite_capture_errors": e2e.get("errors_despite_matching_capture_organs"),
        }

    # Conditional Orchestrate ceiling: among reports where hybrid Capture organ
    # set matches reference actionable organs, measure modality/timeframe match.
    match_ids = []
    mod_agree = tf_agree = 0
    compared = 0
    for i in ids:
        if _organs_actionable(reference[i]) != _organs_actionable(hybrid[i]):
            continue
        match_ids.append(i)
        ref_recs = list((reference[i].get("follow_up_recommendations") or {}).values())
        hyb_recs = list((hybrid[i].get("follow_up_recommendations") or {}).values())
        if not ref_recs and not hyb_recs:
            mod_agree += 1
            tf_agree += 1
            compared += 1
            continue
        if not ref_recs or not hyb_recs:
            compared += 1
            continue
        compared += 1
        if modality_family(ref_recs[0].get("recommended_modality")) == modality_family(
            hyb_recs[0].get("recommended_modality")
        ):
            mod_agree += 1
        if timeframe_class(ref_recs[0].get("recommended_timeframe")) == timeframe_class(
            hyb_recs[0].get("recommended_timeframe")
        ):
            tf_agree += 1

    llm_miss = arm_metrics["llm_only"]["followup_needed"].get("false_negatives") or 0
    hyb_miss = arm_metrics["full_hybrid"]["followup_needed"].get("false_negatives") or 0
    ora_miss = arm_metrics["oracle_capture"]["followup_needed"].get("false_negatives") or 0

    return {
        "model": model_name,
        "n": len(ids),
        "arms": arm_metrics,
        "interpretation": (
            "Oracle Capture uses the ACR reference Capture output then the shared "
            "Orchestrate engine. Near-ceiling oracle concordance implies remaining "
            "end-to-end error is extraction (Capture), not guideline mapping "
            "(Orchestrate)."
        ),
        "capture_attributable_missed_followups": {
            "llm_only_fn": llm_miss,
            "hybrid_fn": hyb_miss,
            "oracle_fn": ora_miss,
            "resolved_by_hybrid_vs_llm": llm_miss - hyb_miss,
            "remaining_vs_oracle": hyb_miss - ora_miss,
        },
        "conditional_on_matching_capture_organs": {
            "n_reports": len(match_ids),
            "modality_agreement_rate": mod_agree / compared if compared else None,
            "timeframe_agreement_rate": tf_agree / compared if compared else None,
        },
    }


def _finding_key(finding: dict) -> Tuple[str, Optional[float]]:
    size = finding.get("size_mm")
    try:
        size_f = round(float(size), 1) if size is not None else None
    except (TypeError, ValueError):
        size_f = None
    return (str(finding.get("organ_system")), size_f)


def _findings_overlap(a: dict, b: dict) -> bool:
    if str(a.get("organ_system")) != str(b.get("organ_system")):
        return False
    ea = (a.get("evidence_span") or "").strip().lower()
    eb = (b.get("evidence_span") or "").strip().lower()
    if ea and eb and (ea in eb or eb in ea):
        return True
    sa, sb = a.get("size_mm"), b.get("size_mm")
    if sa is not None and sb is not None:
        try:
            return abs(float(sa) - float(sb)) <= 2.0
        except (TypeError, ValueError):
            return False
    return False


def _text_supports_finding(evidence: str, report_text: str) -> dict:
    evidence = (evidence or "").strip()
    text = report_text or ""
    if not evidence:
        return {
            "in_report": False,
            "negated_or_negative": True,
            "supported": False,
            "reason": "empty_evidence_span",
        }
    in_report = evidence.lower() in text.lower()
    negated = bool(NEGATION_SPAN.search(evidence) or NEGATIVE_FINDING.search(evidence))
    # Organ header + negative statement is not a finding
    supported = in_report and not negated
    reason = "supported"
    if not in_report:
        reason = "evidence_not_in_report"
    elif negated:
        reason = "negated_or_negative_language"
    return {
        "in_report": in_report,
        "negated_or_negative": negated,
        "supported": supported,
        "reason": reason,
    }


def recovery_module_audit(
    gold: Dict[str, dict],
    llm_only: Dict[str, dict],
    hybrid: Dict[str, dict],
    report_texts: Dict[str, str],
    model_name: str,
) -> dict:
    """Audit missed-finding recovery (supplemented_from_rules) separately."""

    ids = _ids(gold, hybrid)
    supplemented: List[dict] = []
    for report_id in ids:
        for finding in prediction_findings(hybrid[report_id]):
            meta = finding.get("metadata") or {}
            if meta.get("supplemented_from_rules"):
                support = _text_supports_finding(
                    finding.get("evidence_span") or "",
                    report_texts.get(report_id, ""),
                )
                matched_gold = any(
                    _findings_overlap(finding, g) for g in gold_findings(gold[report_id])
                )
                supplemented.append(
                    {
                        "report_id": report_id,
                        "organ_system": finding.get("organ_system"),
                        "actionability": finding.get("actionability"),
                        "size_mm": finding.get("size_mm"),
                        "evidence_span": (finding.get("evidence_span") or "")[:240],
                        "matched_gold": matched_gold,
                        **support,
                    }
                )

    n_sup = len(supplemented)
    n_supported = sum(1 for s in supplemented if s["supported"])
    n_matched_gold = sum(1 for s in supplemented if s["matched_gold"])
    n_fabricated = sum(1 for s in supplemented if not s["supported"])

    # Recall of recovery: gold findings absent from llm_only that appear in hybrid
    recovered_tp = 0
    missed_unrecovered = 0
    for report_id in ids:
        g_finds = gold_findings(gold[report_id])
        llm_finds = prediction_findings(llm_only[report_id])
        hyb_finds = prediction_findings(hybrid[report_id])
        for g in g_finds:
            in_llm = any(_findings_overlap(g, p) for p in llm_finds)
            if in_llm:
                continue
            in_hyb = any(_findings_overlap(g, p) for p in hyb_finds)
            if in_hyb:
                recovered_tp += 1
            else:
                missed_unrecovered += 1

    denom_recall = recovered_tp + missed_unrecovered
    unsupported_examples = [s for s in supplemented if not s["supported"]]

    return {
        "model": model_name,
        "n_reports": len(ids),
        "n_supplemented_findings": n_sup,
        "precision_vs_gold": n_matched_gold / n_sup if n_sup else None,
        "text_support_precision": n_supported / n_sup if n_sup else None,
        "fabrication_rate_text_audit": n_fabricated / n_sup if n_sup else None,
        "n_fabricated_or_unsupported": n_fabricated,
        "recovery_recall_of_llm_misses": (
            recovered_tp / denom_recall if denom_recall else None
        ),
        "n_llm_misses_recovered": recovered_tp,
        "n_llm_misses_unrecovered": missed_unrecovered,
        "note": (
            "Precision vs gold is partly circular (same ACR rule engine). "
            "Text-support precision / fabrication rate are the non-circular audit: "
            "every supplemented evidence_span checked against report text for "
            "presence and non-negation."
        ),
        "unsupported_examples": unsupported_examples[:100],
        "all_unsupported": unsupported_examples,
    }


def abstention_curve(
    gold: Dict[str, dict],
    predictions: Dict[str, dict],
    model_name: str,
    configuration: str,
    thresholds: Optional[Sequence[float]] = None,
) -> dict:
    """Route uncertain cases to human review; plot referral fraction vs safety.

    Report-level confidence = minimum finding confidence (1.0 if no findings).
    A report is referred if min_confidence < threshold OR any finding has
    human_review_required.
    Safety metric among auto-accepted cases: sensitivity for actionable class
    and for high-urgency (urgent|short_interval).
    """

    thresholds = list(thresholds) if thresholds is not None else [
        round(x, 2) for x in [i / 20 for i in range(0, 21)]
    ]
    ids = _ids(gold, predictions)
    rows = []
    for report_id in ids:
        finds = prediction_findings(predictions[report_id])
        confs = [float(f["confidence"]) for f in finds if f.get("confidence") is not None]
        min_conf = min(confs) if confs else 1.0
        hr = any(bool(f.get("human_review_required")) for f in finds)
        true_cat = report_actionability_category(gold_findings(gold[report_id]))
        pred_cat = report_actionability_category(finds)
        true_high = report_urgency_category(gold_findings(gold[report_id])) in {
            "urgent",
            "short_interval",
        }
        pred_high = report_urgency_category(finds) in {"urgent", "short_interval"}
        rows.append(
            {
                "report_id": report_id,
                "min_confidence": min_conf,
                "human_review_flag": hr,
                "true_actionable": true_cat == "actionable",
                "pred_actionable": pred_cat == "actionable",
                "true_high_urgency": true_high,
                "pred_high_urgency": pred_high,
            }
        )

    curve = []
    n = len(rows)
    for t in thresholds:
        referred = [
            r for r in rows if r["human_review_flag"] or r["min_confidence"] < t
        ]
        accepted = [
            r for r in rows if not (r["human_review_flag"] or r["min_confidence"] < t)
        ]
        # Safety on auto-accepted: sensitivity among accepted that are true positives
        # Also: overall missed actionable that were NOT referred (dangerous misses)
        true_act = [r for r in rows if r["true_actionable"]]
        dangerous_miss = sum(
            1
            for r in true_act
            if (not r["pred_actionable"])
            and not (r["human_review_flag"] or r["min_confidence"] < t)
        )
        if accepted:
            yt = [r["true_actionable"] for r in accepted]
            yp = [r["pred_actionable"] for r in accepted]
            act_m = binary_metrics(yt, yp)
            yh_t = [r["true_high_urgency"] for r in accepted]
            yh_p = [r["pred_high_urgency"] for r in accepted]
            urg_m = binary_metrics(yh_t, yh_p)
        else:
            act_m = {k: None for k in ("sensitivity", "specificity", "cohen_kappa")}
            urg_m = act_m
        curve.append(
            {
                "threshold": t,
                "fraction_referred": len(referred) / n if n else None,
                "n_referred": len(referred),
                "n_auto_accepted": len(accepted),
                "auto_actionable_sensitivity": act_m.get("sensitivity"),
                "auto_actionable_specificity": act_m.get("specificity"),
                "auto_high_urgency_sensitivity": urg_m.get("sensitivity"),
                "dangerous_actionable_misses_not_referred": dangerous_miss,
                "dangerous_miss_rate_among_true_actionable": (
                    dangerous_miss / len(true_act) if true_act else None
                ),
            }
        )

    return {
        "model": model_name,
        "configuration": configuration,
        "n": n,
        "curve": curve,
        "policy": (
            "Refer if any finding has human_review_required OR report min finding "
            "confidence < threshold. Safety = actionable/high-urgency sensitivity "
            "on auto-accepted cases; dangerous misses = true actionable not predicted "
            "and not referred."
        ),
    }


def run_to_run_stability(
    runs: Dict[str, Dict[str, dict]],
    model_name: str,
) -> dict:
    """Pairwise self-agreement across independently generated Capture JSONLs."""

    from itertools import combinations

    names = sorted(runs)
    ids = None
    for name in names:
        ids = set(runs[name]) if ids is None else (ids & set(runs[name]))
    ids = sorted(ids or [])
    pairs = []
    for a, b in combinations(names, 2):
        ca = [report_actionability_category(prediction_findings(runs[a][i])) for i in ids]
        cb = [report_actionability_category(prediction_findings(runs[b][i])) for i in ids]
        pa = [bool(prediction_findings(runs[a][i])) for i in ids]
        pb = [bool(prediction_findings(runs[b][i])) for i in ids]
        aa = [has_actionable_prediction(runs[a][i]) for i in ids]
        ab = [has_actionable_prediction(runs[b][i]) for i in ids]
        pairs.append(
            {
                "run_a": a,
                "run_b": b,
                "n": len(ids),
                "presence_kappa": binary_metrics(pa, pb)["cohen_kappa"],
                "actionable_kappa": binary_metrics(aa, ab)["cohen_kappa"],
                "actionability_linear_kappa": cohen_kappa(
                    ca, cb, labels=list(ACTIONABILITY_SCALE), weights="linear"
                ),
                "actionability_exact_agreement": (
                    sum(x == y for x, y in zip(ca, cb)) / len(ids) if ids else None
                ),
            }
        )
    return {
        "model": model_name,
        "n_runs": len(names),
        "n_shared_reports": len(ids),
        "run_names": names,
        "pairwise": pairs,
    }
