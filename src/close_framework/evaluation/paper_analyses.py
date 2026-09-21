"""Paper-facing evaluation analyses beyond the base gold/agent packs.

All agreement metrics are vs the deterministic ACR-derived reference standard
(``data/gold_labels`` / ``outputs/reference_rule_based``). These quantify
operational concordance with a guideline engine, not independent clinical
accuracy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from close_framework.evaluation.bootstrap import (
    bootstrap_binary_metrics,
    bootstrap_linear_kappa,
    paired_metric_difference_ci,
)
from close_framework.evaluation.gold_eval import (
    ACTIONABILITY_SCALE,
    IN_SCOPE,
    gold_findings,
    has_actionable_prediction,
    organs_from_prediction,
    prediction_findings,
    report_actionability_category,
    organ_actionability_categories,
)
from close_framework.evaluation.metrics import (
    binary_metrics,
    cohen_kappa,
    multiclass_agreement_metrics,
)
from close_framework.evaluation.agent_eval import (
    evaluate_agents,
    modality_family,
    timeframe_class,
)

ACTION_RANK = {label: i for i, label in enumerate(ACTIONABILITY_SCALE)}


def _ids(reference: Dict[str, dict], predictions: Dict[str, dict]) -> List[str]:
    return sorted(set(reference) & set(predictions))


def filter_ids(
    gold: Dict[str, dict],
    ids: Sequence[str],
    *,
    positive_only: bool = False,
    organs: Optional[Set[str]] = None,
    min_findings: Optional[int] = None,
    max_findings: Optional[int] = None,
    multi_organ: Optional[bool] = None,
) -> List[str]:
    out: List[str] = []
    for report_id in ids:
        label = gold[report_id]
        n_findings = int(label.get("n_findings") or len(gold_findings(label)))
        organs_present = set(label.get("organs_present") or [])
        if positive_only and not label.get("has_in_scope_finding"):
            continue
        if organs is not None and not (organs_present & organs):
            continue
        if min_findings is not None and n_findings < min_findings:
            continue
        if max_findings is not None and n_findings > max_findings:
            continue
        if multi_organ is True and len(organs_present) < 2:
            continue
        if multi_organ is False and len(organs_present) != 1:
            continue
        out.append(report_id)
    return out


def evaluate_capture_subset(
    gold: Dict[str, dict],
    predictions: Dict[str, dict],
    model_name: str,
    ids: Optional[Sequence[str]] = None,
    *,
    n_boot: int = 400,
    seed: int = 0,
    bootstrap_organs: bool = True,
) -> dict:
    """Capture agreement on an arbitrary report subset, with bootstrap CIs."""

    ids = list(ids) if ids is not None else _ids(gold, predictions)
    gold_finds = {i: gold_findings(gold[i]) for i in ids}
    pred_finds = {i: prediction_findings(predictions[i]) for i in ids}

    y_presence_true = [bool(gold[i]["has_in_scope_finding"]) for i in ids]
    y_presence_pred = [bool(pred_finds[i]) for i in ids]
    y_action_true = [bool(gold[i]["has_actionable_finding"]) for i in ids]
    y_action_pred = [has_actionable_prediction(predictions[i]) for i in ids]

    report_action_true = [report_actionability_category(gold_finds[i]) for i in ids]
    report_action_pred = [report_actionability_category(pred_finds[i]) for i in ids]

    organ_metrics = {}
    for organ in IN_SCOPE:
        yt = [organ in (gold[i].get("organs_present") or []) for i in ids]
        yp = [organ in organs_from_prediction(predictions[i]) for i in ids]
        organ_metrics[organ] = {
            "metrics": binary_metrics(yt, yp),
            "bootstrap": (
                bootstrap_binary_metrics(yt, yp, n_boot=n_boot, seed=seed)
                if bootstrap_organs and n_boot > 0
                else {
                    k: {"estimate": binary_metrics(yt, yp).get(k), "ci_low": None, "ci_high": None, "n_boot": 0.0}
                    for k in (
                        "sensitivity",
                        "specificity",
                        "positive_predictive_value",
                        "f1",
                        "balanced_accuracy",
                        "cohen_kappa",
                        "accuracy",
                    )
                }
            ),
        }

    # Finding-count / size agreement among shared organs
    count_pairs: List[Tuple[int, int]] = []
    size_pairs: List[Tuple[float, float]] = []
    size_within_2mm = 0
    size_comparable = 0
    for report_id in ids:
        gf = gold_finds[report_id]
        pf = pred_finds[report_id]
        count_pairs.append((len(gf), len(pf)))
        gold_by_organ: Dict[str, List[dict]] = defaultdict(list)
        pred_by_organ: Dict[str, List[dict]] = defaultdict(list)
        for finding in gf:
            gold_by_organ[str(finding.get("organ_system"))].append(finding)
        for finding in pf:
            pred_by_organ[str(finding.get("organ_system"))].append(finding)
        for organ in set(gold_by_organ) & set(pred_by_organ):
            g_sizes = [f.get("size_mm") for f in gold_by_organ[organ] if f.get("size_mm") is not None]
            p_sizes = [f.get("size_mm") for f in pred_by_organ[organ] if f.get("size_mm") is not None]
            for g_size, p_size in zip(sorted(g_sizes), sorted(p_sizes)):
                size_comparable += 1
                size_pairs.append((float(g_size), float(p_size)))
                if abs(float(g_size) - float(p_size)) <= 2.0:
                    size_within_2mm += 1

    count_mae = (
        sum(abs(a - b) for a, b in count_pairs) / len(count_pairs) if count_pairs else None
    )
    size_mae = (
        sum(abs(a - b) for a, b in size_pairs) / len(size_pairs) if size_pairs else None
    )

    categorical = multiclass_agreement_metrics(
        report_action_true,
        report_action_pred,
        labels=list(ACTIONABILITY_SCALE),
        ordinal=True,
    )

    return {
        "model": model_name,
        "n_reports": len(ids),
        "reference_note": (
            "Agreement with deterministic ACR-derived Capture reference; "
            "not radiologist-adjudicated clinical accuracy."
        ),
        "presence": {
            "metrics": binary_metrics(y_presence_true, y_presence_pred),
            "bootstrap": bootstrap_binary_metrics(
                y_presence_true, y_presence_pred, n_boot=n_boot, seed=seed
            ),
        },
        "actionable": {
            "metrics": binary_metrics(y_action_true, y_action_pred),
            "bootstrap": bootstrap_binary_metrics(
                y_action_true, y_action_pred, n_boot=n_boot, seed=seed
            ),
        },
        "report_actionability": {
            "metrics": categorical,
            "bootstrap_linear_kappa": bootstrap_linear_kappa(
                report_action_true,
                report_action_pred,
                ACTIONABILITY_SCALE,
                n_boot=n_boot,
                seed=seed,
            ),
        },
        "organ_presence": organ_metrics,
        "finding_count_mae": count_mae,
        "exact_finding_count_accuracy": (
            sum(a == b for a, b in count_pairs) / len(count_pairs) if count_pairs else None
        ),
        "size_mae_mm_matched_organs": size_mae,
        "size_within_2mm_rate": (
            size_within_2mm / size_comparable if size_comparable else None
        ),
        "size_comparable_pairs": float(size_comparable),
        "n_reference_positive": float(sum(y_presence_true)),
        "n_reference_actionable": float(sum(y_action_true)),
    }


def cohort_characterization(gold: Dict[str, dict], reference_pipeline: Dict[str, dict]) -> dict:
    ids = sorted(gold)
    n_findings = [int(gold[i].get("n_findings") or 0) for i in ids]
    organs = [set(gold[i].get("organs_present") or []) for i in ids]
    multi_finding = sum(n > 1 for n in n_findings)
    multi_organ = sum(len(o) > 1 for o in organs)
    organ_counts = Counter()
    for organs_present in organs:
        for organ in organs_present:
            organ_counts[organ] += 1

    ref_recs = sum(len(reference_pipeline.get(i, {}).get("follow_up_recommendations") or {}) for i in ids)
    ref_assign = sum(len(reference_pipeline.get(i, {}).get("provider_assignments") or {}) for i in ids)
    status = Counter(
        str(reference_pipeline.get(i, {}).get("final_status") or "missing") for i in ids
    )

    return {
        "n_reports": len(ids),
        "n_no_in_scope_finding": sum(not gold[i].get("has_in_scope_finding") for i in ids),
        "n_with_in_scope_finding": sum(bool(gold[i].get("has_in_scope_finding")) for i in ids),
        "n_with_actionable_finding": sum(bool(gold[i].get("has_actionable_finding")) for i in ids),
        "n_multi_finding_reports": multi_finding,
        "n_multi_organ_reports": multi_organ,
        "n_single_finding_reports": sum(n == 1 for n in n_findings),
        "mean_findings_among_positives": (
            sum(n for n in n_findings if n > 0) / max(1, sum(n > 0 for n in n_findings))
        ),
        "organ_report_counts": dict(organ_counts),
        "report_actionability_distribution": dict(
            Counter(gold[i].get("report_actionability_category") for i in ids)
        ),
        "reference_pipeline_recommendations": ref_recs,
        "reference_pipeline_assignments": ref_assign,
        "reference_final_status_counts": dict(status),
        "disclaimer": (
            "Cohort labels from deterministic ACR Capture reference; "
            "not radiologist chart review."
        ),
    }


def hybrid_ablation_metrics(
    gold: Dict[str, dict],
    reference_pipeline: Dict[str, dict],
    arms: Dict[str, Dict[str, dict]],
    model_name: str,
    *,
    n_boot: int = 800,
    seed: int = 0,
) -> dict:
    """Compare LLM-only / calibrate-only / full-hybrid arms."""

    out = {"model": model_name, "arms": {}}
    for arm_name, predictions in arms.items():
        capture = evaluate_capture_subset(
            gold,
            predictions,
            f"{model_name}/{arm_name}",
            n_boot=n_boot,
            seed=seed,
            bootstrap_organs=False,
        )
        agents = evaluate_agents(reference_pipeline, predictions, f"{model_name}/{arm_name}")
        n_actionable = 0
        n_recs = 0
        for result in predictions.values():
            findings = prediction_findings(result)
            n_actionable += sum(f.get("actionability") == "actionable" for f in findings)
            n_recs += len(result.get("follow_up_recommendations") or {})
        out["arms"][arm_name] = {
            "capture": capture,
            "orchestrate_guideline_kappa": agents["orchestrate"]["guideline_category"]["cohen_kappa"],
            "orchestrate_any_recommendation": agents["orchestrate"]["any_recommendation"],
            "n_actionable_findings": n_actionable,
            "n_recommendations": n_recs,
            "rule_id_exact_match_when_both_recommend": agents["orchestrate"][
                "rule_id_exact_match_when_both_recommend"
            ],
        }

    # Incremental deltas on linear actionability κ
    if set(arms) >= {"llm_only", "calibrate", "full_hybrid"}:
        ids = _ids(gold, arms["llm_only"])

        def _labels(preds: Dict[str, dict]) -> List[str]:
            return [
                report_actionability_category(prediction_findings(preds[i])) for i in ids
            ]

        yt = [report_actionability_category(gold_findings(gold[i])) for i in ids]

        def _kappa(true_seq, pred_seq):
            return cohen_kappa(list(true_seq), list(pred_seq), labels=list(ACTIONABILITY_SCALE), weights="linear")

        out["incremental"] = {
            "calibrate_minus_llm_only": paired_metric_difference_ci(
                yt, _labels(arms["llm_only"]), _labels(arms["calibrate"]), _kappa, n_boot=n_boot, seed=seed
            ),
            "full_hybrid_minus_calibrate": paired_metric_difference_ci(
                yt, _labels(arms["calibrate"]), _labels(arms["full_hybrid"]), _kappa, n_boot=n_boot, seed=seed
            ),
            "full_hybrid_minus_llm_only": paired_metric_difference_ci(
                yt, _labels(arms["llm_only"]), _labels(arms["full_hybrid"]), _kappa, n_boot=n_boot, seed=seed
            ),
        }
    return out


def actionability_error_analysis(
    gold: Dict[str, dict],
    predictions: Dict[str, dict],
    model_name: str,
) -> dict:
    ids = _ids(gold, predictions)
    true_labels: List[str] = []
    pred_labels: List[str] = []
    distance_counts = Counter()
    taxonomy = Counter()
    organ_misses = Counter()
    organ_false_alarms = Counter()

    for report_id in ids:
        g_finds = gold_findings(gold[report_id])
        p_finds = prediction_findings(predictions[report_id])
        t = report_actionability_category(g_finds)
        p = report_actionability_category(p_finds)
        true_labels.append(t)
        pred_labels.append(p)

        dist = ACTION_RANK[p] - ACTION_RANK[t]
        if dist == 0:
            distance_counts["exact"] += 1
        elif abs(dist) == 1:
            distance_counts["off_by_1"] += 1
            taxonomy["actionability_adjacent_error"] += 1
        else:
            distance_counts["off_by_ge_2"] += 1
            taxonomy["actionability_distant_error"] += 1
        if dist < 0:
            taxonomy["undercall_actionability"] += 1
        elif dist > 0:
            taxonomy["overcall_actionability"] += 1

        g_organs = set(gold[report_id].get("organs_present") or [])
        p_organs = organs_from_prediction(predictions[report_id])
        if g_organs and not p_organs:
            taxonomy["finding_missed_report"] += 1
        if (not g_organs) and p_organs:
            taxonomy["false_positive_report"] += 1
        for organ in g_organs - p_organs:
            organ_misses[organ] += 1
            taxonomy["organ_miss"] += 1
        for organ in p_organs - g_organs:
            organ_false_alarms[organ] += 1
            taxonomy["organ_false_alarm"] += 1

        # Shared organs with severity mismatch
        for organ in g_organs & p_organs:
            gt = organ_actionability_categories(g_finds)[organ]
            pt = organ_actionability_categories(p_finds)[organ]
            if gt != pt:
                delta = ACTION_RANK[pt] - ACTION_RANK[gt]
                if delta < 0:
                    taxonomy["organ_actionability_undercall"] += 1
                elif delta > 0:
                    taxonomy["organ_actionability_overcall"] += 1

    confusion = multiclass_agreement_metrics(
        true_labels, pred_labels, labels=list(ACTIONABILITY_SCALE), ordinal=True
    )
    n = len(ids)
    return {
        "model": model_name,
        "n_reports": n,
        "confusion": confusion,
        "ordinal_distance_counts": dict(distance_counts),
        "ordinal_distance_rates": {k: v / n for k, v in distance_counts.items()},
        "taxonomy_counts": dict(taxonomy),
        "taxonomy_rates": {k: v / n for k, v in taxonomy.items()},
        "organ_miss_counts": dict(organ_misses),
        "organ_false_alarm_counts": dict(organ_false_alarms),
        "mean_signed_severity_bias": (
            sum(ACTION_RANK[p] - ACTION_RANK[t] for t, p in zip(true_labels, pred_labels)) / n
            if n
            else None
        ),
    }


def adnexal_failure_analysis(
    gold: Dict[str, dict],
    predictions: Dict[str, dict],
    model_name: str,
    max_examples: int = 25,
) -> dict:
    ids = _ids(gold, predictions)
    categories = Counter()
    examples: List[dict] = []

    for report_id in ids:
        g_organs = set(gold[report_id].get("organs_present") or [])
        p_organs = organs_from_prediction(predictions[report_id])
        g_finds = [f for f in gold_findings(gold[report_id]) if f.get("organ_system") == "adnexal"]
        p_finds = [f for f in prediction_findings(predictions[report_id]) if f.get("organ_system") == "adnexal"]
        ref_has = "adnexal" in g_organs
        pred_has = "adnexal" in p_organs
        if not ref_has and not pred_has:
            continue

        if ref_has and not pred_has:
            cat = "finding_missed"
        elif (not ref_has) and pred_has:
            cat = "false_positive_detection"
        else:
            gt = organ_actionability_categories(gold_findings(gold[report_id]))["adnexal"]
            pt = organ_actionability_categories(prediction_findings(predictions[report_id]))["adnexal"]
            if gt == pt:
                cat = "agree"
            elif ACTION_RANK[pt] < ACTION_RANK[gt]:
                # Heuristic sub-causes when both detected
                g_size = next((f.get("size_mm") for f in g_finds if f.get("size_mm") is not None), None)
                p_size = next((f.get("size_mm") for f in p_finds if f.get("size_mm") is not None), None)
                if g_size is not None and p_size is not None and abs(float(g_size) - float(p_size)) > 5:
                    cat = "size_mismatch_with_undercall"
                elif any((f.get("metadata") or {}).get("llm_actionability_before_recalibration") for f in p_finds):
                    cat = "actionability_threshold_undercall"
                else:
                    cat = "actionability_undercall"
            else:
                cat = "actionability_overcall"

        # Context flags that often matter clinically but may be unavailable
        texts = " ".join(
            str(f.get("evidence_span") or f.get("description") or "") for f in g_finds + p_finds
        ).lower()
        if "menopaus" in texts or "postmenopausal" in texts or "premenopausal" in texts:
            categories["mention_menopausal_context"] += 1
        else:
            categories["no_menopausal_context_in_evidence"] += 1
        if any(tok in texts for tok in ("cyst", "cystic", "follicle", "hemorrhagic")):
            categories["cyst_morphology_language"] += 1

        categories[cat] += 1
        if len(examples) < max_examples and cat != "agree":
            examples.append(
                {
                    "report_id": report_id,
                    "category": cat,
                    "reference_actionability": organ_actionability_categories(
                        gold_findings(gold[report_id])
                    ).get("adnexal"),
                    "predicted_actionability": organ_actionability_categories(
                        prediction_findings(predictions[report_id])
                    ).get("adnexal"),
                    "reference_evidence": [
                        (f.get("evidence_span") or "")[:160] for f in g_finds
                    ],
                    "predicted_evidence": [
                        (f.get("evidence_span") or f.get("description") or "")[:160]
                        for f in p_finds
                    ],
                    "needs_manual_review": cat
                    in {
                        "finding_missed",
                        "false_positive_detection",
                        "actionability_undercall",
                        "actionability_overcall",
                        "size_mismatch_with_undercall",
                    },
                }
            )

    total_discordant = sum(v for k, v in categories.items() if k not in {"agree", "mention_menopausal_context", "no_menopausal_context_in_evidence", "cyst_morphology_language"})
    return {
        "model": model_name,
        "category_counts": dict(categories),
        "n_discordant_or_positive": total_discordant + categories.get("agree", 0),
        "examples": examples,
        "note": (
            "Automated adnexal categories are heuristic. Cases labeled needs_manual_review "
            "require human adjudication; menopausal status is often absent from report text."
        ),
    }


def end_to_end_clinical_errors(
    reference: Dict[str, dict],
    candidate: Dict[str, dict],
    model_name: str,
) -> dict:
    """Clinically oriented error counts vs the deterministic reference pipeline."""

    ids = _ids(reference, candidate)
    counts = Counter()
    capture_driven = Counter()
    despite_capture = Counter()

    for report_id in ids:
        ref = reference[report_id]
        cand = candidate[report_id]
        ref_recs = ref.get("follow_up_recommendations") or {}
        cand_recs = cand.get("follow_up_recommendations") or {}
        ref_needs = bool(ref_recs)
        cand_needs = bool(cand_recs)

        ref_organs = {
            str(f.get("organ_system"))
            for f in ((ref.get("capture_result") or {}).get("findings") or [])
            if str(f.get("organ_system")) in IN_SCOPE
            and f.get("actionability") in {"actionable", "uncertain"}
        }
        cand_organs = {
            str(f.get("organ_system"))
            for f in ((cand.get("capture_result") or {}).get("findings") or [])
            if str(f.get("organ_system")) in IN_SCOPE
            and f.get("actionability") in {"actionable", "uncertain"}
        }
        capture_match = ref_organs == cand_organs

        if ref_needs and not cand_needs:
            counts["missed_necessary_followup"] += 1
            (capture_driven if not capture_match else despite_capture)["missed_necessary_followup"] += 1
        if (not ref_needs) and cand_needs:
            counts["unnecessary_followup"] += 1
            (capture_driven if not capture_match else despite_capture)["unnecessary_followup"] += 1

        # Organ-level recommendation component mismatches where both recommend
        for organ in IN_SCOPE:
            ref_entries = [
                r
                for fid, r in ref_recs.items()
                if any(
                    str(f.get("finding_id")) == str(fid) and str(f.get("organ_system")) == organ
                    for f in ((ref.get("capture_result") or {}).get("findings") or [])
                )
            ]
            cand_entries = [
                r
                for fid, r in cand_recs.items()
                if any(
                    str(f.get("finding_id")) == str(fid) and str(f.get("organ_system")) == organ
                    for f in ((cand.get("capture_result") or {}).get("findings") or [])
                )
            ]
            if not ref_entries and not cand_entries:
                continue
            if ref_entries and cand_entries:
                ref_mod = modality_family(ref_entries[0].get("recommended_modality"))
                cand_mod = modality_family(cand_entries[0].get("recommended_modality"))
                ref_tf = timeframe_class(ref_entries[0].get("recommended_timeframe"))
                cand_tf = timeframe_class(cand_entries[0].get("recommended_timeframe"))
                ref_rule = ref_entries[0].get("rule_id")
                cand_rule = cand_entries[0].get("rule_id")
                if ref_mod != cand_mod:
                    counts["wrong_modality"] += 1
                    (capture_driven if not capture_match else despite_capture)["wrong_modality"] += 1
                if ref_tf != cand_tf:
                    counts["wrong_interval"] += 1
                    (capture_driven if not capture_match else despite_capture)["wrong_interval"] += 1
                if ref_rule and cand_rule and str(ref_rule) != str(cand_rule):
                    counts["wrong_guideline_pathway"] += 1
                    (capture_driven if not capture_match else despite_capture)[
                        "wrong_guideline_pathway"
                    ] += 1
                if (
                    ref_mod == cand_mod
                    and ref_tf == cand_tf
                    and (not ref_rule or not cand_rule or str(ref_rule) == str(cand_rule))
                ):
                    counts["complete_recommendation_agree_organ"] += 1

    agents = evaluate_agents(reference, candidate, model_name)
    return {
        "model": model_name,
        "n_reports": len(ids),
        "error_counts": dict(counts),
        "errors_with_capture_mismatch": dict(capture_driven),
        "errors_despite_matching_capture_organs": dict(despite_capture),
        "followup_required_metrics": agents["orchestrate"]["any_recommendation"],
        "imaging_followup_metrics": agents["orchestrate"]["imaging_followup_recommended"],
        "guideline_kappa": agents["orchestrate"]["guideline_category"]["cohen_kappa"],
        "modality_kappa": agents["orchestrate"]["modality_family_category"]["cohen_kappa"],
        "timeframe_kappa": agents["orchestrate"]["timeframe_category"]["cohen_kappa"],
        "rule_id_exact_match_when_both_recommend": agents["orchestrate"][
            "rule_id_exact_match_when_both_recommend"
        ],
        "note": (
            "Downstream engines are shared with the reference pipeline; residual "
            "errors largely reflect propagated Capture differences."
        ),
    }


def link_failsafe_summary(predictions: Dict[str, dict], model_name: str) -> dict:
    assignments = []
    for result in predictions.values():
        assignments.extend((result.get("provider_assignments") or {}).values())
    n = len(assignments)
    failsafe = sum(bool(a.get("failsafe_triggered")) for a in assignments)
    review = sum(bool(a.get("human_review_required")) for a in assignments)
    types = Counter(str(a.get("provider_type") or "unknown") for a in assignments)
    return {
        "model": model_name,
        "n_assignments": n,
        "n_failsafe": failsafe,
        "failsafe_rate": failsafe / n if n else None,
        "n_human_review": review,
        "human_review_rate": review / n if n else None,
        "provider_type_counts": dict(types),
        "evaluation_mode": "failsafe_only",
        "note": (
            "No EHR/PCP specialty context was available; Link evaluation is a "
            "failsafe-behavior summary, not validation of provider assignment quality."
        ),
    }


def robustness_slices(
    gold: Dict[str, dict],
    predictions: Dict[str, dict],
    model_name: str,
    *,
    n_boot: int = 600,
    seed: int = 0,
) -> dict:
    ids = _ids(gold, predictions)
    slices = {
        "all": ids,
        "positive_only": filter_ids(gold, ids, positive_only=True),
        "single_finding": filter_ids(gold, ids, positive_only=True, min_findings=1, max_findings=1),
        "multi_finding": filter_ids(gold, ids, positive_only=True, min_findings=2),
        "single_organ": filter_ids(gold, ids, positive_only=True, multi_organ=False),
        "multi_organ": filter_ids(gold, ids, positive_only=True, multi_organ=True),
    }
    # Discordant: report actionability differs
    discordant = []
    for report_id in ids:
        t = report_actionability_category(gold_findings(gold[report_id]))
        p = report_actionability_category(prediction_findings(predictions[report_id]))
        if t != p:
            discordant.append(report_id)
    slices["actionability_discordant"] = discordant

    out = {"model": model_name, "slices": {}}
    for name, subset in slices.items():
        if not subset:
            out["slices"][name] = {"n_reports": 0}
            continue
        metrics = evaluate_capture_subset(
            gold,
            predictions,
            f"{model_name}/{name}",
            subset,
            n_boot=n_boot,
            seed=seed,
            bootstrap_organs=False,
        )
        out["slices"][name] = {
            "n_reports": metrics["n_reports"],
            "presence_balanced_accuracy": metrics["presence"]["metrics"]["balanced_accuracy"],
            "presence_kappa": metrics["presence"]["metrics"]["cohen_kappa"],
            "actionable_kappa": metrics["actionable"]["metrics"]["cohen_kappa"],
            "actionability_linear_kappa": metrics["report_actionability"]["metrics"]["cohen_kappa"],
            "actionability_linear_kappa_ci": metrics["report_actionability"]["bootstrap_linear_kappa"],
        }
    return out


def sample_expert_adjudication_ids(
    gold: Dict[str, dict],
    llm_predictions: Dict[str, Dict[str, dict]],
    target_n: int = 250,
    seed: int = 17,
) -> List[str]:
    """Stratified sample enriched for positives and disagreements."""

    import random

    rng = random.Random(seed)
    ids = sorted(gold)
    by_organ_any: Dict[str, List[str]] = defaultdict(list)
    pure_organ: Dict[str, List[str]] = defaultdict(list)
    none_ids: List[str] = []
    complex_ids: List[str] = []
    discordant_ids: List[str] = []

    for report_id in ids:
        organs = set(gold[report_id].get("organs_present") or [])
        n_findings = int(gold[report_id].get("n_findings") or 0)
        discordant = False
        for preds in llm_predictions.values():
            if report_id not in preds:
                continue
            t = report_actionability_category(gold_findings(gold[report_id]))
            p = report_actionability_category(prediction_findings(preds[report_id]))
            if t != p:
                discordant = True
                break
        if discordant:
            discordant_ids.append(report_id)
        if not organs:
            none_ids.append(report_id)
            continue
        for organ in organs:
            by_organ_any[organ].append(report_id)
        if len(organs) > 1 or n_findings > 1:
            complex_ids.append(report_id)
        elif len(organs) == 1:
            pure_organ[next(iter(organs))].append(report_id)

    selected: List[str] = []
    seen: Set[str] = set()

    def _take(pool: List[str], quota: int) -> None:
        candidates = [x for x in pool if x not in seen]
        rng.shuffle(candidates)
        for report_id in candidates[:quota]:
            selected.append(report_id)
            seen.add(report_id)

    # Ensure organ coverage first (prefer pure, then any containing organ)
    for organ in IN_SCOPE:
        need = 25 if organ != "adnexal" else 15
        _take(pure_organ.get(organ, []), need)
        remaining = need - sum(1 for x in selected if organ in set(gold[x].get("organs_present") or []))
        if remaining > 0:
            _take(by_organ_any.get(organ, []), remaining)

    _take(complex_ids, 40)
    _take(discordant_ids, 40)
    _take(none_ids, 60)

    if len(selected) < target_n:
        remaining = [x for x in ids if x not in seen]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_n - len(selected)])
    return selected[:target_n]
