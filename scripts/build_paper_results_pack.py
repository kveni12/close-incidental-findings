#!/usr/bin/env python
"""Build the paper-ready CLOSE evaluation package in paper_results/.

Reuses existing saved LLM-only / calibrate / full-hybrid JSONL outputs.
Does not call LLMs.

Example:
  PYTHONPATH=src python scripts/build_paper_results_pack.py
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.data.schemas import WorkflowResult  # noqa: E402
from close_framework.evaluation.agent_eval import evaluate_agents  # noqa: E402
from close_framework.evaluation.gold_eval import (  # noqa: E402
    ACTIONABILITY_SCALE,
    IN_SCOPE,
    evaluate_predictions_against_gold,
    load_jsonl_by_report_id,
)
from close_framework.evaluation.metrics import descriptive_summary_metrics  # noqa: E402
from close_framework.evaluation.paper_analyses import (  # noqa: E402
    actionability_error_analysis,
    adnexal_failure_analysis,
    cohort_characterization,
    end_to_end_clinical_errors,
    evaluate_capture_subset,
    filter_ids,
    hybrid_ablation_metrics,
    link_failsafe_summary,
    robustness_slices,
)

OUT = ROOT / "paper_results"
FIGS = OUT / "figures"
TABLES = OUT / "tables"
ZIP_PATH = ROOT / "close_paper_results.zip"
GOLD_PATH = ROOT / "data" / "gold_labels" / "gold_labels.jsonl"
REFERENCE_PIPELINE = ROOT / "outputs" / "reference_rule_based" / "close_results.jsonl"

MODELS = {
    "Llama-4 Scout": {
        "llm_only": ROOT / "outputs/model_comparison_9283993/llama/close_results.jsonl",
        "calibrate": ROOT / "outputs/model_comparison_9283993/llama/close_results_calibrate.jsonl",
        "full_hybrid": ROOT / "outputs/model_comparison_9283993/llama/close_results_hybrid.jsonl",
    },
    "Qwen3.6-35B": {
        "llm_only": ROOT / "outputs/model_comparison_9500762/qwen/close_results.jsonl",
        "calibrate": ROOT / "outputs/model_comparison_9500762/qwen/close_results_calibrate.jsonl",
        "full_hybrid": ROOT / "outputs/model_comparison_9500762/qwen/close_results_hybrid.jsonl",
    },
    "Mistral-Small-3.2": {
        "llm_only": ROOT / "outputs/model_comparison_9500762/mistral/close_results.jsonl",
        "calibrate": ROOT / "outputs/model_comparison_9500762/mistral/close_results_calibrate.jsonl",
        "full_hybrid": ROOT / "outputs/model_comparison_9500762/mistral/close_results_hybrid.jsonl",
    },
}
SHORT = {
    "Llama-4 Scout": "Llama",
    "Qwen3.6-35B": "Qwen",
    "Mistral-Small-3.2": "Mistral",
}
ARM_LABELS = {
    "llm_only": "A: LLM-only",
    "calibrate": "B: + ACR calibrate",
    "full_hybrid": "C: Full hybrid",
}
COLORS = {
    "Llama-4 Scout": "#2E86AB",
    "Qwen3.6-35B": "#A23B72",
    "Mistral-Small-3.2": "#F18F01",
}
N_BOOT = 400


def _fmt(x, digits=3):
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def _fmt_ci(ci: dict | None, digits=3) -> str:
    if not ci or ci.get("estimate") is None:
        return ""
    low, high = ci.get("ci_low"), ci.get("ci_high")
    if low is None or high is None:
        return _fmt(ci["estimate"], digits)
    return f"{ci['estimate']:.{digits}f} ({low:.{digits}f}–{high:.{digits}f})"


def _write_csv(path: Path, headers, rows):
    body = "\n".join(",".join("" if c is None else str(c) for c in row) for row in rows)
    path.write_text(",".join(headers) + "\n" + body + ("\n" if rows else ""), encoding="utf-8")


def _save_fig(fig, stem: str):
    fig.tight_layout()
    fig.savefig(FIGS / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_results(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(WorkflowResult.model_validate_json(line))
    return rows


def main() -> None:
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

    if OUT.exists():
        shutil.rmtree(OUT)
    for path in (FIGS, TABLES, OUT / "expert_adjudication"):
        path.mkdir(parents=True)

    gold = load_jsonl_by_report_id(GOLD_PATH)
    gold_meta = json.loads((GOLD_PATH.parent / "gold_labels_meta.json").read_text())
    reference = load_jsonl_by_report_id(REFERENCE_PIPELINE)

    preds: dict[str, dict[str, dict]] = {}
    missing_arms = []
    for model, arms in MODELS.items():
        preds[model] = {}
        for arm, path in arms.items():
            if not path.exists():
                missing_arms.append(str(path))
                continue
            preds[model][arm] = load_jsonl_by_report_id(path)
    if missing_arms:
        raise FileNotFoundError(
            "Missing required JSONL arms (generate calibrate with "
            "`run_hybrid_downstream.py --no-supplement`):\n" + "\n".join(missing_arms)
        )

    # --- Core analyses ---
    cohort = cohort_characterization(gold, reference)
    ablations = {}
    capture_full = {}
    capture_positive = {}
    errors = {}
    adnexal = {}
    e2e = {}
    link = {}
    robustness = {}
    organ_action = {}

    positive_ids = filter_ids(gold, sorted(gold), positive_only=True)

    for model in MODELS:
        print(f"Analyzing {model}...", flush=True)
        ablations[model] = hybrid_ablation_metrics(
            gold, reference, preds[model], model, n_boot=N_BOOT, seed=0
        )
        # Primary paper Capture metrics = full hybrid
        capture_full[model] = evaluate_capture_subset(
            gold, preds[model]["full_hybrid"], model, n_boot=N_BOOT, seed=0
        )
        capture_positive[model] = evaluate_capture_subset(
            gold, preds[model]["full_hybrid"], model, positive_ids, n_boot=N_BOOT, seed=1
        )
        errors[model] = {
            "llm_only": actionability_error_analysis(gold, preds[model]["llm_only"], model),
            "full_hybrid": actionability_error_analysis(gold, preds[model]["full_hybrid"], model),
        }
        adnexal[model] = {
            "llm_only": adnexal_failure_analysis(gold, preds[model]["llm_only"], model),
            "full_hybrid": adnexal_failure_analysis(gold, preds[model]["full_hybrid"], model),
        }
        e2e[model] = {
            arm: end_to_end_clinical_errors(reference, preds[model][arm], f"{model}/{arm}")
            for arm in ("llm_only", "calibrate", "full_hybrid")
        }
        link[model] = link_failsafe_summary(preds[model]["full_hybrid"], model)
        robustness[model] = robustness_slices(
            gold, preds[model]["full_hybrid"], model, n_boot=200, seed=2
        )
        organ_action[model] = evaluate_predictions_against_gold(
            gold, preds[model]["full_hybrid"], model
        )["categorical"]["organ_actionability_by_organ"]

    # --- Tables ---
    _write_csv(
        TABLES / "table01_cohort_characteristics.csv",
        [
            "metric",
            "value",
        ],
        [
            ["n_reports", cohort["n_reports"]],
            ["n_no_in_scope_finding", cohort["n_no_in_scope_finding"]],
            ["n_with_in_scope_finding", cohort["n_with_in_scope_finding"]],
            ["n_with_actionable_finding", cohort["n_with_actionable_finding"]],
            ["n_single_finding_reports", cohort["n_single_finding_reports"]],
            ["n_multi_finding_reports", cohort["n_multi_finding_reports"]],
            ["n_multi_organ_reports", cohort["n_multi_organ_reports"]],
            ["mean_findings_among_positives", _fmt(cohort["mean_findings_among_positives"])],
            ["reference_pipeline_recommendations", cohort["reference_pipeline_recommendations"]],
            *[
                [f"organ_reports_{organ}", cohort["organ_report_counts"].get(organ, 0)]
                for organ in IN_SCOPE
            ],
            *[
                [f"actionability_{k}", v]
                for k, v in sorted(cohort["report_actionability_distribution"].items())
            ],
        ],
    )

    # Table 02 overall Capture (hybrid primary)
    rows = []
    for model in MODELS:
        m = capture_full[model]
        rows.append(
            [
                model,
                m["n_reports"],
                _fmt_ci(m["presence"]["bootstrap"]["balanced_accuracy"]),
                _fmt_ci(m["presence"]["bootstrap"]["sensitivity"]),
                _fmt_ci(m["presence"]["bootstrap"]["specificity"]),
                _fmt_ci(m["presence"]["bootstrap"]["positive_predictive_value"]),
                _fmt_ci(m["presence"]["bootstrap"]["f1"]),
                _fmt_ci(m["presence"]["bootstrap"]["cohen_kappa"]),
                _fmt_ci(m["actionable"]["bootstrap"]["balanced_accuracy"]),
                _fmt_ci(m["actionable"]["bootstrap"]["cohen_kappa"]),
                _fmt_ci(m["report_actionability"]["bootstrap_linear_kappa"]),
                _fmt(m["finding_count_mae"]),
                _fmt(m["size_within_2mm_rate"]),
            ]
        )
    _write_csv(
        TABLES / "table02_capture_overall_hybrid.csv",
        [
            "model",
            "n",
            "presence_balanced_acc_ci",
            "presence_sens_ci",
            "presence_spec_ci",
            "presence_ppv_ci",
            "presence_f1_ci",
            "presence_kappa_ci",
            "actionable_balanced_acc_ci",
            "actionable_kappa_ci",
            "actionability_linear_kappa_ci",
            "finding_count_mae",
            "size_within_2mm_rate",
        ],
        rows,
    )

    # Table 03 organ-level presence
    organ_rows = []
    for model in MODELS:
        for organ in IN_SCOPE:
            block = capture_full[model]["organ_presence"][organ]
            organ_rows.append(
                [
                    model,
                    organ,
                    _fmt_ci(block["bootstrap"]["balanced_accuracy"]),
                    _fmt_ci(block["bootstrap"]["sensitivity"]),
                    _fmt_ci(block["bootstrap"]["specificity"]),
                    _fmt_ci(block["bootstrap"]["f1"]),
                    _fmt_ci(block["bootstrap"]["cohen_kappa"]),
                    int(block["metrics"]["true_positives"] + block["metrics"]["false_negatives"]),
                ]
            )
    _write_csv(
        TABLES / "table03_capture_by_organ.csv",
        [
            "model",
            "organ",
            "balanced_acc_ci",
            "sensitivity_ci",
            "specificity_ci",
            "f1_ci",
            "kappa_ci",
            "n_reference_positive",
        ],
        organ_rows,
    )

    # Table 04 positive-only
    pos_rows = []
    for model in MODELS:
        m = capture_positive[model]
        pos_rows.append(
            [
                model,
                m["n_reports"],
                _fmt_ci(m["presence"]["bootstrap"]["sensitivity"]),
                _fmt_ci(m["actionable"]["bootstrap"]["balanced_accuracy"]),
                _fmt_ci(m["actionable"]["bootstrap"]["cohen_kappa"]),
                _fmt_ci(m["report_actionability"]["bootstrap_linear_kappa"]),
            ]
        )
    _write_csv(
        TABLES / "table04_capture_positive_only.csv",
        [
            "model",
            "n_positive_reports",
            "presence_sensitivity_ci",
            "actionable_balanced_acc_ci",
            "actionable_kappa_ci",
            "actionability_linear_kappa_ci",
        ],
        pos_rows,
    )

    # Table 05 ablation
    abl_rows = []
    for model in MODELS:
        for arm in ("llm_only", "calibrate", "full_hybrid"):
            arm_block = ablations[model]["arms"][arm]
            cap = arm_block["capture"]
            abl_rows.append(
                [
                    model,
                    ARM_LABELS[arm],
                    _fmt_ci(cap["report_actionability"]["bootstrap_linear_kappa"]),
                    _fmt_ci(cap["actionable"]["bootstrap"]["balanced_accuracy"]),
                    _fmt_ci(cap["actionable"]["bootstrap"]["sensitivity"]),
                    _fmt_ci(cap["actionable"]["bootstrap"]["specificity"]),
                    _fmt_ci(cap["actionable"]["bootstrap"]["positive_predictive_value"]),
                    _fmt_ci(cap["actionable"]["bootstrap"]["f1"]),
                    arm_block["n_actionable_findings"],
                    arm_block["n_recommendations"],
                    _fmt(arm_block["orchestrate_guideline_kappa"]),
                ]
            )
    _write_csv(
        TABLES / "table05_hybrid_ablation.csv",
        [
            "model",
            "configuration",
            "actionability_linear_kappa_ci",
            "actionable_balanced_acc_ci",
            "actionable_sens_ci",
            "actionable_spec_ci",
            "actionable_ppv_ci",
            "actionable_f1_ci",
            "n_actionable_findings",
            "n_recommendations",
            "orchestrate_guideline_kappa",
        ],
        abl_rows,
    )

    # Incremental ablation deltas
    delta_rows = []
    for model in MODELS:
        inc = ablations[model].get("incremental") or {}
        for key, label in [
            ("calibrate_minus_llm_only", "B_minus_A_calibrate_gain"),
            ("full_hybrid_minus_calibrate", "C_minus_B_supplement_gain"),
            ("full_hybrid_minus_llm_only", "C_minus_A_total_hybrid_gain"),
        ]:
            block = inc.get(key) or {}
            delta_rows.append([model, label, _fmt_ci(block)])
    _write_csv(
        TABLES / "table05b_hybrid_ablation_deltas.csv",
        ["model", "contrast", "delta_linear_kappa_ci"],
        delta_rows,
    )

    # Table 06 organ actionability
    oa_rows = []
    for model in MODELS:
        for organ in IN_SCOPE:
            block = organ_action[model][organ]
            oa_rows.append(
                [
                    model,
                    organ,
                    _fmt(block.get("cohen_kappa")),
                    _fmt(block.get("cohen_kappa_nominal")),
                    _fmt(block.get("macro_f1")),
                    _fmt(block.get("accuracy")),
                ]
            )
    _write_csv(
        TABLES / "table06_organ_actionability.csv",
        ["model", "organ", "linear_kappa", "nominal_kappa", "macro_f1", "accuracy"],
        oa_rows,
    )

    # Table 07 E2E orchestrate
    e2e_rows = []
    for model in MODELS:
        for arm in ("llm_only", "calibrate", "full_hybrid"):
            block = e2e[model][arm]
            fu = block["followup_required_metrics"]
            e2e_rows.append(
                [
                    model,
                    ARM_LABELS[arm],
                    _fmt(fu.get("balanced_accuracy")),
                    _fmt(fu.get("sensitivity")),
                    _fmt(fu.get("specificity")),
                    _fmt(fu.get("positive_predictive_value")),
                    _fmt(fu.get("f1")),
                    _fmt(fu.get("cohen_kappa")),
                    _fmt(block["guideline_kappa"]),
                    _fmt(block["modality_kappa"]),
                    _fmt(block["timeframe_kappa"]),
                    _fmt(block["rule_id_exact_match_when_both_recommend"]),
                    block["error_counts"].get("missed_necessary_followup", 0),
                    block["error_counts"].get("unnecessary_followup", 0),
                ]
            )
    _write_csv(
        TABLES / "table07_orchestrate_end_to_end.csv",
        [
            "model",
            "configuration",
            "followup_balanced_acc",
            "followup_sens",
            "followup_spec",
            "followup_ppv",
            "followup_f1",
            "followup_kappa",
            "guideline_kappa",
            "modality_kappa",
            "timeframe_kappa",
            "rule_id_exact_match",
            "missed_necessary_followup",
            "unnecessary_followup",
        ],
        e2e_rows,
    )

    # Table 08 recommendation components / clinical errors
    rec_rows = []
    for model in MODELS:
        block = e2e[model]["full_hybrid"]
        for key in [
            "missed_necessary_followup",
            "unnecessary_followup",
            "wrong_modality",
            "wrong_interval",
            "wrong_guideline_pathway",
            "complete_recommendation_agree_organ",
        ]:
            rec_rows.append(
                [
                    model,
                    key,
                    block["error_counts"].get(key, 0),
                    block["errors_with_capture_mismatch"].get(key, 0),
                    block["errors_despite_matching_capture_organs"].get(key, 0),
                ]
            )
    _write_csv(
        TABLES / "table08_recommendation_error_components.csv",
        [
            "model",
            "error_type",
            "count",
            "with_capture_organ_mismatch",
            "despite_matching_capture_organs",
        ],
        rec_rows,
    )

    # Table 09 error taxonomy
    tax_rows = []
    for model in MODELS:
        for arm in ("llm_only", "full_hybrid"):
            block = errors[model][arm]
            for key, count in sorted(block["taxonomy_counts"].items()):
                tax_rows.append(
                    [
                        model,
                        arm,
                        key,
                        count,
                        _fmt(block["taxonomy_rates"].get(key)),
                    ]
                )
            for key, count in sorted(block["ordinal_distance_counts"].items()):
                tax_rows.append(
                    [
                        model,
                        arm,
                        f"ordinal_distance_{key}",
                        count,
                        _fmt(block["ordinal_distance_rates"].get(key)),
                    ]
                )
            tax_rows.append(
                [
                    model,
                    arm,
                    "mean_signed_severity_bias",
                    "",
                    _fmt(block["mean_signed_severity_bias"]),
                ]
            )
    _write_csv(
        TABLES / "table09_actionability_error_taxonomy.csv",
        ["model", "configuration", "category", "count", "rate_or_value"],
        tax_rows,
    )

    # Table 10 adnexal
    adn_rows = []
    for model in MODELS:
        for arm in ("llm_only", "full_hybrid"):
            block = adnexal[model][arm]
            for key, count in sorted(block["category_counts"].items()):
                adn_rows.append([model, arm, key, count])
    _write_csv(
        TABLES / "table10_adnexal_error_analysis.csv",
        ["model", "configuration", "category", "count"],
        adn_rows,
    )

    # Table 11 link failsafe
    link_rows = []
    for model in MODELS:
        block = link[model]
        link_rows.append(
            [
                model,
                block["n_assignments"],
                block["n_failsafe"],
                _fmt(block["failsafe_rate"]),
                block["n_human_review"],
                _fmt(block["human_review_rate"]),
                json.dumps(block["provider_type_counts"]),
                block["evaluation_mode"],
            ]
        )
    _write_csv(
        TABLES / "table11_link_failsafe_summary.csv",
        [
            "model",
            "n_assignments",
            "n_failsafe",
            "failsafe_rate",
            "n_human_review",
            "human_review_rate",
            "provider_type_counts_json",
            "evaluation_mode",
        ],
        link_rows,
    )

    # Table 12 runtime placeholder / descriptive volumes
    runtime_rows = []
    for model in MODELS:
        for arm, path in MODELS[model].items():
            summary = descriptive_summary_metrics(_load_results(path))
            runtime_rows.append(
                [
                    model,
                    ARM_LABELS[arm],
                    summary["reports_processed"],
                    summary["findings_detected"],
                    summary["actionable_findings"],
                    summary["recommendations_generated"],
                    summary["capture_error_reports"],
                    "offline_from_saved_outputs",
                ]
            )
    _write_csv(
        TABLES / "table12_run_volumes.csv",
        [
            "model",
            "configuration",
            "reports",
            "findings",
            "actionable_findings",
            "recommendations",
            "capture_errors",
            "runtime_note",
        ],
        runtime_rows,
    )

    # Robustness table
    rob_rows = []
    for model in MODELS:
        for slice_name, block in robustness[model]["slices"].items():
            rob_rows.append(
                [
                    model,
                    slice_name,
                    block.get("n_reports", 0),
                    _fmt(block.get("presence_balanced_accuracy")),
                    _fmt(block.get("actionable_kappa")),
                    _fmt_ci(block.get("actionability_linear_kappa_ci")),
                ]
            )
    _write_csv(
        TABLES / "table13_robustness_slices.csv",
        [
            "model",
            "slice",
            "n_reports",
            "presence_balanced_acc",
            "actionable_kappa",
            "actionability_linear_kappa_ci",
        ],
        rob_rows,
    )

    # Confusion matrices long
    conf_rows = []
    for model in MODELS:
        for arm in ("llm_only", "full_hybrid"):
            matrix = errors[model][arm]["confusion"]["confusion_matrix"]
            for truth in ACTIONABILITY_SCALE:
                for pred in ACTIONABILITY_SCALE:
                    conf_rows.append([model, arm, truth, pred, matrix[truth][pred]])
    _write_csv(
        TABLES / "table14_actionability_confusion_long.csv",
        ["model", "configuration", "reference", "predicted", "count"],
        conf_rows,
    )

    # Adnexal examples
    ex_rows = []
    for model in MODELS:
        for arm in ("llm_only", "full_hybrid"):
            for ex in adnexal[model][arm]["examples"]:
                ex_rows.append(
                    [
                        model,
                        arm,
                        ex["report_id"],
                        ex["category"],
                        ex.get("reference_actionability"),
                        ex.get("predicted_actionability"),
                        " | ".join(ex.get("reference_evidence") or []),
                        " | ".join(ex.get("predicted_evidence") or []),
                        ex.get("needs_manual_review"),
                    ]
                )
    _write_csv(
        TABLES / "table15_adnexal_examples.csv",
        [
            "model",
            "configuration",
            "report_id",
            "category",
            "reference_actionability",
            "predicted_actionability",
            "reference_evidence",
            "predicted_evidence",
            "needs_manual_review",
        ],
        ex_rows,
    )

    # --- Figures ---
    # fig01 workflow
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    boxes = ["Capture", "Link", "Orchestrate"]
    for i, label in enumerate(boxes):
        ax.add_patch(plt.Rectangle((0.12 + i * 0.28, 0.35), 0.2, 0.35, fill=False, lw=2))
        ax.text(0.22 + i * 0.28, 0.52, label, ha="center", va="center", fontsize=12)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(0.12 + (i + 1) * 0.28, 0.52),
                xytext=(0.32 + i * 0.28, 0.52),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )
    ax.text(
        0.5,
        0.12,
        "Eval vs deterministic ACR-derived reference (not radiologist chart review)",
        ha="center",
        fontsize=9,
        style="italic",
    )
    ax.set_title("CLOSE evaluation workflow")
    _save_fig(fig, "fig01_close_evaluation_workflow")

    # fig02 cohort
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["No finding", "≥1 finding"],
        [cohort["n_no_in_scope_finding"], cohort["n_with_in_scope_finding"]],
        color=["#94A3B8", "#2E86AB"],
    )
    axes[0].set_ylabel("Reports")
    axes[0].set_title("In-scope finding presence")
    organs = list(IN_SCOPE)
    axes[1].bar(organs, [cohort["organ_report_counts"].get(o, 0) for o in organs], color="#A23B72")
    axes[1].set_title("Reports with organ finding")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Cohort distribution (ACR-derived reference)")
    _save_fig(fig, "fig02_cohort_distribution")

    # fig03 capture by model
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(MODELS))
    width = 0.25
    metrics = [
        ("presence", "balanced_accuracy", "Presence bal-acc"),
        ("actionable", "cohen_kappa", "Actionable κ"),
        ("report_actionability", None, "Actionability linear κ"),
    ]
    for i, (section, key, _) in enumerate(metrics):
        vals = []
        for model in MODELS:
            if section == "report_actionability":
                vals.append(capture_full[model][section]["metrics"]["cohen_kappa"])
            else:
                vals.append(capture_full[model][section]["metrics"][key])
        ax.bar(x + (i - 1) * width, vals, width, label=metrics[i][2])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Hybrid Capture agreement with ACR-derived reference")
    _save_fig(fig, "fig03_capture_by_model")

    # fig04 organ presence kappa
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(IN_SCOPE))
    width = 0.25
    for i, model in enumerate(MODELS):
        vals = [
            capture_full[model]["organ_presence"][organ]["metrics"]["cohen_kappa"] or 0
            for organ in IN_SCOPE
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=SHORT[model], color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(IN_SCOPE)
    ax.set_ylabel("Cohen κ")
    ax.set_title("Organ presence agreement (hybrid)")
    ax.legend()
    _save_fig(fig, "fig04_capture_by_organ")

    # fig05 confusion matrices hybrid
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, model in zip(axes, MODELS):
        matrix = errors[model]["full_hybrid"]["confusion"]["confusion_matrix"]
        arr = np.array([[matrix[t][p] for p in ACTIONABILITY_SCALE] for t in ACTIONABILITY_SCALE])
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(ACTIONABILITY_SCALE, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ACTIONABILITY_SCALE, fontsize=8)
        ax.set_title(SHORT[model])
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("ACR reference")
    fig.suptitle("Hybrid 4-level actionability confusion")
    _save_fig(fig, "fig05_actionability_confusion_hybrid")

    # fig06 ablation
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(3)
    width = 0.25
    arm_order = ("llm_only", "calibrate", "full_hybrid")
    for i, model in enumerate(MODELS):
        vals = [
            ablations[model]["arms"][arm]["capture"]["report_actionability"]["metrics"]["cohen_kappa"]
            for arm in arm_order
        ]
        ax.plot(x, vals, marker="o", linewidth=2, label=SHORT[model], color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arm_order], rotation=15, ha="right")
    ax.set_ylabel("4-level linear κ")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Hybrid ablation: LLM-only → calibrate → + missed-finding recovery")
    _save_fig(fig, "fig06_hybrid_ablation")

    # fig07 llm vs hybrid actionability
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(MODELS))
    width = 0.35
    before = [
        ablations[m]["arms"]["llm_only"]["capture"]["report_actionability"]["metrics"]["cohen_kappa"]
        for m in MODELS
    ]
    after = [
        ablations[m]["arms"]["full_hybrid"]["capture"]["report_actionability"]["metrics"]["cohen_kappa"]
        for m in MODELS
    ]
    ax.bar(x - width / 2, before, width, label="LLM-only", color="#94A3B8")
    ax.bar(x + width / 2, after, width, label="Full hybrid", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Actionability linear κ: LLM-only vs full hybrid")
    _save_fig(fig, "fig07_llm_vs_hybrid_actionability")

    # fig08 orchestrate
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(3)
    for model in MODELS:
        vals = [e2e[model][arm]["guideline_kappa"] or 0 for arm in arm_order]
        ax.plot(x, vals, marker="o", linewidth=2, label=SHORT[model], color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arm_order], rotation=15, ha="right")
    ax.set_ylabel("Guideline category κ")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Orchestrate guideline agreement across hybrid arms")
    _save_fig(fig, "fig08_orchestrate_performance")

    # fig09 error categories
    fig, ax = plt.subplots(figsize=(10, 4.8))
    cats = [
        "undercall_actionability",
        "overcall_actionability",
        "finding_missed_report",
        "false_positive_report",
    ]
    x = np.arange(len(cats))
    width = 0.25
    for i, model in enumerate(MODELS):
        vals = [errors[model]["llm_only"]["taxonomy_counts"].get(c, 0) for c in cats]
        ax.bar(x + (i - 1) * width, vals, width, label=f"{SHORT[model]} LLM-only", color=COLORS[model], alpha=0.45)
    for i, model in enumerate(MODELS):
        vals = [errors[model]["full_hybrid"]["taxonomy_counts"].get(c, 0) for c in cats]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            bottom=0,
            label=f"{SHORT[model]} hybrid",
            color=COLORS[model],
            fill=False,
            linewidth=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Report counts")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Actionability error categories (outline=hybrid, fill=LLM-only)")
    _save_fig(fig, "fig09_error_category_distribution")

    # fig10 adnexal
    fig, ax = plt.subplots(figsize=(9, 4.5))
    focus = [
        "finding_missed",
        "false_positive_detection",
        "actionability_undercall",
        "actionability_overcall",
        "agree",
    ]
    x = np.arange(len(focus))
    width = 0.25
    for i, model in enumerate(MODELS):
        vals = [adnexal[model]["full_hybrid"]["category_counts"].get(c, 0) for c in focus]
        ax.bar(x + (i - 1) * width, vals, width, label=SHORT[model], color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(focus, rotation=20, ha="right", fontsize=8)
    ax.legend()
    ax.set_title("Adnexal outcomes (hybrid)")
    _save_fig(fig, "fig10_adnexal_failure_analysis")

    # --- Expert sample ---
    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts/create_expert_adjudication_sample.py"),
            "--output",
            str(OUT / "expert_adjudication/adjudication_sample.csv"),
            "--target-n",
            "250",
        ]
    )

    # --- metrics.json ---
    payload = {
        "reference_standard": {
            "type": "deterministic_acr_derived_reference",
            "path": str(GOLD_PATH.relative_to(ROOT)),
            "disclaimer": gold_meta.get("disclaimer"),
            "not_clinical_ground_truth": True,
        },
        "cohort": cohort,
        "capture_full_hybrid": capture_full,
        "capture_positive_only": capture_positive,
        "hybrid_ablation": ablations,
        "actionability_errors": errors,
        "adnexal": adnexal,
        "end_to_end": e2e,
        "link_failsafe": link,
        "robustness": robustness,
        "n_bootstrap": N_BOOT,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # Copy gold labels for auditability
    shutil.copytree(GOLD_PATH.parent, OUT / "rule_reference_labels")

    # --- README / SUMMARY ---
    def _k(model, arm):
        return ablations[model]["arms"][arm]["capture"]["report_actionability"]["metrics"]["cohen_kappa"]

    def _ak(model, arm):
        return ablations[model]["arms"][arm]["capture"]["actionable"]["metrics"]["cohen_kappa"]

    def _gk(model, arm):
        return e2e[model][arm]["guideline_kappa"]

    readme = f"""# CLOSE paper-ready evaluation package

## Question
Can LLMs reliably extract incidental abdominal findings from radiology reports, and does
combining semantic LLM extraction with deterministic guideline calibration produce a more
reliable end-to-end follow-up workflow?

## Critical framing
All agreement metrics are vs a **deterministic ACR-derived reference**
(`rule_reference_labels/`, built by rule-based Capture / reference pipeline).
This is **operational concordance with a guideline engine**, not independent clinical
validation. Expert adjudication is required for clinical claims
(`expert_adjudication/`).

## Experiments included
1. Capture detection vs characterization (overall, organ-level, positive-only)
2. Hybrid ablation A/B/C (LLM-only → ACR calibrate → + missed-finding recovery)
3. Actionability error taxonomy + confusion matrices
4. Adnexal failure analysis
5. End-to-end Orchestrate follow-up / modality / timeframe / rule concordance
6. Link failsafe-only summary
7. Robustness slices (positive-only, single/multi finding/organ, discordant)
8. Expert adjudication sampling template (250 reports; labels blank)

## Inputs
| Arm | Paths |
|---|---|
| A LLM-only | `outputs/model_comparison_*/{{llama,qwen,mistral}}/close_results.jsonl` |
| B Calibrate | `.../close_results_calibrate.jsonl` (ACR recalibration, no supplement) |
| C Full hybrid | `.../close_results_hybrid.jsonl` (recalibrate + supplement + Link/Orchestrate) |
| Reference Capture | `data/gold_labels/gold_labels.jsonl` |
| Reference pipeline | `outputs/reference_rule_based/close_results.jsonl` |

## Regenerate
```bash
# If calibrate arm missing:
PYTHONPATH=src python scripts/run_hybrid_downstream.py \\
  --input outputs/model_comparison_9500762/qwen/close_results.jsonl \\
  --output outputs/model_comparison_9500762/qwen/close_results_calibrate.jsonl \\
  --no-supplement
# (repeat for llama/mistral)

PYTHONPATH=src python scripts/build_paper_results_pack.py
PYTHONPATH=src python scripts/create_expert_adjudication_sample.py
```

## Metric definitions
- **Balanced accuracy**: (sensitivity + specificity) / 2 — preferred for imbalanced binary tasks
- **Linear Cohen's κ**: primary ordinal agreement on 4-level actionability
  (`none` < `non_actionable` < `uncertain` < `actionable`)
- Bootstrap CIs: report-level percentile intervals ({N_BOOT} resamples) where shown as
  `estimate (low–high)`
- Undefined metrics are left blank (not zero-filled)

## Status
| Item | Status |
|---|---|
| Offline analyses from saved outputs | Complete |
| Expert labels | Template only — not filled |
| Multi-run LLM consistency | Scriptable later; not run (expensive) |
| Independent clinical validation | Not available yet |
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    summary = f"""# RESULTS_SUMMARY

## Main findings (vs ACR-derived reference; n=1000)

Hybrid Capture (LLM extraction + ACR calibration + missed-finding recovery) substantially
improves actionability agreement and downstream Orchestrate concordance relative to LLM-only.

### 4-level actionability linear κ
| Model | A LLM-only | B +calibrate | C Full hybrid |
|---|--:|--:|--:|
| Llama | {_fmt(_k('Llama-4 Scout','llm_only'))} | {_fmt(_k('Llama-4 Scout','calibrate'))} | {_fmt(_k('Llama-4 Scout','full_hybrid'))} |
| Qwen | {_fmt(_k('Qwen3.6-35B','llm_only'))} | {_fmt(_k('Qwen3.6-35B','calibrate'))} | {_fmt(_k('Qwen3.6-35B','full_hybrid'))} |
| Mistral | {_fmt(_k('Mistral-Small-3.2','llm_only'))} | {_fmt(_k('Mistral-Small-3.2','calibrate'))} | {_fmt(_k('Mistral-Small-3.2','full_hybrid'))} |

### Binary actionable κ
| Model | A | B | C |
|---|--:|--:|--:|
| Llama | {_fmt(_ak('Llama-4 Scout','llm_only'))} | {_fmt(_ak('Llama-4 Scout','calibrate'))} | {_fmt(_ak('Llama-4 Scout','full_hybrid'))} |
| Qwen | {_fmt(_ak('Qwen3.6-35B','llm_only'))} | {_fmt(_ak('Qwen3.6-35B','calibrate'))} | {_fmt(_ak('Qwen3.6-35B','full_hybrid'))} |
| Mistral | {_fmt(_ak('Mistral-Small-3.2','llm_only'))} | {_fmt(_ak('Mistral-Small-3.2','calibrate'))} | {_fmt(_ak('Mistral-Small-3.2','full_hybrid'))} |

### Orchestrate guideline κ
| Model | A | B | C |
|---|--:|--:|--:|
| Llama | {_fmt(_gk('Llama-4 Scout','llm_only'))} | {_fmt(_gk('Llama-4 Scout','calibrate'))} | {_fmt(_gk('Llama-4 Scout','full_hybrid'))} |
| Qwen | {_fmt(_gk('Qwen3.6-35B','llm_only'))} | {_fmt(_gk('Qwen3.6-35B','calibrate'))} | {_fmt(_gk('Qwen3.6-35B','full_hybrid'))} |
| Mistral | {_fmt(_gk('Mistral-Small-3.2','llm_only'))} | {_fmt(_gk('Mistral-Small-3.2','calibrate'))} | {_fmt(_gk('Mistral-Small-3.2','full_hybrid'))} |

## Ablation interpretation
Gains come from **both** steps, but not equally for every metric/model:
- **ACR calibration (A→B)** mainly fixes severity downgrading on already-detected findings
  (largest lift in binary actionable κ / balanced accuracy).
- **Missed-finding recovery (B→C)** drives most of the remaining linear-κ gain by adding
  rule-detected findings the LLM missed.
- For Mistral, calibrate-alone does not improve linear κ (CI includes 0); almost all gain
  is from supplementation.

See `tables/table05_hybrid_ablation.csv` and `tables/table05b_hybrid_ablation_deltas.csv`.

## What changed vs prior pack
- Added **3-way ablation** (calibrate-only arm), not just LLM vs full hybrid
- Added bootstrap **95% CIs**
- Added positive-only, robustness slices, error taxonomy, adnexal deep dive
- Added end-to-end follow-up decision + clinical error components with Capture-propagation split
- Added expert adjudication sampling infrastructure (blank labels)
- Paper-facing language uses **ACR-derived reference**, not unqualified “gold/ground truth”

## Strongest results
- Finding presence detection is high for hybrid (near-ceiling sensitivity)
- Kidney/liver/adrenal/pancreas actionability agreement is strong under hybrid
- Hybrid improves both Capture actionability and Orchestrate guideline concordance
- Downstream recommendation volume rises substantially once Capture is calibrated/supplemented

## Remaining weaknesses
- **Adnexal** remains the weakest organ (low prevalence; missing menopausal context)
- Link is **failsafe-only** (no EHR PCP context) — do not claim provider-routing validity
- Reference circularity: hybrid uses the same ACR heuristics used to score Capture agreement
- No radiologist-adjudicated labels yet

## Do NOT interpret as clinical validation
Agreement with the deterministic ACR pipeline measures operational concordance with a
guideline engine. Independent expert adjudication (`expert_adjudication/`) is required
before claiming clinical accuracy.

## Recommended next experiments
1. Complete dual-reviewer adjudication on the 250-report sample
2. Held-out / temporal split to reduce evaluation leakage from prompt/hybrid tuning
3. Targeted adnexal guideline/prompt work + optional age/menopausal context features
4. Optional multi-seed LLM consistency study (`scripts/analyze_llm_run_consistency.py`; expensive)
"""
    (OUT / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")

    # Zip
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(ROOT)))
    print(
        json.dumps(
            {
                "paper_results": str(OUT),
                "zip": str(ZIP_PATH),
                "models": list(MODELS),
                "missing_arms_checked": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
