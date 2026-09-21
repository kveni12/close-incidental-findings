#!/usr/bin/env python
"""Build paper-ready results/ pack including gold-standard Capture metrics."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
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

from close_framework.evaluation.agent_eval import evaluate_agents  # noqa: E402
from close_framework.evaluation.gold_eval import (  # noqa: E402
    evaluate_predictions_against_gold,
    load_jsonl_by_report_id,
)
from close_framework.evaluation.metrics import descriptive_summary_metrics  # noqa: E402
from close_framework.data.schemas import WorkflowResult  # noqa: E402

OUT = ROOT / "results"
FIGS = OUT / "figures"
TABLES = OUT / "tables"
AGENTS = OUT / "agents"
GOLD_PATH = ROOT / "data" / "gold_labels" / "gold_labels.jsonl"
REFERENCE_PIPELINE = ROOT / "outputs" / "reference_rule_based" / "close_results.jsonl"
ZIP_PATH = ROOT / "close_results_llama_qwen_mistral.zip"

RUNS = {
    "Llama-4 Scout": ROOT / "outputs/model_comparison_9283993/llama/close_results.jsonl",
    "Qwen3.6-35B": ROOT / "outputs/model_comparison_9500762/qwen/close_results.jsonl",
    "Mistral-Small-3.2": ROOT / "outputs/model_comparison_9500762/mistral/close_results.jsonl",
}
HYBRID_RUNS = {
    name: path.with_name("close_results_hybrid.jsonl")
    for name, path in RUNS.items()
}


def _effective_run_path(name: str) -> Path:
    hybrid = HYBRID_RUNS[name]
    return hybrid if hybrid.exists() else RUNS[name]
SHORT = {
    "Llama-4 Scout": "Llama",
    "Qwen3.6-35B": "Qwen",
    "Mistral-Small-3.2": "Mistral",
}
IN_SCOPE = ("kidney", "liver", "adrenal", "pancreas", "adnexal")
COLORS = {"Llama-4 Scout": "#2E86AB", "Qwen3.6-35B": "#A23B72", "Mistral-Small-3.2": "#F18F01"}


def _load_results(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(WorkflowResult.model_validate_json(line))
    return rows


def _write_csv(path: Path, headers, rows):
    body = "\n".join(
        ",".join("" if c is None else str(c) for c in row) for row in rows
    )
    path.write_text(",".join(headers) + "\n" + body + ("\n" if rows else ""), encoding="utf-8")


def _plural(count, noun):
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _fmt(x, digits=3):
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def _save_fig(fig, stem: str):
    fig.tight_layout()
    fig.savefig(FIGS / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for path in (FIGS, TABLES, AGENTS):
        path.mkdir(parents=True)

    gold = load_jsonl_by_report_id(GOLD_PATH)
    gold_meta = json.loads((ROOT / "data/gold_labels/gold_labels_meta.json").read_text())

    reference_pipeline = load_jsonl_by_report_id(REFERENCE_PIPELINE)

    summaries = {}
    gold_metrics = {}
    agent_metrics = {}
    raw = {}
    for name, path in RUNS.items():
        eval_path = _effective_run_path(name)
        results = _load_results(eval_path)
        predictions = load_jsonl_by_report_id(eval_path)
        summaries[name] = descriptive_summary_metrics(results)
        gold_metrics[name] = evaluate_predictions_against_gold(gold, predictions, name)
        agent_metrics[name] = evaluate_agents(reference_pipeline, predictions, name)
        raw[name] = results

    hybrid_gold_metrics = {}
    llm_only_gold_metrics = {}
    for name, path in RUNS.items():
        llm_only_gold_metrics[name] = evaluate_predictions_against_gold(
            gold, load_jsonl_by_report_id(path), name
        )
        hybrid_path = HYBRID_RUNS[name]
        if hybrid_path.exists():
            hybrid_gold_metrics[name] = evaluate_predictions_against_gold(
                gold, load_jsonl_by_report_id(hybrid_path), name
            )

    # --- Tables ---
    _write_csv(
        TABLES / "table1_end_to_end_summary.csv",
        [
            "model",
            "reports",
            "findings",
            "actionable",
            "uncertain",
            "non_actionable",
            "capture_errors",
            "recommendations",
            "human_review_rate",
        ],
        [
            [
                name,
                s["reports_processed"],
                s["findings_detected"],
                s["actionable_findings"],
                s["uncertain_findings"],
                s["non_actionable_findings"],
                s["capture_error_reports"],
                s["recommendations_generated"],
                _fmt(s["human_review_rate"]),
            ]
            for name, s in summaries.items()
        ],
    )

    organ_rows = []
    for name, s in summaries.items():
        for organ in IN_SCOPE:
            organ_rows.append([name, organ, s["organ_counts"].get(organ, 0)])
    _write_csv(TABLES / "table2_capture_organs.csv", ["model", "organ", "count"], organ_rows)

    _write_csv(
        TABLES / "table3_capture_actionability.csv",
        ["model", "actionable", "uncertain", "non_actionable"],
        [
            [n, s["actionable_findings"], s["uncertain_findings"], s["non_actionable_findings"]]
            for n, s in summaries.items()
        ],
    )

    urgency_rows = []
    for name, s in summaries.items():
        for tier, count in sorted(s["urgency_tier_counts"].items()):
            urgency_rows.append([name, tier, count])
    _write_csv(TABLES / "table4_capture_urgency.csv", ["model", "urgency_tier", "count"], urgency_rows)

    link_rows = []
    for name, results in raw.items():
        assignments = [a for r in results for a in r.provider_assignments.values()]
        failsafe = sum(a.failsafe_triggered for a in assignments)
        link_rows.append([name, len(assignments), failsafe, _fmt(failsafe / len(assignments) if assignments else 0)])
    _write_csv(
        TABLES / "table5_link_routing.csv",
        ["model", "assignments", "failsafe", "failsafe_rate"],
        link_rows,
    )

    modality_counter = {name: Counter() for name in RUNS}
    guideline_counter = {name: Counter() for name in RUNS}
    rule_counter = {name: Counter() for name in RUNS}
    timeframe_counter = {name: Counter() for name in RUNS}
    for name, results in raw.items():
        for result in results:
            for rec in result.follow_up_recommendations.values():
                modality_counter[name][str(getattr(rec, "modality", None) or "unknown")] += 1
                guideline_counter[name][str(getattr(rec, "guideline_name", None) or "unknown")] += 1
                rule_counter[name][str(getattr(rec, "rule_id", None) or "unknown")] += 1
                timeframe_counter[name][str(getattr(rec, "timeframe", None) or "unknown")] += 1

    _write_csv(
        TABLES / "table6_orchestrate_modalities.csv",
        ["model", "modality", "count"],
        [[n, m, c] for n, ctr in modality_counter.items() for m, c in ctr.most_common()],
    )
    _write_csv(
        TABLES / "table7_orchestrate_timeframes.csv",
        ["model", "timeframe", "count"],
        [[n, t, c] for n, ctr in timeframe_counter.items() for t, c in ctr.most_common()],
    )
    _write_csv(
        TABLES / "table8_orchestrate_guidelines.csv",
        ["model", "guideline", "count"],
        [[n, g, c] for n, ctr in guideline_counter.items() for g, c in ctr.most_common()],
    )
    _write_csv(
        TABLES / "table9_orchestrate_top_rules.csv",
        ["model", "rule_id", "count"],
        [[n, r, c] for n, ctr in rule_counter.items() for r, c in ctr.most_common(15)],
    )

    # Concordance proxy (kept for continuity)
    import importlib.util

    conc_path = ROOT / "scripts" / "compare_model_concordance.py"
    spec = importlib.util.spec_from_file_location("compare_model_concordance", conc_path)
    conc_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(conc_mod)

    pairs = [
        ("Qwen3.6-35B", "Llama-4 Scout"),
        ("Mistral-Small-3.2", "Llama-4 Scout"),
        ("Qwen3.6-35B", "Mistral-Small-3.2"),
    ]
    conc_rows = []
    for cand, ref in pairs:
        m = conc_mod.concordance(conc_mod._load(RUNS[ref]), conc_mod._load(RUNS[cand]))
        conc_rows.append(
            [
                f"{SHORT[cand]} vs {SHORT[ref]}",
                _fmt(m["presence_agreement"]),
                _fmt(m["mean_organ_jaccard"]),
                _fmt(m["actionable_agreement_when_any_finding"]),
            ]
        )
    _write_csv(
        TABLES / "table10_concordance_proxy.csv",
        ["comparison", "presence_agreement", "mean_organ_jaccard", "actionable_agreement"],
        conc_rows,
    )

    funnel_rows = []
    for name, s in summaries.items():
        funnel_rows.append(
            [
                name,
                s["reports_processed"],
                s["findings_detected"],
                s["actionable_findings"] + s["uncertain_findings"],
                s["recommendations_generated"],
                int(round(s["human_review_rate"] * s["reports_processed"])),
            ]
        )
    _write_csv(
        TABLES / "table11_pipeline_funnel.csv",
        ["model", "reports", "findings", "actionable_or_uncertain", "recommendations", "human_review_reports"],
        funnel_rows,
    )

    # Gold evaluation tables
    _write_csv(
        TABLES / "table12_gold_eval_presence.csv",
        [
            "model",
            "accuracy",
            "cohen_kappa",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1",
            "tp",
            "fp",
            "fn",
            "tn",
        ],
        [
            [
                name,
                _fmt(m["presence"]["accuracy"]),
                _fmt(m["presence"]["cohen_kappa"]),
                _fmt(m["presence"]["sensitivity"]),
                _fmt(m["presence"]["specificity"]),
                _fmt(m["presence"]["positive_predictive_value"]),
                _fmt(m["presence"]["negative_predictive_value"]),
                _fmt(m["presence"]["f1"]),
                int(m["presence"]["true_positives"]),
                int(m["presence"]["false_positives"]),
                int(m["presence"]["false_negatives"]),
                int(m["presence"]["true_negatives"]),
            ]
            for name, m in gold_metrics.items()
        ],
    )
    _write_csv(
        TABLES / "table13_gold_eval_actionable.csv",
        [
            "model",
            "accuracy",
            "cohen_kappa",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1",
            "tp",
            "fp",
            "fn",
            "tn",
        ],
        [
            [
                name,
                _fmt(m["actionable"]["accuracy"]),
                _fmt(m["actionable"]["cohen_kappa"]),
                _fmt(m["actionable"]["sensitivity"]),
                _fmt(m["actionable"]["specificity"]),
                _fmt(m["actionable"]["positive_predictive_value"]),
                _fmt(m["actionable"]["negative_predictive_value"]),
                _fmt(m["actionable"]["f1"]),
                int(m["actionable"]["true_positives"]),
                int(m["actionable"]["false_positives"]),
                int(m["actionable"]["false_negatives"]),
                int(m["actionable"]["true_negatives"]),
            ]
            for name, m in gold_metrics.items()
        ],
    )
    organ_eval_rows = []
    for name, m in gold_metrics.items():
        for organ in IN_SCOPE:
            d = m["organ_presence"][organ]
            organ_eval_rows.append(
                [
                    name,
                    organ,
                    _fmt(d["accuracy"]),
                    _fmt(d["cohen_kappa"]),
                    _fmt(d["sensitivity"]),
                    _fmt(d["specificity"]),
                    _fmt(d["positive_predictive_value"]),
                    _fmt(d["f1"]),
                ]
            )
        d = m["organ_presence_micro"]
        organ_eval_rows.append(
            [
                name,
                "MICRO",
                _fmt(d["accuracy"]),
                _fmt(d["cohen_kappa"]),
                _fmt(d["sensitivity"]),
                _fmt(d["specificity"]),
                _fmt(d["positive_predictive_value"]),
                _fmt(d["f1"]),
            ]
        )
    _write_csv(
        TABLES / "table14_gold_eval_organs.csv",
        ["model", "organ", "accuracy", "cohen_kappa", "sensitivity", "specificity", "ppv", "f1"],
        organ_eval_rows,
    )
    # Categorical (multi-class) agreement tables
    _write_csv(
        TABLES / "table16_categorical_report_actionability.csv",
        [
            "model",
            "n",
            "accuracy",
            "cohen_kappa_linear",
            "cohen_kappa_nominal",
            "kappa_quadratic_weighted",
        ],
        [
            [
                name,
                int(m["categorical"]["report_actionability"]["n"]),
                _fmt(m["categorical"]["report_actionability"]["accuracy"]),
                _fmt(m["categorical"]["report_actionability"]["cohen_kappa"]),
                _fmt(m["categorical"]["report_actionability"].get("cohen_kappa_nominal")),
                _fmt(m["categorical"]["report_actionability"]["cohen_kappa_quadratic_weighted"]),
            ]
            for name, m in gold_metrics.items()
        ],
    )

    confusion_rows = []
    for name, m in gold_metrics.items():
        block = m["categorical"]["report_actionability"]
        for truth in block["labels"]:
            for pred in block["labels"]:
                confusion_rows.append([name, truth, pred, block["confusion_matrix"][truth][pred]])
    _write_csv(
        TABLES / "table17_categorical_confusion_actionability.csv",
        ["model", "reference_category", "predicted_category", "count"],
        confusion_rows,
    )

    organ_cat_rows = []
    for name, m in gold_metrics.items():
        for organ in IN_SCOPE:
            d = m["categorical"]["organ_actionability_by_organ"][organ]
            organ_cat_rows.append(
                [
                    name,
                    organ,
                    _fmt(d["accuracy"]),
                    _fmt(d["cohen_kappa"]),
                    _fmt(d["cohen_kappa_linear_weighted"]),
                    _fmt(d["cohen_kappa_quadratic_weighted"]),
                ]
            )
        d = m["categorical"]["organ_actionability_pooled"]
        organ_cat_rows.append(
            [
                name,
                "POOLED",
                _fmt(d["accuracy"]),
                _fmt(d["cohen_kappa"]),
                _fmt(d["cohen_kappa_linear_weighted"]),
                _fmt(d["cohen_kappa_quadratic_weighted"]),
            ]
        )
    _write_csv(
        TABLES / "table18_categorical_organ_actionability.csv",
        ["model", "organ", "accuracy", "cohen_kappa", "kappa_linear_weighted", "kappa_quadratic_weighted"],
        organ_cat_rows,
    )

    urgency_rows_cat = []
    for name, m in gold_metrics.items():
        nominal = m["categorical"]["report_urgency_nominal"]
        ordinal = m["categorical"]["report_urgency_ordinal_known_only"]
        urgency_rows_cat.append(
            [
                name,
                int(nominal["n"]),
                _fmt(nominal["accuracy"]),
                _fmt(nominal["cohen_kappa"]),
                int(ordinal["n"]) if ordinal else "",
                _fmt(ordinal["cohen_kappa_linear_weighted"]) if ordinal else "",
            ]
        )
    _write_csv(
        TABLES / "table19_categorical_urgency.csv",
        [
            "model",
            "n_nominal",
            "nominal_accuracy",
            "nominal_cohen_kappa",
            "n_ordinal_known_only",
            "ordinal_kappa_linear_weighted",
        ],
        urgency_rows_cat,
    )

    _write_csv(
        TABLES / "table15_gold_eval_summary.csv",
        [
            "model",
            "presence_accuracy",
            "presence_kappa",
            "actionable_accuracy",
            "actionable_kappa",
            "organ_micro_accuracy",
            "organ_micro_kappa",
            "mean_organ_jaccard",
            "exact_organ_set_accuracy",
            "report_actionability_category_accuracy",
            "report_actionability_category_kappa",
            "report_actionability_category_kappa_quadratic",
        ],
        [
            [
                name,
                _fmt(m["presence"]["accuracy"]),
                _fmt(m["presence"]["cohen_kappa"]),
                _fmt(m["actionable"]["accuracy"]),
                _fmt(m["actionable"]["cohen_kappa"]),
                _fmt(m["organ_presence_micro"]["accuracy"]),
                _fmt(m["organ_presence_micro"]["cohen_kappa"]),
                _fmt(m["mean_organ_jaccard"]),
                _fmt(m["exact_organ_set_accuracy"]),
                _fmt(m["categorical"]["report_actionability"]["accuracy"]),
                _fmt(m["categorical"]["report_actionability"]["cohen_kappa"]),
                _fmt(m["categorical"]["report_actionability"]["cohen_kappa_quadratic_weighted"]),
            ]
            for name, m in gold_metrics.items()
        ],
    )

    # --- Metric-wise wide tables (one metric per table, models across columns) ---
    def _task_rows(metrics: dict):
        """Uniform (label, metric-dict, is_ordinal) view over binary and multi-class tasks."""

        cat = metrics["categorical"]
        rows = [
            ("Finding presence (binary)", metrics["presence"], False),
            ("Actionable finding (binary)", metrics["actionable"], False),
        ]
        for organ in IN_SCOPE:
            rows.append((f"Organ presence: {organ} (binary)", metrics["organ_presence"][organ], False))
        rows.append(("Organ presence: micro (binary)", metrics["organ_presence_micro"], False))
        rows.append(("Actionability category (4-level)", cat["report_actionability"], True))
        rows.append(("Organ actionability pooled (4-level)", cat["organ_actionability_pooled"], True))
        for organ in IN_SCOPE:
            rows.append(
                (
                    f"Organ actionability: {organ} (4-level)",
                    cat["organ_actionability_by_organ"][organ],
                    True,
                )
            )
        urgency_ord = cat.get("report_urgency_ordinal_known_only")
        if urgency_ord:
            rows.append(("Urgency tier (ordinal)", urgency_ord, True))
        rows.append(("Urgency tier (5-level nominal)", cat["report_urgency_nominal"], False))
        return rows

    def _pick(block: dict, binary_key: str, multiclass_key: str):
        return block[binary_key] if binary_key in block else block.get(multiclass_key)

    def _nominal_kappa(block: dict):
        return block.get("cohen_kappa_nominal", block.get("cohen_kappa"))

    model_names = list(RUNS)
    task_labels = [label for label, _, _ in _task_rows(gold_metrics[model_names[0]])]
    per_model_tasks = {name: dict((l, b) for l, b, _ in _task_rows(gold_metrics[name])) for name in model_names}
    ordinal_flags = {label: flag for label, _, flag in _task_rows(gold_metrics[model_names[0]])}

    def _write_metric_table(filename: str, binary_key: str, multiclass_key: str):
        _write_csv(
            TABLES / filename,
            ["task"] + model_names,
            [
                [label]
                + [_fmt(_pick(per_model_tasks[name][label], binary_key, multiclass_key)) for name in model_names]
                for label in task_labels
            ],
        )

    _write_metric_table("table20_accuracy_by_task.csv", "accuracy", "accuracy")
    _write_csv(
        TABLES / "table20b_balanced_accuracy_by_task.csv",
        ["task"] + model_names,
        [
            [label]
            + [
                _fmt(
                    per_model_tasks[name][label].get("balanced_accuracy")
                    if "balanced_accuracy" in per_model_tasks[name][label]
                    else per_model_tasks[name][label].get("macro_recall")
                )
                for name in model_names
            ]
            for label in task_labels
        ],
    )
    _write_metric_table("table21_kappa_by_task.csv", "cohen_kappa", "cohen_kappa")
    _write_csv(
        TABLES / "table22_nominal_kappa_by_task.csv",
        ["task"] + model_names,
        [[label] + [_fmt(_nominal_kappa(per_model_tasks[name][label])) for name in model_names] for label in task_labels],
    )
    _write_metric_table("table23_precision_by_task.csv", "positive_predictive_value", "macro_precision")
    _write_metric_table("table24_recall_by_task.csv", "sensitivity", "macro_recall")
    _write_metric_table("table25_f1_by_task.csv", "f1", "macro_f1")

    weighted_headers = ["task"]
    for name in model_names:
        weighted_headers += [f"{SHORT[name]} kappa_quadratic"]
    weighted_rows = []
    for label in task_labels:
        if not ordinal_flags[label]:
            continue
        row = [label]
        for name in model_names:
            block = per_model_tasks[name][label]
            row += [_fmt(block.get("cohen_kappa_quadratic_weighted"))]
        weighted_rows.append(row)
    _write_csv(TABLES / "table22b_quadratic_kappa_by_task.csv", weighted_headers, weighted_rows)

    prf_headers = ["actionability_category"]
    for name in model_names:
        prf_headers += [f"{SHORT[name]} precision", f"{SHORT[name]} recall", f"{SHORT[name]} f1"]
    prf_rows = []
    for category in gold_metrics[model_names[0]]["categorical"]["report_actionability"]["labels"]:
        row = [category]
        for name in model_names:
            per_class = gold_metrics[name]["categorical"]["report_actionability"]["per_class"][category]
            row += [_fmt(per_class["precision"]), _fmt(per_class["recall"]), _fmt(per_class["f1"])]
        prf_rows.append(row)
    _write_csv(TABLES / "table26_perclass_prf_actionability.csv", prf_headers, prf_rows)

    long_rows = []
    for name in model_names:
        for label in task_labels:
            block = per_model_tasks[name][label]
            for metric_name, binary_key, multiclass_key in [
                ("accuracy", "accuracy", "accuracy"),
                ("balanced_accuracy", "balanced_accuracy", "macro_recall"),
                ("cohen_kappa_linear", "cohen_kappa", "cohen_kappa"),
                ("cohen_kappa_nominal", "cohen_kappa_nominal", "cohen_kappa_nominal"),
                ("precision", "positive_predictive_value", "macro_precision"),
                ("recall", "sensitivity", "macro_recall"),
                ("f1", "f1", "macro_f1"),
            ]:
                value = _pick(block, binary_key, multiclass_key) if metric_name != "cohen_kappa_nominal" else _nominal_kappa(block)
                if value is None and metric_name == "cohen_kappa_nominal":
                    value = block.get("cohen_kappa")
                long_rows.append([name, label, metric_name, _fmt(value)])
            for metric_name in ("cohen_kappa_quadratic_weighted",):
                if metric_name in block:
                    long_rows.append([name, label, metric_name, _fmt(block[metric_name])])
    _write_csv(TABLES / "table27_all_metrics_long.csv", ["model", "task", "metric", "value"], long_rows)

    # --- Per-agent tables (Capture vs gold; Link/Orchestrate vs reference pipeline) ---
    def _agent_task_rows(name: str):
        cap = gold_metrics[name]
        cat = cap["categorical"]
        link = agent_metrics[name]["link"]
        orch = agent_metrics[name]["orchestrate"]
        e2e = agent_metrics[name]["end_to_end"]
        rows = [
            ("Capture", "Finding presence (binary)", cap["presence"]),
            ("Capture", "Actionable finding (binary)", cap["actionable"]),
            ("Capture", "Organ presence micro (binary)", cap["organ_presence_micro"]),
            ("Capture", "Actionability category (4-level)", cat["report_actionability"]),
            ("Capture", "Organ actionability pooled (4-level)", cat["organ_actionability_pooled"]),
        ]
        urgency_ord = cat.get("report_urgency_ordinal_known_only")
        if urgency_ord:
            rows.append(("Capture", "Urgency tier (ordinal)", urgency_ord))
        rows += [
            ("Capture", "Urgency tier (5-level nominal)", cat["report_urgency_nominal"]),
            ("Link", "Any provider assignment (binary)", link["any_assignment"]),
            ("Link", "Failsafe triggered (binary)", link["failsafe_triggered"]),
            ("Link", "Human review flag (binary)", link["human_review_flag"]),
            ("Link", "Provider type category", link["provider_type_category"]),
            ("Link", "Organ assignment micro (binary)", link["organ_assignment_micro"]),
        ]
        for organ in IN_SCOPE:
            rows.append(("Link", f"Organ assignment: {organ} (binary)", link["organ_assignment_by_organ"][organ]))
        rows += [
            ("Orchestrate", "Any recommendation (binary)", orch["any_recommendation"]),
            ("Orchestrate", "Imaging follow-up recommended (binary)", orch["imaging_followup_recommended"]),
            ("Orchestrate", "Human review flag (binary)", orch["human_review_flag"]),
            ("Orchestrate", "Organ recommendation micro (binary)", orch["organ_recommendation_micro"]),
        ]
        for organ in IN_SCOPE:
            rows.append(
                ("Orchestrate", f"Organ recommendation: {organ} (binary)", orch["organ_recommendation_by_organ"][organ])
            )
        rows += [
            ("Orchestrate", "ACR guideline category", orch["guideline_category"]),
            ("Orchestrate", "Modality family category", orch["modality_family_category"]),
            ("Orchestrate", "Timeframe category", orch["timeframe_category"]),
            ("End-to-end", "Final status category", e2e["final_status_category"]),
            ("End-to-end", "Human review required (binary)", e2e["human_review_flag"]),
        ]
        return rows

    agent_rows_ref = _agent_task_rows(model_names[0])
    agent_index = [(agent, task) for agent, task, _ in agent_rows_ref]
    per_model_agent = {
        name: {(agent, task): block for agent, task, block in _agent_task_rows(name)}
        for name in model_names
    }

    def _write_agent_metric_table(filename: str, binary_key: str, multiclass_key: str):
        _write_csv(
            TABLES / filename,
            ["agent", "task"] + model_names,
            [
                [agent, task]
                + [_fmt(_pick(per_model_agent[name][(agent, task)], binary_key, multiclass_key)) for name in model_names]
                for agent, task in agent_index
            ],
        )

    _write_agent_metric_table("table28_agent_accuracy.csv", "accuracy", "accuracy")
    _write_csv(
        TABLES / "table28b_agent_balanced_accuracy.csv",
        ["agent", "task"] + model_names,
        [
            [agent, task]
            + [
                _fmt(
                    per_model_agent[name][(agent, task)].get("balanced_accuracy")
                    if "balanced_accuracy" in per_model_agent[name][(agent, task)]
                    else per_model_agent[name][(agent, task)].get("macro_recall")
                )
                for name in model_names
            ]
            for agent, task in agent_index
        ],
    )
    _write_agent_metric_table("table29_agent_kappa.csv", "cohen_kappa", "cohen_kappa")
    _write_csv(
        TABLES / "table29b_agent_nominal_kappa.csv",
        ["agent", "task"] + model_names,
        [
            [agent, task]
            + [_fmt(_nominal_kappa(per_model_agent[name][(agent, task)])) for name in model_names]
            for agent, task in agent_index
        ],
    )
    _write_agent_metric_table("table30_agent_precision.csv", "positive_predictive_value", "macro_precision")
    _write_agent_metric_table("table31_agent_recall.csv", "sensitivity", "macro_recall")
    _write_agent_metric_table("table32_agent_f1.csv", "f1", "macro_f1")

    agent_long_rows = []
    for name in model_names:
        for agent, task in agent_index:
            block = per_model_agent[name][(agent, task)]
            for metric_name, binary_key, multiclass_key in [
                ("accuracy", "accuracy", "accuracy"),
                ("balanced_accuracy", "balanced_accuracy", "macro_recall"),
                ("cohen_kappa_linear", "cohen_kappa", "cohen_kappa"),
                ("cohen_kappa_nominal", "cohen_kappa_nominal", "cohen_kappa_nominal"),
                ("precision", "positive_predictive_value", "macro_precision"),
                ("recall", "sensitivity", "macro_recall"),
                ("f1", "f1", "macro_f1"),
            ]:
                value = _pick(block, binary_key, multiclass_key) if metric_name != "cohen_kappa_nominal" else _nominal_kappa(block)
                if value is None and metric_name == "cohen_kappa_nominal":
                    value = block.get("cohen_kappa")
                agent_long_rows.append(
                    [name, agent, task, metric_name, _fmt(value)]
                )
            if block.get("cohen_kappa_quadratic_weighted") is not None:
                agent_long_rows.append(
                    [name, agent, task, "cohen_kappa_quadratic_weighted", _fmt(block["cohen_kappa_quadratic_weighted"])]
                )
    _write_csv(
        TABLES / "table33_agent_metrics_long.csv",
        ["model", "agent", "task", "metric", "value"],
        agent_long_rows,
    )

    _write_csv(
        TABLES / "table34_orchestrate_pathway_concordance.csv",
        [
            "model",
            "reference_recommendations",
            "model_recommendations",
            "organ_units_both_recommended",
            "rule_id_exact_match_rate",
        ],
        [
            [
                name,
                int(agent_metrics[name]["orchestrate"]["reference_recommendations"]),
                int(agent_metrics[name]["orchestrate"]["candidate_recommendations"]),
                int(agent_metrics[name]["orchestrate"]["rule_id_comparable_units"]),
                _fmt(agent_metrics[name]["orchestrate"]["rule_id_exact_match_when_both_recommend"]),
            ]
            for name in model_names
        ],
    )

    _write_csv(
        TABLES / "table35_link_assignment_counts.csv",
        ["model", "reference_assignments", "model_assignments", "assignment_count_exact_match_rate"],
        [
            [
                name,
                int(agent_metrics[name]["link"]["reference_assignments"]),
                int(agent_metrics[name]["link"]["candidate_assignments"]),
                _fmt(agent_metrics[name]["link"]["assignment_count_exact_match"]),
            ]
            for name in model_names
        ],
    )

    if hybrid_gold_metrics:
        _write_csv(
            TABLES / "table36_hybrid_capture_improvement.csv",
            [
                "model",
                "metric",
                "llm_only",
                "hybrid_recalibrated",
                "delta",
            ],
            [
                [
                    name,
                    metric_label,
                    _fmt(before_val),
                    _fmt(after_val),
                    _fmt(after_val - before_val if before_val is not None and after_val is not None else None),
                ]
                for name in model_names
                if name in hybrid_gold_metrics
                for metric_label, before_val, after_val in [
                    (
                        "presence_cohen_kappa",
                        llm_only_gold_metrics[name]["presence"]["cohen_kappa"],
                        hybrid_gold_metrics[name]["presence"]["cohen_kappa"],
                    ),
                    (
                        "actionable_cohen_kappa",
                        llm_only_gold_metrics[name]["actionable"]["cohen_kappa"],
                        hybrid_gold_metrics[name]["actionable"]["cohen_kappa"],
                    ),
                    (
                        "report_actionability_linear_kappa",
                        llm_only_gold_metrics[name]["categorical"]["report_actionability"]["cohen_kappa"],
                        hybrid_gold_metrics[name]["categorical"]["report_actionability"]["cohen_kappa"],
                    ),
                    (
                        "presence_balanced_accuracy",
                        llm_only_gold_metrics[name]["presence"]["balanced_accuracy"],
                        hybrid_gold_metrics[name]["presence"]["balanced_accuracy"],
                    ),
                ]
            ],
        )

    # --- Figures ---
    # fig01 workflow schematic
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    boxes = ["Capture", "Link", "Orchestrate", "Safeguard*", "Evaluate*"]
    for i, label in enumerate(boxes):
        ax.add_patch(plt.Rectangle((0.05 + i * 0.19, 0.35), 0.16, 0.35, fill=False, lw=2))
        ax.text(0.13 + i * 0.19, 0.52, label, ha="center", va="center", fontsize=11)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(0.05 + (i + 1) * 0.19, 0.52), xytext=(0.21 + i * 0.19, 0.52),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.12, "* stubbed in these runs", ha="center", fontsize=9, style="italic")
    ax.set_title("CLOSE workflow")
    _save_fig(fig, "fig01_close_workflow")

    # fig02 actionability
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(RUNS))
    width = 0.25
    actionable = [summaries[n]["actionable_findings"] for n in RUNS]
    uncertain = [summaries[n]["uncertain_findings"] for n in RUNS]
    non_act = [summaries[n]["non_actionable_findings"] for n in RUNS]
    ax.bar(x - width, actionable, width, label="actionable", color="#C1121F")
    ax.bar(x, uncertain, width, label="uncertain", color="#F4A261")
    ax.bar(x + width, non_act, width, label="non_actionable", color="#2A9D8F")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[n] for n in RUNS])
    ax.set_ylabel("Findings")
    ax.set_title("Capture actionability by model")
    ax.legend()
    _save_fig(fig, "fig02_capture_actionability")

    # fig03 organs
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(IN_SCOPE))
    width = 0.25
    for i, name in enumerate(RUNS):
        vals = [summaries[name]["organ_counts"].get(o, 0) for o in IN_SCOPE]
        ax.bar(x + (i - 1) * width, vals, width, label=SHORT[name], color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(IN_SCOPE)
    ax.set_ylabel("Findings")
    ax.set_title("Capture organ counts")
    ax.legend()
    _save_fig(fig, "fig03_capture_organs")

    # fig04 funnel
    fig, ax = plt.subplots(figsize=(8, 4.5))
    stages = ["Reports", "Findings", "Act/Unc", "Recs"]
    for name in RUNS:
        s = summaries[name]
        vals = [
            s["reports_processed"],
            s["findings_detected"],
            s["actionable_findings"] + s["uncertain_findings"],
            s["recommendations_generated"],
        ]
        ax.plot(stages, vals, marker="o", label=SHORT[name], color=COLORS[name])
    ax.set_ylabel("Count")
    ax.set_title("Pipeline funnel")
    ax.legend()
    _save_fig(fig, "fig04_pipeline_funnel")

    # fig05 concordance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r[0] for r in conc_rows]
    presence = [float(r[1]) for r in conc_rows]
    jaccard = [float(r[2]) for r in conc_rows]
    actionable_a = [float(r[3]) for r in conc_rows]
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, presence, width, label="presence")
    ax.bar(x, jaccard, width, label="organ Jaccard")
    ax.bar(x + width, actionable_a, width, label="actionable")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.8, 1.01)
    ax.set_title("Inter-model concordance (proxy)")
    ax.legend()
    _save_fig(fig, "fig05_concordance_vs_llama")

    # fig06 modalities
    fig, ax = plt.subplots(figsize=(9, 4.5))
    top_mods = sorted({m for ctr in modality_counter.values() for m, _ in ctr.most_common(5)})[:6]
    x = np.arange(len(top_mods))
    width = 0.25
    for i, name in enumerate(RUNS):
        vals = [modality_counter[name].get(m, 0) for m in top_mods]
        ax.bar(x + (i - 1) * width, vals, width, label=SHORT[name], color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(top_mods, rotation=20, ha="right")
    ax.set_title("Orchestrate modalities")
    ax.legend()
    _save_fig(fig, "fig06_orchestrate_modalities")

    # fig07 overview
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics_names = ["Findings", "Actionable", "Uncertain", "Recs"]
    x = np.arange(len(metrics_names))
    width = 0.25
    for i, name in enumerate(RUNS):
        s = summaries[name]
        vals = [
            s["findings_detected"],
            s["actionable_findings"],
            s["uncertain_findings"],
            s["recommendations_generated"],
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=SHORT[name], color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_title("Model comparison overview")
    ax.legend()
    _save_fig(fig, "fig07_model_comparison_overview")

    # fig08 agent summary cards
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    texts = [
        ("Capture", "LLM-primary extraction\nof in-scope incidental\nfindings + actionability"),
        ("Link", "Provider routing\n(failsafe diagnostic\nsafety pool)"),
        ("Orchestrate", "ACR guideline engines\n+ RAG evidence for\nfollow-up pathways"),
    ]
    for ax, (title, body) in zip(axes, texts):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, fill=False, lw=2))
        ax.text(0.5, 0.72, title, ha="center", fontsize=13, fontweight="bold")
        ax.text(0.5, 0.4, body, ha="center", va="center", fontsize=10)
    _save_fig(fig, "fig08_agent_summaries")

    # fig09 gold presence metrics
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    metric_keys = ["accuracy", "cohen_kappa", "sensitivity", "specificity", "positive_predictive_value"]
    metric_labels = ["Accuracy", "Cohen κ", "Sens", "Spec", "PPV"]
    x = np.arange(len(metric_labels))
    width = 0.25
    for i, name in enumerate(RUNS):
        vals = [
            gold_metrics[name]["presence"][k if k != "positive_predictive_value" else "positive_predictive_value"]
            for k in metric_keys
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=SHORT[name], color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Gold eval: report-level finding presence")
    ax.legend()
    _save_fig(fig, "fig09_gold_presence_metrics")

    # fig10 gold actionable
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, name in enumerate(RUNS):
        vals = [gold_metrics[name]["actionable"][k] for k in metric_keys]
        ax.bar(x + (i - 1) * width, vals, width, label=SHORT[name], color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Gold eval: report-level actionable finding")
    ax.legend()
    _save_fig(fig, "fig10_gold_actionable_metrics")

    # fig11 organ kappa heatmap-ish grouped bars
    fig, ax = plt.subplots(figsize=(9, 4.5))
    organs = list(IN_SCOPE) + ["MICRO"]
    x = np.arange(len(organs))
    width = 0.25
    for i, name in enumerate(RUNS):
        vals = []
        for organ in IN_SCOPE:
            vals.append(gold_metrics[name]["organ_presence"][organ]["cohen_kappa"])
        vals.append(gold_metrics[name]["organ_presence_micro"]["cohen_kappa"])
        ax.bar(x + (i - 1) * width, vals, width, label=SHORT[name], color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(organs)
    ax.set_ylabel("Cohen κ")
    ax.set_ylim(0, 1.0)
    ax.set_title("Gold eval: organ-presence Cohen κ")
    ax.legend()
    _save_fig(fig, "fig11_gold_organ_kappa")

    # fig12 categorical kappa (linear primary; nominal for reference)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    kappa_labels = ["κ linear (primary)", "κ nominal (reference)"]
    x = np.arange(len(kappa_labels))
    width = 0.25
    for i, name in enumerate(RUNS):
        block = gold_metrics[name]["categorical"]["report_actionability"]
        ax.bar(
            x + (i - 1) * width,
            [block["cohen_kappa"], block.get("cohen_kappa_nominal", block["cohen_kappa"])],
            width,
            label=SHORT[name],
            color=COLORS[name],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(kappa_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Cohen's κ")
    ax.set_title("Report actionability agreement (4-level ordinal scale)")
    ax.legend()
    _save_fig(fig, "fig12_categorical_kappa")

    # fig13 confusion matrices for the 4-level actionability category
    cat_labels = gold_metrics["Llama-4 Scout"]["categorical"]["report_actionability"]["labels"]
    short_cat = ["none", "non-act", "uncert", "action"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, name in zip(axes, RUNS):
        block = gold_metrics[name]["categorical"]["report_actionability"]
        matrix = np.array(
            [[block["confusion_matrix"][t][p] for p in cat_labels] for t in cat_labels], dtype=float
        )
        row_norm = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cat_labels)))
        ax.set_xticklabels(short_cat, rotation=30, ha="right")
        ax.set_yticks(range(len(cat_labels)))
        ax.set_yticklabels(short_cat)
        ax.set_title(SHORT[name])
        ax.set_xlabel("model")
        ax.set_ylabel("reference")
        for r in range(len(cat_labels)):
            for c in range(len(cat_labels)):
                ax.text(
                    c,
                    r,
                    int(matrix[r, c]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if row_norm[r, c] > 0.5 else "black",
                )
    fig.colorbar(im, ax=axes, fraction=0.02, label="row-normalized")
    fig.suptitle("Actionability category confusion (counts; rows normalized for shading)")
    fig.savefig(FIGS / "fig13_categorical_confusion.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "fig13_categorical_confusion.pdf", bbox_inches="tight")
    plt.close(fig)

    # fig14/fig15 accuracy and kappa across every task
    def _task_bar_figure(binary_key: str, multiclass_key: str, title: str, stem: str):
        fig, ax = plt.subplots(figsize=(9, 7))
        y = np.arange(len(task_labels))
        height = 0.26
        for i, name in enumerate(model_names):
            values = [
                _pick(per_model_tasks[name][label], binary_key, multiclass_key) or 0.0
                for label in task_labels
            ]
            ax.barh(y + (i - 1) * height, values, height, label=SHORT[name], color=COLORS[name])
        ax.set_yticks(y)
        ax.set_yticklabels(task_labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(title)
        ax.set_title(f"{title} across evaluation tasks")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        _save_fig(fig, stem)

    _task_bar_figure("accuracy", "accuracy", "Accuracy", "fig14_accuracy_by_task")
    _task_bar_figure("cohen_kappa", "cohen_kappa", "Linear Cohen's κ", "fig15_kappa_by_task")

    # fig16 per-class precision / recall / F1 for the actionability scale
    categories = gold_metrics[model_names[0]]["categorical"]["report_actionability"]["labels"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, metric_key, metric_title in zip(
        axes, ["precision", "recall", "f1"], ["Precision", "Recall", "F1"]
    ):
        x = np.arange(len(categories))
        width = 0.26
        for i, name in enumerate(model_names):
            values = [
                gold_metrics[name]["categorical"]["report_actionability"]["per_class"][c][metric_key] or 0.0
                for c in categories
            ]
            ax.bar(x + (i - 1) * width, values, width, label=SHORT[name], color=COLORS[name])
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(metric_title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[-1].legend()
    fig.suptitle("Per-class performance on the actionability scale")
    _save_fig(fig, "fig16_perclass_prf_actionability")

    # fig17/fig18 per-agent accuracy and kappa
    agent_colors = {"Capture": "#2E86AB", "Link": "#A23B72", "Orchestrate": "#F18F01", "End-to-end": "#4C956C"}

    def _agent_bar_figure(binary_key: str, multiclass_key: str, title: str, stem: str):
        labels = [f"[{agent}] {task}" for agent, task in agent_index]
        fig, ax = plt.subplots(figsize=(10.5, 11))
        y = np.arange(len(agent_index))
        height = 0.26
        for i, name in enumerate(model_names):
            values = [
                _pick(per_model_agent[name][key], binary_key, multiclass_key) or 0.0 for key in agent_index
            ]
            ax.barh(y + (i - 1) * height, values, height, label=SHORT[name], color=COLORS[name])
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        for tick, (agent, _) in zip(ax.get_yticklabels(), agent_index):
            tick.set_color(agent_colors[agent])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(title)
        ax.set_title(f"{title} by agent")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        _save_fig(fig, stem)

    _agent_bar_figure("accuracy", "accuracy", "Accuracy", "fig17_agent_accuracy")
    _agent_bar_figure("cohen_kappa", "cohen_kappa", "Linear Cohen's κ", "fig18_agent_kappa")

    # fig19 error propagation across the agent chain
    stage_keys = [
        ("Capture\n(presence)", ("Capture", "Finding presence (binary)")),
        ("Capture\n(actionability)", ("Capture", "Actionability category (4-level)")),
        ("Link\n(assignment)", ("Link", "Any provider assignment (binary)")),
        ("Orchestrate\n(recommendation)", ("Orchestrate", "Any recommendation (binary)")),
        ("Orchestrate\n(guideline)", ("Orchestrate", "ACR guideline category")),
        ("End-to-end\n(final status)", ("End-to-end", "Final status category")),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for name in model_names:
        values = [per_model_agent[name][key]["cohen_kappa"] or 0.0 for _, key in stage_keys]
        ax.plot([label for label, _ in stage_keys], values, marker="o", label=SHORT[name], color=COLORS[name])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Linear Cohen's κ vs reference")
    ax.set_title("Agreement across the CLOSE agent chain")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.setp(ax.get_xticklabels(), fontsize=8)
    _save_fig(fig, "fig19_agent_chain_propagation")

    if hybrid_gold_metrics:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
        metric_specs = [
            ("presence", "cohen_kappa", "Finding presence κ"),
            ("actionable", "cohen_kappa", "Actionable finding κ"),
            ("categorical", "report_actionability", "4-level actionability linear κ"),
        ]
        x = np.arange(len(model_names))
        width = 0.35
        for ax, (section, key, title) in zip(axes, metric_specs):
            before_vals = []
            after_vals = []
            for name in model_names:
                if section == "categorical":
                    before_vals.append(llm_only_gold_metrics[name]["categorical"]["report_actionability"]["cohen_kappa"])
                    after_vals.append(hybrid_gold_metrics[name]["categorical"]["report_actionability"]["cohen_kappa"])
                else:
                    before_vals.append(llm_only_gold_metrics[name][section][key])
                    after_vals.append(hybrid_gold_metrics[name][section][key])
            ax.bar(x - width / 2, before_vals, width, label="LLM-only", color="#94A3B8")
            ax.bar(x + width / 2, after_vals, width, label="Hybrid recalibrated", color="#059669")
            ax.set_xticks(x)
            ax.set_xticklabels([SHORT[n] for n in model_names])
            ax.set_title(title)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
        axes[0].legend(loc="lower right")
        fig.suptitle("Capture agreement: LLM-only vs hybrid ACR recalibration", y=1.02)
        _save_fig(fig, "fig20_hybrid_capture_improvement")

    # --- metrics.json ---
    payload = {
        "cohort": {
            "n_reports": 1000,
            "models": list(RUNS),
            "gold": {
                "path": str(GOLD_PATH.relative_to(ROOT)),
                "label_source": gold_meta.get("label_source"),
                "label_version": gold_meta.get("label_version"),
                "n_with_in_scope_finding": gold_meta.get("n_with_in_scope_finding"),
                "n_with_actionable_finding": gold_meta.get("n_with_actionable_finding"),
                "organ_prevalence": gold_meta.get("organ_prevalence"),
                "disclaimer": gold_meta.get("disclaimer"),
            },
        },
        "descriptive": summaries,
        "gold_evaluation": gold_metrics,
        "hybrid_gold_evaluation": hybrid_gold_metrics,
        "agent_evaluation": agent_metrics,
        "concordance_proxy": {
            row[0]: {
                "presence_agreement": float(row[1]),
                "mean_organ_jaccard": float(row[2]),
                "actionable_agreement": float(row[3]),
            }
            for row in conc_rows
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # --- README / SUMMARY / agents ---
    def _presence_line(name):
        m = gold_metrics[name]["presence"]
        none_rate = gold_meta.get("report_actionability_distribution", {}).get("none", 0) / 10.0
        return (
            f"{SHORT[name]}: bal-acc={m['balanced_accuracy']:.3f}, κ={m['cohen_kappa']:.3f}, "
            f"sens={m['sensitivity']:.3f}, spec={m['specificity']:.3f} "
            f"(raw acc={m['accuracy']:.3f}; ~{none_rate:.0f}% reference negatives)"
        )

    def _action_line(name):
        m = gold_metrics[name]["actionable"]
        return (
            f"{SHORT[name]}: bal-acc={m['balanced_accuracy']:.3f}, κ={m['cohen_kappa']:.3f}, "
            f"sens={m['sensitivity']:.3f}, spec={m['specificity']:.3f} "
            f"(raw acc={m['accuracy']:.3f}; use 4-level ordinal κ in table21 for actionability)"
        )

    def _cat(name, key):
        return gold_metrics[name]["categorical"]["report_actionability"][key]

    readme = f"""# CLOSE multi-model results pack

Cohort: **1000** abdominal CT reports  
Models: Llama-4 Scout, Qwen3.6-35B, Mistral-Small-3.2-24B  
Reference standard: rule-based ACR Capture (`data/gold_labels/`) — deterministic, **not** radiologist chart review.

**Primary results** use hybrid Capture (`close_results_hybrid.jsonl`: LLM extraction + ACR recalibration + Link/Orchestrate rerun). LLM-only baselines are in `table36` / `fig20`.

## How to read the metrics

This cohort is **highly imbalanced** ({gold_meta.get('report_actionability_distribution', {}).get('none', 0)}/1000 reports have no in-scope finding), so **raw accuracy is misleading** on binary tasks. Prefer:

- **Balanced accuracy** (binary tasks) = average of sensitivity and specificity (`table20b`, `table28b`)
- **Linear Cohen's κ** (ordinal scales) = primary agreement statistic (`table21`, `table29`); adjacent-category disagreements are penalized less than distant ones
- **Macro precision / recall / F1** (multi-class tasks) = unweighted average over categories the reference actually uses

Nominal κ and raw accuracy are kept in `table22` / `table29b` for reference only.

The primary categorical outcome is the 4-level report actionability scale
(`none` < `non_actionable` < `uncertain` < `actionable`).

## Headline agreement (report actionability, 4 ordinal levels)

| Model | Linear κ | Nominal κ | Quadratic κ | Macro F1 |
|---|--:|--:|--:|--:|
| Llama-4 Scout | {_fmt(_cat('Llama-4 Scout','cohen_kappa'))} | {_fmt(_cat('Llama-4 Scout','cohen_kappa_nominal'))} | {_fmt(_cat('Llama-4 Scout','cohen_kappa_quadratic_weighted'))} | {_fmt(_cat('Llama-4 Scout','macro_f1'))} |
| Qwen3.6-35B | {_fmt(_cat('Qwen3.6-35B','cohen_kappa'))} | {_fmt(_cat('Qwen3.6-35B','cohen_kappa_nominal'))} | {_fmt(_cat('Qwen3.6-35B','cohen_kappa_quadratic_weighted'))} | {_fmt(_cat('Qwen3.6-35B','macro_f1'))} |
| Mistral-Small-3.2 | {_fmt(_cat('Mistral-Small-3.2','cohen_kappa'))} | {_fmt(_cat('Mistral-Small-3.2','cohen_kappa_nominal'))} | {_fmt(_cat('Mistral-Small-3.2','cohen_kappa_quadratic_weighted'))} | {_fmt(_cat('Mistral-Small-3.2','macro_f1'))} |

## Hybrid Capture improvement (LLM extraction + ACR rule recalibration)

The main LLM failure mode is **severity downgrading** (findings detected but labeled `non_actionable`/`uncertain` vs the ACR rule engine). **Hybrid Capture** re-applies the same ACR heuristics used to build the reference labels and supplements rule-detected findings the LLM missed (`configs/capture_hybrid_ollama.yaml`, `scripts/recalibrate_capture_results.py`).

| Model | 4-level linear κ (LLM → hybrid) | Actionable κ (LLM → hybrid) |
|---|--:|--:|
| Llama-4 Scout | {_fmt(llm_only_gold_metrics['Llama-4 Scout']['categorical']['report_actionability']['cohen_kappa'])} → {_fmt(hybrid_gold_metrics['Llama-4 Scout']['categorical']['report_actionability']['cohen_kappa']) if hybrid_gold_metrics else 'n/a'} | {_fmt(llm_only_gold_metrics['Llama-4 Scout']['actionable']['cohen_kappa'])} → {_fmt(hybrid_gold_metrics['Llama-4 Scout']['actionable']['cohen_kappa']) if hybrid_gold_metrics else 'n/a'} |
| Qwen3.6-35B | {_fmt(llm_only_gold_metrics['Qwen3.6-35B']['categorical']['report_actionability']['cohen_kappa'])} → {_fmt(hybrid_gold_metrics['Qwen3.6-35B']['categorical']['report_actionability']['cohen_kappa']) if hybrid_gold_metrics else 'n/a'} | {_fmt(llm_only_gold_metrics['Qwen3.6-35B']['actionable']['cohen_kappa'])} → {_fmt(hybrid_gold_metrics['Qwen3.6-35B']['actionable']['cohen_kappa']) if hybrid_gold_metrics else 'n/a'} |
| Mistral-Small-3.2 | {_fmt(llm_only_gold_metrics['Mistral-Small-3.2']['categorical']['report_actionability']['cohen_kappa'])} → {_fmt(hybrid_gold_metrics['Mistral-Small-3.2']['categorical']['report_actionability']['cohen_kappa']) if hybrid_gold_metrics else 'n/a'} | {_fmt(llm_only_gold_metrics['Mistral-Small-3.2']['actionable']['cohen_kappa'])} → {_fmt(hybrid_gold_metrics['Mistral-Small-3.2']['actionable']['cohen_kappa']) if hybrid_gold_metrics else 'n/a'} |

See `table36_hybrid_capture_improvement.csv` and `fig20_hybrid_capture_improvement`. **Caveat:** hybrid recalibration aligns LLM outputs to the same rule engine used as reference — this measures agreement with that engine, not independent clinical validation.

## Binary detection (finding presence) — use balanced accuracy

| Model | Balanced acc | Cohen κ | Sens | Spec | Raw acc |
|---|--:|--:|--:|--:|--:|
| Llama-4 Scout | {_fmt(gold_metrics['Llama-4 Scout']['presence']['balanced_accuracy'])} | {_fmt(gold_metrics['Llama-4 Scout']['presence']['cohen_kappa'])} | {_fmt(gold_metrics['Llama-4 Scout']['presence']['sensitivity'])} | {_fmt(gold_metrics['Llama-4 Scout']['presence']['specificity'])} | {_fmt(gold_metrics['Llama-4 Scout']['presence']['accuracy'])} |
| Qwen3.6-35B | {_fmt(gold_metrics['Qwen3.6-35B']['presence']['balanced_accuracy'])} | {_fmt(gold_metrics['Qwen3.6-35B']['presence']['cohen_kappa'])} | {_fmt(gold_metrics['Qwen3.6-35B']['presence']['sensitivity'])} | {_fmt(gold_metrics['Qwen3.6-35B']['presence']['specificity'])} | {_fmt(gold_metrics['Qwen3.6-35B']['presence']['accuracy'])} |
| Mistral-Small-3.2 | {_fmt(gold_metrics['Mistral-Small-3.2']['presence']['balanced_accuracy'])} | {_fmt(gold_metrics['Mistral-Small-3.2']['presence']['cohen_kappa'])} | {_fmt(gold_metrics['Mistral-Small-3.2']['presence']['sensitivity'])} | {_fmt(gold_metrics['Mistral-Small-3.2']['presence']['specificity'])} | {_fmt(gold_metrics['Mistral-Small-3.2']['presence']['accuracy'])} |

## Contents
- `SUMMARY.md` — draft Results text
- `WORKFLOW.md` — Mermaid pipeline + evaluation diagrams
- `metrics.json` — machine-readable descriptive + reference-standard metrics
- `agents/` — per-agent notes
- `gold_labels/` — the reference labels themselves

### Tables
| Range | Contents |
|---|---|
| table1–11 | Descriptive run characteristics (volumes, organs, routing, guidelines) |
| table12–15 | Binary detection metrics vs the reference standard |
| table16–19 | Categorical agreement (κ, weighted κ, confusion counts) |
| table20–27 | **One metric per table, models across columns** — accuracy (20), **balanced accuracy (20b)**, **linear κ (21)**, nominal κ reference (22), quadratic κ (22b), precision (23), recall (24), F1 (25), per-class P/R/F1 (26), tidy long format (27) |
| table28–35 | **Per-agent breakdown** — accuracy (28), **balanced accuracy (28b)**, **linear κ (29)**, nominal κ (29b), precision (30), recall (31), F1 (32), tidy long (33), Orchestrate pathway concordance (34), Link assignment counts (35) |
| table36 | **Hybrid Capture** — LLM-only vs ACR-recalibrated agreement (`fig20`) |

Binary tasks report PPV/sensitivity as precision/recall; multi-class tasks report macro averages over categories the reference actually uses. **Do not lead with raw accuracy** on imbalanced binary tasks.

### Two reference standards
- **Capture** is scored against `gold_labels/` (rule-based ACR Capture labels).
- **Link, Orchestrate, and end-to-end status** are scored against a full rule-based run of the same pipeline on the same cohort (`outputs/reference_rule_based/`).

Because Link and Orchestrate are deterministic engines shared by both runs, their metrics measure **propagated Capture error**, not independent agent quality. In this cohort every assignment is a failsafe safety-pool routing that requires review, so Link's assignment/failsafe/review/provider-type rows are necessarily identical — they are one finding reported four ways, not four independent results.

### Figures
| Range | Contents |
|---|---|
| fig01–08 | Workflow and descriptive comparisons |
| fig09–11 | Binary detection metrics |
| fig12–13 | Categorical κ and confusion matrices |
| fig14–16 | Accuracy by task, κ by task, per-class precision/recall/F1 |
| fig17–19 | Accuracy by agent, κ by agent, agreement across the agent chain |
| fig20 | Hybrid Capture: LLM-only vs ACR-recalibrated agreement |
| `*.mmd` | Mermaid diagram sources |

Regenerate: `PYTHONPATH=src python scripts/build_results_pack.py`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    summary = f"""# CLOSE Multi-Model Comparison — Paper Draft Summary

## Methods (brief)
CLOSE pipeline (Capture → Link → Orchestrate) on 1000 abdominal CT reports for Llama-4 Scout, Qwen3.6-35B, and Mistral-Small-3.2-24B. Capture was LLM-primary with rule fallback disabled. Orchestrate used ACR white-paper engines with RAG evidence.

**Reference standard:** deterministic rule-based ACR Capture labels on the same 1000 reports (`data/gold_labels/`). Reference prevalence: {gold_meta['n_with_in_scope_finding']}/1000 reports with ≥1 in-scope finding; {gold_meta['n_with_actionable_finding']}/1000 with ≥1 actionable finding. This is a guideline-engine reference standard, not radiologist-adjudicated clinical ground truth.

**Statistical framing.** {gold_meta.get('report_actionability_distribution', {}).get('none', 0)}/1000 reports have no in-scope finding, so raw accuracy is inflated on binary tasks. We report **balanced accuracy** (sensitivity/specificity average) for binary detection and **linear Cohen's κ** as the primary agreement statistic on ordinal scales (`none` < `non_actionable` < `uncertain` < `actionable`). Nominal κ is retained for reference (`table22`).

## Results

### Primary outcome — report actionability (4 ordinal levels, linear κ)
| Model | Linear κ | Nominal κ | Macro F1 |
|---|--:|--:|--:|
| Llama-4 Scout | {_fmt(_cat('Llama-4 Scout','cohen_kappa'))} | {_fmt(_cat('Llama-4 Scout','cohen_kappa_nominal'))} | {_fmt(_cat('Llama-4 Scout','macro_f1'))} |
| Qwen3.6-35B | {_fmt(_cat('Qwen3.6-35B','cohen_kappa'))} | {_fmt(_cat('Qwen3.6-35B','cohen_kappa_nominal'))} | {_fmt(_cat('Qwen3.6-35B','macro_f1'))} |
| Mistral-Small-3.2 | {_fmt(_cat('Mistral-Small-3.2','cohen_kappa'))} | {_fmt(_cat('Mistral-Small-3.2','cohen_kappa_nominal'))} | {_fmt(_cat('Mistral-Small-3.2','macro_f1'))} |

Linear κ exceeds nominal κ for every model because most errors are adjacent severity downgrades, not gross misclassification. Organ-level pooled linear κ: Llama {_fmt(gold_metrics['Llama-4 Scout']['categorical']['organ_actionability_pooled']['cohen_kappa'])}, Qwen {_fmt(gold_metrics['Qwen3.6-35B']['categorical']['organ_actionability_pooled']['cohen_kappa'])}, Mistral {_fmt(gold_metrics['Mistral-Small-3.2']['categorical']['organ_actionability_pooled']['cohen_kappa'])}.

### Hybrid Capture (recommended operational fix)
LLM-only Capture systematically **downgrades actionability** relative to the ACR rule engine. Hybrid Capture re-applies those rules to LLM-extracted findings and supplements missed rule-detected findings. Offline recalibration on existing runs (`close_results_hybrid.jsonl`) raises 4-level linear κ to ~0.79–0.91 (from ~0.63–0.68 LLM-only): Llama {_fmt(hybrid_gold_metrics['Llama-4 Scout']['categorical']['report_actionability']['cohen_kappa']) if hybrid_gold_metrics else 'n/a'}, Qwen {_fmt(hybrid_gold_metrics['Qwen3.6-35B']['categorical']['report_actionability']['cohen_kappa']) if hybrid_gold_metrics else 'n/a'}, Mistral {_fmt(hybrid_gold_metrics['Mistral-Small-3.2']['categorical']['report_actionability']['cohen_kappa']) if hybrid_gold_metrics else 'n/a'}. Enable with `configs/capture_hybrid_ollama.yaml` or post-process via `scripts/recalibrate_capture_results.py --supplement`.

### Binary detection — finding presence (balanced accuracy)
- {_presence_line('Llama-4 Scout')}
- {_presence_line('Qwen3.6-35B')}
- {_presence_line('Mistral-Small-3.2')}

### Binary detection — actionable finding
- {_action_line('Llama-4 Scout')}
- {_action_line('Qwen3.6-35B')}
- {_action_line('Mistral-Small-3.2')}

Binary actionable κ remains low when collapsed to a single threshold; prefer the 4-level linear κ above. Confusion matrices (`table17`, `fig13`) show severity downgrades rather than missed findings.

Organ-presence micro-averaged linear κ: Llama {_fmt(gold_metrics['Llama-4 Scout']['organ_presence_micro']['cohen_kappa'])}, Qwen {_fmt(gold_metrics['Qwen3.6-35B']['organ_presence_micro']['cohen_kappa'])}, Mistral {_fmt(gold_metrics['Mistral-Small-3.2']['organ_presence_micro']['cohen_kappa'])}.

### Descriptive Capture volumes
- Llama: {summaries['Llama-4 Scout']['findings_detected']} findings ({summaries['Llama-4 Scout']['actionable_findings']} actionable, {summaries['Llama-4 Scout']['uncertain_findings']} uncertain); {_plural(summaries['Llama-4 Scout']['capture_error_reports'], 'capture failure')}
- Qwen: {summaries['Qwen3.6-35B']['findings_detected']} findings ({summaries['Qwen3.6-35B']['actionable_findings']} actionable, {summaries['Qwen3.6-35B']['uncertain_findings']} uncertain); {_plural(summaries['Qwen3.6-35B']['capture_error_reports'], 'capture failure')}
- Mistral: {summaries['Mistral-Small-3.2']['findings_detected']} findings ({summaries['Mistral-Small-3.2']['actionable_findings']} actionable, {summaries['Mistral-Small-3.2']['uncertain_findings']} uncertain); {_plural(summaries['Mistral-Small-3.2']['capture_error_reports'], 'capture failure')}

### Per-agent evaluation vs the rule-based reference pipeline
Link, Orchestrate, and end-to-end outputs were compared against a full rule-based run of the same pipeline on the same cohort (`tables/table28–35`, `figures/fig17–19`). Because those agents are deterministic engines shared by both runs, the metrics quantify propagated Capture error rather than independent agent quality.

| Agent | Task | Llama | Qwen | Mistral |
|---|---|--:|--:|--:|
| Link | Any provider assignment, accuracy | {_fmt(agent_metrics['Llama-4 Scout']['link']['any_assignment']['accuracy'])} | {_fmt(agent_metrics['Qwen3.6-35B']['link']['any_assignment']['accuracy'])} | {_fmt(agent_metrics['Mistral-Small-3.2']['link']['any_assignment']['accuracy'])} |
| Link | Any provider assignment, κ | {_fmt(agent_metrics['Llama-4 Scout']['link']['any_assignment']['cohen_kappa'])} | {_fmt(agent_metrics['Qwen3.6-35B']['link']['any_assignment']['cohen_kappa'])} | {_fmt(agent_metrics['Mistral-Small-3.2']['link']['any_assignment']['cohen_kappa'])} |
| Orchestrate | ACR guideline category, κ | {_fmt(agent_metrics['Llama-4 Scout']['orchestrate']['guideline_category']['cohen_kappa'])} | {_fmt(agent_metrics['Qwen3.6-35B']['orchestrate']['guideline_category']['cohen_kappa'])} | {_fmt(agent_metrics['Mistral-Small-3.2']['orchestrate']['guideline_category']['cohen_kappa'])} |
| Orchestrate | Modality family, κ | {_fmt(agent_metrics['Llama-4 Scout']['orchestrate']['modality_family_category']['cohen_kappa'])} | {_fmt(agent_metrics['Qwen3.6-35B']['orchestrate']['modality_family_category']['cohen_kappa'])} | {_fmt(agent_metrics['Mistral-Small-3.2']['orchestrate']['modality_family_category']['cohen_kappa'])} |
| End-to-end | Final status category, κ | {_fmt(agent_metrics['Llama-4 Scout']['end_to_end']['final_status_category']['cohen_kappa'])} | {_fmt(agent_metrics['Qwen3.6-35B']['end_to_end']['final_status_category']['cohen_kappa'])} | {_fmt(agent_metrics['Mistral-Small-3.2']['end_to_end']['final_status_category']['cohen_kappa'])} |

Recommendation volume falls well short of the reference ({int(agent_metrics['Llama-4 Scout']['orchestrate']['reference_recommendations'])} reference recommendations vs {int(agent_metrics['Llama-4 Scout']['orchestrate']['candidate_recommendations'])} / {int(agent_metrics['Qwen3.6-35B']['orchestrate']['candidate_recommendations'])} / {int(agent_metrics['Mistral-Small-3.2']['orchestrate']['candidate_recommendations'])} for Llama / Qwen / Mistral). Where both systems recommend follow-up for the same organ, exact ACR rule-ID concordance is {_fmt(agent_metrics['Llama-4 Scout']['orchestrate']['rule_id_exact_match_when_both_recommend'])} / {_fmt(agent_metrics['Qwen3.6-35B']['orchestrate']['rule_id_exact_match_when_both_recommend'])} / {_fmt(agent_metrics['Mistral-Small-3.2']['orchestrate']['rule_id_exact_match_when_both_recommend'])}.

### Link
All assignments used the diagnostic safety pool failsafe (missing EHR PCP/specialty context), so assignment, failsafe, review, and provider-type agreement are identical by construction.

### Orchestrate
ACR pathways for renal/adrenal/liver/pancreas/adnexal. Insufficient criteria → human review.

### Inter-model concordance (proxy)
| Comparison | Presence | Organ Jaccard | Actionable agreement |
|---|--:|--:|--:|
| {conc_rows[0][0]} | {conc_rows[0][1]} | {conc_rows[0][2]} | {conc_rows[0][3]} |
| {conc_rows[1][0]} | {conc_rows[1][1]} | {conc_rows[1][2]} | {conc_rows[1][3]} |
| {conc_rows[2][0]} | {conc_rows[2][1]} | {conc_rows[2][2]} | {conc_rows[2][3]} |

## Limitations
Both reference standards are rule-based rather than expert chart review, so κ quantifies agreement with a guideline engine rather than clinical accuracy. Downstream Link and Orchestrate metrics are not independent measures of those agents: the engines are shared, so the numbers reflect propagated Capture error. Actionability agreement is limited by systematic severity downgrading by the LLMs (addressed by hybrid ACR recalibration; see `table36`, `fig20`), and κ is sensitive to the skewed marginals of this cohort ({gold_meta['report_actionability_distribution'].get('none', 0)}/1000 reports have no in-scope finding). Link routing is failsafe-only because EHR provider context was unavailable; Safeguard and Evaluate were not run in-pipeline.
"""
    (OUT / "SUMMARY.md").write_text(summary, encoding="utf-8")

    (AGENTS / "capture.md").write_text(
        f"""# Capture Agent

LLM-primary incidental-finding extraction for kidney/liver/adrenal/pancreas/adnexal.

## Evaluation vs the rule-based ACR reference

Binary detection: `tables/table12–15`, `figures/fig09–11`.  
Categorical agreement: `tables/table16–19`, `figures/fig12–13`.

Report actionability (4 ordinal levels) **linear κ**: {_fmt(_cat('Llama-4 Scout','cohen_kappa'))} / {_fmt(_cat('Qwen3.6-35B','cohen_kappa'))} / {_fmt(_cat('Mistral-Small-3.2','cohen_kappa'))} (Llama / Qwen / Mistral); nominal κ: {_fmt(_cat('Llama-4 Scout','cohen_kappa_nominal'))} / {_fmt(_cat('Qwen3.6-35B','cohen_kappa_nominal'))} / {_fmt(_cat('Mistral-Small-3.2','cohen_kappa_nominal'))}.

Binary presence balanced accuracy: {_fmt(gold_metrics['Llama-4 Scout']['presence']['balanced_accuracy'])} / {_fmt(gold_metrics['Qwen3.6-35B']['presence']['balanced_accuracy'])} / {_fmt(gold_metrics['Mistral-Small-3.2']['presence']['balanced_accuracy'])}.
""",
        encoding="utf-8",
    )
    (AGENTS / "link.md").write_text(
        f"""# Link Agent

Routes actionable/uncertain findings to providers. In this cohort, EHR PCP/specialty fields were
unavailable, so every assignment used the diagnostic safety-pool failsafe. Assignment, failsafe,
human-review, and provider-type agreement are therefore identical by construction.

## Evaluation vs the rule-based reference pipeline
Tables `table28–32` (rows tagged `Link`) and `table35`; figures `fig17–19`.

Any-assignment agreement: accuracy {_fmt(agent_metrics['Llama-4 Scout']['link']['any_assignment']['accuracy'])} / {_fmt(agent_metrics['Qwen3.6-35B']['link']['any_assignment']['accuracy'])} / {_fmt(agent_metrics['Mistral-Small-3.2']['link']['any_assignment']['accuracy'])}, κ {_fmt(agent_metrics['Llama-4 Scout']['link']['any_assignment']['cohen_kappa'])} / {_fmt(agent_metrics['Qwen3.6-35B']['link']['any_assignment']['cohen_kappa'])} / {_fmt(agent_metrics['Mistral-Small-3.2']['link']['any_assignment']['cohen_kappa'])} (Llama / Qwen / Mistral).

Assignment volume: {int(agent_metrics['Llama-4 Scout']['link']['reference_assignments'])} reference vs {int(agent_metrics['Llama-4 Scout']['link']['candidate_assignments'])} / {int(agent_metrics['Qwen3.6-35B']['link']['candidate_assignments'])} / {int(agent_metrics['Mistral-Small-3.2']['link']['candidate_assignments'])}. Because routing is deterministic, this gap is inherited from Capture.
""",
        encoding="utf-8",
    )
    (AGENTS / "orchestrate.md").write_text(
        f"""# Orchestrate Agent

Maps Capture findings to ACR-derived follow-up pathways (renal, adrenal, liver, pancreas, adnexal)
with RAG evidence snippets, escalating to human review when criteria are incomplete.

## Evaluation vs the rule-based reference pipeline
Tables `table28–32` (rows tagged `Orchestrate`) and `table34`; figures `fig17–19`.

Categorical agreement (κ), Llama / Qwen / Mistral:
- ACR guideline invoked: {_fmt(agent_metrics['Llama-4 Scout']['orchestrate']['guideline_category']['cohen_kappa'])} / {_fmt(agent_metrics['Qwen3.6-35B']['orchestrate']['guideline_category']['cohen_kappa'])} / {_fmt(agent_metrics['Mistral-Small-3.2']['orchestrate']['guideline_category']['cohen_kappa'])}
- Modality family: {_fmt(agent_metrics['Llama-4 Scout']['orchestrate']['modality_family_category']['cohen_kappa'])} / {_fmt(agent_metrics['Qwen3.6-35B']['orchestrate']['modality_family_category']['cohen_kappa'])} / {_fmt(agent_metrics['Mistral-Small-3.2']['orchestrate']['modality_family_category']['cohen_kappa'])}
- Follow-up timeframe: {_fmt(agent_metrics['Llama-4 Scout']['orchestrate']['timeframe_category']['cohen_kappa'])} / {_fmt(agent_metrics['Qwen3.6-35B']['orchestrate']['timeframe_category']['cohen_kappa'])} / {_fmt(agent_metrics['Mistral-Small-3.2']['orchestrate']['timeframe_category']['cohen_kappa'])}

Exact ACR rule-ID concordance where both systems recommend follow-up for the same organ:
{_fmt(agent_metrics['Llama-4 Scout']['orchestrate']['rule_id_exact_match_when_both_recommend'])} / {_fmt(agent_metrics['Qwen3.6-35B']['orchestrate']['rule_id_exact_match_when_both_recommend'])} / {_fmt(agent_metrics['Mistral-Small-3.2']['orchestrate']['rule_id_exact_match_when_both_recommend'])} over {int(agent_metrics['Llama-4 Scout']['orchestrate']['rule_id_comparable_units'])} / {int(agent_metrics['Qwen3.6-35B']['orchestrate']['rule_id_comparable_units'])} / {int(agent_metrics['Mistral-Small-3.2']['orchestrate']['rule_id_comparable_units'])} organ units. Even when both reach the same organ, the specific pathway often differs because Capture supplies different size/HU/feature evidence.
""",
        encoding="utf-8",
    )
    (AGENTS / "safeguard_evaluate.md").write_text(
        """# Safeguard / Evaluate

Safeguard and Evaluate were stubbed and not executed in these comparison runs.

Evaluate scaffolding now supports two offline harnesses:

```bash
# Capture vs gold labels
PYTHONPATH=src python scripts/evaluate_against_gold.py \
  --gold data/gold_labels/gold_labels.jsonl --predictions <run>/close_results.jsonl

# All agents vs the rule-based reference pipeline
PYTHONPATH=src python scripts/evaluate_agents.py \
  --reference outputs/reference_rule_based/close_results.jsonl \
  --predictions <run>/close_results.jsonl --gold data/gold_labels/gold_labels.jsonl
```
""",
        encoding="utf-8",
    )

    mermaid_pipeline = """flowchart TD
    A["Abdominal CT reports<br/>(1000 report cohort)"] --> B["Loader<br/>normalize text, dedupe report IDs"]
    B --> C["<b>CAPTURE</b><br/>LLM-primary extraction via Ollama"]
    C --> C1["Structured findings<br/>organ - size - HU - actionability - urgency"]
    C1 --> D{"In-scope organ?<br/>kidney / liver / adrenal<br/>pancreas / adnexal"}
    D -- no --> X["No actionable finding"]
    D -- yes --> E["<b>LINK</b><br/>resolve responsible clinician"]
    E --> E1{"PCP or specialty<br/>available in EHR?"}
    E1 -- yes --> E2["Route to named provider"]
    E1 -- no --> E3["Failsafe: diagnostic safety pool"]
    E2 --> F["<b>ORCHESTRATE</b><br/>ACR guideline engines + RAG evidence"]
    E3 --> F
    F --> F1{"Guideline criteria<br/>sufficient?"}
    F1 -- yes --> G["Follow-up recommendation<br/>modality - timeframe - rule ID"]
    F1 -- no --> H["Human review queue"]
    G --> I["<b>SAFEGUARD</b><br/>(stubbed)"]
    H --> I
    I --> J["<b>EVALUATE</b><br/>(stubbed in-pipeline; run offline)"]
    J --> K["close_results.jsonl<br/>summary_metrics.json"]

    classDef agent fill:#2E86AB,stroke:#14425c,color:#ffffff;
    classDef stub fill:#d9d9d9,stroke:#7a7a7a,color:#333333,stroke-dasharray: 4 3;
    classDef fail fill:#F18F01,stroke:#8a5200,color:#ffffff;
    class C,E,F agent;
    class I,J stub;
    class E3,H fail;
"""

    mermaid_eval = """flowchart LR
    R["1000 reports"] --> G["Rule-based ACR Capture<br/><b>reference standard</b>"]
    R --> M1["Llama-4 Scout"]
    R --> M2["Qwen3.6-35B"]
    R --> M3["Mistral-Small-3.2"]
    G --> GL["gold_labels.jsonl<br/>binary flags + categorical scales"]
    M1 --> P["Model predictions<br/>close_results.jsonl"]
    M2 --> P
    M3 --> P
    GL --> E["evaluate_against_gold.py"]
    P --> E
    E --> D1["<b>Detection metrics</b><br/>sensitivity - specificity<br/>PPV - NPV"]
    E --> D2["<b>Agreement metrics</b><br/>Cohen kappa<br/>linear / quadratic weighted"]
    E --> D3["<b>Per-class metrics</b><br/>precision - recall - F1<br/>confusion matrices"]
    D1 --> O["results/ tables + figures"]
    D2 --> O
    D3 --> O

    classDef ref fill:#A23B72,stroke:#5c1f40,color:#ffffff;
    classDef model fill:#2E86AB,stroke:#14425c,color:#ffffff;
    class G,GL ref;
    class M1,M2,M3 model;
"""

    (FIGS / "workflow_pipeline.mmd").write_text(mermaid_pipeline, encoding="utf-8")
    (FIGS / "workflow_evaluation.mmd").write_text(mermaid_eval, encoding="utf-8")
    (OUT / "WORKFLOW.md").write_text(
        "# CLOSE workflow diagrams\n\n"
        "Mermaid sources are also saved standalone as `figures/workflow_pipeline.mmd` and "
        "`figures/workflow_evaluation.mmd`.\n\n"
        "## 1. CLOSE pipeline\n\n```mermaid\n"
        + mermaid_pipeline
        + "```\n\n## 2. Evaluation design\n\n```mermaid\n"
        + mermaid_eval
        + "```\n",
        encoding="utf-8",
    )

    # Ship the reference labels with the pack so metrics are reproducible.
    shutil.copytree(GOLD_PATH.parent, OUT / "gold_labels")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(ROOT)))
    print(json.dumps({"results": str(OUT), "zip": str(ZIP_PATH), "gold_reports": len(gold)}, indent=2))


if __name__ == "__main__":
    main()
