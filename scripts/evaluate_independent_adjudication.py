#!/usr/bin/env python
"""Evaluate CLOSE Capture arms against independent (non-rule-engine) adjudication labels.

Example:
  PYTHONPATH=src python scripts/evaluate_independent_adjudication.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.evaluation.bootstrap import bootstrap_binary_metrics, bootstrap_linear_kappa  # noqa: E402
from close_framework.evaluation.gold_eval import (  # noqa: E402
    ACTIONABILITY_SCALE,
    IN_SCOPE,
    has_actionable_prediction,
    load_jsonl_by_report_id,
    organs_from_prediction,
    prediction_findings,
    report_actionability_category,
)
from close_framework.evaluation.metrics import binary_metrics, multiclass_agreement_metrics  # noqa: E402

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


def _load_labels(path: Path) -> dict:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows[str(payload["report_id"])] = payload
    return rows


def _fmt_ci(ci: dict | None, digits: int = 3) -> str:
    if not ci or ci.get("estimate") is None:
        return ""
    low, high = ci.get("ci_low"), ci.get("ci_high")
    if low is None or high is None:
        return f"{ci['estimate']:.{digits}f}"
    return f"{ci['estimate']:.{digits}f} ({low:.{digits}f}–{high:.{digits}f})"


def evaluate_against_independent(labels: dict, predictions: dict, model_name: str, n_boot: int = 400) -> dict:
    ids = sorted(set(labels) & set(predictions))
    y_pres_t = [bool(labels[i].get("finding_present")) for i in ids]
    y_pres_p = [bool(prediction_findings(predictions[i])) for i in ids]
    y_act_t = [bool(labels[i].get("followup_required")) or labels[i].get("actionability") == "actionable" for i in ids]
    y_act_p = [has_actionable_prediction(predictions[i]) for i in ids]
    cat_t = []
    cat_p = []
    for i in ids:
        act = labels[i].get("actionability") or "none"
        if act not in ACTIONABILITY_SCALE:
            act = "uncertain" if labels[i].get("finding_present") else "none"
        cat_t.append(act)
        cat_p.append(report_actionability_category(prediction_findings(predictions[i])))

    organ_metrics = {}
    for organ in IN_SCOPE:
        yt = [organ in (labels[i].get("organs_present") or []) for i in ids]
        yp = [organ in organs_from_prediction(predictions[i]) for i in ids]
        organ_metrics[organ] = binary_metrics(yt, yp)

    return {
        "model": model_name,
        "n": len(ids),
        "presence": {
            "metrics": binary_metrics(y_pres_t, y_pres_p),
            "bootstrap": bootstrap_binary_metrics(y_pres_t, y_pres_p, n_boot=n_boot, seed=0),
        },
        "actionable_or_followup": {
            "metrics": binary_metrics(y_act_t, y_act_p),
            "bootstrap": bootstrap_binary_metrics(y_act_t, y_act_p, n_boot=n_boot, seed=1),
        },
        "actionability": {
            "metrics": multiclass_agreement_metrics(
                cat_t, cat_p, labels=list(ACTIONABILITY_SCALE), ordinal=True
            ),
            "bootstrap_linear_kappa": bootstrap_linear_kappa(
                cat_t, cat_p, ACTIONABILITY_SCALE, n_boot=n_boot, seed=2
            ),
        },
        "organ_presence": organ_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "paper_results/independent_adjudication/independent_labels.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "paper_results",
    )
    args = parser.parse_args()

    labels = _load_labels(args.labels)
    if not labels:
        raise SystemExit(f"No labels found at {args.labels}")

    # Agreement of independent labels vs ACR-derived reference (diagnostic only)
    gold = load_jsonl_by_report_id(ROOT / "data/gold_labels/gold_labels.jsonl")
    shared = sorted(set(labels) & set(gold))
    ref_pres = [bool(gold[i].get("has_in_scope_finding")) for i in shared]
    ind_pres = [bool(labels[i].get("finding_present")) for i in shared]
    ref_act = [str(gold[i].get("report_actionability_category") or "none") for i in shared]
    ind_act = []
    for i in shared:
        act = labels[i].get("actionability") or "none"
        if act not in ACTIONABILITY_SCALE:
            act = "uncertain" if labels[i].get("finding_present") else "none"
        ind_act.append(act)
    reference_concordance = {
        "n": len(shared),
        "presence": binary_metrics(ref_pres, ind_pres),
        "actionability": multiclass_agreement_metrics(
            ref_act, ind_act, labels=list(ACTIONABILITY_SCALE), ordinal=True
        ),
        "note": (
            "Concordance between independent adjudication and ACR-derived reference. "
            "Not used as the evaluation target for model scores below."
        ),
    }

    results = {"reference_concordance": reference_concordance, "models": {}}
    table_rows = []
    for model, arms in MODELS.items():
        results["models"][model] = {}
        for arm, path in arms.items():
            preds = load_jsonl_by_report_id(path)
            block = evaluate_against_independent(labels, preds, f"{model}/{arm}")
            results["models"][model][arm] = block
            table_rows.append(
                [
                    model,
                    arm,
                    block["n"],
                    _fmt_ci(block["presence"]["bootstrap"]["balanced_accuracy"]),
                    _fmt_ci(block["presence"]["bootstrap"]["cohen_kappa"]),
                    _fmt_ci(block["actionable_or_followup"]["bootstrap"]["balanced_accuracy"]),
                    _fmt_ci(block["actionable_or_followup"]["bootstrap"]["cohen_kappa"]),
                    _fmt_ci(block["actionability"]["bootstrap_linear_kappa"]),
                ]
            )

    out_dir = args.out_dir
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (out_dir / "independent_adjudication" / "metrics_vs_independent.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    header = [
        "model",
        "configuration",
        "n",
        "presence_balanced_acc_ci",
        "presence_kappa_ci",
        "followup_balanced_acc_ci",
        "followup_kappa_ci",
        "actionability_linear_kappa_ci",
    ]
    body = "\n".join(",".join(str(c) for c in row) for row in table_rows)
    (tables / "table16_independent_adjudication.csv").write_text(
        ",".join(header) + "\n" + body + "\n", encoding="utf-8"
    )

    # Short markdown summary
    lines = [
        "# Independent adjudication evaluation",
        "",
        f"N={len(labels)} reports labeled independently from report text excerpts "
        "(not using the ACR rule engine as the label source).",
        "",
        "## Concordance of independent labels vs ACR-derived reference",
        f"- Presence κ: {reference_concordance['presence']['cohen_kappa']:.3f}",
        f"- Actionability linear κ: {reference_concordance['actionability']['cohen_kappa']:.3f}",
        "",
        "## Model agreement vs independent labels",
        "",
        "| Model | Config | Presence κ | Follow-up κ | Actionability linear κ |",
        "|---|---|--:|--:|--:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[4]} | {row[6]} | {row[7]} |"
        )
    lines.append("")
    lines.append(
        "**Caveat:** Labels are AI-agent adjudication from report excerpts, independent of "
        "the deterministic Capture code path, but **not** radiologist chart review."
    )
    (out_dir / "independent_adjudication" / "INDEPENDENT_EVAL.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_labels": len(labels), "table": str(tables / "table16_independent_adjudication.csv")}, indent=2))


if __name__ == "__main__":
    main()
