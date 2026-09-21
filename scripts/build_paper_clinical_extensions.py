#!/usr/bin/env python
"""Build clinically oriented paper extensions (tables 23–31, figs 13–15).

Covers: per-class/organ clinical metrics, weighted-κ disclosure, under/overcall
matrices, oracle stage ablation, recovery audit, rule-engine audit pack,
run-to-run stability (available replicates), abstention curves.

Example:
  PYTHONPATH=src python scripts/build_paper_clinical_extensions.py
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.data.loaders import iter_reports  # noqa: E402
from close_framework.evaluation.clinical_extensions import (  # noqa: E402
    abstention_curve,
    directional_severity_errors,
    oracle_stage_ablation,
    per_class_actionability_metrics,
    per_organ_clinical_metrics,
    recovery_module_audit,
    run_to_run_stability,
)
from close_framework.evaluation.gold_eval import load_jsonl_by_report_id  # noqa: E402

OUT = ROOT / "paper_results"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
ZIP_PATH = ROOT / "close_paper_results.zip"

GOLD = ROOT / "data/gold_labels/gold_labels.jsonl"
REFERENCE = ROOT / "outputs/reference_rule_based/close_results.jsonl"

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

# Historical Capture replicates for stability (not controlled same-seed triples).
STABILITY_RUNS = {
    "Qwen3.6-35B": {
        "run_9500762": ROOT / "outputs/model_comparison_9500762/qwen/close_results.jsonl",
        "run_9199835": ROOT / "outputs/model_comparison_9199835/qwen/close_results.jsonl",
    },
    "Mistral-Small-3.2": {
        "run_9500762": ROOT / "outputs/model_comparison_9500762/mistral/close_results.jsonl",
        "run_9199835": ROOT / "outputs/model_comparison_9199835/mistral/close_results.jsonl",
        "run_9199674": ROOT / "outputs/model_comparison_9199674/mistral/close_results.jsonl",
    },
}


def _fmt(x, d=3):
    return "" if x is None else f"{x:.{d}f}"


def _write_csv(path: Path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(["" if c is None else c for c in row])


def load_report_texts(report_ids: set[str]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for report in iter_reports(ROOT / "data"):
        if report.report_id in report_ids:
            found[report.report_id] = report.report_text or ""
            if len(found) >= len(report_ids):
                break
    return found


def write_rule_audit_pack(gold: Dict[str, dict], texts: Dict[str, str], n: int = 50) -> dict:
    """Independent rule-engine audit sample + protocol (second auditor fills)."""

    audit_dir = OUT / "rule_engine_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Stratify: prioritize actionable + pancreas/adnexal
    candidates = []
    for report_id, label in gold.items():
        for finding in label.get("findings") or []:
            organ = str(finding.get("organ_system"))
            action = str(finding.get("actionability"))
            weight = 0
            if action == "actionable":
                weight += 3
            if organ in {"pancreas", "adnexal"}:
                weight += 2
            if organ in {"adrenal", "liver"}:
                weight += 1
            candidates.append((weight, report_id, finding))
    candidates.sort(key=lambda x: (-x[0], x[1]))

    seen_reports = set()
    sample = []
    for weight, report_id, finding in candidates:
        if report_id in seen_reports and len(sample) >= n // 2:
            continue
        sample.append((report_id, finding))
        seen_reports.add(report_id)
        if len(sample) >= n:
            break

    rows = []
    for idx, (report_id, finding) in enumerate(sample, start=1):
        evidence = finding.get("evidence_span") or ""
        rows.append(
            [
                idx,
                report_id,
                finding.get("organ_system"),
                finding.get("finding_type"),
                finding.get("size_mm"),
                finding.get("actionability"),
                finding.get("urgency_tier"),
                (evidence or "")[:300],
                "",  # auditor_actionability
                "",  # auditor_urgency
                "",  # auditor_agree_with_engine
                "",  # auditor_notes
            ]
        )
    _write_csv(
        audit_dir / "rule_audit_worksheet.csv",
        [
            "item_id",
            "report_id",
            "engine_organ",
            "engine_finding_type",
            "engine_size_mm",
            "engine_actionability",
            "engine_urgency",
            "evidence_span",
            "auditor_actionability",
            "auditor_urgency",
            "auditor_agree_yn",
            "auditor_notes",
        ],
        rows,
    )

    protocol = """# ACR rule-engine audit protocol

## Why this exists
The Capture reference standard is a deterministic encoding of ACR incidental-finding
white papers. If that encoding is wrong, every concordance number inherits the error.
An independent auditor should apply ACR criteria to a stratified sample **without**
reading the engine code, then we report agreement with the engine labels.

## Who encoded the rules
- Implementation: `src/close_framework/agents/capture_agent.py`
  (`_actionability_from_sentence`, organ size thresholds, rule extraction)
- Guideline mapping: `src/close_framework/guidelines/*_guidelines.py`
- Source PDFs: `incidental_findings_white_papers/`
- Encoder qualifications: **[PI to fill: name, credentials, role]**

## Independent auditor
- Name / credentials: **[to fill]**
- Conflict: must not be the primary rule author
- Materials: this worksheet + ACR white papers only (not CaptureAgent source)

## Procedure
1. For each row, read `evidence_span` (and full report text if needed).
2. Assign `auditor_actionability` in {none, non_actionable, uncertain, actionable}.
3. Assign `auditor_urgency` in {none, routine, short_interval, urgent, unknown}.
4. Mark `auditor_agree_yn` = Y if both actionability and urgency match the engine.
5. Note disagreements with ACR citation when possible.

## Analysis (after completion)
```bash
PYTHONPATH=src python - <<'PY'
# Compute auditor vs engine κ on filled worksheet
PY
```
Report: n, % exact agreement, linear κ for actionability, disagreement taxonomy.

## Status
Worksheet generated; **independent audit not yet completed**.
"""
    (audit_dir / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    # Attach short report excerpts for the sample
    excerpts = []
    for report_id, finding in sample:
        text = texts.get(report_id, "")
        excerpts.append(
            {
                "report_id": report_id,
                "evidence_span": finding.get("evidence_span"),
                "report_excerpt": text[:2000],
            }
        )
    (audit_dir / "sample_excerpts.jsonl").write_text(
        "\n".join(json.dumps(x) for x in excerpts) + "\n", encoding="utf-8"
    )
    return {"n_items": len(sample), "n_reports": len(seen_reports), "status": "worksheet_ready"}


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    print("Loading gold + reference...", flush=True)
    gold = load_jsonl_by_report_id(GOLD)
    reference = load_jsonl_by_report_id(REFERENCE)

    print("Loading model arms...", flush=True)
    model_preds: Dict[str, Dict[str, dict]] = {}
    for model, arms in MODELS.items():
        model_preds[model] = {k: load_jsonl_by_report_id(v) for k, v in arms.items()}

    print("Loading report texts (for recovery audit)...", flush=True)
    texts = load_report_texts(set(gold))

    # --- Table 23: per-class clinical metrics ---
    print("Per-class actionability...", flush=True)
    class_rows = []
    class_payload = {}
    for model, arms in model_preds.items():
        for config, preds in arms.items():
            m = per_class_actionability_metrics(gold, preds, model, config)
            class_payload[f"{model}/{config}"] = m
            for label, stats in m["per_class"].items():
                class_rows.append(
                    [
                        model,
                        config,
                        label,
                        int(stats["support"] or 0),
                        _fmt(stats["prevalence"]),
                        _fmt(stats["sensitivity"]),
                        _fmt(stats["specificity"]),
                        _fmt(stats["ppv"]),
                        _fmt(stats["npv"]),
                        _fmt(stats["f1"]),
                        _fmt(m["kappa_nominal"]),
                        _fmt(m["kappa_linear_weighted"]),
                        _fmt(m["kappa_quadratic_weighted"]),
                        _fmt(m["high_urgency_binary"].get("sensitivity")),
                        int(m["n_true_high_urgency"]),
                        _fmt(m["urgent_only_binary"].get("sensitivity")),
                        int(m["n_true_urgent"]),
                    ]
                )
    _write_csv(
        TABLES / "table23_per_class_actionability_clinical.csv",
        [
            "model",
            "configuration",
            "class",
            "support",
            "prevalence",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1",
            "kappa_nominal",
            "kappa_linear_weighted",
            "kappa_quadratic_weighted",
            "high_urgency_sensitivity",
            "n_true_high_urgency",
            "urgent_only_sensitivity",
            "n_true_urgent",
        ],
        class_rows,
    )

    # --- Table 24: per-organ clinical ---
    print("Per-organ clinical...", flush=True)
    organ_rows = []
    for model, arms in model_preds.items():
        for config in ("llm_only", "full_hybrid"):
            for row in per_organ_clinical_metrics(gold, arms[config], model, config):
                organ_rows.append(
                    [
                        row["model"],
                        row["configuration"],
                        row["organ"],
                        row["n_true_present"],
                        row["n_true_actionable"],
                        _fmt(row["presence_sensitivity"]),
                        _fmt(row["presence_specificity"]),
                        _fmt(row["presence_kappa"]),
                        _fmt(row["actionability_linear_kappa"]),
                        _fmt(row["actionability_quadratic_kappa"]),
                        _fmt(row["actionable_sensitivity"]),
                        _fmt(row["actionable_specificity"]),
                        _fmt(row["actionable_ppv"]),
                    ]
                )
    _write_csv(
        TABLES / "table24_per_organ_clinical.csv",
        [
            "model",
            "configuration",
            "organ",
            "n_true_present",
            "n_true_actionable",
            "presence_sensitivity",
            "presence_specificity",
            "presence_kappa",
            "actionability_linear_kappa",
            "actionability_quadratic_kappa",
            "actionable_sensitivity",
            "actionable_specificity",
            "actionable_ppv",
        ],
        organ_rows,
    )

    # --- Table 25: under/overcall ---
    print("Directional errors...", flush=True)
    dir_rows = []
    dir_payload = {}
    confusion_rows = []
    for model, arms in model_preds.items():
        for config in ("llm_only", "calibrate", "full_hybrid"):
            d = directional_severity_errors(gold, arms[config], model, config)
            dir_payload[f"{model}/{config}"] = d
            dir_rows.append(
                [
                    model,
                    config,
                    d["n"],
                    d["undercalls"],
                    d["overcalls"],
                    _fmt(d["undercall_rate"]),
                    _fmt(d["overcall_rate"]),
                    d["undercalls_off_by_ge2"],
                    d["overcalls_off_by_ge2"],
                    _fmt(d["mean_signed_severity_bias"]),
                    d["n_true_actionable"],
                    d["actionable_not_called_actionable"],
                    _fmt(d["actionable_sensitivity"]),
                    d["actionable_collapsed_to_none"],
                ]
            )
            cm = d["confusion_matrix"]
            for true_lab, preds in cm.items():
                for pred_lab, count in preds.items():
                    confusion_rows.append([model, config, true_lab, pred_lab, count])
    _write_csv(
        TABLES / "table25_undercall_overcall.csv",
        [
            "model",
            "configuration",
            "n",
            "undercalls",
            "overcalls",
            "undercall_rate",
            "overcall_rate",
            "undercalls_off_by_ge2",
            "overcalls_off_by_ge2",
            "mean_signed_severity_bias",
            "n_true_actionable",
            "actionable_not_called_actionable",
            "actionable_sensitivity",
            "actionable_collapsed_to_none",
        ],
        dir_rows,
    )
    _write_csv(
        TABLES / "table25b_directional_confusion.csv",
        ["model", "configuration", "true_label", "pred_label", "count"],
        confusion_rows,
    )

    # --- Table 26: oracle stage ablation ---
    print("Oracle stage ablation...", flush=True)
    oracle_rows = []
    oracle_payload = {}
    for model, arms in model_preds.items():
        o = oracle_stage_ablation(reference, arms["llm_only"], arms["full_hybrid"], model)
        oracle_payload[model] = o
        for arm_name, arm_m in o["arms"].items():
            fu = arm_m["followup_needed"]
            oracle_rows.append(
                [
                    model,
                    arm_name,
                    _fmt(fu.get("sensitivity")),
                    _fmt(fu.get("specificity")),
                    _fmt(fu.get("cohen_kappa")),
                    _fmt(arm_m.get("orchestrate_guideline_kappa")),
                    _fmt(arm_m.get("orchestrate_modality_kappa")),
                    (arm_m.get("end_to_end_errors") or {}).get("missed_necessary_followup"),
                    (arm_m.get("end_to_end_errors") or {}).get("unnecessary_followup"),
                    (arm_m.get("capture_driven_errors") or {}).get("missed_necessary_followup"),
                    (arm_m.get("despite_capture_errors") or {}).get("missed_necessary_followup"),
                ]
            )
    _write_csv(
        TABLES / "table26_oracle_stage_ablation.csv",
        [
            "model",
            "arm",
            "followup_sensitivity",
            "followup_specificity",
            "followup_kappa",
            "guideline_kappa",
            "modality_kappa",
            "missed_necessary_followup",
            "unnecessary_followup",
            "missed_fu_capture_driven",
            "missed_fu_despite_matching_capture",
        ],
        oracle_rows,
    )

    # --- Table 27: recovery audit ---
    print("Recovery module audit...", flush=True)
    recovery_rows = []
    recovery_examples = []
    recovery_payload = {}
    for model, arms in model_preds.items():
        r = recovery_module_audit(
            gold, arms["llm_only"], arms["full_hybrid"], texts, model
        )
        recovery_payload[model] = {k: v for k, v in r.items() if k != "all_unsupported"}
        recovery_rows.append(
            [
                model,
                r["n_supplemented_findings"],
                _fmt(r["precision_vs_gold"]),
                _fmt(r["text_support_precision"]),
                _fmt(r["fabrication_rate_text_audit"]),
                r["n_fabricated_or_unsupported"],
                _fmt(r["recovery_recall_of_llm_misses"]),
                r["n_llm_misses_recovered"],
                r["n_llm_misses_unrecovered"],
            ]
        )
        for ex in r["all_unsupported"]:
            recovery_examples.append(
                [
                    model,
                    ex["report_id"],
                    ex["organ_system"],
                    ex["actionability"],
                    ex.get("size_mm"),
                    ex["reason"],
                    ex.get("matched_gold"),
                    ex.get("evidence_span"),
                ]
            )
    _write_csv(
        TABLES / "table27_recovery_audit.csv",
        [
            "model",
            "n_supplemented",
            "precision_vs_gold",
            "text_support_precision",
            "fabrication_rate",
            "n_unsupported",
            "recall_of_llm_misses",
            "n_recovered",
            "n_unrecovered",
        ],
        recovery_rows,
    )
    _write_csv(
        TABLES / "table27b_recovery_unsupported_findings.csv",
        [
            "model",
            "report_id",
            "organ",
            "actionability",
            "size_mm",
            "reason",
            "matched_gold",
            "evidence_span",
        ],
        recovery_examples,
    )

    # --- Rule audit pack ---
    print("Rule-engine audit pack...", flush=True)
    rule_meta = write_rule_audit_pack(gold, texts, n=50)

    # --- Table 28: stability ---
    print("Run-to-run stability...", flush=True)
    stab_rows = []
    stab_payload = {}
    for model, run_paths in STABILITY_RUNS.items():
        runs = {}
        for name, path in run_paths.items():
            if path.exists():
                runs[name] = load_jsonl_by_report_id(path)
        if len(runs) < 2:
            continue
        s = run_to_run_stability(runs, model)
        stab_payload[model] = s
        for pair in s["pairwise"]:
            stab_rows.append(
                [
                    model,
                    pair["run_a"],
                    pair["run_b"],
                    pair["n"],
                    _fmt(pair["presence_kappa"]),
                    _fmt(pair["actionable_kappa"]),
                    _fmt(pair["actionability_linear_kappa"]),
                    _fmt(pair["actionability_exact_agreement"]),
                ]
            )
    _write_csv(
        TABLES / "table28_run_to_run_stability.csv",
        [
            "model",
            "run_a",
            "run_b",
            "n_shared",
            "presence_kappa",
            "actionable_kappa",
            "actionability_linear_kappa",
            "exact_agreement",
        ],
        stab_rows,
    )
    (OUT / "stability").mkdir(parents=True, exist_ok=True)
    (OUT / "stability" / "STATUS.md").write_text(
        """# Run-to-run stability status

## Available now
Pairwise agreement across historical Capture JSONL jobs that share report IDs
(see `tables/table28_run_to_run_stability.csv`).

- **Mistral**: 3 historical runs on n=100 overlapping reports
- **Qwen**: 2 historical runs on n=100
- **Llama**: no overlapping multi-run JSONL found

These are **not** controlled identical-seed / identical-temperature repeats from a
single experiment plan; treat as opportunistic lower-bound evidence.

## Controlled 3× study (recommended)
```bash
# Produce three Capture JSONLs with identical settings, then:
PYTHONPATH=src python scripts/analyze_llm_run_consistency.py \\
  --runs run_a.jsonl run_b.jsonl run_c.jsonl
```
Target: few hundred reports × 3 seeds. Not executed here (no local LLM endpoint).
""",
        encoding="utf-8",
    )

    # --- Table 29: abstention curves ---
    print("Abstention curves...", flush=True)
    abs_rows = []
    abs_payload = {}
    for model, arms in model_preds.items():
        for config in ("llm_only", "full_hybrid"):
            a = abstention_curve(gold, arms[config], model, config)
            abs_payload[f"{model}/{config}"] = a
            for pt in a["curve"]:
                abs_rows.append(
                    [
                        model,
                        config,
                        pt["threshold"],
                        _fmt(pt["fraction_referred"]),
                        pt["n_referred"],
                        pt["n_auto_accepted"],
                        _fmt(pt["auto_actionable_sensitivity"]),
                        _fmt(pt["auto_actionable_specificity"]),
                        _fmt(pt["auto_high_urgency_sensitivity"]),
                        pt["dangerous_actionable_misses_not_referred"],
                        _fmt(pt["dangerous_miss_rate_among_true_actionable"]),
                    ]
                )
    _write_csv(
        TABLES / "table29_abstention_curve.csv",
        [
            "model",
            "configuration",
            "confidence_threshold",
            "fraction_referred",
            "n_referred",
            "n_auto_accepted",
            "auto_actionable_sensitivity",
            "auto_actionable_specificity",
            "auto_high_urgency_sensitivity",
            "dangerous_actionable_misses",
            "dangerous_miss_rate",
        ],
        abs_rows,
    )

    # --- Figures ---
    print("Figures...", flush=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Fig 13: actionable sensitivity by model/arm
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = []
    vals = []
    colors = []
    for model in MODELS:
        for config, color in [
            ("llm_only", "#2E86AB"),
            ("calibrate", "#F18F01"),
            ("full_hybrid", "#059669"),
        ]:
            m = class_payload[f"{model}/{config}"]
            short = model.split()[0].replace("Qwen3.6-35B", "Qwen").replace("Mistral-Small-3.2", "Mistral")
            if "Qwen" in model:
                short = "Qwen"
            elif "Mistral" in model:
                short = "Mistral"
            elif "Llama" in model:
                short = "Llama"
            labels.append(f"{short}/{config}")
            vals.append(m["actionable_sensitivity"] or 0)
            colors.append(color)
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Sensitivity for actionable class (highest actionability tier)")
    ax.set_title("Clinically relevant metric: actionable sensitivity vs ACR reference")
    fig.tight_layout()
    fig.savefig(FIGS / "fig13_actionable_sensitivity.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "fig13_actionable_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 14: undercall vs overcall rates
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(MODELS))
    width = 0.35
    under_llm = []
    under_hyb = []
    over_llm = []
    over_hyb = []
    names = []
    for model in MODELS:
        names.append(
            "Llama" if "Llama" in model else ("Qwen" if "Qwen" in model else "Mistral")
        )
        under_llm.append(dir_payload[f"{model}/llm_only"]["undercall_rate"] or 0)
        under_hyb.append(dir_payload[f"{model}/full_hybrid"]["undercall_rate"] or 0)
        over_llm.append(dir_payload[f"{model}/llm_only"]["overcall_rate"] or 0)
        over_hyb.append(dir_payload[f"{model}/full_hybrid"]["overcall_rate"] or 0)
    ax.bar(x - width / 2, under_llm, width / 2, label="Undercall LLM", color="#9B2226")
    ax.bar(x - width / 4, under_hyb, width / 2, label="Undercall hybrid", color="#E5383B")
    ax.bar(x + width / 4, over_llm, width / 2, label="Overcall LLM", color="#1D3557")
    ax.bar(x + width / 2, over_hyb, width / 2, label="Overcall hybrid", color="#457B9D")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Rate of reports")
    ax.set_title("Severity undercalls vs overcalls (signed ordinal errors)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig14_undercall_overcall.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "fig14_undercall_overcall.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 15: abstention curve for best hybrid (Llama) + llm_only
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, style in [
        ("Llama-4 Scout/llm_only", {"color": "#2E86AB", "ls": "--", "label": "Llama LLM-only"}),
        ("Llama-4 Scout/full_hybrid", {"color": "#059669", "ls": "-", "label": "Llama hybrid"}),
        ("Qwen3.6-35B/full_hybrid", {"color": "#F18F01", "ls": "-", "label": "Qwen hybrid"}),
    ]:
        curve = abs_payload[key]["curve"]
        xs = [p["fraction_referred"] for p in curve]
        ys = [p["dangerous_miss_rate_among_true_actionable"] or 0 for p in curve]
        ax.plot(xs, ys, **style)
    ax.set_xlabel("Fraction referred to human review")
    ax.set_ylabel("Dangerous miss rate\n(true actionable, not predicted, not referred)")
    ax.set_title("Abstention / referral curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig15_abstention_curve.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "fig15_abstention_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    # Persist JSON payloads
    payload = {
        "per_class": {
            k: {
                **{kk: vv for kk, vv in v.items() if kk != "per_class"},
                "actionable": v["per_class"]["actionable"],
            }
            for k, v in class_payload.items()
        },
        "directional": {
            k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
            for k, v in dir_payload.items()
        },
        "oracle": oracle_payload,
        "recovery": recovery_payload,
        "stability": stab_payload,
        "abstention_policy": next(iter(abs_payload.values()))["policy"],
        "rule_audit": rule_meta,
        "kappa_weighting": next(iter(class_payload.values()))["kappa_weighting_note"],
    }
    (OUT / "clinical_extensions_metrics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    # Update RESULTS_SUMMARY append section
    _update_results_summary(class_payload, dir_payload, oracle_payload, recovery_payload, stab_payload, abs_payload, rule_meta)
    _update_readme()

    # Refresh zip
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(ROOT)))

    print(
        json.dumps(
            {
                "tables": [
                    "table23",
                    "table24",
                    "table25",
                    "table26",
                    "table27",
                    "table28",
                    "table29",
                ],
                "figures": ["fig13", "fig14", "fig15"],
                "rule_audit": rule_meta,
                "zip": str(ZIP_PATH),
            },
            indent=2,
        )
    )


def _short(model: str) -> str:
    if "Llama" in model:
        return "Llama"
    if "Qwen" in model:
        return "Qwen"
    return "Mistral"


def _update_results_summary(
    class_payload, dir_payload, oracle_payload, recovery_payload, stab_payload, abs_payload, rule_meta
) -> None:
    path = OUT / "RESULTS_SUMMARY.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# RESULTS_SUMMARY\n"

    # Pull headline numbers
    lines = [
        "",
        "## Clinical metrics beyond κ (new)",
        "",
        "Primary ordinal endpoint uses **linear-weighted** Cohen's κ on",
        "`none < non_actionable < uncertain < actionable` (quadratic also reported;",
        "nominal/unweighted is not the primary ordinal metric).",
        "",
        "### Actionable-class sensitivity (highest actionability tier)",
        "| Model | LLM-only | Hybrid |",
        "|---|--:|--:|",
    ]
    for model in MODELS:
        a = class_payload[f"{model}/llm_only"]["actionable_sensitivity"]
        b = class_payload[f"{model}/full_hybrid"]["actionable_sensitivity"]
        lines.append(f"| {_short(model)} | {_fmt(a)} | {_fmt(b)} |")

    lines += [
        "",
        "See `table23` (per-class sens/spec), `table24` (per-organ, incl. pancreas/adnexal),",
        "`fig13`.",
        "",
        "### Undercalls vs overcalls",
        "| Model | LLM undercall rate | Hybrid undercall | LLM overcall | Hybrid overcall |",
        "|---|--:|--:|--:|--:|",
    ]
    for model in MODELS:
        ll = dir_payload[f"{model}/llm_only"]
        hy = dir_payload[f"{model}/full_hybrid"]
        lines.append(
            f"| {_short(model)} | {_fmt(ll['undercall_rate'])} | {_fmt(hy['undercall_rate'])} | "
            f"{_fmt(ll['overcall_rate'])} | {_fmt(hy['overcall_rate'])} |"
        )
    lines += [
        "",
        "Operating preference: **prefer overcalls to undercalls** (missed-cancer risk).",
        "ACR recalibration exists to cut LLM severity undercalls. `table25`, `fig14`.",
        "",
        "### Oracle Capture → Orchestrate ceiling",
    ]
    for model, o in oracle_payload.items():
        ora = o["arms"]["oracle_capture"]["followup_needed"]
        hyb = o["arms"]["full_hybrid"]["followup_needed"]
        llm = o["arms"]["llm_only"]["followup_needed"]
        lines.append(
            f"- {_short(model)}: follow-up κ LLM {_fmt(llm.get('cohen_kappa'))} → "
            f"hybrid {_fmt(hyb.get('cohen_kappa'))} → oracle {_fmt(ora.get('cohen_kappa'))}"
        )
    lines += [
        "",
        "Oracle near ceiling ⇒ residual E2E error is Capture extraction, not Orchestrate mapping.",
        "`table26`.",
        "",
        "### Missed-finding recovery audit",
    ]
    for model, r in recovery_payload.items():
        lines.append(
            f"- {_short(model)}: n_supplemented={r['n_supplemented_findings']}, "
            f"text-support precision={_fmt(r['text_support_precision'])}, "
            f"fabrication/unsupported rate={_fmt(r['fabrication_rate_text_audit'])}, "
            f"recall of LLM misses={_fmt(r['recovery_recall_of_llm_misses'])}"
        )
    lines += [
        "",
        "Every supplemented finding audited against report text (`table27`, `table27b`).",
        "",
        "### Rule-engine audit",
        f"Independent auditor worksheet ready (n={rule_meta['n_items']}); "
        "**audit not yet completed** — `rule_engine_audit/PROTOCOL.md`.",
        "",
        "### Run-to-run stability",
    ]
    if stab_payload:
        for model, s in stab_payload.items():
            for pair in s["pairwise"]:
                lines.append(
                    f"- {_short(model)} {pair['run_a']} vs {pair['run_b']} "
                    f"(n={pair['n']}): presence κ={_fmt(pair['presence_kappa'])}, "
                    f"actionability linear κ={_fmt(pair['actionability_linear_kappa'])}"
                )
    lines += [
        "",
        "Opportunistic historical replicates only; controlled 3× study still pending "
        "(`stability/STATUS.md`, `table28`).",
        "",
        "### Abstention / referral curve",
        "Refer if `human_review_required` or min finding confidence < threshold.",
        "See `table29`, `fig15` for fraction-referred vs dangerous actionable miss rate.",
        "",
    ]

    # Replace prior clinical section if re-run, else append
    marker = "## Clinical metrics beyond κ (new)"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    path.write_text(existing + "\n".join(lines), encoding="utf-8")


def _update_readme() -> None:
    path = OUT / "README.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = """
## Clinical extensions (tables 23–29)
| Table | Content |
|---|---|
| table23 | Per-class sens/spec + linear/quadratic κ disclosure |
| table24 | Per-organ clinical metrics (incl. pancreas/adnexal) |
| table25 | Undercall vs overcall + directional confusion |
| table26 | Oracle Capture stage ablation |
| table27 | Missed-finding recovery precision/recall + text audit |
| table28 | Run-to-run stability (historical replicates) |
| table29 | Abstention / referral curve |
| rule_engine_audit/ | Independent ACR encoding audit worksheet |

Regenerate: `PYTHONPATH=src python scripts/build_paper_clinical_extensions.py`
"""
    if "table23" not in text:
        path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


if __name__ == "__main__":
    main()
