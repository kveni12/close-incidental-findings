#!/usr/bin/env python
"""Paper extensions: dumb baseline, Link scoping/sim, Capture failure taxonomy, template shift.

Example:
  PYTHONPATH=src python scripts/build_paper_extensions.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.agents.link_agent import LinkAgent  # noqa: E402
from close_framework.config.settings import CloseSettings  # noqa: E402
from close_framework.data.loaders import iter_report_file, iter_reports  # noqa: E402
from close_framework.data.schemas import IncidentalFinding, PatientContext, RadiologyReport  # noqa: E402
from close_framework.evaluation.gold_eval import (  # noqa: E402
    ACTIONABILITY_SCALE,
    IN_SCOPE,
    evaluate_predictions_against_gold,
    gold_findings,
    has_actionable_prediction,
    load_jsonl_by_report_id,
    organs_from_prediction,
    prediction_findings,
    report_actionability_category,
)
from close_framework.evaluation.metrics import binary_metrics, cohen_kappa, multiclass_agreement_metrics  # noqa: E402
from close_framework.evaluation.bootstrap import bootstrap_binary_metrics, bootstrap_linear_kappa  # noqa: E402

OUT = ROOT / "paper_results"
TABLES = OUT / "tables"
FIGS = OUT / "figures"

GOLD = ROOT / "data/gold_labels/gold_labels.jsonl"
MODELS = {
    "Llama-4 Scout": {
        "llm_only": ROOT / "outputs/model_comparison_9283993/llama/close_results.jsonl",
        "full_hybrid": ROOT / "outputs/model_comparison_9283993/llama/close_results_hybrid.jsonl",
    },
    "Qwen3.6-35B": {
        "llm_only": ROOT / "outputs/model_comparison_9500762/qwen/close_results.jsonl",
        "full_hybrid": ROOT / "outputs/model_comparison_9500762/qwen/close_results_hybrid.jsonl",
    },
    "Mistral-Small-3.2": {
        "llm_only": ROOT / "outputs/model_comparison_9500762/mistral/close_results.jsonl",
        "full_hybrid": ROOT / "outputs/model_comparison_9500762/mistral/close_results_hybrid.jsonl",
    },
}

ORGAN_TERMS = {
    "kidney": ["kidney", "renal", "kidneys"],
    "liver": ["liver", "hepatic"],
    "adrenal": ["adrenal"],
    "pancreas": ["pancreas", "pancreatic"],
    "adnexal": ["adnexal", "adnexa", "ovary", "ovarian", "ovaries"],
}
FINDING_TERMS = [
    "cyst", "cysts", "lesion", "lesions", "mass", "masses", "nodule", "nodules",
    "hypodensity", "hypodensities", "hyperdensity", "hyperdense", "ipmn",
    "dilatation", "dilation", "thickening",
]
NEGATION = re.compile(
    r"\b(no|without|absent|unremarkable|normal)\b.{0,40}\b("
    + "|".join(FINDING_TERMS)
    + r")\b",
    re.I,
)
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|mm)\b", re.I)
# Dumb thresholds (mm)
DUMB_THRESHOLDS = {
    "kidney": 10.0,
    "liver": 10.0,
    "adrenal": 10.0,
    "pancreas": 5.0,
    "adnexal": 30.0,
}


def _fmt(x, d=3):
    return "" if x is None else f"{x:.{d}f}"


def _fmt_ci(ci, d=3):
    if not ci or ci.get("estimate") is None:
        return ""
    if ci.get("ci_low") is None:
        return _fmt(ci["estimate"], d)
    return f"{ci['estimate']:.{d}f} ({ci['ci_low']:.{d}f}–{ci['ci_high']:.{d}f})"


def _write_csv(path: Path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(",".join("" if c is None else str(c) for c in row) for row in rows)
    path.write_text(",".join(headers) + "\n" + body + ("\n" if rows else ""), encoding="utf-8")


def dumb_baseline_label(text: str) -> dict:
    """Regex organ+finding + size-threshold actionability (intentionally naive)."""

    organs = []
    organ_actions = {}
    sentences = re.split(r"(?<=[.!?;:\n])\s+", text)
    for sentence in sentences:
        lower = sentence.lower()
        if NEGATION.search(sentence):
            continue
        has_finding = any(re.search(rf"\b{re.escape(t)}\b", lower) for t in FINDING_TERMS)
        if not has_finding:
            continue
        for organ, terms in ORGAN_TERMS.items():
            if any(re.search(rf"\b{re.escape(t)}\b", lower) for t in terms):
                sizes = []
                for m in SIZE_RE.finditer(sentence):
                    val = float(m.group(1))
                    sizes.append(val * 10 if m.group(2).lower() == "cm" else val)
                max_size = max(sizes) if sizes else None
                if max_size is not None and max_size >= DUMB_THRESHOLDS[organ]:
                    action = "actionable"
                elif max_size is None:
                    action = "uncertain"
                else:
                    action = "non_actionable"
                organs.append(organ)
                prev = organ_actions.get(organ)
                rank = {"non_actionable": 1, "uncertain": 2, "actionable": 3}
                if prev is None or rank[action] > rank[prev]:
                    organ_actions[organ] = action
    organs = sorted(set(organs))
    if not organs:
        return {
            "has_in_scope_finding": False,
            "has_actionable_finding": False,
            "organs_present": [],
            "report_actionability_category": "none",
            "findings": [],
        }
    rank = {"non_actionable": 1, "uncertain": 2, "actionable": 3}
    top = max(organ_actions.values(), key=lambda a: rank[a])
    findings = [
        {
            "organ_system": organ,
            "actionability": organ_actions[organ],
            "finding_type": "dumb_regex",
            "evidence_span": "",
            "size_mm": None,
            "urgency_tier": "unknown",
        }
        for organ in organs
    ]
    return {
        "has_in_scope_finding": True,
        "has_actionable_finding": top == "actionable",
        "organs_present": organs,
        "report_actionability_category": top,
        "findings": findings,
    }


def load_report_texts(report_ids: set[str]) -> Dict[str, str]:
    found = {}
    for report in iter_reports(ROOT / "data"):
        if report.report_id in report_ids:
            found[report.report_id] = report.report_text or ""
            if len(found) == len(report_ids):
                break
    return found


def evaluate_label_dict(gold: dict, labels: dict, name: str) -> dict:
    """Evaluate prediction-like label dicts shaped like gold records."""

    # Wrap into fake workflow prediction structure expected by gold_eval helpers
    predictions = {}
    for rid, lab in labels.items():
        findings = lab.get("findings") or []
        # ensure actionability fields
        predictions[rid] = {
            "report_id": rid,
            "capture_result": {"findings": findings},
        }
        # if findings empty but has_in_scope flags, synthesize minimal findings for organ presence
        if not findings and lab.get("organs_present"):
            predictions[rid]["capture_result"]["findings"] = [
                {
                    "organ_system": organ,
                    "actionability": lab.get("report_actionability_category") or "uncertain",
                    "finding_type": "label",
                    "evidence_span": "",
                    "urgency_tier": "unknown",
                }
                for organ in lab["organs_present"]
            ]
    return evaluate_predictions_against_gold(gold, predictions, name)


def run_dumb_baseline(gold: dict, texts: Dict[str, str]) -> Tuple[dict, dict]:
    labels = {}
    for rid in gold:
        labels[rid] = dumb_baseline_label(texts.get(rid, ""))
        labels[rid]["report_id"] = rid
    metrics = evaluate_label_dict(gold, labels, "DumbRegexSizeBaseline")
    return labels, metrics


def template_shift_analysis(gold: dict, texts: Dict[str, str], model_preds: dict) -> dict:
    """Within-site split by reporting template richness (structured organ headers)."""

    structured = []
    sparse = []
    header_re = re.compile(
        r"^\s*(Liver|Kidneys?(?:/Ureters)?|Adrenal(?:\s+Glands?)?|Pancreas|Adnexa|Ovar(?:y|ies))\s*:",
        re.I | re.M,
    )
    for rid, text in texts.items():
        n_headers = len(header_re.findall(text))
        if n_headers >= 3:
            structured.append(rid)
        else:
            sparse.append(rid)

    def _subset_metrics(ids: List[str], preds: dict, name: str) -> dict:
        sub_gold = {i: gold[i] for i in ids if i in gold and i in preds}
        sub_pred = {i: preds[i] for i in sub_gold}
        if not sub_gold:
            return {"n": 0}
        m = evaluate_predictions_against_gold(sub_gold, sub_pred, name)
        return {
            "n": len(sub_gold),
            "presence_kappa": m["presence"]["cohen_kappa"],
            "actionability_linear_kappa": m["categorical"]["report_actionability"]["cohen_kappa"],
            "actionable_kappa": m["actionable"]["cohen_kappa"],
        }

    out = {
        "definition": (
            "Structured = ≥3 organ-section headers (Liver:/Kidneys:/etc.). "
            "Sparse = fewer headers. Single-site institutional template-style split; not an external institution."
        ),
        "n_structured": len(structured),
        "n_sparse": len(sparse),
        "models": {},
    }
    for model, arms in model_preds.items():
        out["models"][model] = {}
        for arm, preds in arms.items():
            out["models"][model][arm] = {
                "structured": _subset_metrics(structured, preds, f"{model}/{arm}/structured"),
                "sparse": _subset_metrics(sparse, preds, f"{model}/{arm}/sparse"),
            }
    return out


def external_holdout_prep() -> dict:
    """Document alternate export-slice availability and sample 100 IDs for future external eval."""

    out_dir = OUT / "generalizability"
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_ids = set(load_jsonl_by_report_id(GOLD))
    external = []
    for path in sorted((ROOT / "data").glob("*Rad.txt")):
        count = 0
        for report in iter_report_file(path):
            if report.report_id in gold_ids:
                continue
            # Prefer CT abdomen-ish text
            text = (report.report_text or "").lower()
            if "ct" not in text and "computed tomography" not in text:
                continue
            if not any(k in text for k in ("abdomen", "liver", "kidney", "adrenal", "pancreas")):
                continue
            external.append(
                {
                    "report_id": report.report_id,
                    "source_file": path.name,
                    "text_chars": len(report.report_text or ""),
                }
            )
            count += 1
            if count >= 80:  # per file cap
                break
    # take 100
    selected = external[:100]
    _write_csv(
        out_dir / "external_candidate_ids.csv",
        ["report_id", "source_file", "text_chars"],
        [[r["report_id"], r["source_file"], r["text_chars"]] for r in selected],
    )
    meta = {
        "n_candidates": len(selected),
        "source_files": sorted({r["source_file"] for r in selected}),
        "status": "ids_ready_llm_outputs_not_run",
        "note": (
            "Cohort LLM runs used only Site-A export. Site-B export/3 provide different export slices from the "
            "same institution family; treat as template/export shift, not a true multi-center external site. "
            "Run CLOSE on these IDs to complete the external slice."
        ),
        "how_to_run": (
            "Restrict pipeline with --report-ids-from after writing a JSONL of these IDs, "
            "or filter iter_reports to this ID set."
        ),
    }
    (out_dir / "external_slice_README.md").write_text(
        "# External / template-shift slice\n\n"
        + meta["note"]
        + "\n\n"
        + f"Prepared **{meta['n_candidates']}** candidate report IDs in "
        + "`external_candidate_ids.csv`.\n\n"
        + "**Not yet scored:** no LLM outputs for these IDs in the current experiment.\n",
        encoding="utf-8",
    )
    (out_dir / "external_slice_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def link_simulation() -> dict:
    """Evaluate Link routing logic under simulated EHR contexts (unit-style)."""

    agent = LinkAgent(CloseSettings())
    finding = IncidentalFinding(
        finding_id="sim-1",
        organ_system="kidney",
        finding_type="mass",
        description="sim",
        actionability="actionable",
        evidence_span="sim",
        human_review_required=False,
    )

    scenarios = [
        (
            "pcp_available",
            PatientContext(active_pcp_id="PCP1", active_pcp_name="Dr PCP", clinical_setting="outpatient"),
            "PCP",
            False,
        ),
        (
            "specialist_available",
            PatientContext(
                specialists={"kidney": {"provider_id": "U1", "provider_name": "Dr Uro"}},
                clinical_setting="outpatient",
            ),
            "specialist",
            False,
        ),
        (
            "ordering_provider_non_ed",
            PatientContext(
                ordering_provider_id="OP1",
                ordering_provider_name="Dr Order",
                clinical_setting="outpatient",
            ),
            "ordering_provider",
            False,
        ),
        (
            "ed_with_ordering_only",
            PatientContext(
                ordering_provider_id="OP1",
                ordering_provider_name="Dr Order",
                clinical_setting="ED",
            ),
            "safety_pool",
            True,
        ),
        (
            "missing_all_context",
            PatientContext(clinical_setting="unknown"),
            "safety_pool",
            True,
        ),
    ]

    rows = []
    correct = 0
    for name, ctx, expected_type, expected_failsafe in scenarios:
        assignment = agent.run(finding, ctx)
        ok = assignment.provider_type == expected_type and assignment.failsafe_triggered == expected_failsafe
        correct += int(ok)
        rows.append(
            [
                name,
                expected_type,
                assignment.provider_type,
                expected_failsafe,
                assignment.failsafe_triggered,
                assignment.human_review_required,
                ok,
            ]
        )

    # Empirical cohort behavior
    hybrid = load_jsonl_by_report_id(MODELS["Qwen3.6-35B"]["full_hybrid"])
    n_assign = 0
    n_failsafe = 0
    for result in hybrid.values():
        for a in (result.get("provider_assignments") or {}).values():
            n_assign += 1
            n_failsafe += int(bool(a.get("failsafe_triggered")))

    summary = {
        "evaluation_mode": "simulated_context_unit_tests_plus_cohort_failsafe_description",
        "simulated_scenarios_n": len(scenarios),
        "simulated_routing_accuracy": correct / len(scenarios),
        "cohort_assignments": n_assign,
        "cohort_failsafe_rate": n_failsafe / n_assign if n_assign else None,
        "scope_statement": (
            "In the primary 1000-report experiment, EHR/PCP specialty context was unavailable, "
            "so Link operated in failsafe-only mode (safety pool + human review). "
            "Link is therefore described and unit-tested under simulated contexts, but "
            "**not clinically validated** on real provider directories in this cohort."
        ),
    }
    _write_csv(
        TABLES / "table17_link_simulation.csv",
        [
            "scenario",
            "expected_provider_type",
            "observed_provider_type",
            "expected_failsafe",
            "observed_failsafe",
            "human_review_required",
            "correct",
        ],
        rows,
    )
    (OUT / "agents" / "link_scoping.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "agents" / "link_scoping.md").write_text(
        "# Link agent scoping\n\n"
        + summary["scope_statement"]
        + f"\n\nSimulated routing accuracy: **{summary['simulated_routing_accuracy']:.2f}** "
        + f"({correct}/{len(scenarios)} scenarios).\n\n"
        + f"Cohort failsafe rate (Qwen hybrid assignments): **{_fmt(summary['cohort_failsafe_rate'])}** "
        + f"({n_failsafe}/{n_assign}).\n",
        encoding="utf-8",
    )
    return summary


def capture_failure_taxonomy(gold: dict, texts: Dict[str, str], llm_preds: dict, model_name: str) -> dict:
    """Characterize Capture errors using linguistic phenotypes on discordant reports."""

    patterns = {
        "hedged_language": re.compile(
            r"\b(probably|likely|possible|suggests?|may represent|cannot exclude|cannot be excluded|"
            r"indeterminate|too small to characterize|tstc)\b",
            re.I,
        ),
        "comparison_to_priors": re.compile(
            r"\b(unchanged|stable|similar to|no significant change|compared with|prior|previously|"
            r"since \d{4}|redemonstrat)\b",
            re.I,
        ),
        "multiple_lesions_language": re.compile(
            r"\b(multiple|numerous|several|scattered|bilateral)\b.{0,40}\b(cyst|lesion|nodule|mass|hypodensit)",
            re.I,
        ),
        "biphasic_or_two_plane_size": re.compile(
            r"\d+(?:\.\d+)?\s*(?:cm|mm)\s*[x×]\s*\d+(?:\.\d+)?\s*(?:cm|mm)",
            re.I,
        ),
        "impression_mentions_finding": re.compile(
            r"(?is)impression:?(.{0,800})",
        ),
    }

    counts = Counter()
    examples = defaultdict(list)
    ids = sorted(set(gold) & set(llm_preds))

    for rid in ids:
        g = gold[rid]
        p = llm_preds[rid]
        g_org = set(g.get("organs_present") or [])
        p_org = organs_from_prediction(p)
        g_act = g.get("report_actionability_category") or "none"
        p_act = report_actionability_category(prediction_findings(p))
        text = texts.get(rid, "")

        missed = g_org - p_org
        false_pos = p_org - g_org
        severity_under = (
            ACTIONABILITY_SCALE.index(p_act) < ACTIONABILITY_SCALE.index(g_act)
            if g_act in ACTIONABILITY_SCALE and p_act in ACTIONABILITY_SCALE
            else False
        )
        severity_over = (
            ACTIONABILITY_SCALE.index(p_act) > ACTIONABILITY_SCALE.index(g_act)
            if g_act in ACTIONABILITY_SCALE and p_act in ACTIONABILITY_SCALE
            else False
        )

        if not (missed or false_pos or g_act != p_act):
            continue

        counts["discordant_reports"] += 1
        if missed:
            counts["organ_miss"] += 1
        if false_pos:
            counts["organ_false_positive"] += 1
        if severity_under:
            counts["actionability_undercall"] += 1
        if severity_over:
            counts["actionability_overcall"] += 1

        # Linguistic tags on the full report text for discordant cases
        tags = []
        if patterns["hedged_language"].search(text):
            tags.append("hedged_language")
        if patterns["comparison_to_priors"].search(text):
            tags.append("comparison_to_priors")
        if patterns["multiple_lesions_language"].search(text):
            tags.append("multiple_lesions_language")
        if patterns["biphasic_or_two_plane_size"].search(text):
            tags.append("two_plane_size")

        # Buried in body / absent from impression heuristic
        imp = patterns["impression_mentions_finding"].search(text)
        impression = imp.group(1) if imp else ""
        body_has = any(
            re.search(rf"\b{re.escape(t)}\b", text, re.I) for t in FINDING_TERMS
        )
        impression_has = any(
            re.search(rf"\b{re.escape(t)}\b", impression, re.I) for t in FINDING_TERMS
        )
        if body_has and impression and not impression_has and missed:
            tags.append("finding_in_body_not_impression")

        for tag in tags:
            counts[tag] += 1
            if len(examples[tag]) < 8:
                examples[tag].append(
                    {
                        "report_id": rid,
                        "reference_organs": sorted(g_org),
                        "pred_organs": sorted(p_org),
                        "reference_actionability": g_act,
                        "pred_actionability": p_act,
                        "missed_organs": sorted(missed),
                    }
                )

        if not tags:
            counts["no_tagged_phenotype"] += 1

    return {
        "model": model_name,
        "n_discordant": counts["discordant_reports"],
        "counts": dict(counts),
        "examples": {k: v for k, v in examples.items()},
        "note": (
            "Taxonomy is applied to LLM-only vs ACR-derived reference discordances. "
            "Tags are linguistic phenotypes co-occurring with errors, not proven causal mechanisms."
        ),
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    gold = load_jsonl_by_report_id(GOLD)
    print("Loading report texts for cohort...", flush=True)
    texts = load_report_texts(set(gold))
    print(f"texts={len(texts)}", flush=True)

    model_preds = {
        model: {arm: load_jsonl_by_report_id(path) for arm, path in arms.items()}
        for model, arms in MODELS.items()
    }

    # --- Dumb baseline ---
    print("Dumb baseline...", flush=True)
    dumb_labels, dumb_metrics = run_dumb_baseline(gold, texts)
    baseline_rows = []
    systems = [("DumbRegex+Size", dumb_metrics)]
    for model, arms in model_preds.items():
        for arm, preds in arms.items():
            systems.append(
                (f"{model}/{arm}", evaluate_predictions_against_gold(gold, preds, f"{model}/{arm}"))
            )
    for name, m in systems:
        baseline_rows.append(
            [
                name,
                m["n_shared_reports"],
                _fmt(m["presence"]["balanced_accuracy"]),
                _fmt(m["presence"]["cohen_kappa"]),
                _fmt(m["actionable"]["balanced_accuracy"]),
                _fmt(m["actionable"]["cohen_kappa"]),
                _fmt(m["categorical"]["report_actionability"]["cohen_kappa"]),
                _fmt(m["categorical"]["report_actionability"]["macro_f1"]),
            ]
        )
    _write_csv(
        TABLES / "table18_dumb_baseline_comparison.csv",
        [
            "system",
            "n",
            "presence_balanced_acc",
            "presence_kappa",
            "actionable_balanced_acc",
            "actionable_kappa",
            "actionability_linear_kappa",
            "actionability_macro_f1",
        ],
        baseline_rows,
    )

    # --- Template shift ---
    print("Template-shift split...", flush=True)
    template = template_shift_analysis(gold, texts, model_preds)
    trows = []
    for model, arms in template["models"].items():
        for arm, blocks in arms.items():
            for slice_name in ("structured", "sparse"):
                b = blocks[slice_name]
                trows.append(
                    [
                        model,
                        arm,
                        slice_name,
                        b.get("n", 0),
                        _fmt(b.get("presence_kappa")),
                        _fmt(b.get("actionable_kappa")),
                        _fmt(b.get("actionability_linear_kappa")),
                    ]
                )
    _write_csv(
        TABLES / "table19_template_shift_generalizability.csv",
        [
            "model",
            "configuration",
            "template_slice",
            "n",
            "presence_kappa",
            "actionable_kappa",
            "actionability_linear_kappa",
        ],
        trows,
    )
    (OUT / "generalizability").mkdir(parents=True, exist_ok=True)
    (OUT / "generalizability" / "template_shift.json").write_text(
        json.dumps(template, indent=2), encoding="utf-8"
    )

    # --- External prep ---
    print("External candidate prep...", flush=True)
    external_meta = external_holdout_prep()

    # --- Link ---
    print("Link simulation...", flush=True)
    link_summary = link_simulation()

    # --- Failure taxonomy (LLM-only, best model Qwen + all models) ---
    print("Failure taxonomy...", flush=True)
    taxonomies = {}
    tax_rows = []
    example_rows = []
    for model, arms in model_preds.items():
        tax = capture_failure_taxonomy(gold, texts, arms["llm_only"], model)
        taxonomies[model] = tax
        for key, count in sorted(tax["counts"].items()):
            tax_rows.append([model, key, count])
        for tag, exs in tax["examples"].items():
            for ex in exs:
                example_rows.append(
                    [
                        model,
                        tag,
                        ex["report_id"],
                        "|".join(ex["reference_organs"]),
                        "|".join(ex["pred_organs"]),
                        ex["reference_actionability"],
                        ex["pred_actionability"],
                        "|".join(ex["missed_organs"]),
                    ]
                )
    _write_csv(
        TABLES / "table20_capture_failure_taxonomy.csv",
        ["model", "category", "count"],
        tax_rows,
    )
    _write_csv(
        TABLES / "table21_capture_failure_examples.csv",
        [
            "model",
            "phenotype",
            "report_id",
            "reference_organs",
            "pred_organs",
            "reference_actionability",
            "pred_actionability",
            "missed_organs",
        ],
        example_rows,
    )
    (OUT / "failure_analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "failure_analysis" / "capture_failure_taxonomy.json").write_text(
        json.dumps(taxonomies, indent=2), encoding="utf-8"
    )

    # --- Figure: baseline comparison ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG = FIGS
    FIG.mkdir(parents=True, exist_ok=True)
    labels = []
    vals = []
    for name, m in systems:
        short = name.replace("Llama-4 Scout", "Llama").replace("Qwen3.6-35B", "Qwen").replace(
            "Mistral-Small-3.2", "Mistral"
        )
        labels.append(short)
        vals.append(m["categorical"]["report_actionability"]["cohen_kappa"] or 0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#94A3B8" if "Dumb" in l else ("#059669" if "full_hybrid" in l else "#2E86AB") for l in labels]
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("4-level actionability linear κ vs ACR reference")
    ax.set_xlim(0, 1.05)
    ax.set_title("Dumb regex+size baseline vs LLM-only vs hybrid")
    fig.tight_layout()
    fig.savefig(FIG / "fig11_dumb_baseline_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig11_dumb_baseline_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # Failure taxonomy stacked bars for one model (Qwen)
    qtax = taxonomies["Qwen3.6-35B"]["counts"]
    focus = [
        "hedged_language",
        "comparison_to_priors",
        "multiple_lesions_language",
        "two_plane_size",
        "finding_in_body_not_impression",
        "actionability_undercall",
        "organ_miss",
    ]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(focus, [qtax.get(k, 0) for k in focus], color="#A23B72")
    ax.set_ylabel("Discordant reports (Qwen LLM-only)")
    ax.set_title("Capture failure phenotypes (co-occurrence with discordance)")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig12_capture_failure_taxonomy.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig12_capture_failure_taxonomy.pdf", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "dumb_baseline": {
            "presence_kappa": dumb_metrics["presence"]["cohen_kappa"],
            "actionability_linear_kappa": dumb_metrics["categorical"]["report_actionability"]["cohen_kappa"],
            "thresholds_mm": DUMB_THRESHOLDS,
        },
        "template_shift": template,
        "external_slice": external_meta,
        "link": link_summary,
        "failure_taxonomy_models": list(taxonomies),
    }
    (OUT / "extensions_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["dumb_baseline"], indent=2))
    print("link", link_summary["simulated_routing_accuracy"], link_summary["cohort_failsafe_rate"])
    print("external candidates", external_meta["n_candidates"])
    print("template", template["n_structured"], template["n_sparse"])


if __name__ == "__main__":
    main()
