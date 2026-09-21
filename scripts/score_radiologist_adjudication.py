#!/usr/bin/env python
"""Score dual-reader radiologist adjudication.

Computes:
  - inter-reader agreement (Cohen's κ / linear κ)
  - disagreement list for attending adjudication
  - optional model-vs-consensus evaluation once consensus labels exist

Example (after readers finish):
  PYTHONPATH=src python scripts/score_radiologist_adjudication.py \\
    --reader-a paper_results/radiologist_adjudication/reader_A_worksheet.csv \\
    --reader-b paper_results/radiologist_adjudication/reader_B_worksheet.csv

Example (after consensus):
  PYTHONPATH=src python scripts/score_radiologist_adjudication.py \\
    --reader-a ... --reader-b ... \\
    --consensus paper_results/radiologist_adjudication/consensus_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.evaluation.bootstrap import (  # noqa: E402
    bootstrap_binary_metrics,
    bootstrap_linear_kappa,
)
from close_framework.evaluation.gold_eval import (  # noqa: E402
    ACTIONABILITY_SCALE,
    IN_SCOPE,
    has_actionable_prediction,
    load_jsonl_by_report_id,
    organs_from_prediction,
    prediction_findings,
    report_actionability_category,
)
from close_framework.evaluation.metrics import (  # noqa: E402
    binary_metrics,
    cohen_kappa,
    multiclass_agreement_metrics,
)

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


def _norm_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "na", "n/a"}:
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _norm_actionability(value: object, finding_present: Optional[bool]) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in ACTIONABILITY_SCALE:
        return text
    if text in {"", "na", "n/a"}:
        if finding_present is False:
            return "none"
        return None
    return None


def _norm_organs(value: object) -> List[str]:
    text = str(value or "").strip().lower()
    if not text or text in {"none", "na", "n/a"}:
        return []
    organs = []
    for part in text.replace(";", ",").split(","):
        organ = part.strip()
        if organ in IN_SCOPE and organ not in organs:
            organs.append(organ)
    return organs


def load_reader_csv(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            report_id = str(raw.get("report_id") or "").strip()
            if not report_id:
                continue
            present = _norm_bool(raw.get("finding_present"))
            action = _norm_actionability(raw.get("actionability"), present)
            follow = _norm_bool(raw.get("followup_required"))
            organs = _norm_organs(raw.get("organs_present"))
            if present is False:
                action = "none"
                follow = False
                organs = []
            rows[report_id] = {
                "sample_id": raw.get("sample_id"),
                "report_id": report_id,
                "finding_present": present,
                "actionability": action,
                "followup_required": follow,
                "organs_present": organs,
                "raw": raw,
            }
    return rows


def _complete(rows: Dict[str, dict]) -> Dict[str, dict]:
    return {
        rid: row
        for rid, row in rows.items()
        if row.get("finding_present") is not None and row.get("actionability") in ACTIONABILITY_SCALE
    }


def interreader_metrics(reader_a: Dict[str, dict], reader_b: Dict[str, dict]) -> dict:
    ids = sorted(set(reader_a) & set(reader_b))
    if not ids:
        return {"n_paired_complete": 0}

    y_pres_a = [bool(reader_a[i]["finding_present"]) for i in ids]
    y_pres_b = [bool(reader_b[i]["finding_present"]) for i in ids]
    y_fu_a = [bool(reader_a[i]["followup_required"]) for i in ids]
    y_fu_b = [bool(reader_b[i]["followup_required"]) for i in ids]
    y_act_a = [reader_a[i]["actionability"] for i in ids]
    y_act_b = [reader_b[i]["actionability"] for i in ids]

    organ_kappas = {}
    for organ in IN_SCOPE:
        ya = [organ in reader_a[i]["organs_present"] for i in ids]
        yb = [organ in reader_b[i]["organs_present"] for i in ids]
        organ_kappas[organ] = binary_metrics(ya, yb)

    return {
        "n_paired_complete": len(ids),
        "finding_presence": {
            "metrics": binary_metrics(y_pres_a, y_pres_b),
            "bootstrap": bootstrap_binary_metrics(y_pres_a, y_pres_b, n_boot=400, seed=0),
        },
        "followup_required": {
            "metrics": binary_metrics(y_fu_a, y_fu_b),
            "bootstrap": bootstrap_binary_metrics(y_fu_a, y_fu_b, n_boot=400, seed=1),
        },
        "actionability": {
            "metrics": multiclass_agreement_metrics(
                y_act_a, y_act_b, labels=list(ACTIONABILITY_SCALE), ordinal=True
            ),
            "bootstrap_linear_kappa": bootstrap_linear_kappa(
                y_act_a, y_act_b, ACTIONABILITY_SCALE, n_boot=400, seed=2
            ),
        },
        "organ_presence": organ_kappas,
        "exact_actionability_agreement": sum(a == b for a, b in zip(y_act_a, y_act_b)) / len(ids),
    }


def disagreement_rows(reader_a: Dict[str, dict], reader_b: Dict[str, dict]) -> List[dict]:
    ids = sorted(set(reader_a) & set(reader_b))
    rows = []
    for report_id in ids:
        a = reader_a[report_id]
        b = reader_b[report_id]
        domains = []
        if a["finding_present"] != b["finding_present"]:
            domains.append("presence")
        if a["actionability"] != b["actionability"]:
            domains.append("actionability")
        if a["followup_required"] != b["followup_required"]:
            domains.append("followup")
        if set(a["organs_present"]) != set(b["organs_present"]):
            domains.append("organs")
        if not domains:
            continue
        rows.append(
            {
                "sample_id": a.get("sample_id") or b.get("sample_id"),
                "report_id": report_id,
                "disagreement_domain": "|".join(domains),
                "reader_A_actionability": a["actionability"],
                "reader_B_actionability": b["actionability"],
                "reader_A_finding_present": a["finding_present"],
                "reader_B_finding_present": b["finding_present"],
                "reader_A_followup_required": a["followup_required"],
                "reader_B_followup_required": b["followup_required"],
                "reader_A_organs": ",".join(a["organs_present"]),
                "reader_B_organs": ",".join(b["organs_present"]),
                "final_value": "",
                "adjudicator_id": "",
                "adjudicator_rationale": "",
            }
        )
    return rows


def build_auto_consensus(reader_a: Dict[str, dict], reader_b: Dict[str, dict]) -> Dict[str, dict]:
    """Consensus where readers agree; leave disagreements unmarked."""

    ids = sorted(set(reader_a) & set(reader_b))
    out = {}
    for report_id in ids:
        a = reader_a[report_id]
        b = reader_b[report_id]
        agree = (
            a["finding_present"] == b["finding_present"]
            and a["actionability"] == b["actionability"]
            and a["followup_required"] == b["followup_required"]
            and set(a["organs_present"]) == set(b["organs_present"])
        )
        if not agree:
            continue
        out[report_id] = {
            **a,
            "consensus_source": "agree_AB",
        }
    return out


def evaluate_models_vs_consensus(consensus: Dict[str, dict]) -> dict:
    results = {}
    for model, arms in MODELS.items():
        results[model] = {}
        for arm, path in arms.items():
            if not path.exists():
                continue
            preds = load_jsonl_by_report_id(path)
            ids = sorted(set(consensus) & set(preds))
            if not ids:
                continue
            y_pres_t = [bool(consensus[i]["finding_present"]) for i in ids]
            y_pres_p = [bool(prediction_findings(preds[i])) for i in ids]
            y_fu_t = [bool(consensus[i]["followup_required"]) for i in ids]
            y_fu_p = [has_actionable_prediction(preds[i]) for i in ids]
            y_act_t = [consensus[i]["actionability"] for i in ids]
            y_act_p = [report_actionability_category(prediction_findings(preds[i])) for i in ids]
            results[model][arm] = {
                "n": len(ids),
                "presence": binary_metrics(y_pres_t, y_pres_p),
                "followup": binary_metrics(y_fu_t, y_fu_p),
                "actionability": multiclass_agreement_metrics(
                    y_act_t, y_act_p, labels=list(ACTIONABILITY_SCALE), ordinal=True
                ),
            }
    return results


def _fmt(x, digits=3):
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-a", type=Path, required=True)
    parser.add_argument("--reader-b", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "paper_results/radiologist_adjudication/results",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_a = load_reader_csv(args.reader_a)
    raw_b = load_reader_csv(args.reader_b)
    reader_a = _complete(raw_a)
    reader_b = _complete(raw_b)

    status = {
        "reader_a_rows": len(raw_a),
        "reader_b_rows": len(raw_b),
        "reader_a_complete": len(reader_a),
        "reader_b_complete": len(reader_b),
        "paired_complete": len(set(reader_a) & set(reader_b)),
    }

    if status["paired_complete"] == 0:
        (args.out_dir / "STATUS.json").write_text(
            json.dumps(
                {
                    **status,
                    "message": (
                        "No paired complete reads yet. Readers must fill finding_present and "
                        "actionability on both worksheets."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(status, indent=2))
        return

    irr = interreader_metrics(reader_a, reader_b)
    disagreements = disagreement_rows(reader_a, reader_b)
    _write_csv(
        args.out_dir / "disagreements_for_adjudication.csv",
        list(disagreements[0].keys()) if disagreements else [
            "sample_id",
            "report_id",
            "disagreement_domain",
        ],
        disagreements,
    )

    auto_consensus = build_auto_consensus(reader_a, reader_b)
    consensus_rows = []
    for report_id, row in sorted(auto_consensus.items()):
        consensus_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "report_id": report_id,
                "finding_present": "yes" if row["finding_present"] else "no",
                "organs_present": ",".join(row["organs_present"]),
                "actionability": row["actionability"],
                "followup_required": "yes" if row["followup_required"] else "no",
                "consensus_source": "agree_AB",
            }
        )
    _write_csv(
        args.out_dir / "auto_consensus_agreements_only.csv",
        [
            "sample_id",
            "report_id",
            "finding_present",
            "organs_present",
            "actionability",
            "followup_required",
            "consensus_source",
        ],
        consensus_rows,
    )

    model_vs = None
    consensus = None
    if args.consensus and args.consensus.exists():
        consensus = _complete(load_reader_csv(args.consensus))
        # If consensus file uses yes/no already handled by loader
        model_vs = evaluate_models_vs_consensus(consensus)

    summary = {
        "status": status,
        "inter_reader": {
            "n": irr["n_paired_complete"],
            "presence_kappa": irr["finding_presence"]["metrics"]["cohen_kappa"],
            "presence_kappa_ci": irr["finding_presence"]["bootstrap"]["cohen_kappa"],
            "followup_kappa": irr["followup_required"]["metrics"]["cohen_kappa"],
            "followup_kappa_ci": irr["followup_required"]["bootstrap"]["cohen_kappa"],
            "actionability_linear_kappa": irr["actionability"]["metrics"]["cohen_kappa"],
            "actionability_linear_kappa_ci": irr["actionability"]["bootstrap_linear_kappa"],
            "exact_actionability_agreement": irr["exact_actionability_agreement"],
            "n_disagreements": len(disagreements),
        },
        "model_vs_radiologist_consensus": model_vs,
        "note": (
            "Inter-reader κ quantifies reliability of the clinical reference. "
            "Model-vs-consensus is the non-circular clinical validity analysis."
        ),
    }
    (args.out_dir / "interreader_metrics.json").write_text(
        json.dumps({"inter_reader_full": irr, **summary}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    md = [
        "# Radiologist adjudication scoring",
        "",
        f"Paired complete reads: **{irr['n_paired_complete']}**",
        f"Disagreements needing adjudication: **{len(disagreements)}**",
        "",
        "## Inter-reader agreement",
        f"- Finding presence κ: {_fmt(summary['inter_reader']['presence_kappa'])}",
        f"- Follow-up required κ: {_fmt(summary['inter_reader']['followup_kappa'])}",
        f"- Actionability linear κ: {_fmt(summary['inter_reader']['actionability_linear_kappa'])}",
        f"- Exact actionability agreement: {_fmt(summary['inter_reader']['exact_actionability_agreement'])}",
        "",
    ]
    if model_vs:
        md.append("## Model vs radiologist consensus")
        md.append("")
        md.append("| Model | Config | N | Presence κ | Follow-up κ | Actionability linear κ |")
        md.append("|---|---|--:|--:|--:|--:|")
        for model, arms in model_vs.items():
            for arm, block in arms.items():
                md.append(
                    f"| {model} | {arm} | {block['n']} | "
                    f"{_fmt(block['presence']['cohen_kappa'])} | "
                    f"{_fmt(block['followup']['cohen_kappa'])} | "
                    f"{_fmt(block['actionability']['cohen_kappa'])} |"
                )
    else:
        md.append(
            "## Model vs consensus\nNot run yet. Provide `--consensus` after attending "
            "reconciliation (or use agreements-only auto consensus for a partial analysis)."
        )
    (args.out_dir / "SCORING_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
