#!/usr/bin/env python
"""Create a stratified expert-adjudication annotation template (no fabricated labels).

Example:
  PYTHONPATH=src python scripts/create_expert_adjudication_sample.py \\
    --output paper_results/expert_adjudication/adjudication_sample.csv \\
    --target-n 250
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.evaluation.gold_eval import (  # noqa: E402
    gold_findings,
    load_jsonl_by_report_id,
    report_actionability_category,
)
from close_framework.evaluation.paper_analyses import sample_expert_adjudication_ids  # noqa: E402

RUNS = {
    "Llama-4 Scout": ROOT / "outputs/model_comparison_9283993/llama/close_results.jsonl",
    "Qwen3.6-35B": ROOT / "outputs/model_comparison_9500762/qwen/close_results.jsonl",
    "Mistral-Small-3.2": ROOT / "outputs/model_comparison_9500762/mistral/close_results.jsonl",
}

ANNOTATOR_FIELDS = [
    "finding_present",
    "organ",
    "finding_type",
    "size_mm",
    "imaging_features",
    "actionability",
    "followup_required",
    "modality",
    "timeframe",
    "guideline_pathway",
    "confidence",
    "comments",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper_results/expert_adjudication/adjudication_sample.csv",
    )
    parser.add_argument("--target-n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gold", type=Path, default=ROOT / "data/gold_labels/gold_labels.jsonl")
    args = parser.parse_args()

    gold = load_jsonl_by_report_id(args.gold)
    llm = {name: load_jsonl_by_report_id(path) for name, path in RUNS.items() if path.exists()}
    selected = sample_expert_adjudication_ids(gold, llm, target_n=args.target_n, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        [
            "sample_id",
            "report_id",
            "stratum_hint",
            "reference_has_in_scope_finding",
            "reference_organs",
            "reference_actionability",
            "reference_n_findings",
            "llm_llama_actionability",
            "llm_qwen_actionability",
            "llm_mistral_actionability",
        ]
        + [f"reviewer1_{f}" for f in ANNOTATOR_FIELDS]
        + [f"reviewer2_{f}" for f in ANNOTATOR_FIELDS]
        + [
            "adjudicator_final_finding_present",
            "adjudicator_final_organ",
            "adjudicator_final_actionability",
            "adjudicator_final_followup_required",
            "adjudicator_final_modality",
            "adjudicator_final_timeframe",
            "adjudicator_notes",
            "agreement_reviewers",
        ]
    )

    rows = []
    for idx, report_id in enumerate(selected, start=1):
        label = gold[report_id]
        organs_list = sorted(label.get("organs_present") or [])
        organs = ",".join(organs_list) or "none"
        if not label.get("has_in_scope_finding"):
            stratum = "none"
        elif len(organs_list) > 1 or int(label.get("n_findings") or 0) > 1:
            stratum = "complex:" + "+".join(organs_list)
        else:
            stratum = organs_list[0]

        llm_acts = {}
        for name, short in [
            ("Llama-4 Scout", "llama"),
            ("Qwen3.6-35B", "qwen"),
            ("Mistral-Small-3.2", "mistral"),
        ]:
            preds = llm.get(name, {})
            if report_id in preds:
                from close_framework.evaluation.gold_eval import prediction_findings

                llm_acts[short] = report_actionability_category(prediction_findings(preds[report_id]))
            else:
                llm_acts[short] = ""

        row = {
            "sample_id": f"ADJ-{idx:03d}",
            "report_id": report_id,
            "stratum_hint": stratum,
            "reference_has_in_scope_finding": bool(label.get("has_in_scope_finding")),
            "reference_organs": organs,
            "reference_actionability": label.get("report_actionability_category"),
            "reference_n_findings": label.get("n_findings"),
            "llm_llama_actionability": llm_acts["llama"],
            "llm_qwen_actionability": llm_acts["qwen"],
            "llm_mistral_actionability": llm_acts["mistral"],
        }
        for prefix in ("reviewer1_", "reviewer2_"):
            for field in ANNOTATOR_FIELDS:
                row[prefix + field] = ""
        row.update(
            {
                "adjudicator_final_finding_present": "",
                "adjudicator_final_organ": "",
                "adjudicator_final_actionability": "",
                "adjudicator_final_followup_required": "",
                "adjudicator_final_modality": "",
                "adjudicator_final_timeframe": "",
                "adjudicator_notes": "",
                "agreement_reviewers": "",
            }
        )
        rows.append(row)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "n_sampled": len(rows),
        "target_n": args.target_n,
        "seed": args.seed,
        "gold": str(args.gold),
        "note": (
            "Annotation template only. Reference_* columns are ACR-derived automated "
            "labels for stratification, not expert truth. No PHI report text is included."
        ),
        "stratum_counts": {},
        "organ_coverage_counts": {},
    }
    for row in rows:
        meta["stratum_counts"][row["stratum_hint"]] = meta["stratum_counts"].get(row["stratum_hint"], 0) + 1
        for organ in (row["reference_organs"] or "none").split(","):
            meta["organ_coverage_counts"][organ] = meta["organ_coverage_counts"].get(organ, 0) + 1
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    guide = args.output.parent / "ANNOTATION_GUIDE.md"
    guide.write_text(
        """# Expert adjudication guide

## Purpose
Independent clinical validation subset for CLOSE Capture / follow-up recommendations.
The automated ACR-derived reference is **not** expert ground truth.

## Workflow
1. Two reviewers annotate independently (`reviewer1_*`, `reviewer2_*`).
2. An adjudicator resolves disagreements into `adjudicator_final_*`.
3. Do not look at model predictions when labeling (columns are for sampling audit only).

## Label definitions
- `finding_present`: yes/no for in-scope incidental finding
- `organ`: kidney/liver/adrenal/pancreas/adnexal
- `actionability`: none / non_actionable / uncertain / actionable
- `followup_required`: yes/no
- `modality` / `timeframe` / `guideline_pathway`: free text guided by ACR white papers
- `confidence`: low/medium/high

## Privacy
This CSV contains report IDs and aggregate reference flags only. Link to local reports
separately; do not paste PHI into comments.
""",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "meta": str(meta_path), **meta}, indent=2))


if __name__ == "__main__":
    main()
