#!/usr/bin/env python
"""Create a blinded dual-reader radiologist adjudication pack (n=200).

Does NOT fabricate clinical labels. Produces:
  - blinded worksheets for Reader A / Reader B
  - adjudicator reconciliation sheet
  - admin key (strata + ACR/model audit columns; not for readers)
  - protocol for a ~2-week resident dual-read study

Example:
  PYTHONPATH=src python scripts/create_radiologist_adjudication_pack.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.evaluation.gold_eval import (  # noqa: E402
    IN_SCOPE,
    gold_findings,
    load_jsonl_by_report_id,
    prediction_findings,
    report_actionability_category,
)

RUNS = {
    "Llama-4 Scout": ROOT / "outputs/model_comparison_9283993/llama/close_results.jsonl",
    "Qwen3.6-35B": ROOT / "outputs/model_comparison_9500762/qwen/close_results.jsonl",
    "Mistral-Small-3.2": ROOT / "outputs/model_comparison_9500762/mistral/close_results.jsonl",
}

# Dual-reader worksheet fields (ACR-criteria clinical labels)
READER_FIELDS = [
    "finding_present",  # yes/no
    "organs_present",  # comma list: kidney,liver,adrenal,pancreas,adnexal
    "dominant_organ",  # if multiple, organ driving actionability
    "finding_type",  # free text, e.g. cyst / solid mass / IPMN
    "size_mm",  # numeric or blank
    "key_imaging_features",  # free text
    "actionability",  # none|non_actionable|uncertain|actionable
    "followup_required",  # yes/no
    "recommended_modality",  # free text / none
    "recommended_timeframe",  # free text / none
    "acr_pathway_if_known",  # optional
    "confidence",  # low|medium|high
    "comments",
]


def sample_radiologist_ids(
    gold: dict,
    llm_predictions: dict,
    target_n: int = 200,
    seed: int = 2026,
) -> tuple[list[str], dict]:
    """Stratified sample for dual-reader clinical validation."""

    rng = random.Random(seed)
    ids = sorted(gold)
    buckets: dict[str, list[str]] = defaultdict(list)

    for report_id in ids:
        organs = set(gold[report_id].get("organs_present") or [])
        n_findings = int(gold[report_id].get("n_findings") or 0)
        actionable = bool(gold[report_id].get("has_actionable_finding"))
        discordant = False
        for preds in llm_predictions.values():
            if report_id not in preds:
                continue
            t = report_actionability_category(gold_findings(gold[report_id]))
            p = report_actionability_category(prediction_findings(preds[report_id]))
            if t != p:
                discordant = True
                break

        if not organs:
            buckets["none"].append(report_id)
        else:
            if actionable:
                buckets["actionable"].append(report_id)
            if "adnexal" in organs:
                buckets["adnexal"].append(report_id)
            if len(organs) > 1 or n_findings > 1:
                buckets["complex"].append(report_id)
            for organ in organs:
                buckets[f"organ_{organ}"].append(report_id)
            if len(organs) == 1 and n_findings <= 1:
                buckets[f"pure_{next(iter(organs))}"].append(report_id)
        if discordant:
            buckets["model_discordant"].append(report_id)

    # Target mix (~200): enriched positives, guaranteed organ coverage, some negatives
    quotas = {
        "none": 50,
        "pure_kidney": 18,
        "pure_liver": 18,
        "pure_adrenal": 18,
        "pure_pancreas": 18,
        "adnexal": 15,  # all available if fewer
        "complex": 35,
        "actionable": 30,  # may overlap; filled with uniqueness
        "model_discordant": 30,
    }

    selected: list[str] = []
    seen: set[str] = set()
    fill_log: dict[str, int] = {}

    def _take(name: str, pool: list[str], quota: int) -> None:
        candidates = [x for x in pool if x not in seen]
        rng.shuffle(candidates)
        take = candidates[:quota]
        selected.extend(take)
        seen.update(take)
        fill_log[name] = len(take)

    for name, quota in quotas.items():
        _take(name, buckets.get(name, []), quota)

    if len(selected) < target_n:
        # Prefer remaining positives, then any
        remaining_pos = [
            i
            for i in ids
            if i not in seen and gold[i].get("has_in_scope_finding")
        ]
        rng.shuffle(remaining_pos)
        need = target_n - len(selected)
        selected.extend(remaining_pos[:need])
        seen.update(remaining_pos[:need])
        fill_log["topup_positive"] = min(need, len(remaining_pos))
    if len(selected) < target_n:
        remaining = [i for i in ids if i not in seen]
        rng.shuffle(remaining)
        need = target_n - len(selected)
        selected.extend(remaining[:need])
        fill_log["topup_any"] = need

    selected = selected[:target_n]
    rng.shuffle(selected)  # presentation order blinded from stratum blocks

    meta = {
        "target_n": target_n,
        "seed": seed,
        "n_selected": len(selected),
        "fill_log": fill_log,
        "quotas": quotas,
        "organ_coverage": {},
        "n_reference_positive": sum(
            1 for i in selected if gold[i].get("has_in_scope_finding")
        ),
        "n_reference_actionable": sum(
            1 for i in selected if gold[i].get("has_actionable_finding")
        ),
        "n_reference_none": sum(
            1 for i in selected if not gold[i].get("has_in_scope_finding")
        ),
    }
    for organ in IN_SCOPE:
        meta["organ_coverage"][organ] = sum(
            1 for i in selected if organ in (gold[i].get("organs_present") or [])
        )
    return selected, meta


def _blank_reader_row(sample_id: str, report_id: str) -> dict:
    row = {
        "sample_id": sample_id,
        "report_id": report_id,  # needed to open the report locally; no model labels
        "reader_id": "",
        "read_date": "",
    }
    for field in READER_FIELDS:
        row[field] = ""
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "paper_results/radiologist_adjudication",
    )
    parser.add_argument("--target-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gold", type=Path, default=ROOT / "data/gold_labels/gold_labels.jsonl")
    args = parser.parse_args()

    gold = load_jsonl_by_report_id(args.gold)
    llm = {name: load_jsonl_by_report_id(path) for name, path in RUNS.items() if path.exists()}
    selected, sample_meta = sample_radiologist_ids(
        gold, llm, target_n=args.target_n, seed=args.seed
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Blinded worksheets
    reader_fields = ["sample_id", "report_id", "reader_id", "read_date", *READER_FIELDS]
    for reader_name, filename in [
        ("A", "reader_A_worksheet.csv"),
        ("B", "reader_B_worksheet.csv"),
    ]:
        rows = []
        for idx, report_id in enumerate(selected, start=1):
            row = _blank_reader_row(f"RAD-{idx:03d}", report_id)
            row["reader_id"] = reader_name
            rows.append(row)
        path = out / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=reader_fields)
            writer.writeheader()
            writer.writerows(rows)

    # Adjudicator sheet (for disagreements only; pre-filled sample/report ids)
    adj_fields = [
        "sample_id",
        "report_id",
        "disagreement_domain",  # presence|organs|actionability|followup|other
        "reader_A_value",
        "reader_B_value",
        "final_value",
        "adjudicator_id",
        "adjudicator_rationale",
    ]
    with (out / "adjudicator_reconciliation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=adj_fields)
        writer.writeheader()
        # empty template — scoring script can also auto-list disagreements later

    # Admin key (NOT for readers): strata + automated audit columns
    admin_fields = [
        "sample_id",
        "report_id",
        "stratum_primary",
        "reference_has_in_scope_finding",
        "reference_organs",
        "reference_actionability",
        "reference_n_findings",
        "llm_llama_actionability",
        "llm_qwen_actionability",
        "llm_mistral_actionability",
        "model_discordant_vs_reference",
    ]
    admin_rows = []
    for idx, report_id in enumerate(selected, start=1):
        label = gold[report_id]
        organs = sorted(label.get("organs_present") or [])
        if not organs:
            stratum = "none"
        elif "adnexal" in organs:
            stratum = "adnexal"
        elif len(organs) > 1 or int(label.get("n_findings") or 0) > 1:
            stratum = "complex"
        else:
            stratum = organs[0]

        llm_acts = {}
        discordant = False
        for name, short in [
            ("Llama-4 Scout", "llama"),
            ("Qwen3.6-35B", "qwen"),
            ("Mistral-Small-3.2", "mistral"),
        ]:
            preds = llm.get(name, {})
            if report_id in preds:
                act = report_actionability_category(prediction_findings(preds[report_id]))
                llm_acts[short] = act
                if act != label.get("report_actionability_category"):
                    discordant = True
            else:
                llm_acts[short] = ""

        admin_rows.append(
            {
                "sample_id": f"RAD-{idx:03d}",
                "report_id": report_id,
                "stratum_primary": stratum,
                "reference_has_in_scope_finding": bool(label.get("has_in_scope_finding")),
                "reference_organs": ",".join(organs) or "none",
                "reference_actionability": label.get("report_actionability_category"),
                "reference_n_findings": label.get("n_findings"),
                "llm_llama_actionability": llm_acts["llama"],
                "llm_qwen_actionability": llm_acts["qwen"],
                "llm_mistral_actionability": llm_acts["mistral"],
                "model_discordant_vs_reference": discordant,
            }
        )
    with (out / "admin_sample_key.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=admin_fields)
        writer.writeheader()
        writer.writerows(admin_rows)

    (out / "sample_ids.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    (out / "sample_meta.json").write_text(
        json.dumps(
            {
                **sample_meta,
                "blinding": (
                    "Reader worksheets contain report_id for chart access only. "
                    "Do NOT share admin_sample_key.csv with readers before independent reads."
                ),
                "label_status": "blank_awaiting_radiologist_reads",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    protocol = f"""# Radiologist dual-reader adjudication protocol

## Goal
Break circular validation of CLOSE vs the deterministic ACR rule engine by obtaining
**independent radiologist labels** on a stratified subsample (**n={args.target_n}**).

Primary clinical endpoints after dual read + adjudication:
1. Report-level finding presence (in-scope incidental finding yes/no)
2. Organs involved
3. 4-level actionability (`none` / `non_actionable` / `uncertain` / `actionable`)
4. Follow-up required (yes/no)
5. Optional: modality / timeframe

Then compute:
- **Inter-reader Cohen’s κ** (linear κ for actionability)
- Consensus / adjudicated labels
- Model agreement vs radiologist consensus (clinical validity analysis)

## Design
- **Two independent readers** (Reader A, Reader B), ideally radiology residents/fellows
  supervised by an attending
- **Blinded** to CLOSE model outputs and to ACR rule-engine labels
- **Adjudicator** (attending) resolves disagreements only
- Stratified enrichment for positives, adnexal, complex, actionable, and model-discordant cases
  (see `sample_meta.json`); still includes ~{sample_meta.get('n_reference_none', '50')} reference-negative reports

### Current sample mix (ACR-reference audit only; not reader truth)
- Reference positives: {sample_meta.get('n_reference_positive')}
- Reference actionable: {sample_meta.get('n_reference_actionable')}
- Reference none: {sample_meta.get('n_reference_none')}
- Organ coverage: {json.dumps(sample_meta.get('organ_coverage'))}

## Files
| File | Who uses it |
|---|---|
| `reader_A_worksheet.csv` | Reader A |
| `reader_B_worksheet.csv` | Reader B |
| `adjudicator_reconciliation.csv` | Attending (after disagreement list generated) |
| `admin_sample_key.csv` | Study admin only (strata + automated audit columns) |
| `PROTOCOL.md` | This document |

## Labeling rules (ACR-criteria clinical read)
Readers should apply **ACR incidental finding white-paper principles** for kidney, liver,
adrenal, pancreas, and adnexal findings on CT.

Allowed values:
- `finding_present`: `yes` / `no`
- `organs_present`: comma-separated from `kidney,liver,adrenal,pancreas,adnexal`
- `actionability`: `none` | `non_actionable` | `uncertain` | `actionable`
  - use `none` only if no in-scope incidental finding
- `followup_required`: `yes` / `no`
- `confidence`: `low` / `medium` / `high`

Guidance:
- Negated statements (“no lesions”) → not a finding
- Simple benign cysts / classic lipid-rich adenoma without concerning features → often `non_actionable`
- Indeterminate lesions needing characterization → `uncertain` or `actionable` per ACR
- Explicit recommended follow-up in the report supports `followup_required=yes`

## Suggested 2-week timeline
| Day | Task |
|---|---|
| 1 | Kickoff + 10-case calibration set (discuss disagreements; not counted) |
| 2–6 | Reader A completes 100 cases |
| 2–6 | Reader B completes 100 cases (parallel) |
| 7–10 | Both finish remaining 100 |
| 11 | Auto-generate disagreement list (`score_radiologist_adjudication.py --list-disagreements`) |
| 12–13 | Attending adjudication |
| 14 | Run scoring: inter-reader κ + model-vs-consensus tables |

Rough effort: ~200 reports × ~3–5 min ≈ **10–17 hours/reader**, plus adjudication.

## Blinding / PHI
- Readers may use `report_id` to open the local report viewer
- Do **not** provide `admin_sample_key.csv` to readers before independent reads
- Do not paste PHI into `comments`

## After labels are complete
```bash
# Inter-reader κ + disagreement list
PYTHONPATH=src python scripts/score_radiologist_adjudication.py \\
  --reader-a paper_results/radiologist_adjudication/reader_A_worksheet.csv \\
  --reader-b paper_results/radiologist_adjudication/reader_B_worksheet.csv \\
  --out-dir paper_results/radiologist_adjudication/results

# After adjudicator finishes consensus file:
PYTHONPATH=src python scripts/score_radiologist_adjudication.py \\
  --reader-a ... --reader-b ... \\
  --consensus paper_results/radiologist_adjudication/consensus_labels.csv \\
  --out-dir paper_results/radiologist_adjudication/results
```

## What this enables in the paper
- Claim **clinical agreement** with radiologist consensus (not only guideline-engine concordance)
- Report inter-reader reliability (κ) as a validity check on the reference standard itself
- Compare LLM-only vs hybrid against a non-circular target
"""
    (out / "PROTOCOL.md").write_text(protocol, encoding="utf-8")

    # Empty consensus template for after adjudication
    consensus_fields = ["sample_id", "report_id", *READER_FIELDS, "consensus_source"]
    with (out / "consensus_labels_TEMPLATE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=consensus_fields)
        writer.writeheader()
        for idx, report_id in enumerate(selected, start=1):
            row = {k: "" for k in consensus_fields}
            row["sample_id"] = f"RAD-{idx:03d}"
            row["report_id"] = report_id
            row["consensus_source"] = ""  # agree_AB | adjudicated
            writer.writerow(row)

    print(
        json.dumps(
            {
                "out_dir": str(out),
                "n": len(selected),
                "n_reference_positive": sample_meta["n_reference_positive"],
                "n_reference_none": sample_meta["n_reference_none"],
                "organ_coverage": sample_meta["organ_coverage"],
                "worksheets": [
                    "reader_A_worksheet.csv",
                    "reader_B_worksheet.csv",
                    "adjudicator_reconciliation.csv",
                    "admin_sample_key.csv",
                    "PROTOCOL.md",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
