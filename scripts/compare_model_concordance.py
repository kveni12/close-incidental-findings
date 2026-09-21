#!/usr/bin/env python
"""Proxy accuracy / concordance between CLOSE model runs (no gold labels required).

True sensitivity/PPV needs a gold label file. Until then, this script measures:
  - report-level presence agreement (any in-scope finding?)
  - organ-set Jaccard vs a reference model (default: llama)
  - actionable-flag agreement on reports where both models extracted something

Example:
  PYTHONPATH=src python scripts/compare_model_concordance.py \\
    --reference outputs/model_comparison_9283993/llama/close_results.jsonl \\
    --candidate outputs/model_comparison_<job>/qwen/close_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


ORGANS = {"kidney", "liver", "adrenal", "pancreas", "adnexal"}


def _load(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows[str(payload.get("report_id"))] = payload
    return rows


def _organs(result: dict) -> Set[str]:
    findings = (result.get("capture_result") or {}).get("findings") or []
    return {
        str(f.get("organ_system"))
        for f in findings
        if str(f.get("organ_system")) in ORGANS
    }


def _has_actionable(result: dict) -> bool:
    findings = (result.get("capture_result") or {}).get("findings") or []
    return any(f.get("actionability") == "actionable" for f in findings)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def concordance(reference: Dict[str, dict], candidate: Dict[str, dict]) -> dict:
    ids = sorted(set(reference) & set(candidate))
    presence_agree = 0
    actionable_agree = 0
    actionable_both_present = 0
    jaccards = []
    for report_id in ids:
        ref = reference[report_id]
        cand = candidate[report_id]
        ref_organs = _organs(ref)
        cand_organs = _organs(cand)
        ref_pos = bool(ref_organs)
        cand_pos = bool(cand_organs)
        presence_agree += int(ref_pos == cand_pos)
        jaccards.append(_jaccard(ref_organs, cand_organs))
        if ref_pos or cand_pos:
            actionable_both_present += 1
            actionable_agree += int(_has_actionable(ref) == _has_actionable(cand))
    n = len(ids) or 1
    return {
        "shared_reports": len(ids),
        "presence_agreement": presence_agree / n,
        "mean_organ_jaccard": (sum(jaccards) / len(jaccards)) if jaccards else None,
        "actionable_agreement_when_any_finding": (
            actionable_agree / actionable_both_present if actionable_both_present else None
        ),
        "reference_only_reports": len(set(reference) - set(candidate)),
        "candidate_only_reports": len(set(candidate) - set(reference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    metrics = concordance(_load(args.reference), _load(args.candidate))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
