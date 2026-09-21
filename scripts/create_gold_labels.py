#!/usr/bin/env python
"""Create reference-standard Capture labels via rule-based ACR Capture.

This is a deterministic guideline-engine reference standard for retrospective
evaluation when expert chart-review labels are unavailable. It is NOT
radiologist-adjudicated clinical ground truth.

Example:
  PYTHONPATH=src python scripts/create_gold_labels.py \\
    --data-dir data \\
    --report-ids-from outputs/model_comparison_9283993/llama/close_results.jsonl \\
    --output-dir data/gold_labels
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.agents.capture_agent import CaptureAgent  # noqa: E402
from close_framework.config.settings import CloseSettings, InferenceMode  # noqa: E402
from close_framework.data.loaders import iter_reports  # noqa: E402
from close_framework.evaluation.gold_eval import (  # noqa: E402
    IN_SCOPE,
    organ_actionability_categories,
    report_actionability_category,
    report_urgency_category,
)

LABEL_SOURCE = "rule_based_acr_capture"
LABEL_VERSION = "1.1"


def _load_report_ids(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    ids: Set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            ids.add(str(payload.get("report_id")))
    return ids


def _label_from_capture(report_id: str, patient_id: Optional[str], findings: List[dict]) -> dict:
    in_scope = [f for f in findings if str(f.get("organ_system")) in IN_SCOPE]
    organs = sorted({str(f["organ_system"]) for f in in_scope})
    actionable_organs = sorted(
        {str(f["organ_system"]) for f in in_scope if f.get("actionability") == "actionable"}
    )
    compact_findings = [
        {
            "organ_system": f.get("organ_system"),
            "finding_type": f.get("finding_type"),
            "actionability": f.get("actionability"),
            "urgency_tier": f.get("urgency_tier"),
            "size_mm": f.get("size_mm"),
            "evidence_span": f.get("evidence_span"),
        }
        for f in in_scope
    ]
    return {
        "report_id": report_id,
        "patient_id": patient_id,
        "label_source": LABEL_SOURCE,
        "label_version": LABEL_VERSION,
        "has_in_scope_finding": bool(organs),
        "has_actionable_finding": bool(actionable_organs),
        "has_uncertain_finding": any(f.get("actionability") == "uncertain" for f in in_scope),
        "organs_present": organs,
        "actionable_organs": actionable_organs,
        "report_actionability_category": report_actionability_category(in_scope),
        "report_urgency_category": report_urgency_category(in_scope),
        "organ_actionability_categories": organ_actionability_categories(in_scope),
        "n_findings": len(in_scope),
        "findings": compact_findings,
    }


def create_gold_labels(
    data_dir: Path,
    output_dir: Path,
    report_ids: Optional[Set[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    agent = CaptureAgent(CloseSettings(inference_mode=InferenceMode.RULE_BASED_ONLY))
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "gold_labels.jsonl"
    json_path = output_dir / "gold_labels.json"
    csv_path = output_dir / "gold_labels.csv"
    meta_path = output_dir / "gold_labels_meta.json"

    labels: List[dict] = []
    seen: Set[str] = set()
    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        for report in iter_reports(data_dir):
            if report_ids is not None and report.report_id not in report_ids:
                continue
            if report.report_id in seen:
                continue
            capture = agent.run(report)
            finding_dicts = [f.model_dump() for f in capture.findings]
            label = _label_from_capture(report.report_id, report.patient_id, finding_dicts)
            labels.append(label)
            seen.add(report.report_id)
            jsonl_handle.write(json.dumps(label, sort_keys=True) + "\n")
            if limit is not None and len(labels) >= limit:
                break
            if report_ids is not None and len(labels) >= len(report_ids):
                break

    json_path.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "report_id",
                "patient_id",
                "has_in_scope_finding",
                "has_actionable_finding",
                "has_uncertain_finding",
                "report_actionability_category",
                "report_urgency_category",
                "organs_present",
                "actionable_organs",
                "n_findings",
                "label_source",
                "label_version",
            ],
        )
        writer.writeheader()
        for label in labels:
            writer.writerow(
                {
                    "report_id": label["report_id"],
                    "patient_id": label.get("patient_id") or "",
                    "has_in_scope_finding": int(label["has_in_scope_finding"]),
                    "has_actionable_finding": int(label["has_actionable_finding"]),
                    "has_uncertain_finding": int(label["has_uncertain_finding"]),
                    "report_actionability_category": label["report_actionability_category"],
                    "report_urgency_category": label["report_urgency_category"],
                    "organs_present": "|".join(label["organs_present"]),
                    "actionable_organs": "|".join(label["actionable_organs"]),
                    "n_findings": label["n_findings"],
                    "label_source": label["label_source"],
                    "label_version": label["label_version"],
                }
            )

    meta = {
        "label_source": LABEL_SOURCE,
        "label_version": LABEL_VERSION,
        "n_reports": len(labels),
        "n_with_in_scope_finding": sum(l["has_in_scope_finding"] for l in labels),
        "n_with_actionable_finding": sum(l["has_actionable_finding"] for l in labels),
        "organ_prevalence": {
            organ: sum(organ in l["organs_present"] for l in labels) for organ in IN_SCOPE
        },
        "report_actionability_distribution": dict(
            Counter(l["report_actionability_category"] for l in labels)
        ),
        "report_urgency_distribution": dict(Counter(l["report_urgency_category"] for l in labels)),
        "missing_requested_ids": sorted((report_ids or set()) - seen),
        "disclaimer": (
            "Deterministic rule-based ACR Capture reference standard. "
            "Not radiologist-adjudicated clinical ground truth."
        ),
        "paths": {
            "jsonl": str(jsonl_path),
            "json": str(json_path),
            "csv": str(csv_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "gold_labels")
    parser.add_argument(
        "--report-ids-from",
        type=Path,
        default=None,
        help="JSONL of CLOSE results whose report_ids define the labeled cohort",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    meta = create_gold_labels(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_ids=_load_report_ids(args.report_ids_from),
        limit=args.limit,
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
