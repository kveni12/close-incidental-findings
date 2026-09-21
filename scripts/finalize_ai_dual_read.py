#!/usr/bin/env python
"""Merge dual AI-reader label batches into worksheets, score κ, build consensus, eval models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BASE = ROOT / "paper_results/radiologist_adjudication"
AI = BASE / "ai_dual_read"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_batches(reader: str) -> dict:
    rows = {}
    for i in range(4):
        path = AI / f"labels_reader{reader}_batch_{i}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            payload = json.loads(line)
            rows[str(payload["report_id"])] = payload
    return rows


def _write_worksheet(reader: str, labels: dict, template: Path, out_path: Path) -> None:
    with template.open(encoding="utf-8", newline="") as handle:
        fieldnames = list(csv.DictReader(handle).fieldnames or [])
        handle.seek(0)
        template_rows = list(csv.DictReader(handle))
    out_rows = []
    for row in template_rows:
        rid = row["report_id"]
        lab = labels[rid]
        present = bool(lab.get("finding_present"))
        action = lab.get("actionability") or ("none" if not present else "uncertain")
        follow = bool(lab.get("followup_required"))
        organs = lab.get("organs_present") or []
        if not present:
            action = "none"
            follow = False
            organs = []
        out = dict(row)
        out.update(
            {
                "reader_id": f"AI-{reader}",
                "finding_present": "yes" if present else "no",
                "organs_present": ",".join(organs),
                "actionability": action,
                "followup_required": "yes" if follow else "no",
                "confidence": lab.get("confidence") or "medium",
                "comments": lab.get("rationale") or "",
            }
        )
        out_rows.append(out)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def _build_consensus(path_a: Path, path_b: Path, out_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import score_radiologist_adjudication as score  # noqa: E402

    a = score._complete(score.load_reader_csv(path_a))
    b = score._complete(score.load_reader_csv(path_b))
    auto = score.build_auto_consensus(a, b)

    consensus = dict(auto)
    rank = {"none": 0, "non_actionable": 1, "uncertain": 2, "actionable": 3}
    for rid in sorted(set(a) & set(b)):
        if rid in consensus:
            continue
        aa, bb = a[rid], b[rid]
        act = aa["actionability"] if rank[aa["actionability"]] >= rank[bb["actionability"]] else bb["actionability"]
        present = bool(aa["finding_present"] or bb["finding_present"])
        if act == "none":
            present = False
        follow = bool(aa["followup_required"] or bb["followup_required"]) if act == "actionable" else False
        organs = sorted(set(aa["organs_present"]) | set(bb["organs_present"])) if present else []
        consensus[rid] = {
            "sample_id": aa.get("sample_id"),
            "report_id": rid,
            "finding_present": present,
            "actionability": act if present else "none",
            "followup_required": follow if present else False,
            "organs_present": organs,
            "consensus_source": "ai_adjudicated_max_severity",
        }

    fieldnames = [
        "sample_id",
        "report_id",
        "finding_present",
        "organs_present",
        "actionability",
        "followup_required",
        "consensus_source",
    ]
    rows = []
    for rid, row in sorted(consensus.items(), key=lambda x: x[1].get("sample_id") or x[0]):
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "report_id": rid,
                "finding_present": "yes" if row["finding_present"] else "no",
                "organs_present": ",".join(row.get("organs_present") or []),
                "actionability": row["actionability"],
                "followup_required": "yes" if row["followup_required"] else "no",
                "consensus_source": row.get("consensus_source", "agree_AB"),
            }
        )
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    labels_a = _load_batches("A")
    labels_b = _load_batches("B")
    assert len(labels_a) == 200, len(labels_a)
    assert len(labels_b) == 200, len(labels_b)

    path_a = BASE / "reader_A_worksheet.csv"
    path_b = BASE / "reader_B_worksheet.csv"
    # keep originals as blank templates already used; overwrite with AI dual-read fills
    _write_worksheet("A", labels_a, path_a, path_a)
    _write_worksheet("B", labels_b, path_b, path_b)

    results = BASE / "results"
    results.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts/score_radiologist_adjudication.py"),
            "--reader-a",
            str(path_a),
            "--reader-b",
            str(path_b),
            "--out-dir",
            str(results),
        ]
    )

    consensus_path = BASE / "consensus_labels.csv"
    _build_consensus(path_a, path_b, consensus_path)

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts/score_radiologist_adjudication.py"),
            "--reader-a",
            str(path_a),
            "--reader-b",
            str(path_b),
            "--consensus",
            str(consensus_path),
            "--out-dir",
            str(results),
        ]
    )

    note = BASE / "AI_DUAL_READ_DISCLAIMER.md"
    note.write_text(
        """# AI dual-read disclaimer

The filled Reader A / Reader B worksheets were produced by **two independent AI readers**
applying ACR-style criteria to report excerpts — **not** by radiologists.

This:
- breaks circularity vs the deterministic Capture rule-engine code path
- yields inter-reader κ and model-vs-consensus tables as an interim analysis

This does **NOT** constitute radiologist chart review or clinical validation.
For a paper claim of clinical validity, replace these with human dual-reads using the same worksheets.
""",
        encoding="utf-8",
    )
    print(json.dumps({"consensus": str(consensus_path), "results": str(results)}, indent=2))


if __name__ == "__main__":
    main()
