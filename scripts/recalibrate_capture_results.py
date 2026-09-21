#!/usr/bin/env python
"""Post-process LLM Capture outputs with ACR-aligned rule recalibration.

Does not re-call the LLM. Re-applies the same actionability / urgency heuristics
used by the rule-based reference standard to each extracted finding, and optionally
adds rule-detected findings the LLM missed.

Example:
  PYTHONPATH=src python scripts/recalibrate_capture_results.py \\
    --input outputs/model_comparison_9500762/qwen/close_results.jsonl \\
    --output outputs/model_comparison_9500762/qwen/close_results_hybrid.jsonl \\
    --supplement
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

from close_framework.agents.capture_agent import CaptureAgent  # noqa: E402
from close_framework.config.settings import CloseSettings, InferenceMode  # noqa: E402
from close_framework.data.loaders import iter_reports  # noqa: E402
from close_framework.data.schemas import RadiologyReport, WorkflowResult  # noqa: E402


def _report_context_from_text(text: str) -> dict:
    agent = CaptureAgent(CloseSettings(inference_mode=InferenceMode.RULE_BASED_ONLY))
    stub = RadiologyReport(report_id="stub", report_text=text)
    return agent._report_context(stub)  # noqa: SLF001 - intentional reuse of rule context


def recalibrate_file(
    input_path: Path,
    output_path: Path,
    supplement: bool = False,
    reports_by_id: dict | None = None,
) -> dict:
    agent = CaptureAgent(
        CloseSettings(
            inference_mode=InferenceMode.RULE_BASED_ONLY,
            recalibrate_llm_capture=True,
            supplement_rule_findings=supplement,
        )
    )
    n_recalibrated = 0
    n_supplemented = 0
    out_rows: list[str] = []

    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            result = WorkflowResult.model_validate(payload)
            report_text = ""
            if reports_by_id and result.report_id in reports_by_id:
                report_text = reports_by_id[result.report_id].get("report_text") or ""
            context = _report_context_from_text(report_text) if report_text else {}

            findings = list(result.capture_result.findings or [])
            before = len(findings)
            recalibrated = [agent.recalibrate_finding(f, context) for f in findings]
            n_recalibrated += sum(
                1
                for old, new in zip(findings, recalibrated)
                if old.actionability != new.actionability
            )
            findings = recalibrated

            if supplement and report_text:
                stub = RadiologyReport(
                    report_id=result.report_id,
                    patient_id=result.patient_id,
                    report_text=report_text,
                )
                full_context = agent._report_context(stub)  # noqa: SLF001
                findings = agent._supplement_with_rule_findings(stub, full_context, findings, 1)  # noqa: SLF001
                n_supplemented += max(0, len(findings) - before)

            result.capture_result.findings = findings
            out_rows.append(result.model_dump_json())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_rows) + "\n", encoding="utf-8")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "reports": len(out_rows),
        "findings_recalibrated_actionability": n_recalibrated,
        "findings_supplemented": n_supplemented,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()

    reports_by_id = {}
    for report in iter_reports(args.data_dir):
        reports_by_id[report.report_id] = {"report_text": report.report_text}

    meta = recalibrate_file(args.input, args.output, supplement=args.supplement, reports_by_id=reports_by_id)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
