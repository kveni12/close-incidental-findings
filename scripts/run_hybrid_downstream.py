#!/usr/bin/env python
"""Build full hybrid CLOSE results without re-calling the LLM.

1. Re-applies ACR actionability rules to LLM Capture outputs (optional).
2. Supplements rule-detected findings the LLM missed (optional).
3. Re-runs Link and Orchestrate on the updated Capture findings.

Example:
  PYTHONPATH=src python scripts/run_hybrid_downstream.py \\
    --input outputs/model_comparison_9500762/qwen/close_results.jsonl \\
    --output outputs/model_comparison_9500762/qwen/close_results_hybrid.jsonl
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
from close_framework.config.settings import CloseSettings, InferenceMode, load_settings  # noqa: E402
from close_framework.data.loaders import iter_reports  # noqa: E402
from close_framework.data.schemas import CaptureResult, RadiologyReport, WorkflowResult  # noqa: E402
from close_framework.orchestrator.central_orchestrator import CentralOrchestrator  # noqa: E402


def _report_context(agent: CaptureAgent, report: RadiologyReport) -> dict:
    return agent._report_context(report)  # noqa: SLF001


def _recalibrate_capture(
    agent: CaptureAgent,
    capture_result: CaptureResult,
    report: RadiologyReport,
    supplement: bool,
) -> tuple[CaptureResult, int, int]:
    findings = list(capture_result.findings or [])
    before = len(findings)
    context = _report_context(agent, report) if report.report_text else {}

    recalibrated = [agent.recalibrate_finding(finding, context) for finding in findings]
    n_recalibrated = sum(
        1 for old, new in zip(findings, recalibrated) if old.actionability != new.actionability
    )
    findings = recalibrated

    n_supplemented = 0
    if supplement and report.report_text:
        full_context = _report_context(agent, report)
        findings = agent._supplement_with_rule_findings(report, full_context, findings, 1)  # noqa: SLF001
        n_supplemented = max(0, len(findings) - before)

    capture_result.findings = findings
    return capture_result, n_recalibrated, n_supplemented


def run_hybrid_downstream(
    input_path: Path,
    output_path: Path,
    data_dir: Path,
    recalibrate: bool = True,
    supplement: bool = True,
) -> dict:
    reports_by_id = {
        report.report_id: report for report in iter_reports(data_dir)
    }

    capture_agent = CaptureAgent(
        CloseSettings(
            inference_mode=InferenceMode.RULE_BASED_ONLY,
            recalibrate_llm_capture=recalibrate,
            supplement_rule_findings=supplement,
        )
    )
    orchestrator = CentralOrchestrator(settings=load_settings(data_dir=str(data_dir)))

    n_recalibrated = 0
    n_supplemented = 0
    n_recommendations = 0
    out_rows: list[str] = []

    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            result = WorkflowResult.model_validate(payload)
            report = reports_by_id.get(result.report_id)
            if report is None:
                report = RadiologyReport(
                    report_id=result.report_id,
                    patient_id=result.patient_id,
                    report_text="",
                )

            capture_result = result.capture_result
            if recalibrate or supplement:
                capture_result, rec_delta, sup_delta = _recalibrate_capture(
                    capture_agent, capture_result, report, supplement=supplement
                )
                n_recalibrated += rec_delta
                n_supplemented += sup_delta

            hybrid = orchestrator.run_from_capture(report, capture_result)
            n_recommendations += len(hybrid.follow_up_recommendations or {})
            out_rows.append(hybrid.model_dump_json())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_rows) + "\n", encoding="utf-8")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "reports": len(out_rows),
        "findings_recalibrated_actionability": n_recalibrated,
        "findings_supplemented": n_supplemented,
        "follow_up_recommendations": n_recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--no-recalibrate", action="store_true")
    parser.add_argument("--no-supplement", action="store_true")
    args = parser.parse_args()

    meta = run_hybrid_downstream(
        args.input,
        args.output,
        args.data_dir,
        recalibrate=not args.no_recalibrate,
        supplement=not args.no_supplement,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
