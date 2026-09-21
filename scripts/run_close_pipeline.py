#!/usr/bin/env python
"""Run the CLOSE pipeline over a local data directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.config.logging_config import configure_logging  # noqa: E402
from close_framework.config.settings import load_settings  # noqa: E402
from close_framework.evaluation.metrics import descriptive_summary_metrics  # noqa: E402
from close_framework.orchestrator.central_orchestrator import CentralOrchestrator  # noqa: E402
from close_framework.outputs.serializers import save_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLOSE incidental findings pipeline")
    parser.add_argument("--data-dir", default="data", help="Directory containing report files")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output files")
    parser.add_argument("--config", default=None, help="YAML/JSON experiment config")
    parser.add_argument("--limit", type=int, default=None, help="Optional report limit for smoke tests")
    parser.add_argument(
        "--report-ids-from",
        default=None,
        help="JSONL of prior results; restrict this run to the same report_id cohort",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write intermediate outputs every N reports (0 disables)",
    )
    args = parser.parse_args()

    configure_logging()
    settings = load_settings(args.data_dir, args.output_dir, args.config)
    orchestrator = CentralOrchestrator(settings=settings)
    from close_framework.data.loaders import iter_reports

    cohort_ids = None
    if args.report_ids_from:
        import json

        cohort_ids = set()
        with open(args.report_ids_from, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    cohort_ids.add(str(json.loads(line)["report_id"]))

    results = []
    for index, report in enumerate(iter_reports(settings.data_dir)):
        if cohort_ids is not None:
            if report.report_id not in cohort_ids:
                continue
        elif args.limit is not None and index >= args.limit:
            break
        results.append(orchestrator.run_report(report))
        if cohort_ids is not None and len(results) >= len(cohort_ids):
            break
        if args.checkpoint_every and len(results) % args.checkpoint_every == 0:
            save_outputs(results, settings.output_dir)
            print(f"Checkpoint: {len(results)} reports written to {settings.output_dir}", flush=True)

    save_outputs(results, settings.output_dir)
    metrics = descriptive_summary_metrics(results)
    print("CLOSE Pipeline Summary")
    print(f"Reports processed: {metrics['reports_processed']}")
    print(f"Findings detected: {metrics['findings_detected']}")
    print(f"In-scope findings: {metrics['in_scope_findings']}")
    print(f"Out-of-scope incidental findings: {metrics['out_of_scope_incidental_findings']}")
    print(f"Actionable findings: {metrics['actionable_findings']}")
    print(f"Uncertain findings: {metrics['uncertain_findings']}")
    print(f"Routed to human review: {metrics['human_review_required']}")
    print(f"Recommendations generated: {metrics['recommendations_generated']}")
    if "capture_mode_counts" in metrics:
        print(f"Capture modes: {metrics['capture_mode_counts']}")
    if "llm_error_counts" in metrics:
        print(f"LLM errors: {metrics['llm_error_counts']}")


if __name__ == "__main__":
    main()
