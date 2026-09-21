#!/usr/bin/env python
"""Run descriptive end-to-end CLOSE evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.config.settings import load_settings  # noqa: E402
from close_framework.evaluation.evaluation_runner import run_descriptive_evaluation  # noqa: E402
from close_framework.orchestrator.central_orchestrator import CentralOrchestrator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLOSE end-to-end pipeline")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    settings = load_settings(args.data_dir, "outputs")
    orchestrator = CentralOrchestrator(settings=settings)
    from close_framework.data.loaders import load_reports

    reports = load_reports(settings.data_dir)[: args.limit]
    metrics = run_descriptive_evaluation(orchestrator.run_reports(reports))
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
