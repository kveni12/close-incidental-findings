#!/usr/bin/env python
"""Evaluate Capture Agent with descriptive metrics or future labels."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.agents.capture_agent import CaptureAgent  # noqa: E402
from close_framework.data.schemas import RadiologyReport  # noqa: E402


def main() -> None:
    agent = CaptureAgent()
    report = RadiologyReport(
        report_id="example",
        report_text="CT abdomen shows an indeterminate 1.5 cm left renal lesion. MRI is recommended.",
    )
    result = agent.run(report)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
