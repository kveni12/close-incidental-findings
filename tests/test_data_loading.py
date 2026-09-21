"""Tests for report loading."""

from __future__ import annotations

from close_framework.data.loaders import load_report_file


def test_pipe_text_export_with_multiline_reports(tmp_path):
    path = tmp_path / "reports.txt"
    path.write_text(
        "EMPI|EPIC_PMRN|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Text\n"
        "1|p1|m1|r1|4/15/2025 8:10:00 AM|CT ABDOMEN|CT ABDOMEN\n"
        "FINDINGS: No suspicious focal liver lesions.\n"
        "[report_end]\n"
        "2|p2|m2|r2|4/16/2025 8:10:00 AM|CT ABDOMEN|CT ABDOMEN\n"
        "FINDINGS: Indeterminate 1.4 cm left renal lesion.\n"
        "[report_end]\n",
        encoding="utf-8",
    )

    reports = load_report_file(path)

    assert len(reports) == 2
    assert reports[0].report_id == "r1"
    assert reports[1].patient_id == "p2"
    assert "left renal lesion" in reports[1].report_text
