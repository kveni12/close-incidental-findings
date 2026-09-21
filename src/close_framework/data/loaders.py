"""Load local radiology report files into normalized schemas."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from close_framework.data.preprocessing import normalize_report_text
from close_framework.data.schemas import RadiologyReport

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".txt", ".csv", ".json", ".jsonl"}


def load_reports(data_dir: str | Path) -> List[RadiologyReport]:
    """Load every supported file from a directory, logging and continuing on errors."""

    return list(iter_reports(data_dir))


def iter_reports(data_dir: str | Path) -> Iterator[RadiologyReport]:
    """Yield supported reports from a directory without materializing the full dataset."""

    directory = Path(data_dir)
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        try:
            yield from iter_report_file(path)
        except Exception as exc:  # noqa: BLE001 - loader must not abort a batch
            LOGGER.exception("Failed to load %s: %s", path, exc)


def load_report_file(path: str | Path) -> List[RadiologyReport]:
    """Load one supported report file."""

    return list(iter_report_file(path))


def iter_report_file(path: str | Path) -> Iterator[RadiologyReport]:
    """Yield reports from one supported report file."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        yield from _load_jsonl(file_path)
        return
    if suffix == ".json":
        yield from _load_json(file_path)
        return
    if suffix == ".csv":
        yield from _load_delimited(file_path, delimiter=",")
        return
    if suffix == ".txt":
        yield from _load_text_export(file_path)
        return
    raise ValueError(f"Unsupported report file type: {file_path}")


def _load_text_export(path: Path) -> Iterator[RadiologyReport]:
    """Load pipe-delimited institutional-style exports with multiline reports and [report_end]."""

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        header_line = handle.readline().rstrip("\n")
        if "|" not in header_line:
            yield _record_from_free_text(path, header_line + handle.read(), row_number=1)
            return

        headers = [header.strip() for header in header_line.split("|")]
        row_buffer: List[str] = []
        row_number = 1
        for line in handle:
            if not line.strip() and not row_buffer:
                continue
            row_buffer.append(line.rstrip("\n"))
            if "[report_end]" not in line:
                continue
            row_number += 1
            raw_record = "\n".join(row_buffer).replace("[report_end]", "").strip()
            row_buffer = []
            try:
                fields = raw_record.split("|", maxsplit=len(headers) - 1)
                if len(fields) < len(headers):
                    raise ValueError(f"Expected {len(headers)} fields, got {len(fields)}")
                yield _record_from_mapping(dict(zip(headers, fields)), path, row_number)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to parse %s row %s: %s", path, row_number, exc)
                yield _record_from_free_text(path, raw_record, row_number=row_number, error=str(exc))

        if row_buffer:
            raw_record = "\n".join(row_buffer).strip()
            LOGGER.warning("File %s ended with an unterminated report", path)
            yield _record_from_free_text(path, raw_record, row_number=row_number + 1, error="unterminated_report")


def _load_delimited(path: Path, delimiter: str) -> Iterator[RadiologyReport]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row_number, row in enumerate(reader, start=2):
            try:
                yield _record_from_mapping(row, path, row_number)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to parse %s row %s: %s", path, row_number, exc)
                yield _record_from_free_text(path, json.dumps(row), row_number=row_number, error=str(exc))


def _load_jsonl(path: Path) -> Iterator[RadiologyReport]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield _record_from_mapping(json.loads(line), path, row_number)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to parse %s line %s: %s", path, row_number, exc)
                yield _record_from_free_text(path, line, row_number=row_number, error=str(exc))


def _load_json(path: Path) -> Iterator[RadiologyReport]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    records: Iterable[Dict[str, Any]]
    if isinstance(payload, list):
        records = payload
    else:
        records = payload.get("reports", [payload])
    for row_number, row in enumerate(records, start=1):
        try:
            yield _record_from_mapping(row, path, row_number)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to parse %s record %s: %s", path, row_number, exc)
            yield _record_from_free_text(path, json.dumps(row), row_number=row_number, error=str(exc))


def _record_from_mapping(row: Dict[str, Any], path: Path, row_number: int) -> RadiologyReport:
    row = {str(key).strip(): value for key, value in row.items()}
    report_text = _first_present(row, ["Report_Text", "Report Text", "report_text", "text", "NoteTXT"]) or ""
    report_id = str(_first_present(row, ["Report_Number", "report_id", "NoteID", "id"]) or "")
    patient_id = _first_present(row, ["EPIC_PMRN", "EMPI", "MRN", "patient_id", "Patient MRN"])
    accession = _first_present(row, ["Report_Number", "accession_number", "acc"])
    exam_type = _first_present(row, ["Report_Description", "Report_Type", "exam_type", "modality"])
    report_date = _parse_datetime(_first_present(row, ["Report_Date_Time", "report_date", "Date", "Exam Started Date"]))
    metadata = {str(key): value for key, value in row.items() if key not in {"Report_Text", "Report Text", "report_text"}}
    metadata.update({"source_file": str(path), "source_row": row_number})
    normalized_text = normalize_report_text(str(report_text))
    if not report_id:
        report_id = _stable_id(str(path), row_number, normalized_text)
    return RadiologyReport(
        report_id=report_id,
        patient_id=str(patient_id) if patient_id not in {None, ""} else None,
        accession_number=str(accession) if accession not in {None, ""} else None,
        report_text=normalized_text,
        exam_type=str(exam_type) if exam_type not in {None, ""} else None,
        report_date=report_date,
        clinical_setting=_infer_clinical_setting(row, normalized_text),
        metadata=metadata,
    )


def _record_from_free_text(path: Path, text: str, row_number: int, error: Optional[str] = None) -> RadiologyReport:
    metadata: Dict[str, Any] = {"source_file": str(path), "source_row": row_number}
    if error:
        metadata["parse_error"] = error
    normalized_text = normalize_report_text(text)
    return RadiologyReport(
        report_id=_stable_id(str(path), row_number, normalized_text),
        report_text=normalized_text,
        clinical_setting="unknown",
        metadata=metadata,
    )


def _first_present(row: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return value
    return None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value)
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _infer_clinical_setting(row: Dict[str, Any], text: str) -> str:
    haystack = " ".join(str(value) for value in row.values()) + " " + text
    lower = haystack.lower()
    if "emergency department" in lower or " ed " in f" {lower} ":
        return "ED"
    if "inpatient" in lower or "admission" in lower or "hospitalized" in lower:
        return "inpatient"
    if "outpatient" in lower or "clinic" in lower:
        return "outpatient"
    return "unknown"


def _stable_id(path: str, row_number: int, text: str) -> str:
    digest = hashlib.sha1(f"{path}:{row_number}:{text[:500]}".encode("utf-8")).hexdigest()[:12]
    return f"report-{digest}"
