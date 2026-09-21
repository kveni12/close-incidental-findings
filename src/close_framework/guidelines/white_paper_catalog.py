"""Catalog of ACR incidental-findings white papers used for RAG and engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class WhitePaperDoc:
    """One guideline PDF in incidental_findings_white_papers/."""

    filename: str
    title: str
    publication_year: int
    organs: tuple[str, ...]
    role: str  # primary | supporting | out_of_scope
    notes: str = ""


WHITE_PAPERS: List[WhitePaperDoc] = [
    WhitePaperDoc(
        "renal.pdf",
        "Management of the Incidental Renal Mass on CT (ACR IFC)",
        2018,
        ("kidney",),
        "primary",
    ),
    WhitePaperDoc(
        "adrenal.pdf",
        "Management of Incidental Adrenal Masses (ACR IFC)",
        2017,
        ("adrenal",),
        "primary",
    ),
    WhitePaperDoc(
        "liver.pdf",
        "Management of Incidental Liver Lesions on CT (ACR IFC)",
        2017,
        ("liver",),
        "primary",
    ),
    WhitePaperDoc(
        "pancreas.pdf",
        "Management of Incidental Pancreatic Cysts (ACR IFC)",
        2017,
        ("pancreas",),
        "primary",
    ),
    WhitePaperDoc(
        "adnexal.pdf",
        "Management of Incidental Adnexal Findings on CT and MRI (ACR IFC)",
        2020,
        ("adnexal",),
        "primary",
    ),
    WhitePaperDoc(
        "PIIS1546144010003303.pdf",
        "Managing Incidental Findings on Abdominal CT (Berland et al., 2010)",
        2010,
        ("kidney", "liver", "adrenal", "pancreas"),
        "supporting",
        notes="Foundational abdominal CT white paper.",
    ),
    WhitePaperDoc(
        "PIIS1546144013002858.pdf",
        "Overview of ACR IFC Committee II White Papers",
        2013,
        ("adnexal", "other"),
        "supporting",
    ),
    WhitePaperDoc(
        "PIIS1546144013003086.pdf",
        "Managing Incidental Findings Part 1: Adnexal (2013)",
        2013,
        ("adnexal",),
        "supporting",
        notes="Superseded for primary routing by adnexal.pdf (2020).",
    ),
    WhitePaperDoc(
        "PIIS1546144013003062.pdf",
        "Managing Incidental Findings Part 2: Vascular",
        2013,
        ("other",),
        "supporting",
    ),
    WhitePaperDoc(
        "PIIS1546144013003050.pdf",
        "Managing Incidental Findings Part 3: Splenic and Nodal",
        2013,
        ("other",),
        "supporting",
    ),
    WhitePaperDoc(
        "PIIS1546144013003074.pdf",
        "Managing Incidental Findings Part 4: Gallbladder and Biliary",
        2013,
        ("other",),
        "supporting",
    ),
    WhitePaperDoc(
        "PIIS1546144018305301.pdf",
        "Managing Incidental Findings on Thoracic CT: Mediastinal and Cardiovascular",
        2018,
        ("other",),
        "out_of_scope",
        notes="Thoracic; indexed for RAG completeness but not an abdominal organ engine.",
    ),
]

PRIMARY_BY_ORGAN: Dict[str, WhitePaperDoc] = {}
for _doc in WHITE_PAPERS:
    if _doc.role != "primary":
        continue
    for _organ in _doc.organs:
        PRIMARY_BY_ORGAN[_organ] = _doc


def white_papers_dir(repo_root: Optional[Path] = None) -> Path:
    """Return the directory containing uploaded ACR white-paper PDFs."""

    if repo_root is None:
        # src/close_framework/guidelines/ -> repo root
        repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "incidental_findings_white_papers"


def rag_index_path(repo_root: Optional[Path] = None) -> Path:
    """Default on-disk RAG corpus path."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "guideline_rag" / "chunks.jsonl"


def doc_for_organ(organ: str) -> Optional[WhitePaperDoc]:
    """Primary white paper for an organ system, if any."""

    return PRIMARY_BY_ORGAN.get(organ)
