"""Build and query a lexical RAG index over ACR white-paper PDFs."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from close_framework.guidelines.white_paper_catalog import (
    WHITE_PAPERS,
    WhitePaperDoc,
    rag_index_path,
    white_papers_dir,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.I)
_WS_RE = re.compile(r"\s+")


@dataclass
class GuidelineChunk:
    """One retrieved text span from a white paper."""

    chunk_id: str
    doc_filename: str
    title: str
    publication_year: int
    organs: List[str]
    role: str
    page: int
    text: str
    score: float = 0.0

    def as_dict(self) -> Dict:
        return asdict(self)


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").replace("\x00", " ")).strip()


def _extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        from PyPDF2 import PdfReader  # type: ignore

    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = _normalize_text(page.extract_text() or "")
        if text:
            pages.append((idx, text))
    return pages


def _chunk_page_text(text: str, page: int, max_chars: int = 1200, overlap: int = 150) -> List[Tuple[int, str]]:
    if len(text) <= max_chars:
        return [(page, text)]
    chunks: List[Tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            # Prefer to break on sentence-ish boundaries.
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("; "), window.rfind("\n"))
            if cut > max_chars // 3:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append((page, piece))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(
    papers_dir: Optional[Path] = None,
    docs: Optional[Sequence[WhitePaperDoc]] = None,
) -> List[GuidelineChunk]:
    """Extract and chunk all catalogued white papers."""

    papers_dir = papers_dir or white_papers_dir()
    docs = list(docs or WHITE_PAPERS)
    chunks: List[GuidelineChunk] = []
    for doc in docs:
        pdf_path = papers_dir / doc.filename
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Missing white paper PDF: {pdf_path}")
        for page, page_text in _extract_pdf_pages(pdf_path):
            for i, (page_num, chunk_text) in enumerate(_chunk_page_text(page_text, page)):
                chunks.append(
                    GuidelineChunk(
                        chunk_id=f"{doc.filename}:p{page_num}:c{i}",
                        doc_filename=doc.filename,
                        title=doc.title,
                        publication_year=doc.publication_year,
                        organs=list(doc.organs),
                        role=doc.role,
                        page=page_num,
                        text=chunk_text,
                    )
                )
    return chunks


def save_chunks(chunks: Sequence[GuidelineChunk], path: Optional[Path] = None) -> Path:
    """Write chunks to JSONL."""

    path = path or rag_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")
    return path


def load_chunks(path: Optional[Path] = None) -> List[GuidelineChunk]:
    """Load previously built chunks."""

    path = path or rag_index_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"RAG index not found at {path}. Run: python scripts/build_guideline_rag.py"
        )
    chunks: List[GuidelineChunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chunks.append(
                GuidelineChunk(
                    chunk_id=payload["chunk_id"],
                    doc_filename=payload["doc_filename"],
                    title=payload["title"],
                    publication_year=int(payload["publication_year"]),
                    organs=list(payload.get("organs") or []),
                    role=payload.get("role", "supporting"),
                    page=int(payload["page"]),
                    text=payload["text"],
                    score=float(payload.get("score") or 0.0),
                )
            )
    return chunks


class WhitePaperRAG:
    """TF-IDF lexical retriever over white-paper chunks."""

    def __init__(self, chunks: Optional[Sequence[GuidelineChunk]] = None) -> None:
        self.chunks: List[GuidelineChunk] = list(chunks) if chunks is not None else []
        self._df: Counter = Counter()
        self._tf: List[Counter] = []
        self._idf: Dict[str, float] = {}
        self._norms: List[float] = []
        if self.chunks:
            self._fit()

    @classmethod
    def from_disk(cls, path: Optional[Path] = None) -> "WhitePaperRAG":
        return cls(load_chunks(path))

    @classmethod
    def build_and_save(
        cls,
        papers_dir: Optional[Path] = None,
        index_path: Optional[Path] = None,
    ) -> "WhitePaperRAG":
        chunks = build_chunks(papers_dir=papers_dir)
        save_chunks(chunks, path=index_path)
        return cls(chunks)

    def _fit(self) -> None:
        self._tf = []
        self._df = Counter()
        for chunk in self.chunks:
            tf = Counter(_tokenize(chunk.text))
            self._tf.append(tf)
            for term in tf:
                self._df[term] += 1
        n_docs = max(len(self.chunks), 1)
        self._idf = {term: math.log((1 + n_docs) / (1 + df)) + 1.0 for term, df in self._df.items()}
        self._norms = []
        for tf in self._tf:
            norm = math.sqrt(sum((tf[t] * self._idf.get(t, 0.0)) ** 2 for t in tf)) or 1.0
            self._norms.append(norm)

    def retrieve(
        self,
        query: str,
        *,
        organ: Optional[str] = None,
        top_k: int = 5,
        prefer_primary: bool = True,
    ) -> List[GuidelineChunk]:
        """Return top-k chunks for a finding query."""

        if not self.chunks:
            return []
        q_tf = Counter(_tokenize(query))
        if not q_tf:
            return []
        q_vec = {t: q_tf[t] * self._idf.get(t, 0.0) for t in q_tf}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: List[Tuple[float, int]] = []
        for idx, tf in enumerate(self._tf):
            chunk = self.chunks[idx]
            if organ and organ not in chunk.organs and chunk.role == "out_of_scope":
                continue
            if organ and organ not in chunk.organs and prefer_primary and chunk.role == "primary":
                # Keep primary for other organs out of the top candidates when organ is known.
                organ_boost = 0.0
            else:
                organ_boost = 0.35 if organ and organ in chunk.organs else 0.0
            role_boost = 0.25 if prefer_primary and chunk.role == "primary" else 0.0
            dot = sum(q_vec.get(t, 0.0) * (tf[t] * self._idf.get(t, 0.0)) for t in tf)
            score = (dot / (q_norm * self._norms[idx])) + organ_boost + role_boost
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: List[GuidelineChunk] = []
        for score, idx in scored[:top_k]:
            chunk = self.chunks[idx]
            results.append(
                GuidelineChunk(
                    chunk_id=chunk.chunk_id,
                    doc_filename=chunk.doc_filename,
                    title=chunk.title,
                    publication_year=chunk.publication_year,
                    organs=list(chunk.organs),
                    role=chunk.role,
                    page=chunk.page,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return results


def finding_query_text(
    organ: str,
    finding_type: str,
    description: str,
    evidence_span: str,
    extra: Optional[Iterable[str]] = None,
) -> str:
    """Build a retrieval query from capture fields."""

    parts = [organ, finding_type, description, evidence_span]
    if extra:
        parts.extend(list(extra))
    return " ".join(p for p in parts if p)
