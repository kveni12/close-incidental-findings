"""Tests for white-paper RAG indexing and retrieval."""

from __future__ import annotations

from pathlib import Path

from close_framework.agents.orchestrate_agent import OrchestrateAgent
from close_framework.data.schemas import IncidentalFinding
from close_framework.guidelines.rag_index import WhitePaperRAG, finding_query_text
from close_framework.guidelines.white_paper_catalog import PRIMARY_BY_ORGAN, white_papers_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "data" / "guideline_rag" / "chunks.jsonl"


def test_primary_white_papers_exist():
    papers = white_papers_dir(REPO_ROOT)
    for organ, doc in PRIMARY_BY_ORGAN.items():
        assert (papers / doc.filename).is_file(), f"missing {doc.filename} for {organ}"


def test_rag_index_built_and_retrieves_renal():
    assert INDEX_PATH.is_file(), "Run scripts/build_guideline_rag.py before tests"
    rag = WhitePaperRAG.from_disk(INDEX_PATH)
    assert len(rag.chunks) > 50
    hits = rag.retrieve(
        finding_query_text("kidney", "renal cyst", "homogeneous 12 HU", "Homogeneous right renal cyst 12 HU"),
        organ="kidney",
        top_k=3,
    )
    assert hits
    assert any("renal" in h.doc_filename or "PIIS1546144010003303" in h.doc_filename for h in hits)


def test_orchestrate_attaches_rag_evidence():
    finding = IncidentalFinding(
        finding_id="f1",
        organ_system="kidney",
        finding_type="renal cyst",
        description="Homogeneous right renal cyst measures 12 HU.",
        actionability="actionable",
        evidence_span="Homogeneous right renal cyst measures 12 HU on CT.",
        attenuation_hu=12.0,
        size_mm=20.0,
        cystic=True,
        confidence=0.9,
    )
    rec = OrchestrateAgent().run(finding)
    assert rec.rule_id
    assert "rag_chunks" in (rec.evidence_used or {})
