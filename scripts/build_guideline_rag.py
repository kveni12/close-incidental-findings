#!/usr/bin/env python3
"""Build the ACR white-paper RAG index under data/guideline_rag/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from close_framework.guidelines.rag_index import WhitePaperRAG  # noqa: E402
from close_framework.guidelines.white_paper_catalog import (  # noqa: E402
    rag_index_path,
    white_papers_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=white_papers_dir(REPO_ROOT),
        help="Directory with ACR white-paper PDFs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=rag_index_path(REPO_ROOT),
        help="JSONL chunk index path",
    )
    args = parser.parse_args()

    rag = WhitePaperRAG.build_and_save(papers_dir=args.papers_dir, index_path=args.output)
    print(f"Indexed {len(rag.chunks)} chunks from {args.papers_dir}")
    print(f"Wrote {args.output}")
    # Smoke retrieval
    sample = rag.retrieve("incidental renal mass 20 HU homogeneous no enhancement", organ="kidney", top_k=2)
    if sample:
        print(f"Sample hit: {sample[0].doc_filename} p{sample[0].page} score={sample[0].score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
