# Data (not included)

This public repository does **not** ship institutional radiology reports or patient-linked labels.

- Place your local report exports under `data/` (see main README).
- Synthetic smoke-test reports: `data/examples/synthetic_reports.jsonl`
- Build the guideline RAG index locally after adding ACR white-paper PDFs:
  `PYTHONPATH=src python scripts/build_guideline_rag.py`
