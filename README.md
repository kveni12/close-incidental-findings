# CLOSE — Capture, Link, Orchestrate for incidental findings

Research prototype for extracting incidental abdominal findings from radiology
reports and mapping them to conservative ACR-style follow-up pathways.

> **Not clinical decision support.** Retrospective research software only.

## What this repo contains

- `src/close_framework/` — Capture / Link / Orchestrate agents, guideline engines, evaluation
- `scripts/` — pipeline runners, hybrid recalibration, paper metric builders
- `configs/` — example experiment configs (Ollama / rule-based / hybrid)
- `tests/` — unit tests + synthetic report fixtures
- `paper_results/` — **aggregate** tables/figures only (no patient-linked files)
- `data/examples/` — synthetic smoke-test reports

## What this repo does **not** contain

- Institutional radiology report exports (PHI)
- Patient-linked labels, adjudication packets, or evidence spans
- ACR white-paper PDFs (add locally; see `incidental_findings_white_papers/README.md`)
- Model weights / Ollama caches / run outputs

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# smoke test on synthetic reports
PYTHONPATH=src python scripts/run_close_pipeline.py \
  --config configs/capture_rule_based.yaml \
  --data-dir data/examples \
  --output-dir outputs/smoke \
  --limit 10
```

LLM Capture (local Ollama):

```bash
bash scripts/install_ollama.sh
bash scripts/pull_ollama_models.sh
PYTHONPATH=src python scripts/run_close_pipeline.py \
  --config configs/capture_ollama.yaml \
  --data-dir data/examples \
  --output-dir outputs/ollama_smoke \
  --limit 10
```

## Hybrid Capture

LLM extraction + deterministic ACR recalibration ± missed-finding recovery:

```bash
PYTHONPATH=src python scripts/run_hybrid_downstream.py \
  --input outputs/ollama_smoke/close_results.jsonl \
  --output outputs/ollama_smoke/close_results_hybrid.jsonl
```

## Evaluation notes

Primary paper metrics compare model outputs to a **deterministic ACR-derived
reference** (guideline concordance). Hybrid Capture shares those heuristics —
disclose circularity. Clinical validity requires independent dual-reader review.

Aggregate results live under `paper_results/` (`RESULTS_SUMMARY.md`).

## Citation

If you use this code, please cite the CLOSE manuscript (link TBD).

## License

MIT — see `LICENSE`.
