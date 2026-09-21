# Run-to-run stability status

## Available now
Pairwise agreement across historical Capture JSONL jobs that share report IDs
(see `tables/table28_run_to_run_stability.csv`).

- **Mistral**: 3 historical runs on n=100 overlapping reports
- **Qwen**: 2 historical runs on n=100
- **Llama**: no overlapping multi-run JSONL found

These are **not** controlled identical-seed / identical-temperature repeats from a
single experiment plan; treat as opportunistic lower-bound evidence.

## Controlled 3× study (recommended)
```bash
# Produce three Capture JSONLs with identical settings, then:
PYTHONPATH=src python scripts/analyze_llm_run_consistency.py \
  --runs run_a.jsonl run_b.jsonl run_c.jsonl
```
Target: few hundred reports × 3 seeds. Not executed here (no local LLM endpoint).
