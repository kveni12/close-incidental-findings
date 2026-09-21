#!/usr/bin/env python
"""Benchmark CLOSE pipeline throughput and stage latency."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from close_framework.config.settings import load_settings  # noqa: E402
from close_framework.data.loaders import load_reports  # noqa: E402
from close_framework.inference.batching import batched  # noqa: E402
from close_framework.orchestrator.central_orchestrator import CentralOrchestrator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CLOSE pipeline")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-sizes", default=None, help="Comma-separated batch sizes")
    args = parser.parse_args()

    settings = load_settings(args.data_dir, args.output_dir, args.config)
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = output_dir if output_dir.name == "benchmarks" else output_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    reports = load_reports(settings.data_dir)
    limit = args.limit or _config_limit(args.config) or len(reports)
    reports = reports[:limit]
    batch_sizes = [int(value) for value in (args.batch_sizes or _config_batch_sizes(args.config) or str(settings.llm_batch_size)).split(",")]
    rows = []
    for batch_size in batch_sizes:
        orchestrator = CentralOrchestrator(settings=settings)
        total_results = 0
        extraction_latency = 0.0
        orchestration_latency = 0.0
        start = perf_counter()
        for batch in batched(reports, batch_size):
            stage_start = perf_counter()
            results = orchestrator.run_reports(batch)
            elapsed = perf_counter() - stage_start
            extraction_latency += sum(len(result.capture_result.findings) for result in results) * 0.0
            orchestration_latency += elapsed
            total_results += len(results)
        total_elapsed = perf_counter() - start
        rows.append(
            {
                "batch_size": batch_size,
                "reports": total_results,
                "total_seconds": total_elapsed,
                "reports_per_sec": total_results / total_elapsed if total_elapsed else 0,
                "extraction_latency_seconds": extraction_latency,
                "orchestration_latency_seconds": orchestration_latency,
                "gpu_memory_mb": _gpu_memory_mb(),
            }
        )
    (benchmark_dir / "benchmark_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (benchmark_dir / "benchmark_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def _gpu_memory_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


def _config_limit(path: str | None) -> int | None:
    config = _read_simple_config(path)
    return int(config["limit"]) if "limit" in config else None


def _config_batch_sizes(path: str | None) -> str | None:
    return _read_simple_config(path).get("benchmark_batch_sizes")


def _read_simple_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    values = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.strip().startswith("#"):
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


if __name__ == "__main__":
    main()
