"""Runtime settings for the CLOSE framework."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os


class InferenceMode(str, Enum):
    """Supported extraction refinement modes."""

    RULE_BASED_ONLY = "RULE_BASED_ONLY"
    HYBRID_LLM = "HYBRID_LLM"
    LOCAL_VLLM = "LOCAL_VLLM"
    LOCAL_OLLAMA = "LOCAL_OLLAMA"
    OPENAI_API = "OPENAI_API"
    AZURE_OPENAI = "AZURE_OPENAI"


@dataclass(frozen=True)
class CloseSettings:
    """Configuration values shared across agents and scripts."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    config_path: Path | None = None
    capture_confidence_threshold: float = 0.65
    human_review_confidence_threshold: float = 0.55
    inference_mode: InferenceMode = InferenceMode.RULE_BASED_ONLY
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_max_tokens: int = 1536
    llm_retries: int = 3
    llm_report_chars: int = 12000
    allow_rule_fallback: bool = False
    recalibrate_llm_capture: bool = False
    supplement_rule_findings: bool = False
    llm_batch_size: int = 8
    max_concurrency: int = 4
    gpu_count: int = 1
    mixed_precision: bool = True
    checkpoint_path: Path | None = None
    safety_pool_id: str = "diagnostic-safety-pool"
    safety_pool_name: str = "Diagnostic Safety Review Pool"


def load_settings(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> CloseSettings:
    """Load settings from explicit arguments and environment variables."""

    config_values = _load_config(config_path)
    mode = str(config_values.get("inference_mode", os.getenv("CLOSE_INFERENCE_MODE", "RULE_BASED_ONLY")))
    return CloseSettings(
        data_dir=Path(data_dir or config_values.get("data_dir") or os.getenv("CLOSE_DATA_DIR", "data")),
        output_dir=Path(output_dir or config_values.get("output_dir") or os.getenv("CLOSE_OUTPUT_DIR", "outputs")),
        config_path=Path(config_path) if config_path else None,
        capture_confidence_threshold=float(
            config_values.get("capture_confidence_threshold", os.getenv("CLOSE_CAPTURE_CONFIDENCE_THRESHOLD", "0.65"))
        ),
        human_review_confidence_threshold=float(
            config_values.get("human_review_confidence_threshold", os.getenv("CLOSE_HUMAN_REVIEW_CONFIDENCE_THRESHOLD", "0.55"))
        ),
        inference_mode=InferenceMode(mode),
        llm_model=os.getenv("CLOSE_LLM_MODEL") or config_values.get("llm_model"),
        llm_base_url=os.getenv("CLOSE_LLM_BASE_URL") or config_values.get("llm_base_url"),
        llm_api_key=os.getenv("CLOSE_LLM_API_KEY") or config_values.get("llm_api_key"),
        llm_max_tokens=int(os.getenv("CLOSE_LLM_MAX_TOKENS", config_values.get("llm_max_tokens", "1536"))),
        llm_retries=int(os.getenv("CLOSE_LLM_RETRIES", config_values.get("llm_retries", "3"))),
        llm_report_chars=int(os.getenv("CLOSE_LLM_REPORT_CHARS", config_values.get("llm_report_chars", "12000"))),
        allow_rule_fallback=str(
            os.getenv("CLOSE_ALLOW_RULE_FALLBACK", config_values.get("allow_rule_fallback", "false"))
        ).lower()
        in {"1", "true", "yes"},
        recalibrate_llm_capture=str(
            os.getenv("CLOSE_RECALIBRATE_LLM_CAPTURE", config_values.get("recalibrate_llm_capture", "false"))
        ).lower()
        in {"1", "true", "yes"},
        supplement_rule_findings=str(
            os.getenv("CLOSE_SUPPLEMENT_RULE_FINDINGS", config_values.get("supplement_rule_findings", "false"))
        ).lower()
        in {"1", "true", "yes"},
        llm_batch_size=int(os.getenv("CLOSE_LLM_BATCH_SIZE", config_values.get("batch_size", "8"))),
        max_concurrency=int(config_values.get("max_concurrency", os.getenv("CLOSE_MAX_CONCURRENCY", "4"))),
        gpu_count=int(config_values.get("gpu_count", os.getenv("CLOSE_GPU_COUNT", "1"))),
        mixed_precision=str(config_values.get("mixed_precision", os.getenv("CLOSE_MIXED_PRECISION", "true"))).lower()
        in {"1", "true", "yes"},
        checkpoint_path=Path(config_values["checkpoint_path"]) if config_values.get("checkpoint_path") else None,
        safety_pool_id=config_values.get("safety_pool_id", os.getenv("CLOSE_SAFETY_POOL_ID", "diagnostic-safety-pool")),
        safety_pool_name=config_values.get("safety_pool_name", os.getenv("CLOSE_SAFETY_POOL_NAME", "Diagnostic Safety Review Pool")),
    )


def _load_config(config_path: str | Path | None) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        return json.loads(text)
    try:
        import yaml

        payload = yaml.safe_load(text)
    except ImportError:
        payload = _parse_simple_yaml(text)
    return payload or {}


def _parse_simple_yaml(text: str) -> dict:
    values = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
