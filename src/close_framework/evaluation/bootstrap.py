"""Fast bootstrap confidence intervals for paired report-level metrics."""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from close_framework.evaluation.metrics import binary_metrics, cohen_kappa


def _percentile_ci(samples: np.ndarray, point: Optional[float], alpha: float = 0.05) -> Dict[str, Optional[float]]:
    if samples.size == 0:
        return {"estimate": point, "ci_low": None, "ci_high": None, "n_boot": 0.0}
    clean = samples[np.isfinite(samples)]
    if clean.size == 0:
        return {"estimate": point, "ci_low": None, "ci_high": None, "n_boot": 0.0}
    return {
        "estimate": point,
        "ci_low": float(np.quantile(clean, alpha / 2)),
        "ci_high": float(np.quantile(clean, 1 - alpha / 2)),
        "n_boot": float(clean.size),
    }


def _binary_counts(yt: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum(yt & yp))
    tn = float(np.sum(~yt & ~yp))
    fp = float(np.sum(~yt & yp))
    fn = float(np.sum(yt & ~yp))
    n = float(yt.size)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    acc = (tp + tn) / n if n else np.nan
    bal = (sens + spec) / 2.0 if np.isfinite(sens) and np.isfinite(spec) else np.nan
    # Nominal kappa for binary
    po = acc
    p_true = (tp + fn) / n if n else np.nan
    p_pred = (tp + fp) / n if n else np.nan
    pe = p_true * p_pred + (1 - p_true) * (1 - p_pred)
    kappa = (po - pe) / (1 - pe) if n and pe != 1 else np.nan
    return {
        "sensitivity": sens,
        "specificity": spec,
        "positive_predictive_value": ppv,
        "f1": f1,
        "balanced_accuracy": bal,
        "cohen_kappa": kappa,
        "accuracy": acc,
    }


def bootstrap_binary_metrics(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
    n_boot: int = 400,
    seed: int = 0,
    keys: Sequence[str] = (
        "sensitivity",
        "specificity",
        "positive_predictive_value",
        "f1",
        "balanced_accuracy",
        "cohen_kappa",
        "accuracy",
    ),
) -> Dict[str, Dict[str, Optional[float]]]:
    yt = np.asarray(list(y_true), dtype=bool)
    yp = np.asarray(list(y_pred), dtype=bool)
    n = yt.size
    base = binary_metrics(list(y_true), list(y_pred))
    if n == 0 or n_boot <= 0:
        return {
            key: {"estimate": base.get(key), "ci_low": None, "ci_high": None, "n_boot": 0.0}
            for key in keys
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = {key: np.empty(n_boot, dtype=float) for key in keys}
    for b in range(n_boot):
        counts = _binary_counts(yt[idx[b]], yp[idx[b]])
        for key in keys:
            samples[key][b] = counts[key]

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for key in keys:
        out[key] = _percentile_ci(samples[key], base.get(key))
    return out


def _linear_kappa_from_codes(yt: np.ndarray, yp: np.ndarray, n_labels: int) -> float:
    n = yt.size
    if n == 0:
        return float("nan")
    conf = np.zeros((n_labels, n_labels), dtype=float)
    for t, p in zip(yt, yp):
        conf[t, p] += 1
    conf /= n
    hist_t = conf.sum(axis=1)
    hist_p = conf.sum(axis=0)
    expected = np.outer(hist_t, hist_p)
    weights = np.abs(np.arange(n_labels)[:, None] - np.arange(n_labels)[None, :]).astype(float)
    weights /= max(n_labels - 1, 1)
    observed = np.sum(weights * conf)
    chance = np.sum(weights * expected)
    if chance == 0:
        return float("nan")
    return float(1.0 - observed / chance)


def bootstrap_linear_kappa(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    n_boot: int = 400,
    seed: int = 0,
) -> Dict[str, Optional[float]]:
    lab = list(labels)
    label_to_i = {label: i for i, label in enumerate(lab)}
    yt = np.asarray([label_to_i[v] for v in y_true], dtype=int)
    yp = np.asarray([label_to_i[v] for v in y_pred], dtype=int)
    n = yt.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point = cohen_kappa(list(y_true), list(y_pred), labels=lab, weights="linear")
    if n == 0 or n_boot <= 0:
        return {"estimate": point, "ci_low": None, "ci_high": None, "n_boot": 0.0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = np.empty(n_boot, dtype=float)
    n_labels = len(lab)
    for b in range(n_boot):
        samples[b] = _linear_kappa_from_codes(yt[idx[b]], yp[idx[b]], n_labels)
    return _percentile_ci(samples, point)


def paired_metric_difference_ci(
    y_true: Sequence,
    y_pred_a: Sequence,
    y_pred_b: Sequence,
    metric_fn: Callable[[Sequence, Sequence], Optional[float]],
    n_boot: int = 400,
    seed: int = 0,
) -> Dict[str, Optional[float]]:
    """Bootstrap CI for metric(B) - metric(A) on the same reports."""

    yt = list(y_true)
    ya = list(y_pred_a)
    yb = list(y_pred_b)
    n = len(yt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point_a = metric_fn(yt, ya)
        point_b = metric_fn(yt, yb)
    point = None if point_a is None or point_b is None else point_b - point_a
    if n == 0 or n_boot <= 0:
        return {
            "estimate": point,
            "ci_low": None,
            "ci_high": None,
            "n_boot": 0.0,
            "metric_a": point_a,
            "metric_b": point_b,
        }

    rng = np.random.default_rng(seed)
    samples = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sub_t = [yt[i] for i in idx]
            a = metric_fn(sub_t, [ya[i] for i in idx])
            b = metric_fn(sub_t, [yb[i] for i in idx])
            if a is None or b is None:
                continue
            samples.append(b - a)
    out = _percentile_ci(np.asarray(samples, dtype=float), point)
    out["metric_a"] = point_a
    out["metric_b"] = point_b
    return out
