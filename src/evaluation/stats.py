"""Paired statistical tests for shared-seed policy evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def paired_ttest(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Two-sided paired t-test on per-episode metrics (same seeds)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired series must share the same length")
    diff = a - b
    n = int(diff.size)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    t_stat, p_value = stats.ttest_rel(a, b)
    # Cohen's dz for paired designs
    cohen_dz = mean_diff / std_diff if std_diff > 0 else 0.0
    se = std_diff / np.sqrt(n) if n > 0 else float("nan")
    ci95 = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=se) if n > 1 else (mean_diff, mean_diff)
    return {
        "n": n,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t": float(t_stat),
        "p": float(p_value),
        "cohen_dz": float(cohen_dz),
        "ci95_low": float(ci95[0]),
        "ci95_high": float(ci95[1]),
    }


def compare_all_policies(
    series: dict[str, np.ndarray],
    reference: str = "PPO",
) -> dict[str, Any]:
    """Paired tests of ``reference`` against every other named series."""
    if reference not in series:
        raise KeyError(f"reference policy {reference!r} missing")
    ref = series[reference]
    out: dict[str, Any] = {"reference": reference, "comparisons": {}}
    for name, arr in series.items():
        if name == reference:
            continue
        out["comparisons"][name] = paired_ttest(ref, arr)
    return out
