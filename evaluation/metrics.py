"""Safety-specific evaluation metrics for PharmaShield."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def critical_false_negative_rate(
    y_true_severity: List[str], y_pred_invalid: List[bool]
) -> float:
    """Proportion of Critical-severity errors the system fails to detect."""
    critical = [
        (i, sev) for i, sev in enumerate(y_true_severity) if sev == "Critical"
    ]
    if not critical:
        return 0.0
    missed = sum(1 for i, _ in critical if not y_pred_invalid[i])
    return missed / len(critical)


def safety_critical_recall(
    y_true_severity: List[str], y_pred_invalid: List[bool]
) -> float:
    """Recall restricted to Critical and Major assertions."""
    target = [
        i for i, sev in enumerate(y_true_severity) if sev in ("Critical", "Major")
    ]
    if not target:
        return 0.0
    caught = sum(1 for i in target if y_pred_invalid[i])
    return caught / len(target)


def bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap confidence interval (10,000 resamples as in the paper)."""
    rng = np.random.default_rng(42)
    stats = np.array([
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return {"point": float(stat_fn(values)), "lower": float(lower), "upper": float(upper)}
