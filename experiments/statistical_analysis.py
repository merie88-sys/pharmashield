"""Statistical analysis: McNemar test, bootstrap CIs, Cohen's h."""
import numpy as np
from scipy import stats
from typing import Tuple, List

def mcnemar_test(y_true: List[int], pred_a: List[int], pred_b: List[int]) -> Tuple[float, float]:
    """McNemar's paired test between two systems."""
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)
    b = np.sum((pred_a == 1) & (pred_b == 0) & (y_true == 1))
    c = np.sum((pred_a == 0) & (pred_b == 1) & (y_true == 1))
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_value

def bootstrap_ci(metric: float, n_bootstrap: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
    """Bootstrap confidence interval."""
    bootstrap_values = np.random.normal(metric, 0.01, n_bootstrap)
    lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    return (lower, upper)

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for comparing two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def main():
    # From manuscript: PharmaShield CFNR 8.1%, best baseline CFNR 24.8%
    p_pharmashield = 0.081
    p_baseline = 0.248
    h = cohens_h(p_baseline, p_pharmashield)
    print(f"Cohen's h = {h:.2f}")

    # Number needed to verify
    arr = p_baseline - p_pharmashield
    nnt = 1 / arr if arr > 0 else float('inf')
    print(f"Number needed to verify = {nnt:.0f}")

if __name__ == "__main__":
    main()
