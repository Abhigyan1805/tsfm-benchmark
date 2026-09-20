"""Paired statistics across series.

Two questions are asked of every accuracy comparison:

* **Is the difference between two models real?** Wilcoxon signed-rank over
  per-series aggregate scores, paired by series. Implemented here without SciPy
  so the data spine has no extra runtime dependency; the exact/normal
  approximation used matches SciPy's default for the sample sizes this
  benchmark produces.
* **Is any difference across many models real?** Friedman test over the
  model x series score matrix, plus the average-rank table needed to draw a
  critical-difference diagram (Nemenyi post-hoc).

Inputs are tidy result rows (dicts with ``series_id``, ``model``, and a metric
column) or an explicit wide matrix. Everything returns plain Python floats /
dicts so the output serialises cleanly into a report.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FriedmanResult",
    "PairedResult",
    "aggregate_by_series",
    "critical_difference",
    "friedman_test",
    "nemenyi_critical_difference",
    "pairwise_matrix",
    "rank_matrix",
    "wilcoxon_signed_rank",
    "worst_case_rank_sum",
]


class StatsError(ValueError):
    """Paired statistics could not be computed from the supplied data."""


def aggregate_by_series(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric: str = "mase",
    model: str | None = None,
    aggregate: str = "mean",
    horizon: int | None = None,
) -> dict[str, float]:
    """Collapse window-level rows to one score per series.

    Windows are averaged within a series (the standard way to compare methods
    that could not all be run everywhere). Non-finite scores are dropped; a
    series with no finite scores is omitted entirely.
    """
    reducer = _reducer(aggregate)
    buckets: dict[str, list[float]] = {}
    for row in rows:
        if model is not None and row.get("model") != model:
            continue
        if horizon is not None and int(row.get("horizon", -1)) != int(horizon):
            continue
        value = row.get(metric)
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        buckets.setdefault(str(row["series_id"]), []).append(numeric)
    return {series: reducer(values) for series, values in buckets.items() if values}


def _reducer(name: str):
    if name == "mean":
        return lambda values: float(np.mean(values))
    if name == "median":
        return lambda values: float(np.median(values))
    if name == "min":
        return lambda values: float(np.min(values))
    if name == "max":
        return lambda values: float(np.max(values))
    raise StatsError(f"unknown aggregate {name!r}")


@dataclass(frozen=True)
class PairedResult:
    """Outcome of a paired test between two models."""

    model_a: str
    model_b: str
    metric: str
    n_pairs: int
    statistic: float
    p_value: float
    median_difference: float
    method: str

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "n_pairs": self.n_pairs,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "median_difference": self.median_difference,
            "method": self.method,
            "significant_at_0.05": self.significant,
        }


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _average_ranks_abs(differences: np.ndarray) -> np.ndarray:
    """Average ranks of ``|difference|`` with ties averaged (no zeros handled)."""
    order = np.argsort(np.abs(differences), kind="stable")
    ranks = np.empty(differences.size, dtype=np.float64)
    ranks[order] = np.arange(1, differences.size + 1, dtype=np.float64)
    # Average ranks within tied absolute values.
    abs_values = np.abs(differences)
    unique, inverse = np.unique(abs_values, return_inverse=True)
    for position, _value in enumerate(unique):
        mask = inverse == position
        if mask.sum() > 1:
            ranks[mask] = np.mean(ranks[mask])
    return ranks


def wilcoxon_signed_rank(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    model_a: str = "a",
    model_b: str = "b",
    metric: str = "mase",
    alternative: str = "two-sided",
) -> PairedResult:
    """Wilcoxon signed-rank test over series shared by both models.

    Zero differences are discarded (the standard Wilcoxon convention). For
    ``n <= 25`` the exact null distribution is enumerated; above that the
    normal approximation with tie correction and continuity correction is used,
    which is what SciPy does by default.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        raise StatsError("no series shared by both models")
    x = np.array([float(a[s]) for s in shared], dtype=np.float64)
    y = np.array([float(b[s]) for s in shared], dtype=np.float64)
    differences = x - y
    differences = differences[np.isfinite(differences)]
    nonzero = differences[np.abs(differences) > 1e-12]
    if nonzero.size == 0:
        return PairedResult(
            model_a, model_b, metric, 0, 0.0, 1.0, 0.0, "all-differences-zero"
        )
    ranks = _average_ranks_abs(nonzero)
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    statistic = min(positive, negative)
    n = int(nonzero.size)
    if alternative not in {"two-sided", "less", "greater"}:
        raise StatsError(f"unknown alternative {alternative!r}")
    if n <= 25:
        p_value = _exact_p(positive, negative, n, ranks, alternative)
        method = "exact"
    else:
        p_value = _normal_approx_p(positive, negative, ranks, alternative)
        method = "normal-approximation"
    return PairedResult(
        model_a=model_a,
        model_b=model_b,
        metric=metric,
        n_pairs=n,
        statistic=float(statistic),
        p_value=float(min(1.0, max(0.0, p_value))),
        median_difference=float(np.median(nonzero)),
        method=method,
    )


def _exact_p(
    positive: float,
    negative: float,
    n: int,
    ranks: np.ndarray,
    alternative: str,
) -> float:
    """Exact null distribution by enumerating all sign assignments.

    For ``n <= 25`` enumerating all ``2**n`` sign flips is at most ~33M; the
    distribution is built by convolving each rank's +rank/0 outcome, which is
    polynomial in the number of distinct rank sums. The two-sided test uses the
    exact distribution of ``min(W+, W-)``; one-sided tests take the requested
    tail of ``W+``, the positive rank sum.
    """
    total = float(ranks.sum())
    reachable: dict[float, int] = {0.0: 1}
    for rank in ranks:
        updated: dict[float, int] = {}
        for value, count in reachable.items():
            updated[value] = updated.get(value, 0) + count
            updated[value + rank] = updated.get(value + rank, 0) + count
        reachable = updated
    denominator = 2 ** n
    if alternative == "less":
        extreme = sum(
            count for value, count in reachable.items() if value <= positive + 1e-9
        )
    elif alternative == "greater":
        extreme = sum(
            count for value, count in reachable.items() if value >= positive - 1e-9
        )
    else:
        observed_min = min(positive, negative)
        extreme = sum(
            count
            for value, count in reachable.items()
            if min(value, total - value) <= observed_min + 1e-9
        )
    return extreme / denominator


def _normal_approx_p(
    positive: float,
    negative: float,
    ranks: np.ndarray,
    alternative: str,
) -> float:
    mean = float(ranks.sum()) / 2.0
    variance = float(np.sum(ranks**2)) / 4.0
    if variance <= 0:
        return 1.0
    sd = math.sqrt(variance)
    if alternative == "less":
        return _normal_cdf((positive - mean + 0.5) / sd)
    if alternative == "greater":
        return 1.0 - _normal_cdf((positive - mean - 0.5) / sd)
    return 2.0 * _normal_cdf((min(positive, negative) - mean + 0.5) / sd)


def worst_case_rank_sum(n: int) -> float:
    """Reference helper: the maximum rank sum for ``n`` nonzero pairs."""
    return float(n * (n + 1) / 2)


@dataclass
class FriedmanResult:
    """Friedman test over a model x series score matrix."""

    models: list[str]
    series: list[str]
    average_ranks: dict[str, float]
    statistic: float
    p_value: float
    n_series: int
    k_models: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "n_series": self.n_series,
            "k_models": self.k_models,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant_at_0.05": self.significant,
            "average_ranks": self.average_ranks,
        }


def _chi_square_sf(statistic: float, degrees: int) -> float:
    """Upper tail of the chi-square distribution (regularized gamma)."""
    if statistic <= 0:
        return 1.0
    return 1.0 - _lower_regularized_gamma(degrees / 2.0, statistic / 2.0)


def _lower_regularized_gamma(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        raise StatsError("gamma arguments must be positive")
    if x == 0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        n = 1
        while n < 1000:
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
            n += 1
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction for the complementary function.
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def rank_matrix(
    scores: Mapping[str, Mapping[str, float]],
    *,
    lower_is_better: bool = True,
) -> tuple[list[str], list[str], np.ndarray]:
    """Build the (models x series) rank matrix from a nested score mapping.

    ``scores[model][series]`` is a scalar; missing entries are excluded by
    returning only the series present for every model (listwise deletion).
    """
    models = list(scores)
    if not models:
        raise StatsError("no models supplied")
    common = set(scores[models[0]])
    for model in models[1:]:
        common &= set(scores[model])
    series = sorted(common)
    if not series:
        raise StatsError("no series shared by all models")
    matrix = np.empty((len(models), len(series)), dtype=np.float64)
    for i, model in enumerate(models):
        matrix[i] = [float(scores[model][s]) for s in series]
    ranks = np.empty_like(matrix)
    for j in range(len(series)):
        column = matrix[:, j]
        order = np.argsort(column if lower_is_better else -column, kind="stable")
        column_ranks = np.empty(len(models), dtype=np.float64)
        column_ranks[order] = np.arange(1, len(models) + 1, dtype=np.float64)
        for value in np.unique(column):
            mask = column == value
            if mask.sum() > 1:
                column_ranks[mask] = np.mean(column_ranks[mask])
        ranks[:, j] = column_ranks
    return models, series, ranks


def friedman_test(
    scores: Mapping[str, Mapping[str, float]],
    *,
    lower_is_better: bool = True,
) -> FriedmanResult:
    """Friedman test over the model x series score matrix."""
    models, series, ranks = rank_matrix(scores, lower_is_better=lower_is_better)
    n = len(series)
    k = len(models)
    if k < 3:
        raise StatsError("Friedman test needs at least 3 models")
    if n < 2:
        raise StatsError("Friedman test needs at least 2 series")
    average = ranks.mean(axis=1)
    mean_rank = (k + 1) / 2.0
    statistic = (12.0 * n / (k * (k + 1))) * float(np.sum((average - mean_rank) ** 2))
    # Tie correction.
    tie_sum = 0.0
    for j in range(n):
        _, counts = np.unique(ranks[:, j], return_counts=True)
        tie_sum += float(np.sum(counts**3 - counts))
    if tie_sum > 0:
        denominator = 1.0 - tie_sum / (n * (k**3 - k))
        if denominator > 0:
            statistic /= denominator
        else:
            statistic = 0.0
    p_value = _chi_square_sf(statistic, k - 1)
    return FriedmanResult(
        models=models,
        series=series,
        average_ranks={model: float(np.mean(ranks[i])) for i, model in enumerate(models)},
        statistic=float(statistic),
        p_value=float(min(1.0, max(0.0, p_value))),
        n_series=n,
        k_models=k,
    )


_Q_ALPHA_005 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
    16: 3.426,
    17: 3.458,
    18: 3.489,
    19: 3.517,
    20: 3.544,
}


def nemenyi_critical_difference(k_models: int, n_series: int, *, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for average ranks (alpha=0.05 table)."""
    if k_models < 2:
        raise StatsError("need at least two models")
    if alpha != 0.05:
        raise StatsError("only the alpha=0.05 Nemenyi table is bundled")
    q = _Q_ALPHA_005.get(k_models)
    if q is None:
        raise StatsError(
            f"no Nemenyi q value for {k_models} models; extend the table to compare more"
        )
    return q * math.sqrt(k_models * (k_models + 1) / (6.0 * n_series))


def critical_difference(
    scores: Mapping[str, Mapping[str, float]],
    *,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Friedman result plus the Nemenyi CD and the rank table a CD diagram needs."""
    friedman = friedman_test(scores, lower_is_better=lower_is_better)
    cd = nemenyi_critical_difference(friedman.k_models, friedman.n_series)
    ordered = sorted(friedman.average_ranks.items(), key=lambda item: item[1])
    return {
        "friedman": friedman.as_dict(),
        "critical_difference": cd,
        "ranked_models": [{"model": m, "average_rank": r} for m, r in ordered],
    }


def pairwise_matrix(
    scores: Mapping[str, Mapping[str, float]],
    *,
    metric: str = "mase",
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """All pairwise Wilcoxon tests on a nested score mapping."""
    models = sorted(scores)
    results = []
    for model_a, model_b in itertools.combinations(models, 2):
        try:
            result = wilcoxon_signed_rank(
                scores[model_a],
                scores[model_b],
                model_a=model_a,
                model_b=model_b,
                metric=metric,
            )
        except StatsError:
            continue
        results.append(result.as_dict())
    return {"metric": metric, "comparisons": results}
