from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable


SDG13_TARGET_ORDER = ["13.1", "13.2", "13.3", "13.a", "13.b"]


def target_distribution_report(
    records: Iterable[Dict[str, Any]],
    concentration_threshold: float = 0.80,
    minimum_records: int = 10,
) -> Dict[str, Any]:
    rows = list(records)
    counts = Counter(str(row.get("sdg_target") or "unassigned") for row in rows)
    total = len(rows)
    shares = {target: round(counts.get(target, 0) / total, 4) if total else 0.0 for target in SDG13_TARGET_ORDER}
    unassigned = counts.get("unassigned", 0)
    ranked = [(target, counts.get(target, 0)) for target in SDG13_TARGET_ORDER]
    dominant_target, dominant_count = max(ranked, key=lambda item: item[1]) if ranked else (None, 0)
    dominant_share = round(dominant_count / total, 4) if total else 0.0
    warnings = []
    if total >= minimum_records and dominant_share >= concentration_threshold:
        warnings.append(f"target_concentration:{dominant_target}:{dominant_share:.1%}")
    if unassigned:
        warnings.append(f"unassigned_targets:{unassigned}")
    by_country: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_country[str(row.get("country") or "unknown")][str(row.get("sdg_target") or "unassigned")] += 1
    return {
        "record_count": total,
        "target_counts": {target: counts.get(target, 0) for target in SDG13_TARGET_ORDER},
        "target_shares": shares,
        "dominant_target": dominant_target,
        "dominant_share": dominant_share,
        "warnings": warnings,
        "by_country": {country: dict(counter) for country, counter in sorted(by_country.items())},
        "interpretation": "A concentration warning is an audit signal, not proof of error and not a request to rebalance targets.",
    }
