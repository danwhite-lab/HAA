"""Pure helpers for combining already-calculated strategy sleeve signals."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


def total_weight(sleeves: Iterable[Mapping[str, object]]) -> float:
    return sum(float(sleeve["weight"]) for sleeve in sleeves)


def aggregate_holdings(sleeves: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Aggregate executable target weights from valid, fully weighted sleeves."""
    holdings: dict[str, dict[str, object]] = defaultdict(lambda: {"weight": 0.0, "sleeves": []})
    for sleeve in sleeves:
        sleeve_weight = float(sleeve["weight"]) / 100
        for asset, target_weight in dict(sleeve["target_weights"]).items():
            row = holdings[asset]
            row["weight"] = float(row["weight"]) + sleeve_weight * float(target_weight)
            row["sleeves"].append(str(sleeve["name"]))
    return dict(holdings)
