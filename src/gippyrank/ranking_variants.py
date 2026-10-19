"""The two Performance variants, including the legacy snapshot default."""

from collections.abc import Mapping
from typing import Any

PERFORMANCE_VARIANTS = {
    "quality_of_play": "Quality of Play",
    "quality_of_record": "Quality of Record",
}


def performance_variant(metadata: Mapping[str, Any]) -> str | None:
    """Interpret older Performance artifacts without rewriting their metadata."""
    if metadata.get("ranking_family") != "performance":
        return None
    variant = metadata.get("performance_variant", "quality_of_play")
    if not isinstance(variant, str) or variant not in PERFORMANCE_VARIANTS:
        raise ValueError(f"Unsupported Performance variant: {variant}")
    return str(variant)
