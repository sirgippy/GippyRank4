"""Canonical provenance for explicitly supplied likelihood parameters."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def supplied_likelihood_parameters(likelihood: Any) -> dict[str, Any]:
    return {
        "beta_hex": [float(value).hex() for value in likelihood.beta],
        "scale_hex": float(likelihood.scale).hex(),
        "degrees_of_freedom_hex": float(likelihood.degrees_of_freedom).hex(),
        "fit_kind": likelihood.fit_kind,
    }


def supplied_likelihood_sha256(parameters: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
