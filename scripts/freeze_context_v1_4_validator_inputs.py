"""Freeze outcome-free 2026 inputs needed by the registered validator.

This one-time construction uses local historical rank and coaching caches that
are intentionally absent from a clean checkout. The resulting small, audited
input bundle is sufficient for later confirmation without those caches.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_preseason_context_prior_v1_2 as context12

from gippyrank.context_prior_v1_3 import (
    MODEL_FEATURE_NAMES,
    attach_transfer_features_to_inference_rows,
    load_validated_committed_2026_reconstruction,
    load_validated_context13_fitted_model,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/context_v1_4_validation/validator_inputs.jsonl"
SCHEDULE_FIELDS = (
    "id",
    "season",
    "week",
    "seasonType",
    "startDate",
    "homeId",
    "awayId",
    "homeClassification",
    "awayClassification",
    "neutralSite",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    annual = ROOT / "data/processed/preseason/context_v1_3/annual/2026"
    model, _, _ = load_validated_context13_fitted_model(
        annual / "fitted_model.json", annual / "fitted_instance.json"
    )
    reconstruction = ROOT / "data/processed/preseason/context_v1_3_2026_reconstruction"
    transfer_rows, transfer = load_validated_committed_2026_reconstruction(
        reconstruction
    )
    features_path = ROOT / "data/processed/preseason/team_season_features.csv"
    index = {
        (int(row["season"]), row["subdivision"], row["team_id"]): row
        for row in _rows(features_path)
    }
    base = context12.inference_rows(2026, 2025, index, context12.cached_tenures())
    base = [
        replace(
            row,
            features={name: row.features.get(name) for name in MODEL_FEATURE_NAMES},
            lag_zs=(),  # Context 1.3 uses only the frozen lag-1 distribution.
        )
        for row in base
    ]
    inference_rows = attach_transfer_features_to_inference_rows(base, transfer_rows)
    predictions = {row["team_id"]: row for row in _rows(annual / "predictions.csv")}
    if len(inference_rows) != len(predictions) or len(inference_rows) != 138:
        raise ValueError("2026 Context 1.3 population is incomplete")
    for row in inference_rows:
        row.require_no_target()
        if row.team_id not in predictions:
            raise ValueError(f"missing Context 1.3 prior: {row.team_id}")
        if row.lag1_z is None:
            if row.team_id not in {"16", "2449"}:
                raise ValueError("unexpected 2026 cold start")
            continue
        transfer.validate_inference_row(row)
        original = np.asarray(json.loads(predictions[row.team_id]["pmf"]), dtype=float)
        exact = model.pmf(
            {name: row.features[name] for name in model.feature_names},
            np.asarray(row.lag1_z),
            row.population,
        )
        # Published PMFs have 12 decimal places; keep the exact model output
        # for the candidate source and require the expected serialization bound.
        if not np.allclose(exact, original, rtol=0, atol=5.01e-13):
            raise ValueError(f"published Context 1.3 PMF mismatch: {row.team_id}")

    schedule_paths = (
        ROOT / "data/raw/cfbd/games/2026.json",
        ROOT / "data/raw/cfbd/games/2026-fcs.json",
    )
    expected: dict[str, dict[str, str]] = {}
    for path in schedule_paths:
        for game in json.loads(path.read_text(encoding="utf-8")):
            if (
                game.get("season") != 2026
                or game.get("seasonType") != "regular"
                or not isinstance(game.get("week"), int)
                or not 1 <= game["week"] <= 5
                or game.get("homeClassification") not in {"fbs", "fcs"}
                or game.get("awayClassification") not in {"fbs", "fcs"}
            ):
                continue
            item = {key: str(game.get(key, "")) for key in SCHEDULE_FIELDS}
            old = expected.setdefault(item["id"], item)
            if old != item:
                raise ValueError(f"conflicting registered schedule rows: {item['id']}")
    if not expected or not any(row["week"] == "5" for row in expected.values()):
        raise ValueError("registered Week 5 schedule is missing")

    rank_path = ROOT / "data/processed/modeling/team_season_rank_distributions.csv"
    tenure_dir = ROOT / "data/raw/cfbd/preseason/coach_tenures"
    tenure_hashes = {
        path.name: _sha(path) for path in sorted(tenure_dir.glob("*.json"))
    }
    if not tenure_hashes:
        raise ValueError("coaching input cache is unavailable")
    bundle = {
        "schema_version": 1,
        "purpose": "outcome_free_context_1_4_validation_inputs",
        "season": 2026,
        "forecast_rows": [
            asdict(row) for row in sorted(inference_rows, key=lambda row: row.team_id)
        ],
        "registered_schedule": [expected[key] for key in sorted(expected)],
        "source_sha256": {
            "historical_rank_distributions": _sha(rank_path),
            "coach_tenures": hashlib.sha256(
                json.dumps(
                    tenure_hashes, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "team_season_features": _sha(features_path),
            "transfer_features": _sha(reconstruction / "transfer_features.csv"),
            "context_1_3_predictions": _sha(annual / "predictions.csv"),
            "fbs_schedule": _sha(schedule_paths[0]),
            "fcs_schedule": _sha(schedule_paths[1]),
        },
    }
    records = [
        {
            "kind": "metadata",
            **{
                key: bundle[key]
                for key in ("schema_version", "purpose", "season", "source_sha256")
            },
        },
        *({"kind": "forecast_row", "value": row} for row in bundle["forecast_rows"]),
        *(
            {"kind": "schedule_game", "value": row}
            for row in bundle["registered_schedule"]
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    print(f"{OUTPUT.relative_to(ROOT)} {_sha(OUTPUT)}")


if __name__ == "__main__":
    main()
