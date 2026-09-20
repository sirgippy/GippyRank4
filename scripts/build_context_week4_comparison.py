"""Build the canonical matched 2026 official Week 4 Context pair."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gippyrank.context_comparison import (
    ContextComparisonError,
    build_official_week4_pair,
)
from gippyrank.performance_snapshot import (
    build_performance_snapshot,
    validate_performance_against_context,
)
from gippyrank.posterior.snapshots import Snapshot

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContextComparisonError(f"{path} must contain a JSON object")
    return value


def _snapshot_record(root: Path, path_value: str, *, label: str) -> dict[str, Any]:
    if not path_value or path_value == "None":
        raise ContextComparisonError(f"{label} snapshot path is null")
    path = root / path_value
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise ContextComparisonError(f"{label} snapshot is missing: {metadata_path}")
    metadata = _read_json(metadata_path)
    expected_version = "1.2" if label == "context_1_2" else "1.3"
    if (
        metadata.get("snapshot_type") != "preseason"
        or metadata.get("prior_family") != "context"
        or metadata.get("prior_model_version") != expected_version
        or metadata.get("included_game_count") != 0
        or metadata.get("source_mode") != "preseason_prior_only"
    ):
        raise ContextComparisonError(
            f"{label} is not the expected canonical preseason artifact"
        )
    if label == "context_1_3" and metadata.get("prior_lineage") != (
        "context_1_3_reconstructed_2026"
    ):
        raise ContextComparisonError(
            "Context 1.3 preseason artifact is not the reconstructed lineage"
        )
    return {
        "snapshot_id": metadata["snapshot_id"],
        "path": path_value,
        "prior_model_version": metadata["prior_model_version"],
        "prior_lineage": metadata.get("prior_lineage"),
        "prior_artifact_sha256": metadata["prior_artifact_sha256"],
    }


def _performance_record(root: Path, context_path_value: str) -> dict[str, Any]:
    context_path = root / context_path_value
    performance_path = context_path.parent.parent / "performance"
    context_metadata = _read_json(context_path / "metadata.json")
    performance_metadata_path = performance_path / "metadata.json"
    if not performance_metadata_path.is_file():
        raise ContextComparisonError(
            "Context 1.3 official Week 4 Performance artifact is missing"
        )
    performance_metadata = _read_json(performance_metadata_path)
    validate_performance_against_context(
        Snapshot(context_metadata["snapshot_id"], context_path, context_metadata),
        Snapshot(
            performance_metadata["snapshot_id"],
            performance_path,
            performance_metadata,
        ),
    )
    return {
        "snapshot_id": performance_metadata["snapshot_id"],
        "path": performance_path.relative_to(root).as_posix(),
        "source_context_snapshot_id": performance_metadata[
            "source_context_snapshot_id"
        ],
        "prior_model_version": performance_metadata["prior_model_version"],
        "included_game_count": performance_metadata["included_game_count"],
        "game_corpus_sha256": performance_metadata["game_corpus_sha256"],
        "valid": performance_metadata["valid"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, help="official Week 4 UTC cutoff")
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root"
    )
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff)
    result = build_official_week4_pair(root=args.root, cutoff=cutoff)
    manifest = (
        args.root
        / "data/processed/preseason/context_v1_3_2026_reconstruction/comparison_quartet.json"
    )
    value = _read_json(manifest) if manifest.exists() else {}
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContextComparisonError("comparison quartet manifest has no artifacts map")
    preseason_1_2 = _snapshot_record(
        args.root, str(artifacts.get("context_1_2_preseason")), label="context_1_2"
    )
    preseason_1_3 = _snapshot_record(
        args.root,
        str(artifacts.get("context_1_3_reconstructed_preseason")),
        label="context_1_3",
    )
    official = {
        **result,
        "context_1_2_path": result["context_1_2_path"],
        "context_1_3_path": result["context_1_3_path"],
    }
    build_performance_snapshot(
        args.root / result["context_1_3_path"],
        root=args.root,
    )
    performance = _performance_record(args.root, result["context_1_3_path"])
    official["context_1_3_performance"] = performance
    value.update(
        {
            "status": "frozen",
            "official_week_4": official,
            "preseason": {
                "context_1_2": preseason_1_2,
                "context_1_3_reconstructed": preseason_1_3,
            },
            "validation": {
                "status": "passed",
                "non_null_artifact_count": 4,
                "matched_week_4_pair": True,
                "retained_genuine_context_1_2_preseason": True,
                "retained_reconstructed_context_1_3_preseason": True,
                "context_1_3_performance_valid": performance["valid"],
                "boundary_sanity": result["boundary_sanity"],
            },
            "artifacts": {
                **artifacts,
                "context_1_2_official_week_4": result["context_1_2_path"],
                "context_1_3_official_week_4": result["context_1_3_path"],
            },
        }
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
