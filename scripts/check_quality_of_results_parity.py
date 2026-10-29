"""Check a retained production snapshot against a frozen issue 269 baseline.

This explicit validation command does not execute the research inventory and is
not part of production snapshot generation or ordinary CI.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from gippyrank.performance_snapshot import _as_snapshot, _pmfs, _summary
from gippyrank.posterior.snapshots import sha256
from gippyrank.quality_of_results import QUALITY_OF_RESULTS_MARGIN_WEIGHT
from gippyrank.quality_of_results_snapshot import (
    validate_quality_of_results_against_context,
)
from gippyrank.research.issue_265_real_seasons import parity, validate_pmfs

ROOT = Path(__file__).resolve().parents[1]


def check(snapshot_path: Path, reference_path: Path, root: Path) -> dict:
    sidecar = reference_path.with_suffix("").with_suffix(".json.sha256.json")
    if sha256(reference_path) != json.loads(sidecar.read_text())["sha256"]:
        raise ValueError("Research checkpoint hash mismatch")
    with gzip.open(reference_path, "rt", encoding="utf-8") as handle:
        reference = json.load(handle)
    if (
        reference["kind"] != "baseline"
        or reference["margin_weight"] != QUALITY_OF_RESULTS_MARGIN_WEIGHT
    ):
        raise ValueError("Expected a frozen λ=.40 baseline")
    frozen = reference["identity"]["source_snapshot"]
    for name, digest in frozen["artifact_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Research source artifact changed: {name}")
    source = root / frozen["context_source"]
    validate_quality_of_results_against_context(source, snapshot_path, root=root)
    snapshot = _as_snapshot(snapshot_path)
    metadata = snapshot.metadata
    for field in (
        "included_game_rows_sha256",
        "included_game_count",
        "prior_pmfs_sha256",
        "prior_model_version",
        "historical_likelihood_sha256",
    ):
        if metadata[field] != frozen[field]:
            raise ValueError(f"Research/production input mismatch: {field}")
    if (
        metadata["posterior_inference_configuration"]
        != frozen["inference_configuration"]
    ):
        raise ValueError("Research/production configuration mismatch")
    if metadata["source_context_snapshot_id"] != frozen["context_snapshot_id"]:
        raise ValueError("Research/production source identity mismatch")
    diagnostics = json.loads((snapshot_path / "diagnostics.json").read_text())
    if diagnostics["eligible_game_counts"] != reference["counts"]:
        raise ValueError("Research/production eligible counts mismatch")
    actual = _pmfs(snapshot_path / "posterior_pmfs.csv")
    expected = {t: np.array(p) for t, p in reference["pmfs"].items()}
    validate_pmfs(expected, {t: len(p) for t, p in actual.items()})
    report = parity(actual, expected, reference["counts"])
    blended = _pmfs(snapshot_path / "blended_context_posterior_pmfs.csv")
    reference_context = {t: np.array(p) for t, p in reference["context"].items()}
    validate_pmfs(reference_context, {t: len(p) for t, p in blended.items()})
    report["max_context_difference"] = max(
        float(np.max(np.abs(p - reference_context[t]))) for t, p in blended.items()
    )
    sd_errors, entropy_errors, summary_errors = [], [], []
    discrete_differences = []
    for team_id, pmf in actual.items():
        other = expected[team_id]
        support = np.arange(1, len(pmf) + 1)
        sds, entropies = [], []
        for p in (pmf, other):
            sds.append(float(np.sqrt(p @ (support - p @ support) ** 2)))
            positive = p[p > 0]
            entropies.append(float(-np.sum(positive * np.log2(positive))))
        sd_errors.append(abs(sds[0] - sds[1]))
        entropy_errors.append(abs(entropies[0] - entropies[1]))
        left, right = _summary(pmf), _summary(other)
        for field, value in left.items():
            if isinstance(value, int):
                if value != right[field]:
                    discrete_differences.append([team_id, field])
            else:
                summary_errors.append(abs(value - right[field]))
    report.update(
        {
            "max_rank_sd_difference": max(sd_errors),
            "max_entropy_bits_difference": max(entropy_errors),
            "max_summary_difference": max(summary_errors),
            "discrete_summary_differences": discrete_differences,
            "full_network_team_count": len(blended),
            "included_game_count": metadata["included_game_count"],
            "eligible_counts_match": True,
            "source_context_snapshot_id": metadata["source_context_snapshot_id"],
            "source_context_metadata_sha256": metadata[
                "source_context_metadata_sha256"
            ],
            "prior_artifact_sha256": metadata["prior_artifact_sha256"],
            "included_game_rows_sha256": metadata["included_game_rows_sha256"],
            "historical_likelihood_sha256": metadata["historical_likelihood_sha256"],
            "posterior_inference_configuration": metadata[
                "posterior_inference_configuration"
            ],
            "model": metadata["model"],
            "margin_weight": metadata["margin_weight"],
            "research_reference": reference_path.relative_to(root).as_posix(),
            "research_reference_sha256": sha256(reference_path),
            "production_posterior_pmfs_sha256": metadata["posterior_pmfs_sha256"],
            "production_converged": diagnostics["converged"],
            "production_iterations": diagnostics["iterations"],
            "production_max_message_delta": diagnostics["max_message_delta"],
        }
    )
    report["passed"] = (
        report["passed"]
        and report["max_context_difference"] <= report["pmf_tolerance"]
        and max(sd_errors + entropy_errors + summary_errors)
        <= report["expected_rank_tolerance"]
        and not discrete_differences
    )
    if not report["passed"]:
        raise ValueError(f"Research/production parity failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check(ROOT / args.snapshot, ROOT / args.reference, ROOT)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
