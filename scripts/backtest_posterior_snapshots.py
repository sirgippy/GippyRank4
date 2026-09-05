"""Evaluate rolling posterior snapshots against frozen final Massey PMFs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from gippyrank.posterior.snapshots import build_snapshot, load_teams
from gippyrank.preseason import pmf_summaries

ROOT = Path(__file__).resolve().parents[1]


def _targets(season: int) -> dict[str, np.ndarray]:
    path = ROOT / "data/processed/modeling/team_season_rank_distributions.csv"
    targets = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["season"]) != season or row["subdivision"] != "fbs":
                continue
            support = json.loads(row["pmf"])
            pmf = np.zeros(int(row["team_population"]))
            for item in support:
                pmf[int(item["rank"]) - 1] = float(item["probability"])
            targets[row["team_id"]] = pmf
    return targets


def _metrics(
    predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> dict[str, float | int]:
    rows = [
        (predictions[key], target)
        for key, target in targets.items()
        if key in predictions and len(predictions[key]) == len(target)
    ]
    if not rows:
        raise ValueError("no FBS posterior rows matched final target distributions")
    nll, crps, expected_mae, median_mae, coverage, width = [], [], [], [], [], []
    brier = {threshold: [] for threshold in (5, 10, 25)}
    for prediction, target in rows:
        ranks = np.arange(1, len(prediction) + 1)
        p_summary, q_summary = pmf_summaries(prediction), pmf_summaries(target)
        nll.append(float(-np.sum(target * np.log(np.maximum(prediction, 1e-15)))))
        crps.append(float(np.mean((np.cumsum(prediction) - np.cumsum(target)) ** 2)))
        expected_mae.append(
            abs(p_summary["expected_rank"] - q_summary["expected_rank"])
        )
        median_mae.append(abs(p_summary["median_rank"] - q_summary["median_rank"]))
        low, high = p_summary["interval_80_low"], p_summary["interval_80_high"]
        coverage.append(float(target[(ranks >= low) & (ranks <= high)].sum()))
        width.append(high - low)
        for threshold, values in brier.items():
            values.append(
                (p_summary[f"top{threshold}_probability"] - target[:threshold].sum())
                ** 2
            )
    return {
        "matched_fbs_teams": len(rows),
        "nll": float(np.mean(nll)),
        "crps": float(np.mean(crps)),
        "expected_rank_mae": float(np.mean(expected_mae)),
        "median_rank_mae": float(np.mean(median_mae)),
        "interval_80_coverage": float(np.mean(coverage)),
        "interval_80_width": float(np.mean(width)),
        **{
            f"top{threshold}_brier": float(np.mean(values))
            for threshold, values in brier.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--cutoff",
        action="append",
        required=True,
        help="ISO-8601 timestamp; repeat for rolling cutoffs",
    )
    args = parser.parse_args()
    targets = _targets(args.season)
    rows = []
    for cutoff_value in args.cutoff:
        cutoff = datetime.fromisoformat(cutoff_value)
        variants = {}
        for family in ("context", "history"):
            snapshot = build_snapshot(
                season=args.season,
                cutoff=cutoff,
                prior_family=family,
                snapshot_type="weekly",
            )
            teams, _, _ = load_teams(ROOT, args.season, family)
            prior = {
                team.team_id: team.prior for team in teams if team.subdivision == "fbs"
            }
            pmfs = {}
            with (snapshot.directory / "posterior_pmfs.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                for item in csv.DictReader(handle):
                    pmfs.setdefault(item["team_id"], []).append(
                        float(item["probability"])
                    )
            posterior = {key: np.asarray(value) for key, value in pmfs.items()}
            variants[family] = {
                "prior": _metrics(prior, targets),
                "posterior": _metrics(posterior, targets),
                "snapshot_id": snapshot.snapshot_id,
                "runtime_seconds": json.loads(
                    (snapshot.directory / "diagnostics.json").read_text()
                )["runtime_seconds"],
            }
        rows.append(
            {
                "cutoff": cutoff.isoformat(),
                **variants,
                "context_minus_history_posterior_nll": variants["context"]["posterior"][
                    "nll"
                ]
                - variants["history"]["posterior"]["nll"],
            }
        )
    output = ROOT / f"data/processed/posterior_backtest/{args.season}_rolling.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "season": args.season,
                "cutoffs": rows,
                "target": "frozen final Massey constituent-rank PMFs",
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
