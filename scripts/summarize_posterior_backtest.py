"""Summarize compact rolling-backtest artifacts without retaining snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/posterior_backtest"
SEASONS = (2022, 2023, 2024, 2025)


def main() -> None:
    panels = {
        str(season): json.loads((OUT / f"{season}_rolling.json").read_text())
        for season in SEASONS
    }
    summary = {"seasons": list(SEASONS), "cutoffs_per_season": 7, "season_end": {}}
    for season, panel in panels.items():
        first, last = panel["cutoffs"][0], panel["cutoffs"][-1]
        summary["season_end"][season] = {
            "context": last["context"]["posterior"],
            "history": last["history"]["posterior"],
            "context_posterior_minus_prior": last["context"]["posterior_minus_prior"],
            "history_posterior_minus_prior": last["history"]["posterior_minus_prior"],
            "context_minus_history_posterior": last["context_minus_history_posterior"],
            "context_width_change": last["context"]["posterior"]["interval_80_width"]
            - first["context"]["posterior"]["interval_80_width"],
            "context_coverage_change": last["context"]["posterior"][
                "interval_80_coverage"
            ]
            - first["context"]["posterior"]["interval_80_coverage"],
        }
    (OUT / "panel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    for metric, title, filename in (
        ("nll", "Posterior NLL by season depth", "nll_by_depth.png"),
        (
            "interval_80_coverage",
            "80% interval coverage by season depth",
            "coverage_by_depth.png",
        ),
        (
            "interval_80_width",
            "80% interval width by season depth",
            "width_by_depth.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for season, panel in panels.items():
            depth = range(1, len(panel["cutoffs"]) + 1)
            axis.plot(
                depth,
                [item["context"]["posterior"][metric] for item in panel["cutoffs"]],
                marker="o",
                label=f"{season} C",
            )
            axis.plot(
                depth,
                [item["history"]["posterior"][metric] for item in panel["cutoffs"]],
                marker="x",
                linestyle="--",
                alpha=0.7,
                label=f"{season} H",
            )
        if metric == "interval_80_coverage":
            axis.axhline(
                0.8, color="black", linewidth=1, linestyle=":", label="nominal 80%"
            )
        axis.set(title=title, xlabel="actual-date cutoff depth", ylabel=metric)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        figure.savefig(plots / filename, dpi=150)
        plt.close(figure)


if __name__ == "__main__":
    main()
