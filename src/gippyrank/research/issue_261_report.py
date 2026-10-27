"""Render issue 261 tables and a static research figure from retained results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gippyrank.research.issue_252_quality_of_results import (
    MARGIN_WEIGHTS,
    _mean_ci,
    _write_csv,
    sha256_file,
)
from gippyrank.research.issue_261_quality_of_results import (
    PANELS,
    PROFILES,
    read_csv,
    write_json,
)


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def ci(row: dict, prefix: str, signed: bool = True) -> str:
    fmt = "+.2f" if signed else ".2f"
    return (
        f"{float(row[prefix + '_mean']):{fmt}} "
        f"[{float(row[prefix + '_mean_ci95_low']):{fmt}}, "
        f"{float(row[prefix + '_mean_ci95_high']):{fmt}}]"
    )


def primary_table(summary: list[dict], panel: str) -> str:
    rows = []
    for profile in PROFILES:
        for weight in MARGIN_WEIGHTS:
            row = next(
                row
                for row in summary
                if row["profile"] == profile
                and row["panel"] == panel
                and row["comparison"] == "schedule_pair"
                and float(row["margin_weight"]) == weight
            )
            rows.append(
                [
                    profile,
                    f"{100 * float(row['achieved_win_probability_separation']):.2f}",
                    f"{weight:g}",
                    ci(row, "rank"),
                    f"{float(row['rank_difference_median']):+.2f}",
                    f"{float(row['rank_difference_sd']):.2f}",
                    f"{float(row['rank_difference_p05']):+.2f} / {float(row['rank_difference_p95']):+.2f}",
                    (
                        f"{float(row['right_outranks_left_probability']):.3f} "
                        f"[{float(row['right_outranks_left_ci95_low']):.3f}, "
                        f"{float(row['right_outranks_left_ci95_high']):.3f}]"
                    ),
                ]
            )
    return table(
        [
            "Profile",
            "Separation (pp)",
            "λ",
            "Mean Δ [95% MC CI]",
            "Median Δ",
            "SD(Δ)",
            "p05 / p95",
            "P(easy outranks) [Wilson 95%]",
        ],
        rows,
    )


def controls_table(summary: list[dict]) -> str:
    rows = []
    for profile in PROFILES:
        for weight in MARGIN_WEIGHTS:
            cells = [profile, f"{weight:g}"]
            for panel in PANELS:
                row = next(
                    row
                    for row in summary
                    if row["profile"] == profile
                    and row["panel"] == panel
                    and row["comparison"] == "same_schedule_control"
                    and float(row["margin_weight"]) == weight
                )
                cells.extend(
                    [
                        ci(row, "rank"),
                        f"{float(row['rank_difference_sd']):.2f}",
                        f"{float(row['rank_difference_p05']):+.2f} / {float(row['rank_difference_p95']):+.2f}",
                    ]
                )
            rows.append(cells)
    return table(
        [
            "Profile",
            "λ",
            "Unconditional mean Δ [CI]",
            "SD",
            "p05 / p95",
            "Primary-seed mean Δ [CI]",
            "SD",
            "p05 / p95",
            "Own-record mean Δ [CI]",
            "SD",
            "p05 / p95",
        ],
        rows,
    )


def uncertainty_table(summary: list[dict]) -> str:
    rows = []
    for profile in PROFILES:
        for panel in ("unconditional", "primary_conditioned"):
            for weight in MARGIN_WEIGHTS:
                row = next(
                    row
                    for row in summary
                    if row["profile"] == profile
                    and row["panel"] == panel
                    and row["comparison"] == "schedule_pair"
                    and float(row["margin_weight"]) == weight
                )
                rows.append(
                    [
                        profile,
                        panel,
                        f"{weight:g}",
                        ci(row, "rank_sd"),
                        ci(row, "entropy_bits"),
                        ci(row, "interval_80_width"),
                        f"{float(row['pmf_total_variation_mean']):.3f}",
                        f"{float(row['pmf_wasserstein_mean']):.2f}",
                    ]
                )
    return table(
        [
            "Profile",
            "Panel",
            "λ",
            "Δ rank SD [CI]",
            "Δ entropy bits [CI]",
            "Δ 80% width [CI]",
            "Mean TV",
            "Mean W1 (ranks)",
        ],
        rows,
    )


def candidate_contrasts(output: Path) -> list[dict]:
    """Compare schedule effects across candidates using the matched seeds."""
    paired = defaultdict(lambda: defaultdict(dict))
    for row in read_csv(output / "pair_trial_summaries.csv.gz"):
        paired[row["profile"], row["panel"], row["comparison"]][row["seed"]][
            float(row["margin_weight"])
        ] = float(row["rank_difference"])
    contrasts = []
    for (profile, panel, comparison), seeds in sorted(paired.items()):
        for left, right in combinations(MARGIN_WEIGHTS, 2):
            differences = [weights[right] - weights[left] for weights in seeds.values()]
            mean, low, high = _mean_ci(differences)
            contrasts.append(
                {
                    "profile": profile,
                    "panel": panel,
                    "comparison": comparison,
                    "left_margin_weight": left,
                    "right_margin_weight": right,
                    "draw_count": len(differences),
                    "effect_change_mean": mean,
                    "effect_change_mean_ci95_low": low,
                    "effect_change_mean_ci95_high": high,
                    "effect_change_median": float(np.median(differences)),
                    "effect_change_p05": float(np.quantile(differences, 0.05)),
                    "effect_change_p95": float(np.quantile(differences, 0.95)),
                }
            )
    _write_csv(output / "candidate_contrasts.csv", contrasts)
    return contrasts


def contrasts_table(contrasts: list[dict]) -> str:
    rows = []
    for profile in PROFILES:
        for panel in ("unconditional", "primary_conditioned"):
            cells = [profile, panel]
            for left, right in ((0.0, 0.25), (0.25, 1.0)):
                row = next(
                    row
                    for row in contrasts
                    if row["profile"] == profile
                    and row["panel"] == panel
                    and row["comparison"] == "schedule_pair"
                    and row["left_margin_weight"] == left
                    and row["right_margin_weight"] == right
                )
                cells.append(ci(row, "effect_change"))
            rows.append(cells)
    return table(
        [
            "Profile",
            "Panel",
            "Change in Δ: λ=0 → 0.25 [CI]",
            "Change in Δ: λ=0.25 → 1 [CI]",
        ],
        rows,
    )


def arms_table(arms: list[dict]) -> str:
    rows = []
    for profile in PROFILES:
        for panel in PANELS:
            for group in (
                "approximately_even",
                "weaker_opponents",
                "same_schedule_a",
                "same_schedule_b",
            ):
                matches = [
                    row
                    for row in arms
                    if row["profile"] == profile
                    and row["panel"] == panel
                    and row["focal_group"] == group
                    and float(row["margin_weight"]) == 0.25
                ]
                if not matches:
                    continue
                row = matches[0]
                label = {
                    "approximately_even": "difficult",
                    "weaker_opponents": "easy",
                    "same_schedule_a": "control A",
                    "same_schedule_b": "control B",
                }[group]
                rows.append(
                    [
                        profile,
                        panel,
                        label,
                        f"{float(row['focal_true_rank_mean']):.2f}",
                        f"{float(row['wins_mean']):.2f}",
                        f"{float(row['wins_p05']):g} / {float(row['wins_p95']):g}",
                        f"{float(row['mean_focal_margin_mean']):+.2f}",
                        f"{float(row['expected_rank_mean']):.2f}",
                        f"{float(row['rank_sd_mean']):.2f}",
                        f"{float(row['entropy_bits_mean']):.2f}",
                        f"{float(row['interval_80_width_mean']):.2f}",
                    ]
                )
    return table(
        [
            "Profile",
            "Panel",
            "Arm",
            "Mean latent rank",
            "Mean wins",
            "Wins p05 / p95",
            "Mean margin",
            "Mean E[rank]",
            "Mean rank SD",
            "Mean entropy bits",
            "Mean 80% width",
        ],
        rows,
    )


def make_plot(summary: list[dict], output: Path) -> None:
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for profile, axis in zip(PROFILES, axes.flat, strict=True):
        for panel, label, color in (
            ("unconditional", "Unconditional (500)", "#2166ac"),
            ("primary_conditioned", "Exact records (200)", "#b35806"),
        ):
            rows = sorted(
                [
                    row
                    for row in summary
                    if row["profile"] == profile
                    and row["panel"] == panel
                    and row["comparison"] == "schedule_pair"
                ],
                key=lambda row: float(row["margin_weight"]),
            )
            weights = [float(row["margin_weight"]) for row in rows]
            means = np.array([float(row["rank_mean"]) for row in rows])
            low = np.array([float(row["rank_mean_ci95_low"]) for row in rows])
            high = np.array([float(row["rank_mean_ci95_high"]) for row in rows])
            axis.errorbar(
                weights,
                means,
                yerr=np.array([means - low, high - means]),
                fmt="o-",
                color=color,
                capsize=4,
                label=label,
            )
        sep = 100 * float(rows[0]["achieved_win_probability_separation"])
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.set(
            title=f"{profile.capitalize()} · schedule separation {sep:.2f} pp",
            xlabel="Margin weight λ",
            ylabel="Mean Δrank (easy − difficult)",
            xticks=MARGIN_WEIGHTS,
        )
        axis.grid(axis="y", alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    figure.suptitle(
        "Quality of Results: frozen schedule contrasts\n95% Monte Carlo intervals for mean effects; negative favors easier schedules"
    )
    figure.savefig(
        output / "schedule_effects.png",
        dpi=160,
        metadata={"Software": "GippyRank4 issue 261"},
    )
    plt.close(figure)


def render(root: Path, output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"result hash mismatch: {name}")
    summary = read_csv(output / "simulation_summary.csv")
    arms = read_csv(output / "arm_summary.csv")
    contrasts = candidate_contrasts(output)
    design_path = root / "data/research/issue_258_quality_of_results"
    profiles = read_csv(design_path / "profile_summary.csv")
    schedules = read_csv(design_path / "schedule_summary.csv")
    setup_rows, execution_rows = [], []
    for profile in PROFILES:
        focal = next(row for row in profiles if row["profile"] == profile)
        pair = {
            row["schedule_arm"]: float(row["mean_win_probability"])
            for row in schedules
            if row["profile"] == profile
        }
        lower, higher = manifest["record_targets"][profile]
        setup_rows.append(
            [
                profile,
                focal["team_name"],
                f"{float(focal['expected_rank']):.2f}",
                f"{float(focal['rank_sd']):.2f}",
                f"{100 * pair['lower_win_probability']:.2f}% / {100 * pair['higher_win_probability']:.2f}%",
                f"{100 * (pair['higher_win_probability'] - pair['lower_win_probability']):.2f}",
                f"{lower}–{10 - lower} / {higher}–{10 - higher}",
            ]
        )
        for panel in PANELS:
            row = manifest["completed_panels"][profile][panel]
            execution_rows.append(
                [
                    profile,
                    panel,
                    str(len(row["seeds"])),
                    str(row["attempts"]),
                    f"{len(row['seeds']) / row['attempts']:.3%}",
                    f"{row['seed_range'][0]}–{row['seeds'][-1]}",
                ]
            )
    make_plot(summary, output)
    interpretation = (root / "docs/issue_261_interpretation.md").read_text()
    text = (
        "\n\n".join(
            [
                "# Issue 261: Multi-strength Quality of Results schedule equivalence",
                interpretation,
                "## Experimental contract and execution",
                "The committed issue 258 contract was used without redesign: 2026 Context 1.4 focal and opponent PMFs on all 138 ranks, Historical Likelihood V1, the frozen ten-opponent slates, and four home/four away/two neutral venues. Colorado remains provenance for issue 252, not a fifth stratum. Each pair shares a sampled focal rank and opponent quantiles; primary arms share Student-t margin draws, and controls have independent margin draws. Wins are the unmodified margin signs. Every λ receives the same generated games. Opponents remain uncertain full-PMF variables in the disconnected inference forest. Synthetic points encode a signed continuous margin and are likelihood observations, not literal football scores.",
                table(
                    [
                        "Profile",
                        "Focal team",
                        "Prior E[rank]",
                        "Prior SD",
                        "Difficult / easy p(win)",
                        "Separation (pp)",
                        "Conditioned records",
                    ],
                    setup_rows,
                ),
                "There are 500 unconditional, 200 primary-record-conditioned, and 200 independently own-record-conditioned control comparisons per stratum. Primary-conditioned seeds retain four arms; control-conditioned seeds infer only the two controls. Controls on primary-conditioned seeds are explicitly labeled `primary_seed_control`; they were not selected on their own records. All eight searches found 200 acceptances before the 20,000-attempt ceilings. Seed search used only records, before candidate ranks. All five existing candidates (0, 0.25, 0.5, 0.75, 1) were evaluated without schedule selection or parameter changes.",
                table(
                    [
                        "Profile",
                        "Panel",
                        "Accepted",
                        "Attempts",
                        "Acceptance",
                        "Searched seeds (inclusive)",
                    ],
                    execution_rows,
                ),
                "Validation checked every frozen artifact, the manifest, design-code, protocol and documentation hashes; reconstructed all schedule probabilities to 1e-12; and smoke-tested all profiles and weights. All focal PMFs were finite, nonnegative, normalized on 138 ranks, and all runs converged using the existing deterministic BP defaults (500 iterations maximum, tolerance 1e-9, damping 0.35). Focal Context priors were stripped by the existing Performance implementation. The research harness clears the engine's unbounded factor cache between trials to bound memory; it does not alter factors or inference. Evidence hashes are identical across weights and rechecked against regenerated observations during aggregation.",
                f"The four-trial smoke benchmark estimated {manifest['smoke']['estimated_serial_inference_minutes']:.1f} serial inference minutes (the frozen planning estimate was 108). The sum of per-trial elapsed inference times was {manifest['total_inference_seconds'] / 60:.1f} minutes across concurrent processes. This includes scheduling and resource contention; it is neither CPU time nor study wall time. The environment followed the repository's `.python-version` pin: Python {manifest['provenance']['versions']['python']}, NumPy {manifest['provenance']['versions']['numpy']}, SciPy {manifest['provenance']['versions']['scipy']}. NumPy and SciPy match the frozen design versions. No scientific deviations or execution limitations were needed.",
                "## Paired schedule effects",
                "Δrank = E[rank easy] − E[rank difficult]. Negative favors easier schedules. Intervals are normal 95% Monte Carlo confidence intervals for paired means; outrank intervals are Wilson 95% intervals for the fraction with strictly smaller easy-arm expected rank. They describe finite-simulation precision, not historical external validity. Individual-season p05/p95 and SD describe realized variation; posterior uncertainty below is a separate quantity. Quantiles use NumPy's default linear interpolation. The 100 rows in `simulation_summary.csv` retain additional quartiles and CIs for every uncertainty and distance metric.",
                "![Schedule effects by profile and margin weight](schedule_effects.png)",
                "### Unconditional populations (500 pairs per profile)",
                primary_table(summary, "unconditional"),
                "### Exact-record primary populations (200 accepted pairs per profile)",
                primary_table(summary, "primary_conditioned"),
                "### Candidate differences on the same evidence",
                "These are within-seed changes in the schedule effect between candidates, with paired 95% Monte Carlo intervals. A positive change shifts the easy-minus-difficult effect toward the difficult schedule; it does not by itself imply an overall difficult-schedule advantage. All ten candidate pairs, including controls, are retained in `candidate_contrasts.csv`. These comparisons do not define a candidate-selection objective or an equivalence threshold.",
                contrasts_table(contrasts),
                "## Same-schedule outcome variation",
                "Control Δ is E[rank B] − E[rank A], with both controls on the difficult slate. B is not an easier-schedule arm. All three control populations are shown separately. They are baselines alongside the schedule contrasts, not bias estimates subtracted from them. Primary arms share margin shocks, while control margins are independent; the spread difference between primary and control pairs therefore is not a clean decomposition of variance into schedule and outcome terms. Controls share latent focal and opponent draws, so this baseline is also conditional on that matching structure.",
                controls_table(summary),
                "Control medians, quartiles, outrank probabilities/Wilson intervals, TV/Wasserstein distances and paired uncertainty CIs are in the machine-readable summary alongside every primary comparison.",
                "## Posterior uncertainty and realized outcomes",
                "The next table reports easy minus difficult mean posterior SD, entropy in bits, and central 80% width, with 95% Monte Carlo intervals. Width is the inclusive number of rank positions between the 10th and 90th percentile ranks. TV is half the L1 PMF distance; W1 is the sum of absolute CDF differences over unit-spaced ranks. Distance means compare entire paired PMFs and need not vanish when expected ranks are close.",
                uncertainty_table(summary),
                "The following arm summaries expose absolute posterior uncertainty, latent-rank selection, records and margins at λ=0.25. All five weights and all arm/population combinations, record-frequency distributions, and per-arm quantiles/CIs are in `arm_summary.csv`; game-level observations and per-trial summaries are retained separately.",
                arms_table(arms),
                "## Reproduction and artifact inventory",
                "Run from the repository root with its pinned environment. The profile commands can execute concurrently, each in a separate process with one BLAS thread. No research inference is added to ordinary CI; tests audit invariants and retained artifacts.",
                "```bash\nuv sync --locked\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\nuv run python scripts/run_issue_261_quality_of_results.py --smoke\nuv run python scripts/run_issue_261_quality_of_results.py --profile elite\nuv run python scripts/run_issue_261_quality_of_results.py --profile strong\nuv run python scripts/run_issue_261_quality_of_results.py --profile middle\nuv run python scripts/run_issue_261_quality_of_results.py --profile weak\nuv run python scripts/run_issue_261_quality_of_results.py --aggregate\nuv run python -m gippyrank.research.issue_261_report\n```",
                "`manifest.json` pins source revision and source hashes, existing input paths/hashes, all accepted seeds and attempted ranges, record criteria, inference settings, completed counts, smoke results and output hashes. It references the committed issue 258 input artifacts rather than duplicating them. `.checkpoints/` is an ignored, atomic per-trial cache; a changed execution fingerprint rejects reuse. A clean run regenerates it. Simulation wall timings may differ; the seeded evidence and numerical result tables are deterministic in the recorded environment. The report/figure can be refreshed from portable results without rerunning inference.",
                "- `simulated_games.csv.gz`: all 128,000 game observations, latent rank draws, venues, source opponents, margins and wins.\n- `team_trial_summaries.csv.gz`: all 64,000 arm/candidate rows with records, margins, full uncertainty summaries, convergence and evidence hashes.\n- `pair_trial_summaries.csv.gz`: all 32,000 pair/candidate rows with rank effects, posterior uncertainty differences, TV and Wasserstein.\n- `simulation_summary.csv`: 100 aggregate profile/panel/comparison/candidate rows.\n- `arm_summary.csv`: all arm/candidate aggregates, outcome distributions and posterior uncertainty.\n- `candidate_contrasts.csv`: 200 paired comparisons of candidate effects on identical trials.\n- `schedule_effects.png`: static figure of primary mean effects and Monte Carlo intervals.\n- `report.md`: this report, with methods and interpretation.\n\nThe legacy machine roles `approximately_even` and `weaker_opponents` mean the frozen difficult and easy arms respectively; they do not imply 50% win probabilities in every stratum. Full per-trial posterior PMFs are reproducible from the saved seed, games, frozen priors, likelihood and source; they are not duplicated in the output. No production ranking, predictive methodology, website or API changed.",
            ]
        )
        + "\n"
    )
    (output / "report.md").write_text(text)
    manifest["artifact_sha256"].update(
        {
            name: sha256_file(output / name)
            for name in ("report.md", "schedule_effects.png", "candidate_contrasts.csv")
        }
    )
    manifest["report_generation"] = {
        "method": "render_saved_results_without_inference",
        "source_sha256": sha256_file(Path(__file__)),
        "interpretation_path": "docs/issue_261_interpretation.md",
        "interpretation_sha256": sha256_file(root / "docs/issue_261_interpretation.md"),
    }
    write_json(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    render(
        args.root,
        args.output_dir or args.root / "data/research/issue_261_quality_of_results",
    )


if __name__ == "__main__":
    main()
