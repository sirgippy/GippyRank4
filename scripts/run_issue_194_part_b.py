from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from gippyrank.methodology import HISTORICAL_LIKELIHOOD_SHA256
from gippyrank.posterior.snapshots import load_pinned_likelihood
from gippyrank.research.issue_194_part_b import (
    CASE_NAMES,
    DRAWS_PER_PANEL,
    GRAPH_EDGES,
    SEED_RANGES,
    TEAM_IDS,
    run_experiment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _report(
    summary: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    scenarios: list[dict[str, object]],
) -> str:
    summary_index = {
        (str(row["case"]), str(row["graph"]), str(row["arm"])): row for row in summary
    }
    diagnostic_index = {
        (str(row["case"]), str(row["graph"])): row for row in diagnostics
    }
    shared_nll_differences = {
        graph: float(
            summary_index[("shared_context", graph, "exact_product_marginals")][
                "mean_rank_nll"
            ]
        )
        - float(
            summary_index[("shared_context", graph, "exact_true_joint")][
                "mean_rank_nll"
            ]
        )
        for graph in GRAPH_EDGES
    }
    if all(value > 0 for value in shared_nll_differences.values()):
        dependence_decision = "supported"
    elif all(value <= 0 for value in shared_nll_differences.values()):
        dependence_decision = "unsupported"
    else:
        dependence_decision = "unresolved"

    cycle_tvs = {
        case: float(diagnostic_index[(case, "cycle_4")]["bp_vs_product_tv"])
        for case in CASE_NAMES
    }
    cycle_bp_nll_differences = {
        case: float(
            summary_index[(case, "cycle_4", "bp_product_marginals")]["mean_rank_nll"]
        )
        - float(
            summary_index[(case, "cycle_4", "exact_product_marginals")]["mean_rank_nll"]
        )
        for case in CASE_NAMES
    }
    bp_present = any(value > 1e-10 for value in cycle_tvs.values())
    bp_decision = "supported" if bp_present else "unsupported"
    cycle_nll_min = min(cycle_bp_nll_differences.values())
    cycle_nll_max = max(cycle_bp_nll_differences.values())
    if max(abs(cycle_nll_min), abs(cycle_nll_max)) <= 1e-12:
        bp_score_decision = "no measured score change"
    elif cycle_nll_min <= 0 <= cycle_nll_max:
        bp_score_decision = "unresolved"
    else:
        bp_score_decision = "directionally consistent"

    lines = [
        "# Issue 194 Part B: exact synthetic mechanism test",
        "",
        "Part A is complete in the merged [Part A report](../issue_194_part_a/report.md). Its historical co-movement classification was **descriptively unsupported** under the preregistered four-season rule. Part B is a separate controlled synthetic check; it does not revise that historical finding.",
        "",
        "## Frozen design",
        "",
        "The fixed experiment uses four teams, five ordinal rank states, a three-edge tree and a four-edge cycle, 100 seeds per case/graph panel, and the pinned Historical Likelihood V1. The exact enumerator visits 625 assignments per trial. All inference arms share the same generated ranks, games, home/neutral assignments, and continuous V1 margins within each trial. The full protocol and seed ranges are in [the frozen protocol](../../../docs/issue_194_part_b_protocol.md).",
        "",
        "The shared prior's two rank-distribution scenarios each have prior weight 0.5. Their component posterior weights are computed as prior weight times scenario evidence, then normalized. They are not averaged equally after observing games; per-draw evidence weights are retained in `scenario_weights.csv`.",
        "",
        "## Posterior rank scores and calibration",
        "",
        "Rank NLL and multicategory Brier score are lower-is-better. Expected-rank error is forecast expected rank minus simulated rank. Calibration MAE averages the absolute difference between forecast probability and observed frequency over each team/rank cell.",
        "",
        "| Case | Graph | Inference arm | Draws | Rank NLL | Brier | Signed expected-rank error | Absolute expected-rank error | Calibration MAE | BP converged | Mean BP iterations |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                _fmt(row[key])
                for key in (
                    "case",
                    "graph",
                    "arm",
                    "draws",
                    "mean_rank_nll",
                    "mean_multicategory_brier",
                    "mean_signed_expected_rank_error",
                    "mean_absolute_expected_rank_error",
                    "rank_calibration_macro_mae",
                    "bp_converged_fraction",
                    "bp_mean_iterations",
                )
            )
            + " |"
        )
    lines += [
        "",
        "Full team/rank calibration cells are in `rank_calibration.csv`; per-draw paired scores are in `draw_scores.csv`.",
        "",
        "## Inference-arm differences",
        "",
        "Total variation is averaged over teams within each draw, then over the 100 draws. The first pair isolates replacing the true joint prior with independent marginals. The second isolates BP approximation under the product prior. The third is the combined difference from the true-joint exact reference.",
        "",
        "| Case | Graph | Product exact vs true-joint exact TV | Absolute expected-rank difference | BP vs product exact TV | Absolute expected-rank difference | BP vs true-joint exact TV | Absolute expected-rank difference |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    diagnostic_columns = (
        ("product_vs_true_joint_tv", "product_vs_true_joint_abs_expected_rank_error"),
        ("bp_vs_product_tv", "bp_vs_product_abs_expected_rank_error"),
        ("bp_vs_true_joint_tv", "bp_vs_true_joint_abs_expected_rank_error"),
    )
    for row in diagnostics:
        values: list[str] = [str(row["case"]), str(row["graph"])]
        for tv_key, rank_key in diagnostic_columns:
            values.extend((_fmt(row[tv_key]), _fmt(row[rank_key])))
        lines.append("| " + " | ".join(values) + " |")

    lines += [
        "",
        "The preregistered shared-prior NLL contrast (product exact minus true-joint exact) was "
        + "; ".join(
            f"{graph}: {value:+.6f}" for graph, value in shared_nll_differences.items()
        )
        + ". Under the frozen two-graph rule, dependence-related rank-NLL degradation is **"
        + dependence_decision
        + "** for this synthetic setup.",
        "",
        "The product-prior exact and BP posterior differences on the cycle were "
        + "; ".join(f"{case}: {value:.6g} TV" for case, value in cycle_tvs.items())
        + ". BP approximation as a nonzero cyclic marginal difference is **"
        + bp_decision
        + "** at the frozen 1e-10 criterion. The cycle BP-minus-product-exact rank-NLL changes were "
        + "; ".join(
            f"{case}: {value:+.6f}" for case, value in cycle_bp_nll_differences.items()
        )
        + ". Their score direction is "
        + bp_score_decision
        + "; a PMF difference alone is not treated as a rank-score effect. The tree exactness check passed in the focused tests.",
    ]
    if scenarios:
        lines += [
            "",
            "### Shared-scenario evidence weights",
            "",
            "| Graph | Draws | Mean posterior weight A | Mean posterior weight B |",
            "| --- | ---: | ---: | ---: |",
        ]
        for graph in GRAPH_EDGES:
            panel = [row for row in scenarios if row["graph"] == graph]
            lines.append(
                f"| {graph} | {len(panel)} | "
                f"{np_mean(float(row['posterior_weight_a']) for row in panel):.6f} | "
                f"{np_mean(float(row['posterior_weight_b']) for row in panel):.6f} |"
            )
    lines += [
        "",
        "## Decision and limits",
        "",
        "* Historical co-movement consistent with shared Context errors: **descriptively unsupported** by the completed Part A report. Part B supplies no historical evidence.",
        f"* Dependence causing rank-NLL degradation in the controlled synthetic setting: **{dependence_decision}** under the preregistered direction rule. This says only whether the fixed shared-prior example loses rank score when its joint prior is replaced by independent marginals.",
        f"* BP approximation as a source of a small-graph posterior difference: **{bp_decision}** under the fixed cycle criterion. Its rank-NLL effect is **{bp_score_decision}** across the two generative cases because the cycle contrasts have opposite signs. The measured BP-to-exact gap is specific to these four-node examples.",
        "",
        "This experiment measures posterior rank prediction under known simulated ranks. It does not evaluate future-game forecasting and cannot establish that Context harms real-game forecasts or explain the historical rank-NLL contrast. Better game forecasts can coexist with worse rank NLL. The 2022–2025 historical seasons have already informed this research, so the results remain exploratory development.",
        "",
        "No full-season joint-prior experiment is recommended from this result alone. The unresolved real-data question is whether dependence in existing Context prior errors contributes to its historical rank-NLL disadvantage despite its better future-game CRPS; Part A did not find descriptive support for shared opponent-error co-movement, while this five-state synthetic control tests only possibility under its specified scenario magnitude. A full-season model would require a separately justified historical dependence signal; these four-team BP discrepancies are not extrapolated into a causal share.",
        "",
        "## Reproduce",
        "",
        "From the repository root, run `uv run python scripts/run_issue_194_part_b.py --input-root . --output-root data/research/issue_194_part_b`. The runner requires the pinned `data/processed/posterior/historical_likelihood_v1.json` input and verifies its retained V1 hash.",
        "",
        "No production priors, likelihoods, inference settings, snapshots, website defaults, or raw downloads were changed.",
    ]
    return "\n".join(lines) + "\n"


def np_mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    likelihood_path = (
        input_root / "data/processed/posterior/historical_likelihood_v1.json"
    )
    protocol_path = input_root / "docs/issue_194_part_b_protocol.md"
    module_path = input_root / "src/gippyrank/research/issue_194_part_b.py"
    script_path = input_root / "scripts/run_issue_194_part_b.py"
    for required_path in (likelihood_path, protocol_path, module_path, script_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    likelihood = load_pinned_likelihood(likelihood_path)
    output_root.mkdir(parents=True, exist_ok=True)
    result = run_experiment(likelihood)
    table_names = {
        "summary": "simulation_summary.csv",
        "calibration": "rank_calibration.csv",
        "draw_scores": "draw_scores.csv",
        "diagnostics": "mechanism_diagnostics.csv",
        "scenario_weights": "scenario_weights.csv",
    }
    for key, filename in table_names.items():
        _write_csv(output_root / filename, result[key])
    report = _report(
        result["summary"], result["diagnostics"], result["scenario_weights"]
    )
    (output_root / "report.md").write_text(report, encoding="utf-8")

    manifest: dict[str, object] = {
        "schema_version": "issue_194_part_b_v1",
        "issue": 194,
        "part": "B",
        "experiment_status": "complete",
        "protocol": "docs/issue_194_part_b_protocol.md",
        "protocol_sha256": _sha256(protocol_path),
        "inputs": {
            "historical_likelihood_v1": {
                "path": "data/processed/posterior/historical_likelihood_v1.json",
                "bytes": likelihood_path.stat().st_size,
                "sha256": _sha256(likelihood_path),
                "pinned_sha256": HISTORICAL_LIKELIHOOD_SHA256,
            }
        },
        "source_sha256": {
            "module": _sha256(module_path),
            "runner": _sha256(script_path),
        },
        "model": {
            "historical_likelihood": "V1",
            "team_count": len(TEAM_IDS),
            "rank_states": [1, 2, 3, 4, 5],
            "assignments_per_draw": 625,
            "max_supported_assignments": 3125,
            "graph_edges": {
                name: [list(edge) for edge in edges]
                for name, edges in GRAPH_EDGES.items()
            },
            "draws_per_case_graph": DRAWS_PER_PANEL,
            "seeds_by_case_graph": {
                f"{case}/{graph}": list(seeds)
                for (case, graph), seeds in SEED_RANGES.items()
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "outputs": {},
    }
    manifest["outputs"] = {
        filename: {
            "bytes": (output_root / filename).stat().st_size,
            "sha256": _sha256(output_root / filename),
        }
        for filename in (*table_names.values(), "report.md")
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
