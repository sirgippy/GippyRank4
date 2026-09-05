"""Add 2025 component-disagreement and team-example views to the ablation study."""

from __future__ import annotations

from pathlib import Path

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior as v1
import build_preseason_prior_v1_1 as h11
import numpy as np
from investigate_context_ablation import (
    ALL_CONTEXT,
    RECRUIT,
    RETURN_ALL,
    RTP,
    TALENT,
    fit_context,
    fit_h,
    predictions,
    restrict_observed,
    standardized_interaction_rows,
    write_csv,
)

from gippyrank.preseason import pmf_summaries

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/context_ablation"
TARGET = 2025


def expected(prediction: v1.PriorPrediction) -> float:
    return pmf_summaries(prediction.pmf)["expected_rank"]


def interaction_comparisons(contextual: list[object]) -> list[dict[str, object]]:
    """Compare each interaction to its exact no-interaction parent by season."""
    definitions = [
        (
            "talent_x_total_returning",
            [*TALENT, "returning_pct_ppa"],
            ("talent_composite", "returning_pct_ppa"),
        ),
        (
            "talent_x_passing_returning",
            [*TALENT, "returning_pct_passing_ppa"],
            ("talent_composite", "returning_pct_passing_ppa"),
        ),
        (
            "coach_x_total_returning",
            ["coach_tenure_seasons", "returning_pct_ppa"],
            ("coach_tenure_seasons", "returning_pct_ppa"),
        ),
    ]
    records = []
    for name, features, pair in definitions:
        source = restrict_observed(contextual, features)
        for target in range(2017, TARGET + 1):
            train = [row for row in source if row.season < target]
            target_rows = [row for row in source if row.season == target]
            if len({row.season for row in train}) < 3 or len(target_rows) < 20:
                continue
            parent = predictions(fit_context(train, features), target_rows, "parent")
            augmented_train, augmented_target, interaction = (
                standardized_interaction_rows(train, target_rows, *pair)
            )
            child = predictions(
                fit_context(augmented_train, [*features, interaction]),
                augmented_target,
                "interaction",
            )
            parent_loss, child_loss = (
                h11.prediction_losses(parent),
                h11.prediction_losses(child),
            )
            assert set(parent_loss) == set(child_loss)
            records.append(
                {
                    "interaction": name,
                    "target_season": target,
                    "same_population_keys": True,
                    "n_team_seasons": len(parent_loss),
                    "interaction_minus_parent_nll": float(
                        np.mean(
                            [
                                child_loss[key][0] - parent_loss[key][0]
                                for key in parent_loss
                            ]
                        )
                    ),
                }
            )
    return records


def main() -> None:
    rows, _cold, _coverage = v1.load_rows(max_season=TARGET)
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _ = c12.attach_context(fbs, c12.feature_index(), c12.cached_tenures())
    rtp = restrict_observed(contextual, RTP)
    train_rtp = [row for row in rtp if row.season < TARGET]
    target_rtp = [row for row in rtp if row.season == TARGET]
    models = {
        "H": fit_h(train_rtp),
        "recruiting": fit_context(train_rtp, RECRUIT),
        "talent": fit_context(train_rtp, TALENT),
        "returning": fit_context(train_rtp, RETURN_ALL),
    }
    predicted = {
        name: {row.key: row for row in predictions(model, target_rtp, name)}
        for name, model in models.items()
    }
    losses = {
        name: h11.prediction_losses(list(values.values()))
        for name, values in predicted.items()
    }
    all_context = restrict_observed(contextual, ALL_CONTEXT)
    train_all = [row for row in all_context if row.season < TARGET]
    target_all = [row for row in all_context if row.season == TARGET]
    h_all = {row.key: row for row in predictions(fit_h(train_all), target_all, "H_all")}
    c_all = {
        row.key: row
        for row in predictions(fit_context(train_all, ALL_CONTEXT), target_all, "C")
    }
    h_all_losses, c_all_losses = (
        h11.prediction_losses(list(h_all.values())),
        h11.prediction_losses(list(c_all.values())),
    )

    component = []
    for key, h_prediction in predicted["H"].items():
        adjustments = {
            name: expected(predicted[name][key]) - expected(h_prediction)
            for name in ("recruiting", "talent", "returning")
        }
        signs = [np.sign(value) for value in adjustments.values() if abs(value) >= 1]
        agreement = "agreement" if len(set(signs)) <= 1 else "conflict"
        component.append(
            {
                "season": TARGET,
                "team_id": key[2],
                "team_name": h_prediction.team_name,
                "recruiting_adjustment": adjustments["recruiting"],
                "talent_adjustment": adjustments["talent"],
                "returning_adjustment": adjustments["returning"],
                "component_pattern": agreement,
                "recruiting_minus_h_nll": losses["recruiting"][key][0]
                - losses["H"][key][0],
                "talent_minus_h_nll": losses["talent"][key][0] - losses["H"][key][0],
                "returning_minus_h_nll": losses["returning"][key][0]
                - losses["H"][key][0],
            }
        )
    write_csv("component_disagreement.csv", component)
    interaction_rows = interaction_comparisons(contextual)
    write_csv("interaction_results.csv", interaction_rows)

    requested = ["New Mexico", "Utah", "James Madison", "North Texas"]
    best_c = sorted(c_all, key=lambda key: c_all_losses[key][0] - h_all_losses[key][0])[
        :4
    ]
    selection = [*requested, *(c_all[key].team_name for key in best_c)]
    raw_2025 = {row.team_name: row for row in contextual if row.season == TARGET}
    rtp_by_name = {row.team_name: row for row in target_rtp}
    all_by_name = {row.team_name: row for row in target_all}
    examples = []
    for name in dict.fromkeys(selection):
        raw = raw_2025.get(name)
        rtp_row = rtp_by_name.get(name)
        all_row = all_by_name.get(name)
        record: dict[str, object] = {
            "season": TARGET,
            "team_name": name,
            "team_id": raw.team_id if raw else None,
            "rtp_observed": rtp_row is not None,
            "all_context_observed": all_row is not None,
            "missing_raw_context_features": ";".join(
                feature
                for feature in ALL_CONTEXT
                if raw is not None and raw.features.get(feature) is None
            ),
        }
        if rtp_row is not None:
            key = (TARGET, "fbs", rtp_row.team_id)
            record.update(
                {
                    "h_expected_rank": expected(predicted["H"][key]),
                    "recruiting_expected_rank": expected(predicted["recruiting"][key]),
                    "talent_expected_rank": expected(predicted["talent"][key]),
                    "returning_expected_rank": expected(predicted["returning"][key]),
                    "realized_rank_mean": float(np.mean(rtp_row.target_ranks)),
                    "realized_rank_min": int(np.min(rtp_row.target_ranks)),
                    "realized_rank_max": int(np.max(rtp_row.target_ranks)),
                }
            )
        if all_row is not None:
            key = (TARGET, "fbs", all_row.team_id)
            record.update(
                {
                    "all_context_h_expected_rank": expected(h_all[key]),
                    "c_expected_rank": expected(c_all[key]),
                    "c_minus_h_nll": c_all_losses[key][0] - h_all_losses[key][0],
                }
            )
        examples.append(record)
    write_csv("team_examples_2025.csv", examples)

    pattern = {}
    for label in ("agreement", "conflict"):
        values = [
            row["returning_minus_h_nll"]
            for row in component
            if row["component_pattern"] == label
        ]
        pattern[label] = {
            "n": len(values),
            "mean_returning_minus_h_nll": float(np.mean(values)) if values else None,
        }
    report = OUT / "report.md"
    report.write_text(
        report.read_text(encoding="utf-8")
        + "\n## 2025 component disagreement\n\n"
        + "Component directions are calculated on the coach-free RTP observed population. "
        + f"Returning-only ΔNLL averaged {pattern['agreement']['mean_returning_minus_h_nll']:.4f} for {pattern['agreement']['n']} agreement cases and {pattern['conflict']['mean_returning_minus_h_nll']:.4f} for {pattern['conflict']['n']} conflict cases. Team examples retain rows with unavailable coach coverage rather than silently excluding them.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
