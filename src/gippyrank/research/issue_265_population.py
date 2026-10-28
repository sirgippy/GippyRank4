"""Deterministic population descriptions for the frozen issue 265 baselines.

These are observed-population descriptions, not causal schedule-effect tests.
No case selection or game perturbation is performed.
"""

from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import stdtr
from scipy.stats import spearmanr

from gippyrank.posterior.predictive import (
    ScheduledGame,
    conditional_margin_location_surface,
)
from gippyrank.posterior.snapshots import sha256
from gippyrank.research.issue_242_quality_of_results import (
    MARGIN_WEIGHTS,
    _read_json,
    _read_rows,
)
from gippyrank.research.issue_261_quality_of_results import write_json
from gippyrank.research.issue_263_real_season_design import ARTIFACT_DIR, check_design
from gippyrank.research.issue_265_real_seasons import (
    cell_fingerprint,
    load_network,
    mean_rank,
    provenance,
    read_checkpoint,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write stable columns/rows, with deterministic gzip where requested."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                k: json.dumps(v, sort_keys=True, separators=(",", ":"))
                if isinstance(v, (list, dict))
                else v
                for k, v in row.items()
            }
        )
    payload = buffer.getvalue().encode()
    path.write_bytes(
        gzip.compress(payload, mtime=0) if path.suffix == ".gz" else payload
    )


def summary(p: np.ndarray) -> dict:
    ranks = np.arange(1, len(p) + 1)
    mean = mean_rank(p)
    cdf = np.cumsum(p)

    def quantile(q):
        return min(len(p), int(np.searchsorted(cdf, q, side="left")) + 1)

    value = {
        "expected_rank": mean,
        "median_rank": quantile(0.5),
        "modal_rank": int(np.argmax(p)) + 1,
        "rank_sd": float(np.sqrt(p @ (ranks - mean) ** 2)),
        "entropy_bits": -float(p[p > 0] @ np.log2(p[p > 0])),
    }
    for coverage in (50, 80, 95):
        tail = (1 - coverage / 100) / 2
        low, high = quantile(tail), quantile(1 - tail)
        value.update(
            {
                f"low_{coverage}": low,
                f"high_{coverage}": high,
                f"width_{coverage}": high - low + 1,
            }
        )
    for boundary in (10, 12, 25):
        value[f"mass_le_{boundary}"] = float(p[:boundary].sum())
    return value


def distribution(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "mean_absolute": None,
            "p05": None,
            "p95": None,
            "max_absolute": None,
        }
    a = np.asarray(values)
    return {
        "n": len(a),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "mean_absolute": float(np.mean(abs(a))),
        "p05": float(np.quantile(a, 0.05)),
        "p95": float(np.quantile(a, 0.95)),
        "max_absolute": float(np.max(abs(a))),
    }


def order(pmfs: dict[str, np.ndarray], counts: dict[str, int]) -> dict[str, int]:
    ids = sorted((t for t in pmfs if counts[t]), key=lambda t: (mean_rank(pmfs[t]), t))
    return {t: i for i, t in enumerate(ids, 1)}


def compare(p: np.ndarray, q: np.ndarray) -> dict:
    """Higher weight (q) minus lower weight (p), or later minus earlier."""
    left, right = summary(p), summary(q)
    fields = (
        "expected_rank",
        "rank_sd",
        "entropy_bits",
        "width_50",
        "width_80",
        "width_95",
        "mass_le_10",
        "mass_le_12",
        "mass_le_25",
    )
    return {
        **{f"delta_{k}": right[k] - left[k] for k in fields},
        "total_variation": float(0.5 * abs(p - q).sum()),
        "wasserstein_rank": float(abs(np.cumsum(p) - np.cumsum(q)).sum()),
    }


def top_changes(left: dict[str, int], right: dict[str, int], k: int) -> dict:
    a, b = {t for t in left if left[t] <= k}, {t for t in right if right[t] <= k}
    return {
        f"top{k}_overlap": len(a & b),
        f"top{k}_symmetric_difference": len(a ^ b),
        f"top{k}_entries": sorted(b - a),
        f"top{k}_exits": sorted(a - b),
        f"top{k}_union_movement": [
            {"team_id": t, "before": left.get(t), "after": right.get(t)}
            for t in sorted(a | b)
        ],
    }


def contrast_summary(rows: list[dict]) -> dict:
    result = {
        f"{field}_{stat}": value
        for field in (
            "delta_expected_rank",
            "delta_ordinal_rank",
            "delta_rank_sd",
            "delta_entropy_bits",
            "delta_width_50",
            "delta_width_80",
            "delta_width_95",
            "total_variation",
            "wasserstein_rank",
        )
        for stat, value in distribution([r[field] for r in rows]).items()
    }
    if rows:
        worst = max(rows, key=lambda r: abs(r["delta_expected_rank"]))
        result["max_absolute_expected_rank_team_id"] = worst["team_id"]
    return result


def frozen_descriptors(root: Path, snapshot_id: str, season: int) -> dict[str, dict]:
    file = (
        "historical_reference_descriptors.csv"
        if season == 2025
        else "reference_descriptors.csv"
    )
    result = {}
    for row in _read_rows(root / ARTIFACT_DIR / file):
        if row.get("snapshot_id", row.get("publication_slot")) != snapshot_id:
            continue
        r = dict(row)
        for key in (
            "games",
            "wins",
            "losses",
            "qop_ordinal",
            "qor_ordinal",
            "reference_ordinal",
        ):
            if key in r:
                r[key] = int(r[key]) if r[key] else None
        for key in (
            "win_fraction",
            "schedule_reference_mean_win_probability",
            "average_margin",
            "context_expected_wins",
            "prior_expected_wins",
        ):
            if key in r:
                r[key] = float(r[key]) if r[key] else None
        r["reference_kind"] = (
            "preseason_context_v1.2"
            if season == 2025
            else "retained_context_v1.4_posterior"
        )
        r["reference_ordinal"] = r.get("reference_ordinal", r.get("qop_ordinal"))
        r["expected_wins"] = r.get(
            "prior_expected_wins", r.get("context_expected_wins")
        )
        r["wins_minus_expectation"] = r["wins"] - r["expected_wins"]
        r["near_expected_record"] = abs(r["wins_minus_expectation"]) <= 1
        rank = r["reference_ordinal"]
        r["strength_band"] = (
            "unrated"
            if rank is None
            else "elite"
            if rank <= 10
            else "strong"
            if rank <= 40
            else "middle"
            if rank <= 95
            else "weak"
        )
        wf = r["win_fraction"]
        r["record_category"] = (
            "unrated"
            if wf is None
            else "[0,.3)"
            if wf < 0.3
            else "[.3,.6]"
            if wf <= 0.6
            else "(.6,.8)"
            if wf < 0.8
            else "[.8,1]"
        )
        result[r["team_id"]] = r
    for band in ("elite", "strong", "middle", "weak"):
        ids = sorted(
            (t for t, r in result.items() if r["strength_band"] == band and r["games"]),
            key=lambda t: (result[t]["schedule_reference_mean_win_probability"], t),
        )
        for label, ids_part in zip(
            ("hard", "middle", "easy"), np.array_split(ids, 3), strict=True
        ):
            for t in ids_part:
                result[str(t)]["schedule_tercile"] = label
    return result


def game_descriptions(
    network, descriptors: dict, context: dict, qop_order: dict
) -> tuple[list[dict], list[dict]]:
    """Integrated V1 probabilities/margins at actual sites using complete PMFs."""
    by_id = {t.team_id: replace(t, prior=context[t.team_id]) for t in network.teams}
    priors = {t.team_id: t for t in network.teams}
    reference_id = next(iter(descriptors.values()))["reference_team_id"]
    reference = priors[reference_id].prior
    surfaces, rows = {}, []
    for g in network.games:
        home, away = by_id[g.home_id], by_id[g.away_id]
        if home.subdivision == away.subdivision == "fcs":
            continue
        key = (
            home.subdivision,
            away.subdivision,
            len(home.prior),
            len(away.prior),
            g.neutral_site,
        )
        if key not in surfaces:
            scheduled = ScheduledGame(
                g.game_id,
                g.home_id,
                g.away_id,
                g.home_subdivision,
                g.away_subdivision,
                g.neutral_site,
            )
            location = conditional_margin_location_surface(
                scheduled, home, away, network.likelihood
            )
            surfaces[key] = (
                location,
                stdtr(
                    network.likelihood.degrees_of_freedom,
                    location / network.likelihood.scale,
                ),
            )
        locations, probabilities = surfaces[key]
        p = float(home.prior @ probabilities @ away.prior)
        margin_mean = float(home.prior @ locations @ away.prior)
        for focal, opponent, is_home, margin in (
            (home, away, True, g.home_points - g.away_points),
            (away, home, False, g.away_points - g.home_points),
        ):
            if focal.subdivision != "fbs":
                continue
            standardized = (
                float(reference @ probabilities @ opponent.prior)
                if is_home
                else 1 - float(opponent.prior @ probabilities @ reference)
            )
            opponent_summary = summary(opponent.prior)
            rows.append(
                {
                    "snapshot_id": network.snapshot_id,
                    "team_id": focal.team_id,
                    "game_id": g.game_id,
                    "opponent_id": opponent.team_id,
                    "opponent_name": opponent.name,
                    "opponent_subdivision": opponent.subdivision,
                    "venue": "neutral"
                    if g.neutral_site
                    else "home"
                    if is_home
                    else "away",
                    "margin": margin,
                    "context_win_probability": p if is_home else 1 - p,
                    "expected_margin": margin_mean if is_home else -margin_mean,
                    "schedule_reference_win_probability": standardized,
                    "opponent_context_expected_rank": opponent_summary["expected_rank"],
                    "opponent_context_rank_sd": opponent_summary["rank_sd"],
                    "opponent_qop_ordinal": qop_order.get(opponent.team_id),
                }
            )
    aggregated = []
    for t, d in descriptors.items():
        own = [r for r in rows if r["team_id"] == t]
        margins = [r["margin"] for r in own]
        aggregated.append(
            {
                **d,
                "snapshot_id": network.snapshot_id,
                "margins": margins,
                "median_margin": float(np.median(margins)) if margins else None,
                "close_wins": sum(0 < m <= 7 for m in margins),
                "close_losses": sum(-7 <= m < 0 for m in margins),
                "dominant_wins": sum(m >= 28 for m in margins),
                "fbs_games": sum(r["opponent_subdivision"] == "fbs" for r in own),
                "fcs_games": sum(r["opponent_subdivision"] == "fcs" for r in own),
                "posterior_schedule_reference_mean": float(
                    np.mean([r["schedule_reference_win_probability"] for r in own])
                )
                if own
                else None,
                "posterior_schedule_expected_wins": sum(
                    r["schedule_reference_win_probability"] for r in own
                ),
                "posterior_context_expected_wins": sum(
                    r["context_win_probability"] for r in own
                ),
                "wins_minus_posterior_context_expectation": d["wins"]
                - sum(r["context_win_probability"] for r in own),
                "schedule_probability_profile": [
                    r["schedule_reference_win_probability"] for r in own
                ],
                "context_probability_profile": [
                    r["context_win_probability"] for r in own
                ],
            }
        )
    return rows, aggregated


def panels(descriptors: dict[str, dict]) -> list[tuple[str, list[str]]]:
    rated = {t: r for t, r in descriptors.items() if r["games"]}
    result = [("all", sorted(rated))]
    for band in ("elite", "strong", "middle", "weak"):
        pool = {t: r for t, r in rated.items() if r["strength_band"] == band}
        result.append((f"band/{band}", sorted(pool)))
        for tercile in ("hard", "middle", "easy"):
            ids = sorted(t for t, r in pool.items() if r["schedule_tercile"] == tercile)
            result.append((f"band/{band}/schedule/{tercile}", ids))
            result.append(
                (
                    f"band/{band}/schedule/{tercile}/near_expected",
                    [t for t in ids if pool[t]["near_expected_record"]],
                )
            )
            for category in ("[0,.3)", "[.3,.6]", "(.6,.8)", "[.8,1]"):
                result.append(
                    (
                        f"band/{band}/schedule/{tercile}/record/{category}",
                        [t for t in ids if pool[t]["record_category"] == category],
                    )
                )
            for label, predicate in (
                (
                    "excellent_5plus",
                    lambda r: r["games"] >= 5 and r["win_fraction"] >= 0.8,
                ),
                (
                    "middling_5plus",
                    lambda r: r["games"] >= 5 and 0.3 <= r["win_fraction"] <= 0.6,
                ),
            ):
                result.append(
                    (
                        f"band/{band}/schedule/{tercile}/{label}",
                        [t for t in ids if predicate(pool[t])],
                    )
                )
        for category in ("[0,.3)", "[.3,.6]", "(.6,.8)", "[.8,1]"):
            result.append(
                (
                    f"band/{band}/record/{category}",
                    sorted(
                        t for t, r in pool.items() if r["record_category"] == category
                    ),
                )
            )
    for record in sorted(
        {(r["wins"], r["losses"], r["games"]) for r in rated.values()}
    ):
        result.append(
            (
                f"exact_record/{record[0]}-{record[1]}/{record[2]}",
                sorted(
                    t
                    for t, r in rated.items()
                    if (r["wins"], r["losses"], r["games"]) == record
                ),
            )
        )
    return result


def analyze(root: Path, output: Path) -> None:
    spec = check_design(root, root / ARTIFACT_DIR)
    origin = provenance(root, spec)
    execution = _read_json(output / "execution.json")
    saved_origin = _read_json(output / "provenance.json")
    if (
        not execution["complete"]
        or saved_origin["fingerprint"] != origin["fingerprint"]
    ):
        raise ValueError("incomplete or incompatible execution cannot be interpreted")
    gates = _read_json(output / "endpoint_parity.json")
    if set(gates) != set(spec["selected_snapshot_ids"]) or not all(
        g["passed"] for s in gates.values() for g in s.values()
    ):
        raise ValueError("numerical gates block interpretation")
    rankings, pmf_rows, contrasts, aggregates, panel_rows = [], [], [], [], []
    boundaries, associations, matched, sensitivity, uncertainty = [], [], [], [], []
    games_all, accomplishments, saved = [], [], {}
    for snapshot_id in spec["selected_snapshot_ids"]:
        network = load_network(root, spec, snapshot_id)
        data = {}
        for weight in MARGIN_WEIGHTS:
            path = output / "checkpoints" / snapshot_id / f"baseline-{weight:g}.json.gz"
            cell = read_checkpoint(
                path, cell_fingerprint(network, origin, "baseline", weight), network
            )
            data[weight] = {t: np.array(p) for t, p in cell["pmfs"].items()}
        counts = network.counts
        orders = {w: order(p, counts) for w, p in data.items()}
        ds = frozen_descriptors(root, snapshot_id, network.specification["season"])
        if set(ds) != set(network.fbs) or any(ds[t]["games"] != counts[t] for t in ds):
            raise ValueError("descriptor population/evidence mismatch")
        if network.context:
            context = network.context
            reference_orders = [
                {t: r[family + "_ordinal"] for t, r in ds.items() if r["games"]}
                for family in ("qop", "qor")
            ]
        else:
            reference_path = (
                output / "checkpoints" / snapshot_id / "reference-1.json.gz"
            )
            reference = read_checkpoint(
                reference_path,
                cell_fingerprint(network, origin, "reference", 1.0),
                network,
            )
            context = {t: np.array(p) for t, p in reference["context"].items()}
            reference_orders = [orders[1.0], orders[0.0]]
        game_rows, descriptions = game_descriptions(
            network, ds, context, reference_orders[0]
        )
        games_all += game_rows
        accomplishments += descriptions
        descriptors = {r["team_id"]: r for r in descriptions}
        groups = panels(ds)
        for weight, pmfs in data.items():
            for t in sorted(pmfs):
                d = ds[t]
                rankings.append(
                    {
                        "snapshot_id": snapshot_id,
                        "season": network.specification["season"],
                        "margin_weight": weight,
                        "team_id": t,
                        "team_name": d["team_name"],
                        "eligible_games": counts[t],
                        "wins": d["wins"],
                        "losses": d["losses"],
                        "rated": bool(counts[t]),
                        "ordinal_rank": orders[weight].get(t),
                        **summary(pmfs[t]),
                    }
                )
                pmf_rows += [
                    {
                        "snapshot_id": snapshot_id,
                        "margin_weight": weight,
                        "team_id": t,
                        "rank": i,
                        "probability": float(p),
                    }
                    for i, p in enumerate(pmfs[t], 1)
                ]
            for label, ids in groups:
                uncertainty.append(
                    {
                        "snapshot_id": snapshot_id,
                        "margin_weight": weight,
                        "panel": label,
                        "n": len(ids),
                        "team_ids": ids,
                        **{
                            f"{field}_{stat}": value
                            for field in (
                                "expected_rank",
                                "rank_sd",
                                "entropy_bits",
                                "width_50",
                                "width_80",
                                "width_95",
                            )
                            for stat, value in distribution(
                                [summary(pmfs[t])[field] for t in ids]
                            ).items()
                        },
                    }
                )
        for low, high in itertools.combinations(MARGIN_WEIGHTS, 2):
            pair_rows = []
            for t in sorted(orders[low]):
                row = {
                    "snapshot_id": snapshot_id,
                    "lambda_low": low,
                    "lambda_high": high,
                    "team_id": t,
                    "team_name": ds[t]["team_name"],
                    "low_expected_rank": mean_rank(data[low][t]),
                    "high_expected_rank": mean_rank(data[high][t]),
                    "low_rank_sd": summary(data[low][t])["rank_sd"],
                    "high_rank_sd": summary(data[high][t])["rank_sd"],
                    "low_ordinal_rank": orders[low][t],
                    "high_ordinal_rank": orders[high][t],
                    "delta_ordinal_rank": orders[high][t] - orders[low][t],
                    **compare(data[low][t], data[high][t]),
                }
                pair_rows.append(row)
            contrasts += pair_rows
            base = {"snapshot_id": snapshot_id, "lambda_low": low, "lambda_high": high}
            inversions = sum(
                (orders[low][a] - orders[low][b]) * (orders[high][a] - orders[high][b])
                < 0
                for a, b in itertools.combinations(orders[low], 2)
            )
            aggregates.append(
                {
                    **base,
                    **contrast_summary(pair_rows),
                    "ordering_inversions": inversions,
                    "ordering_inversion_fraction": inversions
                    / (len(pair_rows) * (len(pair_rows) - 1) / 2),
                    **top_changes(orders[low], orders[high], 10),
                    **top_changes(orders[low], orders[high], 25),
                }
            )
            by_id = {r["team_id"]: r for r in pair_rows}
            for boundary in (10, 12, 25):
                ids = sorted(
                    t
                    for t in by_id
                    if any(
                        abs(reference[t] - boundary) <= 3
                        for reference in reference_orders
                    )
                )
                for t in ids:
                    row = by_id[t]
                    neighbors = {}
                    for weight, prefix in ((low, "low"), (high, "high")):
                        ranked = list(orders[weight])
                        i = ranked.index(t)
                        neighbors[prefix + "_preceding_gap"] = (
                            mean_rank(data[weight][t])
                            - mean_rank(data[weight][ranked[i - 1]])
                            if i
                            else None
                        )
                        neighbors[prefix + "_following_gap"] = (
                            mean_rank(data[weight][ranked[i + 1]])
                            - mean_rank(data[weight][t])
                            if i + 1 < len(ranked)
                            else None
                        )
                    boundaries.append(
                        {
                            **row,
                            "boundary": boundary,
                            "crossed": (row["low_ordinal_rank"] <= boundary)
                            != (row["high_ordinal_rank"] <= boundary),
                            **neighbors,
                        }
                    )
            for label, ids in groups:
                members = [by_id[t] for t in ids]
                panel_rows.append(
                    {
                        **base,
                        "panel": label,
                        "reference_kind": next(iter(ds.values()))["reference_kind"],
                        "n": len(ids),
                        "team_ids": ids,
                        **contrast_summary(members),
                        "delta_expected_rank_distribution": [
                            r["delta_expected_rank"] for r in members
                        ],
                        "top10_crossing_ids": [
                            t
                            for t in ids
                            if (orders[low][t] <= 10) != (orders[high][t] <= 10)
                        ],
                        "top25_crossing_ids": [
                            t
                            for t in ids
                            if (orders[low][t] <= 25) != (orders[high][t] <= 25)
                        ],
                        "boundary12_crossing_ids": [
                            t
                            for t in ids
                            if (orders[low][t] <= 12) != (orders[high][t] <= 12)
                        ],
                    }
                )
            for band in ("all", "elite", "strong", "middle", "weak"):
                ids = sorted(
                    t for t in by_id if band == "all" or ds[t]["strength_band"] == band
                )
                for field in (
                    "win_fraction",
                    "schedule_reference_mean_win_probability",
                    "average_margin",
                    "wins_minus_expectation",
                    "close_wins",
                    "close_losses",
                    "dominant_wins",
                ):
                    x = [descriptors[t][field] for t in ids]
                    y = [by_id[t]["delta_expected_rank"] for t in ids]
                    defined = len(x) >= 2 and len(set(x)) > 1 and len(set(y)) > 1
                    associations.append(
                        {
                            **base,
                            "strength_band": band,
                            "descriptor": field,
                            "n": len(ids),
                            "spearman_rho": float(spearmanr(x, y).statistic)
                            if defined
                            else None,
                            "undefined_reason": None
                            if defined
                            else "insufficient population or constant column",
                        }
                    )
            for a, b in itertools.combinations(sorted(by_id), 2):
                if ds[a]["strength_band"] != ds[b]["strength_band"] or (
                    ds[a]["wins"],
                    ds[a]["losses"],
                ) != (ds[b]["wins"], ds[b]["losses"]):
                    continue
                hard, easy = sorted(
                    (a, b),
                    key=lambda t: (ds[t]["schedule_reference_mean_win_probability"], t),
                )
                matched.append(
                    {
                        **base,
                        "strength_band": ds[hard]["strength_band"],
                        "wins": ds[hard]["wins"],
                        "losses": ds[hard]["losses"],
                        "hard_team_id": hard,
                        "easy_team_id": easy,
                        "schedule_probability_gap_easy_minus_hard": ds[easy][
                            "schedule_reference_mean_win_probability"
                        ]
                        - ds[hard]["schedule_reference_mean_win_probability"],
                        "low_expected_rank_gap_easy_minus_hard": mean_rank(
                            data[low][easy]
                        )
                        - mean_rank(data[low][hard]),
                        "high_expected_rank_gap_easy_minus_hard": mean_rank(
                            data[high][easy]
                        )
                        - mean_rank(data[high][hard]),
                        "contrast_gap_easy_minus_hard": by_id[easy][
                            "delta_expected_rank"
                        ]
                        - by_id[hard]["delta_expected_rank"],
                        "low_sd_gap_easy_minus_hard": summary(data[low][easy])[
                            "rank_sd"
                        ]
                        - summary(data[low][hard])["rank_sd"],
                        "high_sd_gap_easy_minus_hard": summary(data[high][easy])[
                            "rank_sd"
                        ]
                        - summary(data[high][hard])["rank_sd"],
                    }
                )
        for t in sorted(orders[0.0]):
            steps = [
                mean_rank(data[b][t]) - mean_rank(data[a][t])
                for a, b in itertools.pairwise(MARGIN_WEIGHTS)
            ]
            sensitivity.append(
                {
                    "snapshot_id": snapshot_id,
                    "team_id": t,
                    "team_name": ds[t]["team_name"],
                    "adjacent_expected_rank_changes": steps,
                    "adjacent_slopes": [d / 0.25 for d in steps],
                    "adjacent_total_variation": [
                        compare(data[a][t], data[b][t])["total_variation"]
                        for a, b in itertools.pairwise(MARGIN_WEIGHTS)
                    ],
                    "nonmonotonic_above_numerical_tolerance": min(steps)
                    < -len(data[0.0][t]) * 1e-8
                    and max(steps) > len(data[0.0][t]) * 1e-8,
                }
            )
        saved[snapshot_id] = (data, orders, counts, network.rows, ds)
    longitudinal, transitions = [], []
    ids = spec["selected_snapshot_ids"]
    for before, after in ((ids[0], ids[1]), (ids[2], ids[3]), (ids[3], ids[4])):
        early, eo, ec, er, ed = saved[before]
        late, lo, lc, lr, _ld = saved[after]
        new_ids = sorted({r["id"] for r in lr} - {r["id"] for r in er})
        new_rows = [r for r in lr if r["id"] in set(new_ids)]
        if {r["id"] for r in er} - {r["id"] for r in lr}:
            raise ValueError("unexpected removed evidence between frozen snapshots")
        for weight in MARGIN_WEIGHTS:
            common = sorted(eo[weight].keys() & lo[weight].keys())
            rows = []
            for t in common:
                row = {
                    "before": before,
                    "after": after,
                    "margin_weight": weight,
                    "team_id": t,
                    "team_name": ed[t]["team_name"],
                    "before_games": ec[t],
                    "after_games": lc[t],
                    "new_own_games": lc[t] - ec[t],
                    "additional_network_games": len(new_ids),
                    "delta_ordinal_rank": lo[weight][t] - eo[weight][t],
                    **compare(early[weight][t], late[weight][t]),
                }
                rows.append(row)
            longitudinal += rows
            transitions.append(
                {
                    "before": before,
                    "after": after,
                    "margin_weight": weight,
                    "rated_intersection_n": len(common),
                    "rated_entrants": sorted(lo[weight].keys() - eo[weight].keys()),
                    "rated_exits": sorted(eo[weight].keys() - lo[weight].keys()),
                    "additional_network_games": len(new_ids),
                    "new_game_ids": new_ids,
                    "new_game_season_types": dict(
                        Counter(r["seasonType"] for r in new_rows)
                    ),
                    "new_fbs_fbs_games": sum(
                        r["homeClassification"] == r["awayClassification"] == "fbs"
                        for r in new_rows
                    ),
                    "new_fcs_fcs_games": sum(
                        r["homeClassification"] == r["awayClassification"] == "fcs"
                        for r in new_rows
                    ),
                    **contrast_summary(rows),
                    **top_changes(eo[weight], lo[weight], 10),
                    **top_changes(eo[weight], lo[weight], 25),
                }
            )
    files = {
        "rankings.csv": rankings,
        "posterior_pmfs/candidates.csv.gz": pmf_rows,
        "team_contrasts.csv.gz": contrasts,
        "population_contrasts.csv": aggregates,
        "accomplishment_panels.csv": panel_rows,
        "uncertainty_panels.csv": uncertainty,
        "boundary_movement.csv": boundaries,
        "descriptive_associations.csv": associations,
        "matched_records.csv.gz": matched,
        "weight_sensitivity.csv": sensitivity,
        "actual_game_descriptors.csv.gz": games_all,
        "accomplishments.csv": accomplishments,
        "longitudinal_team_changes.csv": longitudinal,
        "longitudinal_summary.csv": transitions,
        "convergence.csv": execution["cells"],
    }
    for name, rows in files.items():
        write_csv(output / name, rows)
    analysis_source = Path(__file__)
    manifest = {
        "issue": 265,
        "complete": True,
        "baseline_solves": 25,
        "independent_reference_solves": 4,
        "order_reversal_solves": 3,
        "failed_cells": execution["failures"],
        "execution_provenance": saved_origin,
        "analysis_source_sha256": sha256(analysis_source),
        "analysis_revision": origin["execution_revision"],
        "interpretation": "descriptive census; no causal schedule identification; no weight selected",
        "output_sha256": {
            str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name not in ("manifest.json", "report.md")
        },
    }
    write_json(output / "manifest.json", manifest)
    print(
        f"Population analysis complete: {len(rankings)} rankings, {len(aggregates)} candidate pairs",
        flush=True,
    )
