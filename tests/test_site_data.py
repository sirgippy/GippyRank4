from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from gippyrank import site_data
from gippyrank.site_data import SiteDataValidationError, build_site_data

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "site/publish_config.json"


def _hash_tree(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_for(source: Path, tmp_path: Path, *, slot: str = "test") -> Path:
    config = tmp_path / "publish.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "snapshots": [
                    {
                        "source": str(source.relative_to(tmp_path)),
                        "display_label": "Test",
                        "publication_slot": slot,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config


def _copied_snapshot(tmp_path: Path, relative_source: str | None = None) -> Path:
    source = ROOT / (
        relative_source
        or "data/processed/snapshots/2026/2026-preseason-context/predictive/context"
    )
    destination = tmp_path / "snapshot"
    shutil.copytree(source, destination)
    return destination


def _pmf_rows(source: Path) -> tuple[Path, list[str], list[dict[str, str]]]:
    path = source / "posterior_pmfs.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    return path, fields, rows


def _write_pmf_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _counterpart(
    entries: list[dict[str, object]], current: dict[str, object], prior: str
) -> dict[str, object]:
    """Mirror the UI's slot-based counterpart lookup contract."""
    return next(
        (
            entry
            for entry in entries
            if entry["season"] == current["season"]
            and entry["ranking_family"] == current["ranking_family"]
            and entry["publication_slot"] == current["publication_slot"]
            and entry["prior_family"] == prior
        ),
        current,
    )


def test_site_data_publishes_all_initial_h_c_preseason_and_current_snapshots(
    tmp_path: Path,
) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    assert manifest["seasons"] == [2026]
    assert manifest["default_publication_slot"] == "2026-sep-05"
    assert {(entry["snapshot_type"], entry["prior_family"]) for entry in manifest["snapshots"]} == {
        ("preseason", "context"),
        ("preseason", "history"),
        ("live", "context"),
        ("live", "history"),
    }
    current = [entry for entry in manifest["snapshots"] if entry["snapshot_type"] == "live"]
    assert {entry["effective_cutoff"] for entry in current} == {
        "2026-09-05T20:11:12.864212+00:00"
    }


@pytest.mark.parametrize(
    ("slot", "from_prior", "to_prior"),
    [
        ("2026-preseason", "context", "history"),
        ("2026-preseason", "history", "context"),
        ("2026-sep-05", "context", "history"),
        ("2026-sep-05", "history", "context"),
    ],
)
def test_logical_publication_slots_pair_context_and_history(
    tmp_path: Path, slot: str, from_prior: str, to_prior: str
) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    current = next(
        entry
        for entry in manifest["snapshots"]
        if entry["publication_slot"] == slot and entry["prior_family"] == from_prior
    )
    counterpart = _counterpart(manifest["snapshots"], current, to_prior)
    assert counterpart["prior_family"] == to_prior
    assert counterpart["publication_slot"] == slot
    assert counterpart["snapshot_id"] != current["snapshot_id"]
    assert counterpart["distribution_path"] != current["distribution_path"]


def test_missing_logical_counterpart_keeps_current_selection(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    current = next(
        entry
        for entry in manifest["snapshots"]
        if entry["publication_slot"] == "2026-sep-05"
        and entry["prior_family"] == "context"
    )
    entries = [
        entry
        for entry in manifest["snapshots"]
        if not (
            entry["publication_slot"] == "2026-sep-05"
            and entry["prior_family"] == "history"
        )
    ]
    assert _counterpart(entries, current, "history") == current


def test_site_data_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "data"
    build_site_data(root=ROOT, config_path=CONFIG, output_directory=output)
    first = _hash_tree(output)
    build_site_data(root=ROOT, config_path=CONFIG, output_directory=output)
    assert _hash_tree(output) == first


def test_distribution_artifact_contains_complete_fbs_pmfs_and_summaries(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    entry = next(item for item in manifest["snapshots"] if item["prior_family"] == "context")
    distribution = json.loads(
        (tmp_path / "data" / entry["distribution_path"].removeprefix("data/")).read_text()
    )
    snapshot = json.loads(
        (tmp_path / "data" / entry["data_path"].removeprefix("data/")).read_text()
    )
    assert distribution["schema_version"] == "1.0"
    assert distribution["snapshot_id"] == entry["snapshot_id"]
    assert distribution["rank_count"] == 138
    assert set(distribution["teams"]) == {row["team_id"] for row in snapshot["rankings"]}
    team = distribution["teams"][snapshot["rankings"][0]["team_id"]]
    assert len(team["pmf"]) == distribution["rank_count"]
    assert sum(team["pmf"]) == pytest.approx(1.0, abs=1e-9)
    assert set(team["summary"]) == {
        "expected_rank", "median_rank", "modal_rank", "interval_50", "interval_80",
        "interval_95", "interval_widths", "rank_1_probability", "top5_probability",
        "top10_probability", "top25_probability",
    }


def test_pmf_summary_uses_established_discrete_quantiles() -> None:
    summary = site_data._pmf_summary([0.1, 0.2, 0.3, 0.2, 0.2])
    assert summary["expected_rank"] == pytest.approx(3.2)
    assert summary["median_rank"] == 3
    assert summary["modal_rank"] == 3
    assert summary["interval_50"] == [2, 4]
    assert summary["interval_80"] == [1, 5]
    assert summary["interval_95"] == [1, 5]
    assert summary["interval_widths"] == {"50": 3, "80": 5, "95": 5}


def test_exported_80_percent_interval_remains_the_ranking_interval(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    for entry in manifest["snapshots"]:
        snapshot = json.loads(
            (tmp_path / "data" / entry["data_path"].removeprefix("data/")).read_text()
        )
        distribution = json.loads(
            (tmp_path / "data" / entry["distribution_path"].removeprefix("data/")).read_text()
        )
        for row in snapshot["rankings"]:
            assert distribution["teams"][row["team_id"]]["summary"]["interval_80"] == row["interval_80"]


@pytest.mark.parametrize(
    "relative_source",
    [
        "data/processed/snapshots/2026/2026-preseason-context/predictive/context",
        "data/processed/snapshots/2026/2026-live-2026-09-05T23-59-59Z-context/predictive/context",
    ],
)
def test_preseason_and_in_season_distribution_exports_work(
    tmp_path: Path, relative_source: str
) -> None:
    source = _copied_snapshot(tmp_path, relative_source)
    manifest = build_site_data(
        root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data"
    )
    entry = manifest["snapshots"][0]
    distribution = json.loads(
        (tmp_path / "data" / entry["distribution_path"].removeprefix("data/")).read_text()
    )
    assert distribution["rank_count"] == 138
    assert len(distribution["teams"]) == 138


def test_context_and_history_export_distinct_distribution_artifacts(tmp_path: Path) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    for slot in {entry["publication_slot"] for entry in manifest["snapshots"]}:
        context, history = (
            next(
                entry
                for entry in manifest["snapshots"]
                if entry["publication_slot"] == slot and entry["prior_family"] == prior
            )
            for prior in ("context", "history")
        )
        context_path = tmp_path / "data" / context["distribution_path"].removeprefix("data/")
        history_path = tmp_path / "data" / history["distribution_path"].removeprefix("data/")
        assert context_path != history_path
        assert context_path.read_bytes() != history_path.read_bytes()


def test_missing_pmf_rank_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    path, fields, rows = _pmf_rows(source)
    team_id = rows[0]["team_id"]
    _write_pmf_rows(path, fields, [row for row in rows if not (row["team_id"] == team_id and row["rank"] == "1")])
    with pytest.raises(SiteDataValidationError, match="missing or unexpected ranks"):
        build_site_data(root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data")


def test_duplicate_pmf_rank_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    path, fields, rows = _pmf_rows(source)
    team_id = rows[0]["team_id"]
    next(row for row in rows if row["team_id"] == team_id and row["rank"] == "2")["rank"] = "1"
    _write_pmf_rows(path, fields, rows)
    with pytest.raises(SiteDataValidationError, match="duplicate PMF rank"):
        build_site_data(root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data")


def test_pmf_team_mismatch_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    path, fields, rows = _pmf_rows(source)
    team_id = rows[0]["team_id"]
    for row in rows:
        if row["team_id"] == team_id:
            row["team_id"] = "not-a-ranking-team"
    _write_pmf_rows(path, fields, rows)
    with pytest.raises(SiteDataValidationError, match=f"missing PMF for FBS team {team_id}"):
        build_site_data(root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data")


@pytest.mark.parametrize(
    ("value", "message"),
    [("nan", "must be finite"), ("-0.01", "must be between 0 and 1"), ("1.01", "must be between 0 and 1")],
)
def test_malformed_pmf_probability_is_refused(
    tmp_path: Path, value: str, message: str
) -> None:
    source = _copied_snapshot(tmp_path)
    path, fields, rows = _pmf_rows(source)
    rows[0]["probability"] = value
    _write_pmf_rows(path, fields, rows)
    with pytest.raises(SiteDataValidationError, match=message):
        build_site_data(root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data")


def test_unnormalized_pmf_is_refused_without_renormalizing(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    path, fields, rows = _pmf_rows(source)
    rows[0]["probability"] = "0.1"
    _write_pmf_rows(path, fields, rows)
    with pytest.raises(SiteDataValidationError, match="sum to"):
        build_site_data(root=tmp_path, config_path=_config_for(source, tmp_path), output_directory=tmp_path / "data")


def test_invalid_snapshot_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    metadata = json.loads((source / "metadata.json").read_text())
    metadata["valid"] = False
    (source / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SiteDataValidationError, match="not valid"):
        build_site_data(
            root=tmp_path,
            config_path=_config_for(source, tmp_path),
            output_directory=tmp_path / "data",
        )


def test_lower_division_game_does_not_change_modeled_record(tmp_path: Path) -> None:
    source = _copied_snapshot(
        tmp_path,
        "data/processed/snapshots/2026/2026-live-2026-09-05T23-59-59Z-context/"
        "predictive/context",
    )
    config = _config_for(source, tmp_path)
    build_site_data(
        root=tmp_path, config_path=config, output_directory=tmp_path / "first"
    )
    snapshot_id = json.loads((source / "metadata.json").read_text())["snapshot_id"]
    first_data = json.loads((tmp_path / f"first/snapshots/{snapshot_id}.json").read_text())
    team = first_data["rankings"][0]
    games = source / "included_games.csv"
    with games.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    lower_division = dict(rows[0])
    lower_division.update(
        {
            "id": "not-eligible",
            "homeId": team["team_id"],
            "homeClassification": "fbs",
            "homePoints": "99",
            "awayId": "division-ii",
            "awayClassification": "ii",
            "awayPoints": "0",
        }
    )
    rows.append(lower_division)
    with games.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    build_site_data(root=tmp_path, config_path=config, output_directory=tmp_path / "second")
    second_data = json.loads(
        (tmp_path / f"second/snapshots/{snapshot_id}.json").read_text()
    )
    assert second_data["rankings"][0]["record"] == team["record"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("display_rank", "1", "duplicate or invalid FBS display rank"),
        ("top25_probability", "1.1", "must be between 0 and 1"),
    ],
)
def test_malformed_ranking_values_are_refused(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    source = _copied_snapshot(tmp_path)
    rankings = source / "rankings.csv"
    with rankings.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[1][field] = value
    with rankings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SiteDataValidationError, match=message):
        build_site_data(
            root=tmp_path,
            config_path=_config_for(source, tmp_path),
            output_directory=tmp_path / "data",
        )


def test_unsupported_snapshot_schema_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = "999.0"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SiteDataValidationError, match="unsupported snapshot schema"):
        build_site_data(
            root=tmp_path,
            config_path=_config_for(source, tmp_path),
            output_directory=tmp_path / "data",
        )


def test_unknown_ranking_family_is_refused(tmp_path: Path) -> None:
    source = _copied_snapshot(tmp_path)
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["ranking_family"] = "unknown"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SiteDataValidationError, match="unsupported ranking family"):
        build_site_data(
            root=tmp_path,
            config_path=_config_for(source, tmp_path),
            output_directory=tmp_path / "data",
        )


def test_registered_published_families_drive_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _copied_snapshot(tmp_path)
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["ranking_family"] = "resume"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setitem(site_data.RANKING_FAMILIES, "resume", {"label": "Résumé"})
    manifest = build_site_data(
        root=tmp_path,
        config_path=_config_for(source, tmp_path),
        output_directory=tmp_path / "data",
    )
    assert manifest["ranking_families"] == [{"id": "resume", "label": "Résumé"}]


def test_site_builder_has_no_model_import_path() -> None:
    source = (ROOT / "src/gippyrank/site_data.py").read_text(encoding="utf-8")
    assert "gippyrank.posterior" not in source
    assert "gippyrank.preseason" not in source


def test_site_uses_base_safe_relative_paths() -> None:
    index = (ROOT / "site/index.html").read_text(encoding="utf-8")
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")
    assert 'href="./assets/style.css"' in index
    assert 'src="./assets/app.js"' in index
    assert 'fetch("./data/manifest.json")' in app
    assert 'fetch(`./${entry.distribution_path}`)' in app
    assert "distributionCache" in app
    assert "selectedEntry()?.snapshot_id !== entry.snapshot_id" in app
    assert "publication_slot" in app
    assert "staying on" in app


def test_uncertainty_copy_uses_central_interval_language() -> None:
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")
    assert "The central 50% interval spans" in app
    assert "The central 80% interval spans" in app
    assert "The central 95% interval spans" in app
    assert "Half of the posterior lies between" not in app
    assert "80% lies between" not in app
    assert "95% lies between" not in app


def test_percentage_formatter_preserves_nonzero_and_noncertainty_distinctions() -> None:
    """Keep lightweight static coverage because this dependency-free site has no JS runner."""
    app = (ROOT / "site/assets/app.js").read_text(encoding="utf-8")
    assert 'if (value === 1) return "100%";' in app
    assert 'if (valueAsPercent < 0.01) return "<0.01%";' in app
    assert 'if (valueAsPercent >= 99.95) return "<100%";' in app
    assert 'if (valueAsPercent >= 95) return `${valueAsPercent.toFixed(1)}%`;' in app
    assert 'return `${Math.round(valueAsPercent)}%`;' in app
