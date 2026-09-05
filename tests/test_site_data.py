from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

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


def _config_for(source: Path, tmp_path: Path) -> Path:
    config = tmp_path / "publish.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "snapshots": [
                    {"source": str(source.relative_to(tmp_path)), "display_label": "Test"}
                ],
            }
        ),
        encoding="utf-8",
    )
    return config


def _copied_snapshot(tmp_path: Path) -> Path:
    source = ROOT / (
        "data/processed/snapshots/2026/2026-preseason-context/predictive/context"
    )
    destination = tmp_path / "snapshot"
    shutil.copytree(source, destination)
    return destination


def test_site_data_publishes_all_initial_h_c_preseason_and_current_snapshots(
    tmp_path: Path,
) -> None:
    manifest = build_site_data(
        root=ROOT, config_path=CONFIG, output_directory=tmp_path / "data"
    )
    assert manifest["seasons"] == [2026]
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


def test_site_data_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "data"
    build_site_data(root=ROOT, config_path=CONFIG, output_directory=output)
    first = _hash_tree(output)
    build_site_data(root=ROOT, config_path=CONFIG, output_directory=output)
    assert _hash_tree(output) == first


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
