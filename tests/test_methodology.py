from __future__ import annotations

import json
from pathlib import Path

from gippyrank.methodology import (
    PREDICTION_SCHEMA_VERSION,
    PRODUCTION_MODEL_VERSIONS,
    PRODUCTION_SCHEMA_VERSIONS,
    production_methodology_metadata,
)
from gippyrank.performance_snapshot import (
    PERFORMANCE_MODEL_VERSION,
    PERFORMANCE_SCHEMA_VERSION,
)
from gippyrank.posterior.game_evidence import (
    TEAM_SEASON_SCHEMA_VERSION as GAME_EVIDENCE_TEAM_SEASON_SCHEMA_VERSION,
)
from gippyrank.posterior.season_simulation import (
    SEASON_SIMULATION_SCHEMA_VERSION,
    SEASON_SIMULATION_VERSION,
)
from gippyrank.posterior.snapshots import SCHEMA_VERSION as SNAPSHOT_SCHEMA_VERSION
from gippyrank.site_data import (
    SITE_SCHEMA_VERSION,
    TEAM_SEASON_SCHEMA_VERSION,
    WEEKLY_GAME_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]


def test_published_methodology_metadata_matches_the_central_contract() -> None:
    published = json.loads(
        (ROOT / "site/data/methodology.json").read_text(encoding="utf-8")
    )

    assert published == production_methodology_metadata()
    assert published["model_versions"] == PRODUCTION_MODEL_VERSIONS
    assert published["schema_versions"] == PRODUCTION_SCHEMA_VERSIONS

    manifest = json.loads(
        (ROOT / "site/data/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["methodology_path"] == "data/methodology.json"
    assert manifest["methodology_schema_version"] == published["schema_version"]


def test_methodology_page_documents_every_production_layer() -> None:
    page = (ROOT / "site/methodology.html").read_text(encoding="utf-8")

    for section in (
        "measure",
        "latent-model",
        "priors",
        "likelihood",
        "posterior",
        "performance",
        "game-performance",
        "future",
        "simulation",
        "uncertainty",
        "snapshots",
        "publications",
        "validation",
        "research",
        "limitations",
        "reproducibility",
    ):
        assert f'id="{section}"' in page

    for phrase in (
        "Predictive Context",
        "Predictive History",
        "Historical Likelihood",
        "posterior PMF",
        "Performance rankings",
        "Future-game predictions",
        "Season simulation",
        "Official and interim publications",
        "Validation and research evidence",
        "Limitations",
        "Shipped",
        "does not change production model outputs",
    ):
        assert phrase in page

    assert 'src="./assets/methodology.js"' in page
    assert 'fetch("./data/methodology.json")' in (
        ROOT / "site/assets/methodology.js"
    ).read_text(encoding="utf-8")


def test_methodology_is_linked_from_all_public_site_surfaces() -> None:
    for filename in ("index.html", "team.html", "week.html"):
        page = (ROOT / "site" / filename).read_text(encoding="utf-8")
        assert 'href="./methodology.html"' in page


def test_serializers_share_the_documented_schema_and_model_identifiers() -> None:
    assert SITE_SCHEMA_VERSION == PRODUCTION_SCHEMA_VERSIONS["site"]
    assert SNAPSHOT_SCHEMA_VERSION == PRODUCTION_SCHEMA_VERSIONS["snapshot"]
    assert TEAM_SEASON_SCHEMA_VERSION == PRODUCTION_SCHEMA_VERSIONS["team_season"]
    assert (
        GAME_EVIDENCE_TEAM_SEASON_SCHEMA_VERSION
        == PRODUCTION_SCHEMA_VERSIONS["team_season"]
    )
    assert WEEKLY_GAME_SCHEMA_VERSION == PRODUCTION_SCHEMA_VERSIONS["weekly_game"]
    assert PREDICTION_SCHEMA_VERSION == PRODUCTION_SCHEMA_VERSIONS["prediction"]
    assert PERFORMANCE_SCHEMA_VERSION == PRODUCTION_SCHEMA_VERSIONS["performance"]
    assert (
        SEASON_SIMULATION_SCHEMA_VERSION
        == PRODUCTION_SCHEMA_VERSIONS["season_simulation"]
    )

    assert PERFORMANCE_MODEL_VERSION == PRODUCTION_MODEL_VERSIONS["performance"]
    assert SEASON_SIMULATION_VERSION == PRODUCTION_MODEL_VERSIONS["season_simulation"]
