"""Central production identifiers used by the public methodology page.

These values describe the contracts that the publication pipeline exposes.  A
small, JSON-ready projection is generated into ``site/data`` so the static
site can display the same identifiers without importing model code in the
browser.
"""

from __future__ import annotations

from typing import Final

METHODOLOGY_SCHEMA_VERSION: Final = "1.0"

SITE_SCHEMA_VERSION: Final = "1.0"
SNAPSHOT_SCHEMA_VERSION: Final = "1.0"
TEAM_SEASON_SCHEMA_VERSION: Final = "1.0"
WEEKLY_GAME_SCHEMA_VERSION: Final = "1.0"
PREDICTION_SCHEMA_VERSION: Final = "1.0"
PERFORMANCE_SCHEMA_VERSION: Final = "1.0"
SEASON_SIMULATION_SCHEMA_VERSION: Final = "1.0"

CONTEXT_PRIOR_VERSION: Final = "1.2"
HISTORY_PRIOR_VERSION: Final = "1.1"
HISTORICAL_LIKELIHOOD_VERSION: Final = "V1"
POSTERIOR_VERSION: Final = "V1"
PERFORMANCE_VERSION: Final = "1.0"
SEASON_SIMULATION_VERSION: Final = "hierarchical_latent_state_v1"

PRODUCTION_MODEL_VERSIONS: Final[dict[str, str]] = {
    "context_prior": CONTEXT_PRIOR_VERSION,
    "history_prior": HISTORY_PRIOR_VERSION,
    "historical_likelihood": HISTORICAL_LIKELIHOOD_VERSION,
    "posterior": POSTERIOR_VERSION,
    "performance": PERFORMANCE_VERSION,
    "season_simulation": SEASON_SIMULATION_VERSION,
}

PRODUCTION_SCHEMA_VERSIONS: Final[dict[str, str]] = {
    "site": SITE_SCHEMA_VERSION,
    "snapshot": SNAPSHOT_SCHEMA_VERSION,
    "team_season": TEAM_SEASON_SCHEMA_VERSION,
    "weekly_game": WEEKLY_GAME_SCHEMA_VERSION,
    "prediction": PREDICTION_SCHEMA_VERSION,
    "performance": PERFORMANCE_SCHEMA_VERSION,
    "season_simulation": SEASON_SIMULATION_SCHEMA_VERSION,
}


def production_methodology_metadata() -> dict[str, object]:
    """Return the stable metadata projection consumed by the static site."""

    return {
        "schema_version": METHODOLOGY_SCHEMA_VERSION,
        "model_versions": dict(PRODUCTION_MODEL_VERSIONS),
        "schema_versions": dict(PRODUCTION_SCHEMA_VERSIONS),
        "feature_status": {"season_simulation": "shipped"},
    }
