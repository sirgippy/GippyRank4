"""Public V1 API response models.

The publication artifacts are deliberately more detailed than this public
contract.  These models make the compatibility surface explicit and keep
implementation-only artifact fields out of HTTP responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

API_VERSION = "v1"

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Percentage = Annotated[float, Field(ge=0.0, le=100.0)]
RankInterval = Annotated[list[float], Field(min_length=2, max_length=2)]
EncodedWeight = Annotated[int, Field(ge=0, le=1000)]
WeekValue = int | str | None
Subdivision = Literal["fbs", "fcs"]
GameState = Literal["completed", "future", "unresolved", "cancelled", "out_of_scope"]
PredictionSource = Literal["predictive_context", "predictive_history"]


class PublicModel(BaseModel):
    """Base class for models returned by the API."""

    model_config = ConfigDict(extra="forbid")


class ModelVersions(PublicModel):
    context_prior: str | None = None
    history_prior: str | None = None
    historical_likelihood: str | None = None
    posterior: str | None = None
    performance: str | None = None
    season_simulation: str | None = None


class PublicationMetadata(PublicModel):
    api_version: Literal["v1"] = API_VERSION
    season: int
    snapshot_id: str
    snapshot_type: Literal["preseason", "weekly", "live"]
    publication_slot: str
    publication_status: Literal["official", "temporary"]
    display_label: str
    ranking_family: Literal["predictive", "performance"]
    prior_family: Literal["context", "history"] | None = None
    requested_cutoff: datetime | None = None
    effective_cutoff: datetime | None = None
    generation_timestamp: datetime
    source_retrieved_at: datetime | None = None
    model_versions: ModelVersions
    model_version: str | None = None
    method: str | None = None
    anchor_family: str | None = None
    source_context_snapshot_id: str | None = None
    included_game_count: int = Field(ge=0)
    excluded_lower_division_games: int = Field(ge=0)
    rank_count: int = Field(ge=1)
    rated_count: int = Field(ge=0)
    unrated_count: int = Field(ge=0)
    default_week: str | None = None
    comparison_snapshot_id: str | None = None
    comparison_display_label: str | None = None


class TeamIdentity(PublicModel):
    team_id: str
    team_name: str
    subdivision: Subdivision
    conference: str


class IntervalWidths(PublicModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fifty: int = Field(alias="50", ge=0)
    eighty: int = Field(alias="80", ge=0)
    ninety_five: int = Field(alias="95", ge=0)


class RankRow(PublicModel):
    team_id: str
    team_name: str
    conference: str
    display_rank: int | Literal["NR"]
    rated: bool
    eligible_games: int = Field(ge=0)
    eligible_evidence_count: int = Field(ge=0)
    expected_rank: float = Field(ge=1)
    median_rank: float = Field(ge=1)
    mode_rank: int = Field(ge=1)
    interval_50: RankInterval
    interval_80: RankInterval
    interval_95: RankInterval
    interval_widths: IntervalWidths
    rank_1_probability: Probability
    top5_probability: Probability
    top10_probability: Probability
    top25_probability: Probability
    record: str
    previous_official_rank: int | None = Field(default=None, ge=1)
    rank_change: int | None = None
    rank_change_status: Literal[
        "ranked", "newly_rated", "became_unrated", "unrated", "no_comparison"
    ]
    rank_change_display: str
    rank_change_accessible: str


class RankDistributionSummary(PublicModel):
    expected_rank: float = Field(ge=1)
    median_rank: int = Field(ge=1)
    modal_rank: int = Field(ge=1)
    interval_50: RankInterval
    interval_80: RankInterval
    interval_95: RankInterval
    interval_widths: IntervalWidths
    rank_1_probability: Probability
    top5_probability: Probability
    top10_probability: Probability
    top25_probability: Probability


class RankDistribution(PublicModel):
    rank_count: int = Field(ge=1)
    pmf: list[Probability]
    summary: RankDistributionSummary


class DisplayEncoding(PublicModel):
    type: Literal["fixed_scale_integer"]
    scale: int = Field(gt=0)
    normalization: Literal["divide weights by scale"]
    total_weight: int = Field(gt=0)


class PerformanceAxis(PublicModel):
    min_rank: int = Field(ge=1)
    max_rank: int = Field(ge=1)
    bins: int = Field(gt=0)
    direction: Literal["best_to_worst"]
    label: str
    probability_encoding: DisplayEncoding


class FutureMarginAxis(PublicModel):
    min_margin: float
    max_margin: float
    bins: int = Field(gt=0)
    direction: Literal["home_minus_away"]
    unit: Literal["points"]
    probability_encoding: DisplayEncoding
    tail_handling: str


class DisplayAxes(PublicModel):
    performance: PerformanceAxis | None = None
    future_margin: FutureMarginAxis | None = None


class PerformanceSummary(PublicModel):
    rank_count: int = Field(ge=1)
    expected_rank: float = Field(ge=1)
    median_rank: int = Field(ge=1)
    mode_rank: int = Field(ge=1)
    interval_50: RankInterval
    interval_80: RankInterval
    interval_95: RankInterval
    top5_probability: Probability
    top10_probability: Probability
    top25_probability: Probability
    display_pmf: list[EncodedWeight] | None = Field(
        default=None, min_length=40, max_length=40
    )
    performance_percentile: Percentage | None = None
    performance_grade: Literal["A", "B", "C", "D", "F"] | None = None


class PredictionDisplayDistribution(PublicModel):
    masses: list[EncodedWeight] = Field(min_length=40, max_length=40)
    lower_tail_probability: EncodedWeight
    upper_tail_probability: EncodedWeight


class FuturePrediction(PublicModel):
    game_id: str
    prediction_source: PredictionSource
    source_snapshot_id: str
    home_team_id: str
    home_team_name: str
    home_subdivision: Subdivision
    away_team_id: str
    away_team_name: str
    away_subdivision: Subdivision
    neutral_site: bool
    expected_home_margin: float
    median_home_margin: float
    home_win_probability: Probability
    away_win_probability: Probability
    tie_probability: Probability
    margin_interval_50: RankInterval
    margin_interval_80: RankInterval
    margin_interval_95: RankInterval
    display_distribution: PredictionDisplayDistribution | None = None


class GameScore(PublicModel):
    home: int = Field(ge=0)
    away: int = Field(ge=0)


class TeamGameScore(PublicModel):
    team: int = Field(ge=0)
    opponent: int = Field(ge=0)


class CanonicalGame(PublicModel):
    game_id: str
    week: WeekValue
    date: datetime | None
    season_type: str
    conference_game: bool
    neutral_site: bool
    state: GameState
    home_team: TeamIdentity
    away_team: TeamIdentity
    home_team_id: str
    away_team_id: str
    score: GameScore | None = None
    winner_team_id: str | None = None
    home_performance: PerformanceSummary | None = None
    away_performance: PerformanceSummary | None = None
    future_prediction_id: str | None = None
    prediction: FuturePrediction | None = None


class TeamSeasonGame(PublicModel):
    game_id: str
    week: WeekValue
    date: datetime | None
    season_type: str
    conference_game: bool
    opponent: TeamIdentity
    site: Literal["home", "away", "neutral"]
    state: GameState
    result: str | None = None
    score: TeamGameScore | None = None
    modeled: bool
    future_prediction_id: str | None = None
    performance: PerformanceSummary | None = None
    prediction: FuturePrediction | None = None


class WeekSummary(PublicModel):
    week: WeekValue
    key: str
    label: str
    scheduled_game_count: int = Field(ge=0)
    completed_game_count: int = Field(ge=0)
    future_game_count: int = Field(ge=0)
    unresolved_game_count: int = Field(ge=0)
    cancelled_game_count: int = Field(ge=0)


class UnsupportedSeasonGame(PublicModel):
    game_id: str
    reason: str
    detail: str | None = None


class ExpectedWinsSummary(PublicModel):
    expected_final_wins: float
    expected_remaining_wins: float
    interval_50: RankInterval
    interval_80: RankInterval
    interval_95: RankInterval
    median: float
    remaining_interval_50: RankInterval
    remaining_interval_80: RankInterval
    remaining_interval_95: RankInterval
    remaining_median: float


class VarianceDecomposition(PublicModel):
    game_randomness: float = Field(ge=0)
    game_randomness_fraction: Probability
    game_randomness_variance: float = Field(ge=0)
    quality_uncertainty_variance: float = Field(ge=0)
    team_quality: float = Field(ge=0)
    team_quality_fraction: Probability
    total: float = Field(ge=0)


class SeasonOutlook(PublicModel):
    team_id: str
    team_name: str
    subdivision: Subdivision
    forecast_status: Literal["available", "unavailable"]
    completed_wins: int = Field(ge=0)
    completed_losses: int = Field(ge=0)
    completed_ties: int = Field(ge=0)
    completed_regular_season_games: int = Field(ge=0)
    remaining_games: int = Field(ge=0)
    forecast_scope_games: int = Field(ge=0)
    expected_final_wins: float | None = None
    expected_remaining_wins: float | None = None
    median_final_wins: int | None = Field(default=None, ge=0)
    median_remaining_wins: int | None = Field(default=None, ge=0)
    final_win_interval_50: RankInterval | None = None
    final_win_interval_80: RankInterval | None = None
    final_win_interval_95: RankInterval | None = None
    remaining_win_interval_50: RankInterval | None = None
    remaining_win_interval_80: RankInterval | None = None
    remaining_win_interval_95: RankInterval | None = None
    final_win_distribution: dict[str, Probability] | None = None
    remaining_win_distribution: dict[str, Probability] | None = None
    record_probabilities: dict[str, Probability] | None = None
    threshold_probabilities: dict[str, Probability] | None = None
    conditional_expected_wins: ExpectedWinsSummary | None = None
    variance_decomposition: VarianceDecomposition | None = None
    simulated_remaining_games: int | None = Field(default=None, ge=0)
    unsupported_games: list[UnsupportedSeasonGame] = Field(default_factory=list)


class SeasonScope(PublicModel):
    season_type: Literal["regular"]
    future_game_count: int = Field(ge=0)
    unresolved_game_count: int = Field(ge=0)
    forecast_scope_game_count: int = Field(ge=0)
    supported_future_game_count: int = Field(ge=0)
    future_game_ids: list[str]
    unresolved_game_ids: list[str]
    forecast_scope_game_ids: list[str]
    supported_future_game_ids: list[str]
    unsupported_team_ids: list[str]
    unsupported_future_games: list[UnsupportedSeasonGame]
    forecast_status: Literal["available", "partial"]


class SimulationConfiguration(PublicModel):
    simulation_version: str
    likelihood_version: str
    latent_sampling_method: str
    conditional_distribution_method: str
    outer_draw_count: int = Field(ge=1)
    inner_rollout_count: int = Field(ge=0)
    seed: int = Field(ge=0)


class SeasonSimulationInfo(PublicModel):
    simulation_version: str
    prediction_source: PredictionSource
    configuration: SimulationConfiguration
    season_scope: SeasonScope


class PublicationLinks(PublicModel):
    rankings: str
    weeks: str
    season_outlook: str


class TeamLinks(PublicModel):
    rankings: str
    season: str


class RootResources(PublicModel):
    publications: str
    rankings: str
    health: str
    openapi: str


class APIDataInfo(PublicModel):
    schema_version: str
    publication_count: int = Field(ge=0)
    season_count: int = Field(ge=0)


class APIRootResponse(PublicModel):
    api_version: Literal["v1"] = API_VERSION
    name: str
    description: str
    read_only: Literal[True]
    data: APIDataInfo
    resources: RootResources


class PublicationsResponse(PublicModel):
    api_version: Literal["v1"] = API_VERSION
    count: int = Field(ge=0)
    publications: list[PublicationMetadata]


class RankingsResponse(PublicModel):
    api_version: Literal["v1"] = API_VERSION
    publication: PublicationMetadata
    ordering: Literal["display_rank_ascending_nr_last"]
    rankings: list[RankRow]


class TeamRankingResponse(PublicModel):
    api_version: Literal["v1"] = API_VERSION
    publication: PublicationMetadata
    team: RankRow
    rank_distribution: RankDistribution
    links: TeamLinks


class TeamSeasonResponse(PublicModel):
    api_version: Literal["v1"] = API_VERSION
    publication: PublicationMetadata
    team: TeamIdentity
    ranking: RankRow
    games: list[TeamSeasonGame]
    season_outlook: SeasonOutlook | None = None
    season_simulation: SeasonSimulationInfo | None = None
    axes: DisplayAxes


class WeeksResponse(PublicModel):
    api_version: Literal["v1"] = API_VERSION
    publication: PublicationMetadata
    default_week: str | None
    week_count: int = Field(ge=0)
    weeks: list[WeekSummary]
    axes: DisplayAxes


class WeekResponse(PublicModel):
    api_version: Literal["v1"] = API_VERSION
    publication: PublicationMetadata
    week: WeekSummary
    games: list[CanonicalGame]
    axes: DisplayAxes


class GameResponse(PublicModel):
    api_version: Literal["v1"] = API_VERSION
    publication: PublicationMetadata
    game: CanonicalGame
    axes: DisplayAxes


class SeasonOutlookResponse(PublicModel):
    api_version: Literal["v1"] = API_VERSION
    publication: PublicationMetadata
    season_simulation: SeasonSimulationInfo | None = None
    outlooks: list[SeasonOutlook]


class HealthResponse(PublicModel):
    status: Literal["ok"]
    service: str
    api_version: Literal["v1"] = API_VERSION
    publication_data_schema_version: str
    publication_count: int = Field(ge=0)
    season_count: int = Field(ge=0)


class ErrorDetail(PublicModel):
    code: str
    message: str


class ErrorResponse(PublicModel):
    error: ErrorDetail
