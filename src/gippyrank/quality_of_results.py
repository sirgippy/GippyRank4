"""Production Quality of Results V1 with the selected, fixed margin weight.

The frozen research implementation remains the single implementation of the
likelihood and inference mathematics. Its source and research artifacts are
unchanged; production callers have no weight-selection parameter.
"""

from __future__ import annotations

from typing import Final

from gippyrank.posterior.engine import Game, LikelihoodV1, Team
from gippyrank.research.quality_of_results import (
    QualityOfResultsInference,
)
from gippyrank.research.quality_of_results import (
    infer_quality_of_results as _infer,
)

QUALITY_OF_RESULTS_MARGIN_WEIGHT: Final = 0.40
QUALITY_OF_RESULTS_MODEL: Final = "quality_of_results_v1"
QUALITY_OF_RESULTS_VERSION: Final = "1.0"
QUALITY_OF_RESULTS_VARIANT: Final = "quality_of_results"
QUALITY_OF_RESULTS_EVIDENCE_MODE: Final = "outcome_and_tempered_margin"


def infer_quality_of_results(
    teams: list[Team] | tuple[Team, ...],
    games: list[Game] | tuple[Game, ...],
    likelihood: LikelihoodV1,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
    damping: float = 0.35,
) -> QualityOfResultsInference:
    """Run the validated network at the immutable production choice of λ=.40."""
    return _infer(
        teams,
        games,
        likelihood,
        QUALITY_OF_RESULTS_MARGIN_WEIGHT,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
