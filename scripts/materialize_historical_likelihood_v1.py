"""Serialize the omitted V1 coefficient vector without altering frozen inputs.

The merged V1 report describes a fitted weighted-pseudo surface but did not
persist its beta vector.  This script reproduces that documented fit from the
immutable historical corpus and writes a new downstream artifact that snapshot
generation can hash and audit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from gippyrank.modeling import design_matrix, fit_robust_surface, read_csv

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "historical_builder", ROOT / "scripts/build_historical_modeling.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen historical-likelihood builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    builder = _load_builder()
    rows = read_csv(ROOT / "data/processed/modeling/historical_modeling_games.csv")
    data = builder.pseudo_data(rows)
    train = data.season < 2022
    matrix = design_matrix(
        data.x, data.y, data.pairing, data.home, data.neutral, True, data.fbs_home
    )
    report = json.loads(
        (ROOT / "data/processed/modeling/margin_model_results.json").read_text()
    )
    model = fit_robust_surface(
        matrix[train],
        data.margin[train],
        data.weight[train],
        df=float(report["specification"]["student_t_df"]),
    )
    artifact = {
        "artifact_kind": "historical_likelihood_v1_coefficients",
        "fit_kind": "weighted_pseudo",
        "semantics": "Exact frozen V1 design matrix, same-system rank pairing, 2003-2021 training subset, eight IRLS iterations.",
        "beta": model["beta"].tolist(),
        "scale": model["scale"],
        "degrees_of_freedom": model["df"],
    }
    output = ROOT / "data/processed/posterior/historical_likelihood_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
