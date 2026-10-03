from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gippyrank.context_prior_v1_3 as candidate_module
import gippyrank.preseason as preseason_module
from gippyrank.context_prior import AnnualFittedInstance, InferenceRow
from gippyrank.context_prior_v1_3 import (
    CONTEXT13_PROVENANCE_SCHEMA_VERSION,
    CONTEXT_1_3_FEATURES,
    CONTEXT_PRIOR_CANDIDATE_VERSION,
    D5_CONTEXT_FEATURES,
    H_FEATURES,
    LOCATION_FEATURE_NAMES,
    MODEL_FEATURE_NAMES,
    PRODUCTION_TRANSFER_PROVENANCE,
    RETROSPECTIVE_2026_PROVENANCE,
    SCALE_FEATURE_NAMES,
    Context13TrainingCorpusSource,
    attach_transfer_features,
    attach_transfer_features_to_inference_rows,
    candidate_guard,
    context13_semantic_specification,
    context13_semantic_specification_sha256,
    fit_model_with_source,
    load_validated_production_transfer_features,
    load_validated_reconstructed_transfer_features,
    model_specification,
    model_specification_metadata,
    validate_feature_contract,
)
from gippyrank.preseason import DirectRankModel, Preprocessor, TeamSeason
from gippyrank.preseason_transfer import (
    ManifestValidationError,
    SnapshotRecord,
)

ROOT = Path(__file__).parents[1]
CANDIDATE = ROOT / "data/processed/preseason/context_v1_3_candidate"


def _context13_builder_module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("build_preseason_context_prior_v1_3")


def _team() -> TeamSeason:
    return TeamSeason(
        2022,
        "fbs",
        "alpha",
        "Alpha",
        2,
        np.asarray([0.0]),
        np.asarray([0.1]),
        np.asarray([1]),
        {name: 0.0 for name in MODEL_FEATURE_NAMES},
    )


def _training_rows(count: int = 8) -> list[TeamSeason]:
    base = np.asarray([-1.5, -0.5, 0.5, 1.5])
    return [
        TeamSeason(
            season=2010 + index,
            subdivision="fbs",
            team_id=f"training-{index}",
            team_name=f"Training {index}",
            population=4,
            lag1_z=base + index * 0.01,
            target_z=base[::-1] + index * 0.02,
            target_ranks=np.asarray([1, 2, 3, 4]),
            features={
                name: float((index + offset) % 5)
                for offset, name in enumerate(MODEL_FEATURE_NAMES)
            },
        )
        for index in range(count)
    ]


def _valid_context13_model() -> DirectRankModel:
    names = list(MODEL_FEATURE_NAMES)
    preprocessor = Preprocessor(
        tuple(names),
        {name: 0.0 for name in names},
        {name: 0.0 for name in names},
        {name: 1.0 for name in names},
    )
    return DirectRankModel(
        feature_names=names,
        preprocessor=preprocessor,
        beta=np.zeros(2 + 2 * len(names)),
        gamma=np.zeros(1 + 2 * len(names)),
        minimum_scale=0.1,
        penalty=0.25,
        optimizer={"success": True},
        lag_count=1,
        family="normal",
        degrees_of_freedom=None,
        location_feature_names=list(MODEL_FEATURE_NAMES),
        scale_feature_names=list(SCALE_FEATURE_NAMES),
    )


def _transfer_row() -> dict[str, object]:
    return {
        "season": 2022,
        "subdivision": "fbs",
        "team_id": "alpha",
        "team_name": "Alpha",
        "transfer_in_prior_usage_sum": 1.5,
        "transfer_in_prior_defensive_impact_db_sum": -0.25,
        "transfer_in_prior_defensive_impact_db_available": 1.0,
        "audit_unresolved_count": 4,
    }


def _write_transfer_validation_inputs(
    root: Path, *, season: int, late: bool = False, usage: float = 1.0
) -> tuple[Path, Path, dict[str, object], list[tuple[int, str, str]]]:
    raw_root = root / "raw"
    raw_root.mkdir(parents=True)
    sources = ("portal", "usage", "stats", "roster", "games_players")
    retrieval = f"{season}-09-01T00:00:00+00:00" if late else f"{season}-08-01T00:00:00+00:00"
    records = []
    for source in sources:
        relative = Path("snapshots") / f"{source}.json"
        raw_path = raw_root / relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_content = f'[{{"source":"{source}"}}]'.encode()
        raw_path.write_bytes(raw_content)
        records.append(
            SnapshotRecord(
                snapshot_id=f"{season}:{source}",
                target_season=season,
                source=source,
                source_season=season if source in {"portal", "usage", "stats"} else season - 1,
                path=relative.as_posix(),
                source_filename=relative.name,
                endpoint=f"/{source}",
                query_parameters={},
                retrieval_timestamp=retrieval,
                target_cutoff=f"{season}-08-15",
                captured_on_or_before_cutoff=not late,
                sha256=hashlib.sha256(raw_content).hexdigest(),
                record_count=1,
                canonical=True,
            ).as_dict()
        )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "raw_root": "raw",
                "cutoff": {"month": 8, "day": 15},
                "snapshots": records,
            }
        ),
        encoding="utf-8",
    )
    feature_path = root / "features.csv"
    feature_path.write_text(
        "season,subdivision,team_id,team_name,transfer_in_prior_usage_sum,"
        "transfer_in_prior_defensive_impact_db_sum,"
        "transfer_in_prior_defensive_impact_db_available\n"
        f"{season},fbs,alpha,Alpha,{usage},0.0,1.0\n",
        encoding="utf-8",
    )
    snapshot_ids = [str(record["snapshot_id"]) for record in records]
    snapshot_hashes = [str(record["sha256"]) for record in records]
    provenance: dict[str, object] = {
        "provenance_class": (
            RETROSPECTIVE_2026_PROVENANCE if late else PRODUCTION_TRANSFER_PROVENANCE
        ),
        "target_season": season,
        "cutoff": f"{season}-08-15",
        "snapshot_ids": snapshot_ids,
        "snapshot_sha256": snapshot_hashes,
        "all_snapshots_on_or_before_cutoff": not late,
    }
    if late:
        provenance.update(
            {
                "raw_source_hashes": snapshot_hashes,
                "retrieval_timestamps": [retrieval] * len(records),
                "source_endpoints": [str(record["endpoint"]) for record in records],
                "derivation_timestamp": f"{season}-09-02T00:00:00+00:00",
                "known_absence_of_archived_august_15_transfer_snapshot": True,
                "provenance_statement": "retrospective reconstructed state from late source retrievals",
            }
        )
    provenance["source_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    return feature_path, manifest_path, provenance, [(season, "fbs", "alpha")]


def test_frozen_contract_has_exact_features_and_equation_placement() -> None:
    assert validate_feature_contract(MODEL_FEATURE_NAMES) == MODEL_FEATURE_NAMES
    assert validate_feature_contract(D5_CONTEXT_FEATURES, exact=False) == D5_CONTEXT_FEATURES
    assert LOCATION_FEATURE_NAMES == MODEL_FEATURE_NAMES
    assert SCALE_FEATURE_NAMES == (
        "lag2_z_mean",
        "lag3_z_mean",
        "long_run_z_mean",
    )
    assert "returning_pct_passing_ppa" not in MODEL_FEATURE_NAMES
    with pytest.raises(ValueError):
        validate_feature_contract((*MODEL_FEATURE_NAMES, "transfer_out_count"))

    spec = model_specification()
    metadata = model_specification_metadata()
    assert spec.spec_version == CONTEXT_PRIOR_CANDIDATE_VERSION == "1.3"
    assert metadata["context_features_affect"] == "location_only"
    assert metadata["history_features_affect"] == ["location", "scale"]
    assert metadata["active_production_version"] == "1.3"


def test_context13_semantic_hash_excludes_lifecycle_metadata(monkeypatch) -> None:
    original = context13_semantic_specification_sha256()
    original_spec = context13_semantic_specification()
    assert "active_production_version" not in original_spec
    assert "candidate" not in original_spec
    assert "status" not in original_spec

    monkeypatch.setattr(candidate_module, "ACTIVE_CONTEXT_PRIOR_VERSION", "1.4")
    monkeypatch.setattr(candidate_module, "CONTEXT13_LIFECYCLE_STATUS", "validated")
    monkeypatch.setattr(candidate_module, "CONTEXT13_IS_CANDIDATE", True)
    lifecycle_spec = model_specification_metadata()
    assert lifecycle_spec["active_production_version"] == "1.4"
    assert lifecycle_spec["status"] == "validated"
    assert lifecycle_spec["candidate"] is True
    assert context13_semantic_specification_sha256() == original


def test_context13_semantic_hash_changes_with_model_semantics(monkeypatch) -> None:
    original = context13_semantic_specification_sha256()
    monkeypatch.setattr(candidate_module, "FROZEN_PENALTY", 0.3)
    assert context13_semantic_specification_sha256() != original


@pytest.mark.parametrize(
    ("constant", "changed"),
    [
        ("DIRECT_RANK_INITIAL_LAG_BETA", 0.56),
        ("DIRECT_RANK_INITIAL_SCALE", 0.71),
        ("DIRECT_RANK_GAMMA_COEFFICIENT_BOUNDS", (-4.0, 4.0)),
        ("DIRECT_RANK_FIXED_ZERO_COEFFICIENT_BOUNDS", (-1e-12, 1e-12)),
        ("DIRECT_RANK_LOG_SCALE_CLIP_BOUNDS", (-5.0, 3.9)),
        ("DIRECT_RANK_BETA_REGULARIZATION_WEIGHT", 1.01),
        ("DIRECT_RANK_GAMMA_REGULARIZATION_WEIGHT", 0.26),
        ("DIRECT_RANK_OPTIMIZER_MAXITER", 501),
    ],
)
def test_context13_semantic_hash_tracks_fit_algorithm_contract(
    monkeypatch, constant: str, changed: object
) -> None:
    original = context13_semantic_specification_sha256()
    spec = context13_semantic_specification()
    regularization = spec["regularization"]
    assert regularization["beta_objective"] == "penalty * sum(beta ** 2) / N"
    assert regularization["beta_gradient"] == "2 * penalty * beta / N"
    assert regularization["gamma_objective"] == (
        "penalty * gamma_regularization_weight * sum(gamma[1:] ** 2) / N"
    )
    assert regularization["gamma_gradient"] == (
        "2 * penalty * gamma_regularization_weight * gamma[1:] / N"
    )
    assert regularization["gamma_intercept_penalized"] is False
    monkeypatch.setattr(preseason_module, constant, changed)
    assert context13_semantic_specification_sha256() != original


def test_context13_semantic_hash_tracks_optimizer_retry_contract(monkeypatch) -> None:
    original = context13_semantic_specification_sha256()
    optimizer = context13_semantic_specification()["optimizer"]
    assert optimizer["retry"] == {
        "trigger_message": "ITERATIONS REACHED LIMIT",
        "maxiter": 2000,
        "attempts": 1,
        "restart_from_initialization": True,
        "otherwise": "raise optimizer failure",
    }
    monkeypatch.setattr(candidate_module, "CONTEXT13_OPTIMIZER_RETRY_MAXITER", 2001)
    assert context13_semantic_specification_sha256() != original


@pytest.mark.parametrize(
    ("module", "constant", "changed"),
    [
        (candidate_module, "CONTEXT13_MINIMUM_SCALE", 0.11),
        (candidate_module, "CONTEXT13_LAG_COUNT", 2),
        (candidate_module, "CONTEXT13_DISTRIBUTION_FAMILY", "student_t"),
        (candidate_module, "CONTEXT13_DEGREES_OF_FREEDOM", 5.0),
        (candidate_module, "CONTEXT13_TEAM_SEASON_WEIGHT", 2.0),
        (candidate_module, "SCALE_FEATURE_NAMES", H_FEATURES[:-1]),
        (preseason_module, "DIRECT_RANK_OPTIMIZER_METHOD", "BFGS"),
        (preseason_module, "DIRECT_RANK_OPTIMIZER_FTOL", 1e-9),
        (preseason_module, "DIRECT_RANK_OPTIMIZER_GTOL", 1e-5),
        (preseason_module, "PREPROCESSOR_SCALE_FLOOR", 1e-7),
        (preseason_module, "PREPROCESSOR_STD_DDOF", 1),
    ],
)
def test_context13_semantic_hash_tracks_explicit_fit_and_preprocessing_contract(
    monkeypatch, module, constant: str, changed: object
) -> None:
    original = context13_semantic_specification_sha256()
    monkeypatch.setattr(module, constant, changed)
    assert context13_semantic_specification_sha256() != original


def test_context13_fit_ignores_poisoned_generic_defaults(monkeypatch) -> None:
    rows = _training_rows()
    calls = []

    def poisoned_fit(
        cls, training, features, penalty=9.0, minimum_scale=9.0,
        optimizer_options=None, lag_count=9, family="student_t",
        degrees_of_freedom=9.0, location_feature_names=None,
        scale_feature_names=None, row_weights=None,
        preprocessor_scale_floor=9.0, preprocessor_std_ddof=9,
    ):
        assert len(training) == len(rows)
        assert all(actual is expected for actual, expected in zip(training, rows, strict=True))
        assert tuple(features) == MODEL_FEATURE_NAMES
        assert penalty == 0.25
        assert minimum_scale == 0.10
        assert lag_count == 1
        assert family == "normal"
        assert degrees_of_freedom is None
        assert location_feature_names == list(LOCATION_FEATURE_NAMES)
        assert scale_feature_names == list(SCALE_FEATURE_NAMES)
        np.testing.assert_array_equal(row_weights, np.ones(len(rows)))
        assert preprocessor_scale_floor == 1e-8
        assert preprocessor_std_ddof == 0
        assert optimizer_options == {"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6}
        calls.append(True)
        return _valid_context13_model()

    monkeypatch.setattr(DirectRankModel, "fit", classmethod(poisoned_fit))
    model, instance = candidate_module.fit_model(
        rows, target_season=2026, trained_through_season=2025
    )
    assert calls == [True]
    assert model.minimum_scale == 0.1
    assert instance.trained_through_season == 2025


def test_context13_optimizer_uses_the_semantic_method_and_tolerances(monkeypatch) -> None:
    expected = context13_semantic_specification()["optimizer"]
    observed = []

    def optimizer_stub(_objective, initial, *, method, jac, bounds, options):
        assert callable(jac)
        assert len(bounds) == len(initial)
        observed.append((method, options))
        return SimpleNamespace(
            x=initial, success=True, status=0, message="converged",
            nit=0, nfev=1, fun=0.0,
        )

    monkeypatch.setattr(preseason_module, "minimize", optimizer_stub)
    candidate_module.fit_model(
        _training_rows(), target_season=2026, trained_through_season=2025
    )
    assert observed == [
        (
            expected["method"],
            {
                "maxiter": expected["maxiter"],
                "ftol": expected["ftol"],
                "gtol": expected["gtol"],
            },
        )
    ]


def test_canonical_training_loader_mints_typed_complete_corpus_source(
    tmp_path: Path, monkeypatch
) -> None:
    builder = _context13_builder_module()
    rows = _training_rows()
    transfer_path = tmp_path / "historical_transfer_features.csv"
    transfer_provenance_path = tmp_path / "historical_transfer_features.provenance.json"
    artifact_names = (
        "historical_rank_distribution_artifact",
        "historical_context_feature_artifact",
        "historical_transfer_feature_artifact",
        "historical_transfer_feature_provenance",
        "context13_semantic_implementation",
        "direct_rank_implementation",
        "historical_row_builder",
        "context_feature_builder",
        "transfer_feature_builder",
    )
    paths = {}
    for name in artifact_names:
        path = {
            "historical_transfer_feature_artifact": transfer_path,
            "historical_transfer_feature_provenance": transfer_provenance_path,
        }.get(name, tmp_path / f"{name}.txt")
        if path not in {transfer_path, transfer_provenance_path}:
            path.write_text(f"{name} source\n", encoding="utf-8")
        paths[name] = path
    transfer_provenance_path.write_text(
        json.dumps(
            {
                "candidate_spec_version": "1.3",
                "context_features": list(CONTEXT_1_3_FEATURES),
                "provenance_class": "retrospective_research_reconstruction",
                "row_count": len(rows),
                "transfer_features": [
                    "transfer_in_prior_usage_sum",
                    "transfer_in_prior_defensive_impact_db_sum",
                    "transfer_in_prior_defensive_impact_db_available",
                ],
            }
        )
    )

    def write_transfer_rows(usage_offset: float = 0.0) -> None:
        fields = (
            "season",
            "subdivision",
            "team_id",
            "team_name",
            "transfer_in_prior_usage_sum",
            "transfer_in_prior_defensive_impact_db_sum",
            "transfer_in_prior_defensive_impact_db_available",
            "provenance_class",
        )
        with transfer_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                {
                    "season": row.season,
                    "subdivision": row.subdivision,
                    "team_id": row.team_id,
                    "team_name": row.team_name,
                    "transfer_in_prior_usage_sum": 0.5 + index + usage_offset,
                    "transfer_in_prior_defensive_impact_db_sum": 0.1 * index,
                    "transfer_in_prior_defensive_impact_db_available": 1.0,
                    "provenance_class": "retrospective_research_reconstruction",
                }
                for index, row in enumerate(rows)
            )

    write_transfer_rows()
    monkeypatch.setattr(builder, "HISTORICAL_TRANSFER_FEATURES", transfer_path)
    monkeypatch.setattr(builder, "_training_source_artifact_paths", lambda: paths)
    monkeypatch.setattr(builder, "base_context_rows", lambda *, max_season: (rows, [], []))
    built_rows, corpus_source, _, _ = builder.load_context13_training_corpus(
        target_season=2026,
        trained_through_season=2025,
    )
    assert len(built_rows) == len(rows)
    assert corpus_source.training_row_count == len(rows)
    assert corpus_source.target_season == 2026
    assert corpus_source.trained_through_season == 2025
    assert corpus_source.to_metadata()["provenance_schema_version"] == (
        CONTEXT13_PROVENANCE_SCHEMA_VERSION
    )
    assert corpus_source.historical_transfer_feature_artifact_sha256 == (
        hashlib.sha256(transfer_path.read_bytes()).hexdigest()
    )
    assert corpus_source.source_identity_sha256
    corpus_source.validate_rows(built_rows)

    model = _valid_context13_model()

    def fit_stub(
        _rows,
        *,
        target_season,
        trained_through_season,
        context_features,
    ):
        assert tuple(context_features) == CONTEXT_1_3_FEATURES
        return model, AnnualFittedInstance(
            "context_prior", "1.3", trained_through_season, target_season
        )

    monkeypatch.setattr(candidate_module, "fit_model", fit_stub)
    fitted_model, fitted_instance, canonical_fit = fit_model_with_source(
        built_rows,
        target_season=2026,
        trained_through_season=2025,
        training_corpus_source=corpus_source,
    )
    assert canonical_fit.provenance_class == "canonical_context13_fit"
    assert canonical_fit.verified_training_corpus_input_sha256 == (
        corpus_source.training_corpus_input_sha256
    )
    assert canonical_fit.verified_training_row_count == len(built_rows)
    assert canonical_fit.training_corpus_source_identity_sha256 == (
        corpus_source.source_identity_sha256
    )
    model_path = tmp_path / "fitted_model.json"
    instance_path = tmp_path / "fitted_instance.json"
    model_path.write_text(
        json.dumps(
            {
                "model_family": "context_prior",
                "spec_version": "1.3",
                "model": fitted_model.metadata(),
                "fit_provenance": canonical_fit.to_metadata(),
            }
        ),
        encoding="utf-8",
    )
    instance_path.write_text(
        json.dumps(fitted_instance.metadata()),
        encoding="utf-8",
    )
    _, loaded_instance, loaded_source = candidate_module.load_validated_context13_fitted_model(
        model_path,
        instance_path,
        training_rows=built_rows,
        training_corpus_source=corpus_source,
    )
    assert loaded_instance.metadata() == fitted_instance.metadata()
    assert loaded_source.to_metadata() == canonical_fit.to_metadata()
    with pytest.raises(ValueError, match="authoritative Context 1.3 training corpus"):
        fit_model_with_source(
            built_rows[:-1],
            target_season=2026,
            trained_through_season=2025,
            training_corpus_source=corpus_source,
        )

    with pytest.raises(TypeError, match="canonical historical loader"):
        Context13TrainingCorpusSource(
            target_season=2026,
            trained_through_season=2025,
            source_artifact_sha256=(),
            canonical_base_historical_source_identity_sha256="a" * 64,
            context_feature_construction_identity_sha256="b" * 64,
            historical_transfer_feature_artifact_sha256="c" * 64,
            historical_transfer_feature_provenance_sha256="d" * 64,
            fbs_team_season_keys_sha256="e" * 64,
            training_corpus_input_sha256="f" * 64,
            training_row_count=len(rows),
        )

    row = built_rows[0]
    changed_corpora = []
    changed_features = dict(row.features)
    changed_features["transfer_in_prior_usage_sum"] = 999.0
    changed_corpora.append([replace(row, features=changed_features), *built_rows[1:]])
    changed_corpora.append([replace(row, lag1_z=row.lag1_z + 0.25), *built_rows[1:]])
    changed_corpora.append([replace(row, target_z=row.target_z + 0.25), *built_rows[1:]])
    changed_corpora.append(
        [replace(row, target_ranks=np.asarray([4, 3, 2, 1])), *built_rows[1:]]
    )
    changed_corpora.append([replace(row, team_id="changed-identity"), *built_rows[1:]])
    for changed_rows in changed_corpora:
        with pytest.raises(ValueError, match="authoritative Context 1.3 training corpus"):
            corpus_source.validate_rows(changed_rows)
    with pytest.raises(ValueError, match="authoritative Context 1.3 training corpus"):
        corpus_source.validate_rows(built_rows[:-1])
    with pytest.raises(ValueError, match="authoritative Context 1.3 training corpus"):
        corpus_source.validate_rows([*built_rows, replace(built_rows[0], team_id="extra")])

    write_transfer_rows(usage_offset=1.0)
    _, changed_source, _, _ = builder.load_context13_training_corpus(
        target_season=2026,
        trained_through_season=2025,
    )
    assert changed_source.source_identity_sha256 != corpus_source.source_identity_sha256
    assert changed_source.training_corpus_input_sha256 != corpus_source.training_corpus_input_sha256


def test_only_typed_canonical_corpus_source_can_mint_canonical_fit(
    monkeypatch,
) -> None:
    rows = _training_rows()
    model = _valid_context13_model()

    def fit_stub(
        _rows,
        *,
        target_season,
        trained_through_season,
        context_features,
    ):
        assert tuple(context_features) == CONTEXT_1_3_FEATURES
        return model, AnnualFittedInstance(
            "context_prior", "1.3", trained_through_season, target_season
        )

    monkeypatch.setattr(candidate_module, "fit_model", fit_stub)
    arbitrary_source_fit = fit_model_with_source(
        rows,
        target_season=2026,
        trained_through_season=2025,
    )
    assert arbitrary_source_fit[2].provenance_class == "research_only"
    assert arbitrary_source_fit[2].research_training_row_count == len(rows)
    assert arbitrary_source_fit[2].research_training_rows_sha256
    assert arbitrary_source_fit[2].verified_training_corpus_input_sha256 is None
    with pytest.raises(TypeError, match="typed training-corpus source"):
        fit_model_with_source(
            rows,
            target_season=2026,
            trained_through_season=2025,
            training_corpus_source=object(),
        )
    truncated_fit = fit_model_with_source(
        rows[:-1],
        target_season=2026,
        trained_through_season=2025,
    )
    assert truncated_fit[2].provenance_class == "research_only"
    extra_fit = fit_model_with_source(
        [*rows, replace(rows[-1], team_id="extra-team")],
        target_season=2026,
        trained_through_season=2025,
    )
    assert extra_fit[2].provenance_class == "research_only"
    with pytest.raises(ValueError, match="includes rows beyond its cutoff"):
        fit_model_with_source(
            [*rows, replace(rows[0], season=2026, team_id="target-season")],
            target_season=2026,
            trained_through_season=2025,
        )


def test_transfer_attachment_is_attach_only_and_preserves_history() -> None:
    row = _team()
    attached = attach_transfer_features([row], [_transfer_row()])
    assert attached[0].features["lag2_z_mean"] == 0.0
    assert attached[0].features["transfer_in_prior_usage_sum"] == 1.5
    assert attached[0].features[
        "transfer_in_prior_defensive_impact_db_sum"
    ] == -0.25
    assert "audit_unresolved_count" not in attached[0].features

    with pytest.raises(ValueError, match="duplicate transfer feature row"):
        attach_transfer_features([row], [_transfer_row(), _transfer_row()])
    with pytest.raises(ValueError, match="no transfer features"):
        attach_transfer_features([row], [], require_all=True)
    leaked = TeamSeason(
        **{
            **row.__dict__,
            "features": {**row.features, "transfer_in_rating_sum": 1.0},
        }
    )
    with pytest.raises(ValueError, match="unapproved feature columns"):
        attach_transfer_features([leaked], [_transfer_row()])


def test_inference_attachment_keeps_target_outcomes_out() -> None:
    row = InferenceRow(
        2022,
        "fbs",
        "alpha",
        "Alpha",
        2,
        (0.0,),
        (),
        {name: 0.0 for name in MODEL_FEATURE_NAMES},
    )
    attached = attach_transfer_features_to_inference_rows([row], [_transfer_row()])
    assert attached[0].features["transfer_in_prior_usage_sum"] == 1.5
    attached[0].require_no_target()


def test_activated_guard_allows_2026_without_relabeling_its_provenance() -> None:
    candidate_guard(2025)
    candidate_guard(2026)
    with pytest.raises(ManifestValidationError):
        from gippyrank.context_prior_v1_3 import (
            load_validated_production_transfer_features,
        )

        load_validated_production_transfer_features(
            Path("missing.csv"),
            Path("missing-manifest.json"),
            target_season=2026,
            expected_team_keys=[(2026, "fbs", "alpha")],
            provenance={},
        )


def test_production_transfer_provenance_is_authoritative_and_content_addressed(
    tmp_path: Path,
) -> None:
    feature_path, manifest_path, provenance, expected = _write_transfer_validation_inputs(
        tmp_path / "checkout-a", season=2027
    )
    loaded, identity = candidate_module.load_validated_production_transfer_features(
        feature_path,
        manifest_path,
        target_season=2027,
        expected_team_keys=expected,
        provenance=provenance,
    )
    assert loaded[0]["team_id"] == "alpha"
    assert identity.provenance_class == PRODUCTION_TRANSFER_PROVENANCE
    assert identity.cutoff_state == "on_time"
    assert identity.transfer_feature_artifact_sha256 == hashlib.sha256(
        feature_path.read_bytes()
    ).hexdigest()
    assert identity.source_manifest_sha256 == provenance["source_manifest_sha256"]
    assert identity.canonical_snapshot_ids
    assert identity.to_metadata()["feature_artifact_id"] == (
        "gippyrank.context1_3.transfer_features.season_2027"
    )

    with pytest.raises(ManifestValidationError, match="canonical snapshot IDs and hashes"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2027,
            expected_team_keys=expected,
            provenance={**provenance, "snapshot_ids": []},
        )
    with pytest.raises(ManifestValidationError, match="source manifest hash is mismatched"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2027,
            expected_team_keys=expected,
            provenance={**provenance, "source_manifest_sha256": "f" * 64},
        )
    with pytest.raises(ManifestValidationError, match="feature artifact hash is mismatched"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2027,
            expected_team_keys=expected,
            provenance={**provenance, "transfer_feature_artifact_sha256": "e" * 64},
        )
    with pytest.raises(ManifestValidationError, match="cutoff state"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2027,
            expected_team_keys=expected,
            provenance={**provenance, "all_snapshots_on_or_before_cutoff": False},
        )
    with pytest.raises(ManifestValidationError, match="population differs"):
        candidate_module.load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2027,
            expected_team_keys=[(2027, "fbs", "different-team")],
            provenance=provenance,
        )

    malformed_root = tmp_path / "malformed-canonical-flag"
    malformed_feature, malformed_manifest, malformed_provenance, malformed_expected = (
        _write_transfer_validation_inputs(malformed_root, season=2027)
    )
    manifest_payload = json.loads(malformed_manifest.read_text(encoding="utf-8"))
    manifest_payload["snapshots"][0]["canonical"] = "false"
    malformed_manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    malformed_provenance["source_manifest_sha256"] = hashlib.sha256(
        malformed_manifest.read_bytes()
    ).hexdigest()
    with pytest.raises(ManifestValidationError, match="canonical flag is not boolean"):
        candidate_module.load_validated_production_transfer_features(
            malformed_feature,
            malformed_manifest,
            target_season=2027,
            expected_team_keys=malformed_expected,
            provenance=malformed_provenance,
        )

    second_root = tmp_path / "checkout-b"
    second_feature, second_manifest, second_provenance, second_expected = (
        _write_transfer_validation_inputs(second_root, season=2027)
    )
    _, second_identity = candidate_module.load_validated_production_transfer_features(
        second_feature,
        second_manifest,
        target_season=2027,
        expected_team_keys=second_expected,
        provenance=second_provenance,
    )
    assert second_identity.source_identity_sha256 == identity.source_identity_sha256
    assert second_feature.resolve() != feature_path.resolve()

    second_feature.write_text(
        second_feature.read_text(encoding="utf-8").replace(",Alpha,1.0,", ",Alpha,2.0,"),
        encoding="utf-8",
    )
    _, changed_identity = candidate_module.load_validated_production_transfer_features(
        second_feature,
        second_manifest,
        target_season=2027,
        expected_team_keys=second_expected,
        provenance=second_provenance,
    )
    assert changed_identity.transfer_feature_artifact_sha256 != identity.transfer_feature_artifact_sha256
    assert changed_identity.source_identity_sha256 != identity.source_identity_sha256
    bad_raw_source = second_root / "raw/snapshots/portal.json"
    bad_raw_source.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="snapshot hash mismatch"):
        candidate_module.load_validated_production_transfer_features(
            second_feature,
            second_manifest,
            target_season=2027,
            expected_team_keys=second_expected,
            provenance=second_provenance,
        )


def test_2026_reconstruction_requires_retrospective_class_and_complete_attestation(
    tmp_path: Path,
) -> None:
    feature_path, manifest_path, provenance, expected = _write_transfer_validation_inputs(
        tmp_path / "retrospective-2026", season=2026, late=True
    )
    loaded, identity = load_validated_reconstructed_transfer_features(
        feature_path,
        manifest_path,
        target_season=2026,
        expected_team_keys=expected,
        provenance=provenance,
    )
    assert loaded[0]["team_id"] == "alpha"
    assert identity.provenance_class == RETROSPECTIVE_2026_PROVENANCE
    assert identity.cutoff_state == "retrospective_reconstruction"
    assert identity.archived_august_15_snapshot_absent is True
    assert identity.retrieval_timestamps
    assert identity.source_endpoints
    assert identity.reconstructed_state_declaration
    assert identity.to_metadata()["provenance_class"] == (
        "retrospective_2026_reconstruction"
    )
    with pytest.raises(TypeError):
        replace(identity, provenance_class=PRODUCTION_TRANSFER_PROVENANCE)
    with pytest.raises(ManifestValidationError, match="must acknowledge the absent"):
        load_validated_reconstructed_transfer_features(
            feature_path,
            manifest_path,
            target_season=2026,
            expected_team_keys=expected,
            provenance={
                **provenance,
                "known_absence_of_archived_august_15_transfer_snapshot": False,
            },
        )
    with pytest.raises(ManifestValidationError, match="must be"):
        load_validated_reconstructed_transfer_features(
            feature_path,
            manifest_path,
            target_season=2026,
            expected_team_keys=expected,
            provenance={**provenance, "provenance_class": PRODUCTION_TRANSFER_PROVENANCE},
        )
    with pytest.raises(ManifestValidationError, match="on-time production snapshot"):
        load_validated_production_transfer_features(
            feature_path,
            manifest_path,
            target_season=2026,
            expected_team_keys=expected,
            provenance={**provenance, "provenance_class": PRODUCTION_TRANSFER_PROVENANCE},
        )


def test_candidate_artifacts_are_separate_and_parity_validated() -> None:
    expected_files = {
        "candidate_report.md",
        "coefficients.csv",
        "context_coverage.csv",
        "evaluation.json",
        "feature_provenance.json",
        "fitted_model.json",
        "historical_transfer_features.csv",
        "historical_transfer_features.provenance.json",
        "model_report.json",
        "model_spec.json",
        "parity_report.json",
        "predictions.csv",
        "rolling_metrics.csv",
    }
    assert expected_files <= {path.name for path in CANDIDATE.iterdir()}
    spec = json.loads((CANDIDATE / "model_spec.json").read_text())
    assert spec["spec_version"] == "1.3"
    assert spec["active_production_version"] == "1.3"
    assert spec["features"] == list(MODEL_FEATURE_NAMES)
    report = json.loads((CANDIDATE / "model_report.json").read_text())
    assert all(value["passed"] for value in report["parity"].values())
    assert report["production_activation"]["ready_for_first_future_season_with_valid_on_time_snapshot"]
    assert "retrospective reconstruction" in report["production_activation"]["2026_guardrail"]

    with (CANDIDATE / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 534
    assert {row["spec_version"] for row in rows} == {"1.3"}
    assert {row["candidate_status"] for row in rows} == {"active_production"}
    assert all(
        str(value) != "" for row in rows for value in json.loads(row["pmf"])
    )

    provenance = json.loads(
        (CANDIDATE / "historical_transfer_features.provenance.json").read_text()
    )
    assert provenance["provenance_class"] == "retrospective_research_reconstruction"
    assert "/home/" not in json.dumps(report["source_artifacts"])
    assert hashlib.sha256(
        (ROOT / "data/processed/preseason/context/annual/2026/predictions.csv").read_bytes()
    ).hexdigest() == "641182890ec88ea8bc6150cc97047ddc688d0c680486d9fcc63a58d7bfae9132"
