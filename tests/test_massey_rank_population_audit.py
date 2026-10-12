import csv
import runpy
from pathlib import Path

import pytest

AUDIT = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "scripts/audit_massey_rank_population_issue_204.py"
    )
)


def _pair(fields: tuple[str, ...]) -> dict[str, object]:
    return {field: index for index, field in enumerate(fields, start=1)}


def _write_pairs(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_pair_audit_accepts_exact_rebuilt_artifact(tmp_path: Path) -> None:
    fields = AUDIT["CONSTITUENT_RANK_PAIR_FIELDS"]
    expected = [_pair(fields)]
    path = tmp_path / "pairs.csv"
    _write_pairs(path, fields, expected)

    assert AUDIT["validate_corrected_pair_artifact"](path, expected) == 1


@pytest.mark.parametrize(
    ("artifact_rows", "artifact_count"),
    [
        pytest.param([], 0, id="missing-row"),
        pytest.param("extra", 2, id="extra-out-of-support-row"),
    ],
)
def test_pair_audit_rejects_missing_or_extra_serialized_rows(
    tmp_path: Path,
    artifact_rows: list[dict[str, object]] | str,
    artifact_count: int,
) -> None:
    fields = AUDIT["CONSTITUENT_RANK_PAIR_FIELDS"]
    expected = _pair(fields)
    rows = [expected]
    if artifact_rows == []:
        rows = []
    else:
        extra = dict(expected)
        extra["home_rank"] = 999
        rows.append(extra)
    path = tmp_path / "pairs.csv"
    _write_pairs(path, fields, rows)

    with pytest.raises(
        ValueError,
        match=rf"row count mismatch .* has {artifact_count} rows, rebuilt 1",
    ):
        AUDIT["validate_corrected_pair_artifact"](path, [expected])
