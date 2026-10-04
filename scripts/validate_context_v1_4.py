"""Execute the frozen Context 1.4 confirmation only on a complete source.

The default path checks the source before constructing any 2026 candidate
forecast state. A complete run also requires a manifest binding the canonical
CSV to preserved raw acquisition files. Analytical settings have no CLI flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gippyrank.context_v1_4_validation_protocol import (
    ProtocolError,
    load_registered_protocol,
)
from gippyrank.context_v1_4_validator import (
    aggregate_scores,
    audit_game_source,
    construct_forecast_states,
    failure_object,
    load_source_manifest,
    load_validator_inputs,
    score_complete_source,
    write_failure,
    write_source_audit,
    write_success,
)

ROOT = Path(__file__).resolve().parents[1]


def run(
    *,
    games: Path,
    output: Path,
    source_manifest: Path | None = None,
    exceptions: Path | None = None,
    check_source_only: bool = False,
    root: Path = ROOT,
) -> int:
    protocol = load_registered_protocol(root)
    inputs = load_validator_inputs(root, protocol)
    report = (
        root / protocol.data["artifacts_for_later_validator"]["report"]
        if output.resolve()
        == (root / "data/processed/context_v1_4_validation").resolve()
        else None
    )
    audit = audit_game_source(games, protocol, inputs, root, exceptions_path=exceptions)
    if audit.status != "SOURCE_COMPLETE":
        write_failure(
            output,
            failure_object(
                audit, protocol, inputs, reason="Week 5 game source is incomplete"
            ),
            report_path=report,
        )
        print(
            f"SOURCE_INCOMPLETE: {len(audit.unresolved_ids)} missing or unresolved games; candidate scoring not executed"
        )
        return 2
    if check_source_only:
        write_source_audit(
            output,
            {
                "status": audit.status,
                "candidate_scores_opened": False,
                "completed_game_count": audit.completed_count,
                "source_sha256": audit.source_sha256,
                "protocol_sha256": protocol.sha256,
            },
            report_path=report,
        )
        print("SOURCE_COMPLETE: source audit passed; candidate scoring not executed")
        return 0
    if source_manifest is None:
        value = failure_object(
            audit,
            protocol,
            inputs,
            reason="completed game source requires a frozen source manifest",
        )
        value["status"] = "SOURCE_UNVERIFIED"
        write_failure(output, value, report_path=report)
        print("SOURCE_UNVERIFIED: supply --source-manifest before scoring")
        return 2
    provenance = load_source_manifest(
        source_manifest, games, exceptions_path=exceptions
    )
    states = construct_forecast_states(root, protocol, inputs)
    rows = score_complete_source(audit, states, protocol, root)
    summary = aggregate_scores(rows, protocol)
    write_success(
        output,
        rows,
        states,
        audit,
        summary,
        protocol,
        provenance,
        report_path=report,
    )
    print(
        f"{summary['decision']}: {summary['full_sample']['eligible_game_count']} games"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", required=True, type=Path, help="canonical current-season games CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/context_v1_4_validation",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="frozen canonical/raw source provenance JSON",
    )
    parser.add_argument(
        "--exceptions", type=Path, help="audited terminal/out-of-scope exception JSON"
    )
    parser.add_argument("--check-source-only", action="store_true")
    args = parser.parse_args()
    try:
        return run(
            games=args.games,
            output=args.output,
            source_manifest=args.source_manifest,
            exceptions=args.exceptions,
            check_source_only=args.check_source_only,
        )
    except (ProtocolError, ValueError, OSError, KeyError) as error:
        print(f"VALIDATION_ABORTED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
