"""Run issue 212's fixed Context likelihood recalibration replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gippyrank.research.issue_212_likelihood_recalibration import (
    run_issue_212_replay,
    write_blocked_result,
    write_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--issue195-root", type=Path, required=True)
    parser.add_argument("--issue200-root", type=Path, required=True)
    parser.add_argument("--issue200-manifest", type=Path)
    parser.add_argument("--issue201-root", type=Path, required=True)
    parser.add_argument("--issue204-final-root", type=Path, required=True)
    parser.add_argument("--issue204-root", type=Path, required=True)
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "docs/issue_212_generic_likelihood_recalibration.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_issue_212_replay(
            repository_root=args.repository_root,
            issue195_root=args.issue195_root,
            issue200_root=args.issue200_root,
            issue201_root=args.issue201_root,
            issue204_final_root=args.issue204_final_root,
            issue204_root=args.issue204_root,
            game_root=args.game_root,
            protocol_path=args.protocol,
            issue200_manifest_path=args.issue200_manifest,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        write_blocked_result(
            output_root=args.output_root,
            reason=str(error),
            input_roots={
                "issue195": args.issue195_root,
                "issue200": args.issue200_root,
                "issue200_manifest": args.issue200_manifest
                or args.issue200_root / "manifest.json",
                "issue201": args.issue201_root,
                "issue204_final": args.issue204_final_root,
                "issue204_pre_bowl": args.issue204_root,
                "raw_games": args.game_root,
            },
        )
        raise SystemExit(
            f"Issue 212 is incomplete; see {args.output_root / 'report.md'}: {error}"
        ) from error
    manifest = write_results(result, args.output_root)
    print(f"Issue 212 result artifacts written to {args.output_root}")
    print(f"Conclusion: {manifest['conclusion']}")


if __name__ == "__main__":
    main()
