# Defensive transfer feature study

Issue 103 freezes and audits two research-only candidate measurements:

- `transfer_in_prior_defensive_experience_sum`: the sum of incoming players' prior-season defensive box-score game appearance rates. CFBD does not provide defensive snaps or snap share in the selected endpoints, so this is explicitly not a snap-share feature.
- `transfer_in_prior_defensive_impact_sum`: the sum of incoming players' prior defensive impact values, kept separate from experience.

The reproducible implementation is [defensive_transfer.py](../src/gippyrank/defensive_transfer.py) and the audit command is:

```text
uv run python scripts/fetch_transfer_oracle.py
uv run python scripts/fetch_defensive_transfer_data.py
uv run python scripts/investigate_defensive_transfer.py
```

The fetchers preserve raw CFBD response bytes under `data/raw/` with provenance sidecars. The derived report and machine-readable artifacts are written to `data/processed/defensive_transfer_audit/`.

## Frozen construction

CFBD `/roster` supplies prior-season athlete IDs and positions. CFBD `/games/players` supplies weekly defensive box-score rows. A player's experience proxy is the number of distinct prior-season team games with a defensive box-score row divided by the number of frozen FBS/FCS team games observed.

Impact uses `log1p` of each nonnegative component, standardizes each component within prior season × defensive position group, and takes a simple equal-weight mean:

| Group | Components |
| --- | --- |
| DL / EDGE | tackles, tackles for loss, sacks, quarterback hurries |
| LB | tackles, tackles for loss, sacks, passes defended |
| DB | tackles, passes defended, interceptions |

The position taxonomy is the machine-readable [defensive_position_groups.json](../data/reference/defensive_position_groups.json). Unknown and hybrid labels remain unknown; they are never silently assigned.

Portal-to-prior-season joins use normalized player name + normalized source-team name because the portal response has no shared athlete ID. Explicit aliases are accepted, fuzzy matching is not, and ambiguous joins fail closed. A roster player with complete team-game coverage but no defensive row is classified as a legitimate zero.

This study does not fit Context variants, compare predictive metrics, tune weights against outcomes, or combine the defensive candidates with D5.
