# Issue 198: historical pre-bowl Massey collection

## Selection rule

For each season, select the latest dated FBS snapshot after the final FBS conference championship and before the first FBS bowl or CFP game. The accepted interval is the full calendar days strictly between those two events. Schedule event timestamps are converted to `America/New_York` before deriving the calendar dates. The selected snapshot must be a CMP date with FBS rows; a later FCS-only date in the same source file does not qualify.

The schedule evidence is archived as December ESPN scoreboard JSON under `data/raw/massey_prebowl/schedules/` with byte hashes and retrieval timestamps. The checked-in `data/research/massey_prebowl_issue_198/schedule_windows.csv` records the event dates and raw-source references. The collection checklist records the selected snapshot date, required export URL, existing file coverage, and collection status for all 23 seasons.

## Current inventory

The existing dated Massey corpus supplies validated FBS snapshots for 2003–2018:

| Season | Snapshot date | Season | Snapshot date |
| --- | --- | --- | --- |
| 2003 | 2003-12-08 | 2011 | 2011-12-05 |
| 2004 | 2004-12-06 | 2012 | 2012-12-10 |
| 2005 | 2005-12-05 | 2013 | 2013-12-16 |
| 2006 | 2006-12-04 | 2014 | 2014-12-08 |
| 2007 | 2007-12-03 | 2015 | 2015-12-07 |
| 2008 | 2008-12-08 | 2016 | 2016-12-05 |
| 2009 | 2009-12-07 | 2017 | 2017-12-04 |
| 2010 | 2010-12-06 | 2018 | 2018-12-03 |

The seven full FBS exports selected for seasons 2019–2025 are listed below. Each retains CMP plus every constituent system column:

| Season | Required file | Official dated page |
| --- | --- | --- |
| 2019 | `fbs2019-20191209.csv` | <https://masseyratings.com/ranks?c=1&d=20191209&s=cf> |
| 2020 | `fbs2020-20201221.csv` | <https://masseyratings.com/ranks?c=1&d=20201221&s=cf> |
| 2021 | `fbs2021-20211206.csv` | <https://masseyratings.com/ranks?c=1&d=20211206&s=cf> |
| 2022 | `fbs2022-20221205.csv` | <https://masseyratings.com/ranks?c=1&d=20221205&s=cf> |
| 2023 | `fbs2023-20231203.csv` | <https://masseyratings.com/ranks?c=1&d=20231203&s=cf> |
| 2024 | `fbs2024-20241208.csv` | <https://masseyratings.com/ranks?c=1&d=20241208&s=cf> |
| 2025 | `fbs2025-20251207.csv` | <https://masseyratings.com/ranks?c=1&d=20251207&s=cf> |

For 2020, the last FBS championship was Dec 19 and the first bowl was Dec 21, leaving Dec 20 as the only qualifying full date. The available Dec 21 snapshot overlaps the first bowl date, and the local Dec 19 snapshot is on the championship date. The user explicitly approved using Dec 21 despite that overlap. The checklist and manifest record this as an approved phase exception; it is included in the targets and is not listed as unresolved.

## Stage existing local inputs

The stage script copies existing ranking files unchanged into a dedicated pre-bowl namespace and stages season-specific CFBD team mappings. It writes an adjacent provenance sidecar with the source URL, selected snapshot date where applicable, import time, byte count, and SHA-256. It refuses to replace different raw bytes.

```sh
uv run python scripts/stage_massey_prebowl_inputs.py \
  --source-root /path/to/existing/data/raw
```

The `--source-root` directory must contain `massey_rankings/` and `cfbd/teams/`. This staging command does not change those source files.

## Import dated exports

Download each full table as CSV. Keep the downloaded file unchanged, then import it with the season and accepted date. The importer copies the raw bytes into `data/raw/massey_prebowl/` and writes a matching `.provenance.json` sidecar.

```sh
uv run python scripts/import_massey_prebowl_export.py /path/to/download.csv \
  --season 2019 --snapshot-date 2019-12-09
```

The official dated page URL is the default provenance source. Use `--source-url` when the export came from a different source and record that source accurately. The resulting CSV must retain the table header and all constituent columns; screenshots, composite-only exports, and final-season files are not valid inputs.

## Build and validate

Build the checklist after staging the historical inputs. The target builder validates dates, FBS identifiers, duplicate team/system keys, CMP separation, raw hashes and provenance sidecars, and then writes full constituent observations and target PMFs. It carries the existing FCS rows in their original order and with their original fields. Its manifest remains `incomplete` while a season is missing or an exception is unresolved. Approved exceptions are listed separately from unresolved exceptions; the approved 2020 snapshot is included in the generated target corpus.

```sh
uv run python scripts/build_massey_prebowl_collection_checklist.py
uv run python scripts/build_massey_prebowl_targets.py \
  --final-observations data/processed/massey/final_observations.csv \
  --output-root data/processed/massey_prebowl
```

The output directory contains `pre_bowl_observations.csv`, full target PMFs in `pre_bowl_target_distributions.csv`, per-system/per-season `coverage_comparison.csv`, `manifest.json`, and `raw_source_manifest.json`. The builder also writes the compact manifests and coverage comparison under `data/research/massey_prebowl_issue_198/` for review and versioning. The manifest reports missing seasons and records the unchanged FCS baseline hash. Raw and processed corpora follow the repository's ignored-data conventions; the checked-in checklist and this protocol document the collection and regeneration path.

## Current status

Army–Navy may occur after an accepted snapshot; this is an agreed target-timing limitation and does not change the date-selection rule.
