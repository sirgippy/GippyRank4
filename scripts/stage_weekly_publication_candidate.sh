#!/usr/bin/env bash
# Stage exactly the durable artifacts emitted by a successful weekly update.
set -euo pipefail

required_paths=(
  FBS_SCHEDULE_PATH FBS_PROVENANCE_PATH FCS_SCHEDULE_PATH FCS_PROVENANCE_PATH
  PROCESSED_GAMES_PATH CONTEXT_SNAPSHOT_PATH HISTORY_SNAPSHOT_PATH
  PERFORMANCE_SNAPSHOT_PATH
  REPORT_MD_PATH REPORT_JSON_PATH PUBLISH_CONFIG_PATH SITE_DATA_PATH
)

for variable_name in "${required_paths[@]}"; do
  value="${!variable_name:-}"
  case "$value" in
    ""|/*|../*|*/../*|.|*/.)
      echo "Unsafe or empty candidate path in $variable_name" >&2
      exit 1
      ;;
  esac
done

case "$FBS_SCHEDULE_PATH" in data/raw/cfbd/games/*.json) ;; *) exit 1 ;; esac
case "$FBS_PROVENANCE_PATH" in data/raw/cfbd/games/*.json.provenance.json) ;; *) exit 1 ;; esac
case "$FCS_SCHEDULE_PATH" in data/raw/cfbd/games/*-fcs.json) ;; *) exit 1 ;; esac
case "$FCS_PROVENANCE_PATH" in data/raw/cfbd/games/*-fcs.json.provenance.json) ;; *) exit 1 ;; esac
case "$PROCESSED_GAMES_PATH" in data/processed/cfbd/games.csv) ;; *) exit 1 ;; esac
case "$CONTEXT_SNAPSHOT_PATH" in data/processed/snapshots/*/predictive/context) ;; *) exit 1 ;; esac
case "$HISTORY_SNAPSHOT_PATH" in data/processed/snapshots/*/predictive/history) ;; *) exit 1 ;; esac
case "$PERFORMANCE_SNAPSHOT_PATH" in data/processed/snapshots/*/performance) ;; *) exit 1 ;; esac
case "$REPORT_MD_PATH" in data/processed/weekly_updates/*.md) ;; *) exit 1 ;; esac
case "$REPORT_JSON_PATH" in data/processed/weekly_updates/*.json) ;; *) exit 1 ;; esac
case "$PUBLISH_CONFIG_PATH" in site/publish_config.json) ;; *) exit 1 ;; esac
case "$SITE_DATA_PATH" in site/data) ;; *) exit 1 ;; esac

# All paths under data/raw and data/processed are ignored by design.  Force
# only this narrow candidate set; normal site outputs retain normal git rules.
git add -f -- \
  "$FBS_SCHEDULE_PATH" "$FBS_PROVENANCE_PATH" \
  "$FCS_SCHEDULE_PATH" "$FCS_PROVENANCE_PATH" \
  "$PROCESSED_GAMES_PATH" "$CONTEXT_SNAPSHOT_PATH" "$HISTORY_SNAPSHOT_PATH" "$PERFORMANCE_SNAPSHOT_PATH" \
  "$REPORT_MD_PATH" "$REPORT_JSON_PATH"
git add -- "$PUBLISH_CONFIG_PATH" "$SITE_DATA_PATH"

if git diff --cached --quiet; then
  echo "Weekly publication candidate has no staged changes" >&2
  exit 1
fi

invalid_path=0
while IFS= read -r -d '' staged_path; do
  case "$staged_path" in
    "$FBS_SCHEDULE_PATH"|"$FBS_PROVENANCE_PATH"|"$FCS_SCHEDULE_PATH"|"$FCS_PROVENANCE_PATH"|"$PROCESSED_GAMES_PATH"|"$REPORT_MD_PATH"|"$REPORT_JSON_PATH"|"$PUBLISH_CONFIG_PATH"|"$CONTEXT_SNAPSHOT_PATH"/*|"$HISTORY_SNAPSHOT_PATH"/*|"$PERFORMANCE_SNAPSHOT_PATH"/*|"$SITE_DATA_PATH"/*)
      ;;
    *)
      echo "Refusing to commit non-publication path: $staged_path" >&2
      invalid_path=1
      ;;
  esac
done < <(git diff --cached --name-only -z)

if [ "$invalid_path" -ne 0 ]; then
  exit 1
fi
