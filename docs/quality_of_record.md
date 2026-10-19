# Quality of Record

Quality of Record is the second variant of the Performance ranking
family. It uses the same Context preseason prior, Historical Likelihood V1 rank
coordinates, site effects, subdivision orientation, and deterministic belief
propagation as Quality of Play. Its game evidence is only the winner.

For every included game, the frozen Student-t margin model supplies a location
for each home/away rank pair. The binary factor is `P(modeled margin > 0)` for
a win in the model's margin orientation, or `P(modeled margin < 0)` for a loss.
Cross-subdivision games retain the model's FBS-minus-FCS orientation. A one-point
win and a 50-point win therefore use identical factors.

The generator reruns BP on the entire eligible Context game network with these
binary factors. It then computes each FBS team's PMF as
`normalize(BinaryContextPosterior / ContextPrior)`. It never reads the Context
margin posterior. Zero-game FBS teams receive a uniform PMF, `rated=false`, and
display rank `NR`. A nonconverged binary run raises an error before any snapshot
files are written.

Build from a valid Predictive Context snapshot with:

```bash
uv run python scripts/build_quality_of_record_snapshot.py \
  --context data/processed/snapshots/<season>/<context-snapshot>/predictive/context
```

The output is written to
`data/processed/snapshots/<season>/<logical-snapshot>/performance/quality_of_record/`.
It contains `metadata.json`, `diagnostics.json`, `rankings.json`, `rankings.csv`,
`posterior_pmfs.csv`, and the copied `included_games.csv`. Metadata declares
`ranking_family=performance`, `performance_variant=quality_of_record`,
`model=quality_of_record_v1`, and `evidence_mode=win_loss`, and records the
source cutoff, included game hashes,
Context prior identity, Historical Likelihood V1 identity, and inference
configuration. The copied game rows retain original scores for provenance;
their magnitudes do not enter binary inference.

## Publication and historical migration

Performance now has two variants: `quality_of_play` (the original scoring-margin
result) and `quality_of_record` (binary win/loss evidence). Older Performance
metadata without a variant is interpreted as `quality_of_play`. Its snapshot ID,
PMFs, ranking values, publication status, and chronology are retained. Legacy
`family=performance` links and API selectors default to Quality of Play. New
links add `variant=quality_of_record`; static API publication metadata includes
`performance_variant` and `evidence_mode`. Official movement baselines stay
within a variant.

Normal weekly publication builds Predictive Context, Predictive History,
Quality of Play, and Quality of Record for the current candidate. All four
share cutoffs and eligible games, are validated and staged together, and enter
unchanged-publication detection. The review includes both Performance Top 25s
and diagnostics. Historical backfill is never invoked by weekly automation or CI.

Run the one-time historical migration explicitly with:

```bash
uv run python scripts/backfill_performance_variants.py
```

It visits every configured in-season Context publication, Official and Interim,
using that publication's Context version, hashed frozen prior, included game
rows, pinned Historical Likelihood V1, and retained inference configuration.
Legacy snapshots predate some semantic hashes: these are derived in memory
from the retained artifacts. Their recorded deterministic BP engine uses its
original defaults (500 iterations, tolerance 1e-9, damping 0.35) when explicit
configuration is absent. No mutable schedule, fresh acquisition, or current
Context version is substituted. Missing inputs are reported with the publication
slot and source snapshot; the configuration is only registered after every
candidate validates. No source snapshots are rewritten. A previously absent
Quality of Play companion is generated once, and existing results are retained.

If some publications lack binary outcomes, `--allow-blocked` registers all
reproducible companions and records every remaining blocker in the audit
inventory. Without that option, any blocker prevents configuration changes.
Neither mode invents a winner or removes a game. The retained Sep. 6, 7, and 8
publications contain game 401866625 (Campbell–Western Carolina) as completed at
0–0 and therefore require cutoff-era winner evidence before binary replay.

The deterministic audit inventory is
`data/processed/quality_of_record_backfill/manifest.json`. It records cutoffs,
Context versions, input hashes, and ranking/PMF file hashes. Repeated backfills
validate and retain existing results byte for byte. Fresh inference repeats
produce identical substantive metadata, ranking files, and PMFs; diagnostic
runtime is observational.

Game predictions, season simulations, and completed-game ratings on either
Performance page continue to use its paired Predictive Context contracts.
Quality of Record is **not ESPN's Strength of Record methodology**: it does not
calculate the chance that a benchmark team would achieve the observed record.
