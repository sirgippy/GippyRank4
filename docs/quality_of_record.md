# Quality of Record

Quality of Record is a standalone second variant of the Performance ranking
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

This generator does not alter Quality of Play, Predictive snapshots, site
publication, or weekly automation.
