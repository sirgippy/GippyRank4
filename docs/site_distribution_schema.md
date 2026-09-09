# Site distribution artifact contract

Site Schema V1 remains compatible for Ranking Uncertainty Visualization: each
published manifest entry additionally exposes a relative `distribution_path`.
The ranking snapshot JSON remains a compact table payload. The linked file is
loaded only when a visitor opens a team's uncertainty detail.

The manifest registry contains `predictive` and `performance`. Predictive
entries carry `prior_family` (`context` or `history`); Performance entries do
not carry a prior family and instead expose `anchor_family: "context"`,
`model_version: "1.0"`, and `method: "prior_stripping"`.

Each `site/data/distributions/<snapshot-id>.json` artifact is deterministic,
compact JSON with this contract:

```json
{
  "schema_version": "1.0",
  "snapshot_id": "2026-preseason-context",
  "rank_count": 138,
  "teams": {
    "194": {
      "pmf": [0.01, 0.03],
      "summary": {
        "expected_rank": 7.4,
        "median_rank": 6,
        "modal_rank": 4,
        "interval_50": [3, 10],
        "interval_80": [1, 19],
        "interval_95": [1, 37],
        "interval_widths": {"50": 8, "80": 19, "95": 37},
        "rank_1_probability": 0.04,
        "top5_probability": 0.41,
        "top10_probability": 0.63,
        "top25_probability": 0.89
      }
    }
  }
}
```

`pmf[i]` is the probability of final/latent rank `i + 1`; rank 1 is always
the leftmost/best rank. Performance rows also preserve `rated` and
`eligible_games`. A zero-game Performance row has a uniform PMF, `rated: false`,
and `display_rank: "NR"`; it is ordered after rated teams and never enters Top
25. Intervals use the established discrete left-CDF
quantiles: 50% = 25th–75th percentiles, 80% = 10th–90th, and 95% =
2.5th–97.5th. Widths are inclusive counts of ranked positions.

The exporter reads `posterior_pmfs.csv` only. It requires one complete,
normalized PMF for every FBS team in the selected ranking summary and rejects
missing, duplicate, non-finite, out-of-range, non-contiguous, or inconsistent
data. It does not infer, alter, or renormalize posterior probabilities.

Future-game predictions are intentionally absent from the initial ranking
snapshot JSON.  They remain in the lazy team-season payload; the manifest
reports `future_prediction_count`, `future_prediction_bytes`, and aggregate
`payload_stats.lazy_future_prediction_bytes` so publication reviews can track
the incremental payload separately from the rankings-page payload. Issue-46
adds fixed-grid display data inside that same lazy payload; exact predictive
summaries remain the source of truth and are not reconstructed from bins.

The completed-game display adds 40 normalized rank bins per modeled FBS
team-game. The future-game display uses 40 exact CDF masses on the fixed
`-40 ... +40` home-minus-away margin axis plus explicit lower and upper tail
probabilities. The manifest's `future_prediction_bytes` includes the display
representation, and `payload_stats.initial_rankings_page_bytes` remains the
ranking snapshot total, so the rankings page does not load these data.
