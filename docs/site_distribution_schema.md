# Site distribution artifact contract

Site Schema V1 remains compatible for Ranking Uncertainty Visualization: each
published manifest entry additionally exposes a relative `distribution_path`.
The ranking snapshot JSON remains a compact table payload. The linked file is
loaded only when a visitor opens a team's uncertainty detail.

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
the leftmost/best rank. Intervals use the established discrete left-CDF
quantiles: 50% = 25th–75th percentiles, 80% = 10th–90th, and 95% =
2.5th–97.5th. Widths are inclusive counts of ranked positions.

The exporter reads `posterior_pmfs.csv` only. It requires one complete,
normalized PMF for every FBS team in the selected ranking summary and rejects
missing, duplicate, non-finite, out-of-range, non-contiguous, or inconsistent
data. It does not infer, alter, or renormalize posterior probabilities.
