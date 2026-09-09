# Predictive display approximation validation

The future-game chart uses a deterministic maximum-256-component location
approximation for large posterior rank-pair mixtures. The exact predictive
mixture remains authoritative for expected margin, win probabilities, median,
and all predictive intervals.

Run the reproducible audit with:

```text
UV_CACHE_DIR=/tmp/gippyrank-uv-cache uv run python scripts/validate_predictive_display.py
```

The audit selects up to eight games from each available home/away subdivision
and neutral-site group in the latest Predictive Context artifact. It compares
the full mixture and the 256-component approximation at all fixed display-bin
edges.

Current result for 23 representative games in
`2026-weekly-2026-09-08T11-43-00.275833Z-context`:

| Metric | Maximum | Mean across games |
| --- | ---: | ---: |
| Absolute CDF error | 4.8936e-6 | 3.7893e-6 |
| Absolute bin-mass error | 1.0766e-6 | 7.3672e-7 |

The integer display encoding is independently quantized after CDF evaluation;
it does not alter these approximation measurements or any exact prediction
field.
