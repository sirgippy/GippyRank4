# Performance V1 research report

This is a research-only descriptive quality analysis. It is not standings, strength of record, postseason selection, a poll emulator, or a claim about deservingness. No Performance family is registered with or published by the website.

## Definition

For target team `i`, let `C_i(r)` be its Context preseason PMF and let `Post_C_i(r | G)` be the Context-started BP marginal using every eligible game factor in the cutoff network. The primary Performance C PMF is:

`Performance_C_i(r | G) = normalize(Post_C_i(r | G) / C_i(r))`.

Performance H uses the identical construction with History priors for the network and the focal denominator. Equivalently, under exact factorized inference this is the game-derived likelihood profile under a uniform focal prior. The implementation leaves the focal team's preseason prior out of the final PMF; Context remains the primary opponent anchor. Opponent quality is allowed to update after the focal game and before the cutoff, so an idle team's Performance can move when its opponent plays.

## Prior-removal validation

The predeclared acceptance rule was p95 PMF TV ≤ 0.05 and worst PMF TV ≤ 0.15, measured against ordinary BP with only the target prior replaced by a uniform PMF. The selected method was **prior_stripping**.

Observed median/p95/worst TV: 0.000024 / 0.002320 / 0.002791. The validation was accepted under that predeclared rule.

## Current 2026 Performance C

Latest checked-in cutoff: `2026-09-06T17:34:56.895103+00:00`. Rated teams: 131 of 138; zero-game teams are marked unrated and kept out of meaningful display ordering.
Observed local build runtime: 840.066 seconds (wall-clock and hardware dependent).

| Display | Team | Expected quality-equivalent rank | Median | 50% | 80% | 95% | Top 5 | Top 10 | Top 25 | Games |
|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|
| 1 | LSU | 21.97 | 13 | 5–31 | 2–55 | 1–87 | 0.255 | 0.424 | 0.697 | 1 |
| 2 | Texas | 26.33 | 18 | 7–38 | 3–63 | 1–93 | 0.191 | 0.336 | 0.615 | 1 |
| 3 | Miami | 27.09 | 19 | 8–39 | 3–64 | 1–94 | 0.181 | 0.322 | 0.602 | 1 |
| 4 | South Carolina | 28.84 | 21 | 9–42 | 3–67 | 1–98 | 0.164 | 0.297 | 0.571 | 1 |
| 5 | Penn State | 28.87 | 21 | 8–42 | 3–68 | 1–97 | 0.167 | 0.301 | 0.573 | 1 |
| 6 | Texas A&M | 29.96 | 22 | 9–44 | 4–69 | 1–99 | 0.156 | 0.284 | 0.553 | 1 |
| 7 | Pittsburgh | 31.27 | 23 | 10–46 | 4–72 | 1–100 | 0.146 | 0.268 | 0.533 | 1 |
| 8 | Florida | 31.77 | 24 | 10–47 | 4–73 | 1–101 | 0.142 | 0.262 | 0.525 | 1 |
| 9 | Ohio State | 31.90 | 24 | 10–47 | 4–73 | 1–102 | 0.140 | 0.258 | 0.522 | 1 |
| 10 | Nevada | 32.87 | 25 | 10–49 | 4–75 | 1–103 | 0.138 | 0.255 | 0.511 | 1 |
| 11 | Kansas State | 33.01 | 24 | 10–50 | 4–75 | 1–104 | 0.144 | 0.269 | 0.522 | 1 |
| 12 | Alabama | 33.10 | 25 | 11–49 | 4–75 | 1–103 | 0.135 | 0.250 | 0.506 | 1 |
| 13 | Virginia | 33.32 | 25 | 10–49 | 4–76 | 1–104 | 0.135 | 0.252 | 0.507 | 1 |
| 14 | Oklahoma | 34.12 | 26 | 11–50 | 4–77 | 1–105 | 0.125 | 0.234 | 0.488 | 1 |
| 15 | Mississippi State | 34.29 | 27 | 11–50 | 4–77 | 1–105 | 0.124 | 0.232 | 0.485 | 1 |
| 16 | USC | 34.61 | 29 | 13–51 | 6–73 | 2–97 | 0.099 | 0.196 | 0.455 | 2 |
| 17 | Indiana | 35.59 | 28 | 12–53 | 5–79 | 1–106 | 0.120 | 0.224 | 0.470 | 1 |
| 18 | Army | 36.02 | 28 | 11–54 | 5–80 | 1–107 | 0.121 | 0.231 | 0.474 | 1 |
| 19 | Louisiana Tech | 36.61 | 28 | 11–56 | 4–82 | 1–109 | 0.125 | 0.237 | 0.472 | 1 |
| 20 | Iowa | 37.80 | 31 | 13–56 | 5–83 | 2–108 | 0.104 | 0.200 | 0.435 | 1 |
| 21 | UCLA | 38.34 | 31 | 13–57 | 5–84 | 2–109 | 0.103 | 0.198 | 0.432 | 1 |
| 22 | Virginia Tech | 38.91 | 32 | 12–59 | 5–85 | 2–111 | 0.112 | 0.214 | 0.440 | 1 |
| 23 | Arizona State | 39.04 | 32 | 13–59 | 5–86 | 2–112 | 0.110 | 0.210 | 0.437 | 1 |
| 24 | UCF | 39.81 | 33 | 13–61 | 5–87 | 2–113 | 0.108 | 0.206 | 0.428 | 1 |
| 25 | Syracuse | 40.77 | 34 | 14–62 | 5–88 | 2–114 | 0.102 | 0.196 | 0.415 | 1 |

Display order is an ordering of expected latent quality; it is not a joint probability permutation over ranks 1…N. The PMF is over quality-equivalent rank, not poll rank, standings position, or postseason probability.

## Anchor sensitivity and predictive comparison

Across the 2022–2025 cutoff panel, Context-minus-History expected-rank disagreement had median/p95/max absolute values 2.072 / 6.661 / 10.904; median/p95/max PMF TV was 0.0568 / 0.1782 / 0.2536. This difference is opponent-quality uncertainty, not focal-team preseason contamination.

Anchor sensitivity is also emitted by games-played bucket and season/week so the early-to-late hypothesis can be checked rather than assumed.

Performance C versus Predictive C expected-rank difference (Performance minus Predictive) had median -0.901 and absolute p95/max 30.401 / 63.505. Performance uncertainty remains broader when evidence is sparse; it is not artificially narrowed.

Largest current Context-versus-History anchor disagreements (C − H expected rank):
- Liberty: +8.38 ranks, PMF TV 0.0989, 1 games.
- Wake Forest: -6.52 ranks, PMF TV 0.0779, 1 games.
- Michigan: -6.17 ranks, PMF TV 0.0730, 1 games.
- Tulsa: +5.99 ranks, PMF TV 0.0711, 1 games.
- San José State: -5.72 ranks, PMF TV 0.0756, 2 games.
- Oregon State: -3.96 ranks, PMF TV 0.0472, 1 games.
- NC State: +3.84 ranks, PMF TV 0.0501, 1 games.
- Nevada: -3.74 ranks, PMF TV 0.0563, 1 games.
- West Virginia: +3.67 ranks, PMF TV 0.0434, 1 games.
- Sacramento State: -3.66 ranks, PMF TV 0.0488, 2 games.

Largest current Performance C versus Predictive C differences:
- Massachusetts: Performance 42.39, Predictive 129.79, difference -87.40, 1 games.
- Notre Dame: Performance 69.50, Predictive 15.78, difference +53.72, 0 games.
- Texas Tech: Performance 78.96, Predictive 29.42, difference +49.54, 1 games.
- Oregon: Performance 56.95, Predictive 9.43, difference +47.52, 1 games.
- Georgia State: Performance 57.12, Predictive 102.66, difference -45.53, 1 games.
- Ole Miss: Performance 69.50, Predictive 24.51, difference +44.99, 0 games.
- Clemson: Performance 103.36, Predictive 59.83, difference +43.54, 1 games.
- Sam Houston: Performance 73.82, Predictive 113.62, difference -39.81, 1 games.
- Michigan: Performance 68.00, Predictive 28.20, difference +39.80, 1 games.
- Nevada: Performance 32.87, Predictive 71.10, difference -38.22, 1 games.

Performance C versus Predictive C by games-played bucket (absolute expected-rank difference):
- early / 0: median 23.170, p95 53.507, n=501.
- early / 1-3: median 8.455, p95 27.352, n=530.
- early / 4-6: median 6.813, p95 19.866, n=37.
- late / 7+: median 2.343, p95 7.932, n=1068.
- mid / 4-6: median 5.000, p95 16.419, n=560.
- mid / 7+: median 3.259, p95 10.978, n=1042.

Current uncertainty extremes (Performance C):
- broadest: Air Force (128-rank 95% width, 1 games); Delaware (127-rank 95% width, 1 games); Georgia Southern (127-rank 95% width, 1 games)
- most concentrated: LSU (86-rank 95% width, 1 games); Rutgers (92-rank 95% width, 1 games); Texas (92-rank 95% width, 1 games)

## Future-game validation

The leakage-safe 2022–2025 panel rates each team only with games at or before its cutoff, then scores later completed FBS/FCS games. The table compares Predictive C/H and Performance C/H descriptively; it does not promote Performance over Predictive based on NLL.

- Performance_C / all_future: n=19045, marginalized NLL=4.3092, margin MAE=14.1065, win Brier=0.2186, calibration absolute error=0.0500.
- Performance_C / next_game: n=3526, marginalized NLL=4.2537, margin MAE=13.6133, win Brier=0.1863, calibration absolute error=0.1001.
- Performance_H / all_future: n=19045, marginalized NLL=4.3091, margin MAE=14.1125, win Brier=0.2185, calibration absolute error=0.0520.
- Performance_H / next_game: n=3526, marginalized NLL=4.2540, margin MAE=13.6294, win Brier=0.1864, calibration absolute error=0.0965.
- Predictive_C / all_future: n=19045, marginalized NLL=4.2456, margin MAE=13.2693, win Brier=0.2025, calibration absolute error=0.0486.
- Predictive_C / next_game: n=3526, marginalized NLL=4.1995, margin MAE=12.7170, win Brier=0.1758, calibration absolute error=0.0962.
- Predictive_H / all_future: n=19045, marginalized NLL=4.2513, margin MAE=13.3747, win Brier=0.2046, calibration absolute error=0.0498.
- Predictive_H / next_game: n=3526, marginalized NLL=4.2040, margin MAE=12.7810, win Brier=0.1760, calibration absolute error=0.1018.

## Illustrative corpus cases

- **close_road_loss_to_excellent_opponent**: Vanderbilt 31–34 vs Texas (away); opponent Context anchor expected rank 18.07; focal Performance expected rank at cutoff 12.24; record through cutoff 10–2.
- **ugly_win_over_weak_opponent**: Wake Forest 10–9 vs Kennesaw State (home); opponent Context anchor expected rank 126.21; focal Performance expected rank at cutoff 64.90; record through cutoff 8–4.
- **dominant_win_over_strong_opponent**: Utah 42–10 vs Arizona State (home); opponent Context anchor expected rank 34.50; focal Performance expected rank at cutoff 9.67; record through cutoff 10–2.
- **good_record_but_weaker_played_performance**: Kennesaw State, record 10–3, Performance expected rank 94.43.
- **loss_but_strong_played_performance**: Texas Tech, record 12–1, Performance expected rank 3.38.
- **largest_uncomfortable_record_performance_gap**: Massachusetts, record 0–12, Performance expected rank 134.98.

The selector is deterministic and rule-based; it includes an uncomfortable record/performance gap rather than only favorable examples. These comparisons describe played football and opponent interpretation, not reward or punishment for winning.

## Structural semantics and limitations

The frozen Historical Likelihood V1 surface is used unchanged: Student-t df 15, rank-percentile surface, site semantics, FBS/FCS orientation, and margin. There is no win indicator, record feature, capped margin, YPP, recency weight, poll input, or selection logic. Structural checks passed: direct focal-prior TV 5.64e-17; loopy-cycle prior-sensitivity residual TV 0.0483 within the predeclared tolerance; explicit-neutralized cycle TV 0; maximum adjacent −1/0/+1 expected-rank jump 0.245; zero-game output is uniform.

Idle-team updates were observed in 12 no-new-game cutoff transitions; the largest expected-rank movement was 1.957. This is expected network updating, not a bug.

FCS opponents use the established full-season support/fallback policy. Historical 2018–2021 priors are not available in the frozen H/C prediction artifacts, so they were not fabricated; the temporal evaluation uses 2022–2025. Current 2026 uses the latest checked-in cached cutoff and does not fetch new data.

Artifacts are research-only and reproducibly generated by `uv run python scripts/build_performance_v1.py`. The website selectors and Predictive H/C/Likelihood V1 production behavior are unchanged.
