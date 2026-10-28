# Frozen team-season case studies

All 27 team seasons and 70 snapshot-case observations are retained. Each table uses the original verified baseline rankings. Categories and questions were frozen before intermediate candidates were solved; the answers below are interpretations after observation. Rank intervals describe latent marginal coordinates, while ordinal ranks sort means. The 2026 expected wins reuse cutoff Context evidence; the 2025 primary expectations use preseason priors.

## 2025 Alabama (333)

Frozen categories: elite.

### 2025-prebowl

Actual record **10–3**, mean margin **+13.85**; close wins/losses 4/1, dominant wins 2. Frozen standardized schedule probability **0.420** (larger is easier), expected wins **9.24**, wins minus expectation **+0.76**. Reference: preseason_context_v1.2, ordinal 5, band elite, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 12.96 | 9 | 10.11 | 5.016 | 6–17 | 3–26 | 1–40 | 0.496/0.587/0.897 | +0.00 / -0.94 |
| 0.25 | 12.91 | 10 | 8.83 | 4.926 | 7–17 | 4–24 | 1–36 | 0.464/0.569/0.916 | -0.05 / -0.99 |
| 0.5 | 13.18 | 12 | 7.99 | 4.851 | 8–17 | 4–23 | 2–33 | 0.422/0.539/0.928 | +0.22 / -0.73 |
| 0.75 | 13.56 | 11 | 7.37 | 4.779 | 8–17 | 5–23 | 3–32 | 0.376/0.503/0.936 | +0.59 / -0.35 |
| 1 | 13.91 | 11 | 6.87 | 4.707 | 9–17 | 6–23 | 3–31 | 0.335/0.469/0.942 | +0.94 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752665 / 2025-08-30 | away | Florida State (fbs) | 17–31 | -14 | 0.806 | 43.55/14.37 | +10.16 |
| 401752681 / 2025-09-06 | home | UL Monroe (fbs) | 73–0 | +73 | 0.946 | 129.22/4.19 | +37.89 |
| 401752696 / 2025-09-13 | home | Wisconsin (fbs) | 38–14 | +24 | 0.812 | 65.19/17.45 | +21.12 |
| 401752718 / 2025-09-27 | away | Georgia (fbs) | 24–21 | +3 | 0.387 | 7.45/4.57 | -6.32 |
| 401752726 / 2025-10-04 | home | Vanderbilt (fbs) | 30–14 | +16 | 0.791 | 13.69/6.51 | +4.85 |
| 401752734 / 2025-10-11 | away | Missouri (fbs) | 27–24 | +3 | 0.665 | 22.58/9.53 | +3.63 |
| 401752738 / 2025-10-18 | home | Tennessee (fbs) | 37–20 | +17 | 0.680 | 23.35/10.31 | +10.14 |
| 401752751 / 2025-10-25 | away | South Carolina (fbs) | 29–22 | +7 | 0.533 | 40.47/13.58 | +9.57 |
| 401752759 / 2025-11-09 | home | LSU (fbs) | 20–9 | +11 | 0.680 | 30.41/12.01 | +12.90 |
| 401752765 / 2025-11-15 | home | Oklahoma (fbs) | 21–23 | -2 | 0.721 | 16.22/7.13 | +6.55 |
| 401752773 / 2025-11-22 | home | Eastern Illinois (fcs) | 56–0 | +56 | 0.990 | 98.27/11.04 | +51.60 |
| 401752784 / 2025-11-30 | away | Auburn (fbs) | 27–20 | +7 | 0.769 | 35.24/13.16 | +8.07 |
| 401777351 / 2025-12-06 | neutral | Georgia (fbs) | 7–28 | -21 | 0.456 | 7.45/4.57 | -3.18 |

Answers to the original questions at every candidate:

**How do winning and margins affect elite-season credit?**

- At lambda=0, E[rank]=12.96, ordinal=9, SD=10.11, entropy=5.016 bits, central 80%=3-26; versus Record/Play the mean-rank changes are +0.00/-0.94. Actual evidence is 10-3, mean margin +13.85, 4 close wins and 1 close losses, standardized schedule probability 0.420, expected wins 9.24, wins minus expectation +0.76. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.25, E[rank]=12.91, ordinal=10, SD=8.83, entropy=4.926 bits, central 80%=4-24; versus Record/Play the mean-rank changes are -0.05/-0.99. Actual evidence is 10-3, mean margin +13.85, 4 close wins and 1 close losses, standardized schedule probability 0.420, expected wins 9.24, wins minus expectation +0.76. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.5, E[rank]=13.18, ordinal=12, SD=7.99, entropy=4.851 bits, central 80%=4-23; versus Record/Play the mean-rank changes are +0.22/-0.73. Actual evidence is 10-3, mean margin +13.85, 4 close wins and 1 close losses, standardized schedule probability 0.420, expected wins 9.24, wins minus expectation +0.76. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.75, E[rank]=13.56, ordinal=11, SD=7.37, entropy=4.779 bits, central 80%=5-23; versus Record/Play the mean-rank changes are +0.59/-0.35. Actual evidence is 10-3, mean margin +13.85, 4 close wins and 1 close losses, standardized schedule probability 0.420, expected wins 9.24, wins minus expectation +0.76. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=1, E[rank]=13.91, ordinal=11, SD=6.87, entropy=4.707 bits, central 80%=6-23; versus Record/Play the mean-rank changes are +0.94/+0.00. Actual evidence is 10-3, mean margin +13.85, 4 close wins and 1 close losses, standardized schedule probability 0.420, expected wins 9.24, wins minus expectation +0.76. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 12.96 to 13.54 (+0.58); ordinal changes +0, SD changes -0.49, entropy +0.011, 80% width -1, TV=0.054. Record changes 10-3 to 11-4, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 12.91 to 14.60 (+1.69); ordinal changes +0, SD changes +0.03, entropy +0.080, 80% width +1, TV=0.100. Record changes 10-3 to 11-4, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 13.18 to 15.56 (+2.38); ordinal changes -1, SD changes +0.25, entropy +0.099, 80% width +1, TV=0.138. Record changes 10-3 to 11-4, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 13.56 to 16.36 (+2.81); ordinal changes +1, SD changes +0.34, entropy +0.101, 80% width +0, TV=0.167. Record changes 10-3 to 11-4, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 13.91 to 16.98 (+3.08); ordinal changes +2, SD changes +0.39, entropy +0.101, 80% width +0, TV=0.191. Record changes 10-3 to 11-4, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **11–4**, mean margin **+10.33**; close wins/losses 4/1, dominant wins 2. Frozen standardized schedule probability **0.405** (larger is easier), expected wins **10.54**, wins minus expectation **+0.46**. Reference: preseason_context_v1.2, ordinal 5, band elite, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 13.54 | 9 | 9.62 | 5.027 | 7–18 | 4–26 | 1–39 | 0.449/0.547/0.895 | +0.00 / -3.44 |
| 0.25 | 14.60 | 10 | 8.86 | 5.006 | 8–19 | 5–26 | 2–37 | 0.364/0.473/0.893 | +1.06 / -2.38 |
| 0.5 | 15.56 | 11 | 8.23 | 4.950 | 10–20 | 6–26 | 3–36 | 0.287/0.401/0.890 | +2.01 / -1.42 |
| 0.75 | 16.36 | 12 | 7.71 | 4.880 | 11–20 | 8–26 | 5–35 | 0.222/0.336/0.887 | +2.82 / -0.62 |
| 1 | 16.98 | 13 | 7.26 | 4.808 | 12–21 | 9–26 | 6–35 | 0.171/0.282/0.885 | +3.44 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752665 / 2025-08-30 | away | Florida State (fbs) | 17–31 | -14 | 0.806 | 41.86/14.19 | +8.03 |
| 401752681 / 2025-09-06 | home | UL Monroe (fbs) | 73–0 | +73 | 0.946 | 129.21/4.20 | +37.09 |
| 401752696 / 2025-09-13 | home | Wisconsin (fbs) | 38–14 | +24 | 0.812 | 65.67/17.44 | +19.84 |
| 401752718 / 2025-09-27 | away | Georgia (fbs) | 24–21 | +3 | 0.387 | 8.78/5.01 | -7.18 |
| 401752726 / 2025-10-04 | home | Vanderbilt (fbs) | 30–14 | +16 | 0.791 | 15.69/6.87 | +4.22 |
| 401752734 / 2025-10-11 | away | Missouri (fbs) | 27–24 | +3 | 0.665 | 26.97/10.31 | +3.60 |
| 401752738 / 2025-10-18 | home | Tennessee (fbs) | 37–20 | +17 | 0.680 | 26.34/10.60 | +9.54 |
| 401752751 / 2025-10-25 | away | South Carolina (fbs) | 29–22 | +7 | 0.533 | 42.20/13.77 | +8.24 |
| 401752759 / 2025-11-09 | home | LSU (fbs) | 20–9 | +11 | 0.680 | 34.12/12.14 | +12.26 |
| 401752765 / 2025-11-15 | home | Oklahoma (fbs) | 21–23 | -2 | 0.721 | 18.48/7.46 | +5.95 |
| 401752773 / 2025-11-22 | home | Eastern Illinois (fcs) | 56–0 | +56 | 0.990 | 97.74/11.29 | +50.55 |
| 401752784 / 2025-11-30 | away | Auburn (fbs) | 27–20 | +7 | 0.769 | 37.47/13.16 | +6.92 |
| 401777351 / 2025-12-06 | neutral | Georgia (fbs) | 7–28 | -21 | 0.456 | 8.78/5.01 | -4.05 |
| 401779840 / 2025-12-20 | away | Oklahoma (fbs) | 34–24 | +10 | 0.597 | 18.48/7.46 | -0.32 |
| 401769072 / 2026-01-01 | neutral | Indiana (fbs) | 3–38 | -35 | 0.707 | 3.46/1.78 | -8.83 |

Answers to the original questions at every candidate:

**How do winning and margins affect elite-season credit?**

- At lambda=0, E[rank]=13.54, ordinal=9, SD=9.62, entropy=5.027 bits, central 80%=4-26; versus Record/Play the mean-rank changes are +0.00/-3.44. Actual evidence is 11-4, mean margin +10.33, 4 close wins and 1 close losses, standardized schedule probability 0.405, expected wins 10.54, wins minus expectation +0.46. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.25, E[rank]=14.60, ordinal=10, SD=8.86, entropy=5.006 bits, central 80%=5-26; versus Record/Play the mean-rank changes are +1.06/-2.38. Actual evidence is 11-4, mean margin +10.33, 4 close wins and 1 close losses, standardized schedule probability 0.405, expected wins 10.54, wins minus expectation +0.46. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.5, E[rank]=15.56, ordinal=11, SD=8.23, entropy=4.950 bits, central 80%=6-26; versus Record/Play the mean-rank changes are +2.01/-1.42. Actual evidence is 11-4, mean margin +10.33, 4 close wins and 1 close losses, standardized schedule probability 0.405, expected wins 10.54, wins minus expectation +0.46. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.75, E[rank]=16.36, ordinal=12, SD=7.71, entropy=4.880 bits, central 80%=8-26; versus Record/Play the mean-rank changes are +2.82/-0.62. Actual evidence is 11-4, mean margin +10.33, 4 close wins and 1 close losses, standardized schedule probability 0.405, expected wins 10.54, wins minus expectation +0.46. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=1, E[rank]=16.98, ordinal=13, SD=7.26, entropy=4.808 bits, central 80%=9-26; versus Record/Play the mean-rank changes are +3.44/+0.00. Actual evidence is 11-4, mean margin +10.33, 4 close wins and 1 close losses, standardized schedule probability 0.405, expected wins 10.54, wins minus expectation +0.46. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 12.96 to 13.54 (+0.58); ordinal changes +0, SD changes -0.49, entropy +0.011, 80% width -1, TV=0.054. Record changes 10-3 to 11-4, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 12.91 to 14.60 (+1.69); ordinal changes +0, SD changes +0.03, entropy +0.080, 80% width +1, TV=0.100. Record changes 10-3 to 11-4, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 13.18 to 15.56 (+2.38); ordinal changes -1, SD changes +0.25, entropy +0.099, 80% width +1, TV=0.138. Record changes 10-3 to 11-4, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 13.56 to 16.36 (+2.81); ordinal changes +1, SD changes +0.34, entropy +0.101, 80% width +0, TV=0.167. Record changes 10-3 to 11-4, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 13.91 to 16.98 (+3.08); ordinal changes +2, SD changes +0.39, entropy +0.101, 80% width +0, TV=0.191. Record changes 10-3 to 11-4, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 2 / 55 | +0.58 | +0 | -0.49 | 0.054 |
| 2025-prebowl → 2025-final | 0.25 | 2 / 55 | +1.69 | +0 | +0.03 | 0.100 |
| 2025-prebowl → 2025-final | 0.5 | 2 / 55 | +2.38 | -1 | +0.25 | 0.138 |
| 2025-prebowl → 2025-final | 0.75 | 2 / 55 | +2.81 | +1 | +0.34 | 0.167 |
| 2025-prebowl → 2025-final | 1 | 2 / 55 | +3.08 | +2 | +0.39 | 0.191 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2025 Boise State (68)

Frozen categories: strong.

### 2025-prebowl

Actual record **9–4**, mean margin **+8.38**; close wins/losses 1/0, dominant wins 3. Frozen standardized schedule probability **0.695** (larger is easier), expected wins **10.45**, wins minus expectation **-1.45**. Reference: preseason_context_v1.2, ordinal 20, band strong, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 62.29 | 52 | 21.33 | 6.448 | 47–78 | 34–91 | 22–103 | 0.003/0.004/0.040 | +0.00 / +1.16 |
| 0.25 | 62.30 | 53 | 19.55 | 6.327 | 48–76 | 37–88 | 25–100 | 0.001/0.002/0.026 | +0.01 / +1.17 |
| 0.5 | 62.00 | 56 | 18.06 | 6.215 | 49–75 | 38–86 | 28–97 | 0.000/0.001/0.017 | -0.29 / +0.87 |
| 0.75 | 61.58 | 57 | 16.81 | 6.113 | 50–73 | 40–84 | 30–95 | 0.000/0.000/0.011 | -0.71 / +0.45 |
| 1 | 61.13 | 57 | 15.74 | 6.019 | 50–72 | 41–82 | 31–92 | 0.000/0.000/0.008 | -1.16 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401762522 / 2025-08-28 | away | South Florida (fbs) | 7–34 | -27 | 0.785 | 30.65/12.02 | -10.65 |
| 401760363 / 2025-09-06 | home | Eastern Washington (fcs) | 51–14 | +37 | 0.973 | 68.66/15.07 | +35.20 |
| 401760372 / 2025-09-20 | away | Air Force (fbs) | 49–37 | +12 | 0.709 | 105.16/10.97 | +10.79 |
| 401760379 / 2025-09-27 | home | App State (fbs) | 47–14 | +33 | 0.867 | 119.56/7.42 | +22.73 |
| 401757165 / 2025-10-04 | away | Notre Dame (fbs) | 7–28 | -21 | 0.232 | 4.70/3.06 | -23.26 |
| 401760387 / 2025-10-12 | home | New Mexico (fbs) | 41–25 | +16 | 0.961 | 92.67/13.64 | +13.02 |
| 401760393 / 2025-10-18 | home | UNLV (fbs) | 56–31 | +25 | 0.826 | 84.27/14.36 | +10.58 |
| 401760397 / 2025-10-25 | away | Nevada (fbs) | 24–3 | +21 | 0.843 | 121.96/7.15 | +17.55 |
| 401760402 / 2025-11-01 | home | Fresno State (fbs) | 7–30 | -23 | 0.864 | 95.76/14.19 | +14.02 |
| 401760410 / 2025-11-16 | away | San Diego State (fbs) | 7–17 | -10 | 0.857 | 74.67/16.11 | +1.78 |
| 401760416 / 2025-11-23 | home | Colorado State (fbs) | 49–21 | +28 | 0.874 | 124.36/6.71 | +24.95 |
| 401760422 / 2025-11-28 | away | Utah State (fbs) | 25–24 | +1 | 0.832 | 88.72/14.50 | +5.59 |
| 401777326 / 2025-12-06 | home | UNLV (fbs) | 38–21 | +17 | 0.826 | 84.27/14.36 | +10.58 |

Answers to the original questions at every candidate:

**How do winning and margins affect this strong prior-reference team?**

- At lambda=0, E[rank]=62.29, ordinal=52, SD=21.33, entropy=6.448 bits, central 80%=34-91; versus Record/Play the mean-rank changes are +0.00/+1.16. Actual evidence is 9-4, mean margin +8.38, 1 close wins and 0 close losses, standardized schedule probability 0.695, expected wins 10.45, wins minus expectation -1.45. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.25, E[rank]=62.30, ordinal=53, SD=19.55, entropy=6.327 bits, central 80%=37-88; versus Record/Play the mean-rank changes are +0.01/+1.17. Actual evidence is 9-4, mean margin +8.38, 1 close wins and 0 close losses, standardized schedule probability 0.695, expected wins 10.45, wins minus expectation -1.45. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.5, E[rank]=62.00, ordinal=56, SD=18.06, entropy=6.215 bits, central 80%=38-86; versus Record/Play the mean-rank changes are -0.29/+0.87. Actual evidence is 9-4, mean margin +8.38, 1 close wins and 0 close losses, standardized schedule probability 0.695, expected wins 10.45, wins minus expectation -1.45. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.75, E[rank]=61.58, ordinal=57, SD=16.81, entropy=6.113 bits, central 80%=40-84; versus Record/Play the mean-rank changes are -0.71/+0.45. Actual evidence is 9-4, mean margin +8.38, 1 close wins and 0 close losses, standardized schedule probability 0.695, expected wins 10.45, wins minus expectation -1.45. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=1, E[rank]=61.13, ordinal=57, SD=15.74, entropy=6.019 bits, central 80%=41-82; versus Record/Play the mean-rank changes are -1.16/+0.00. Actual evidence is 9-4, mean margin +8.38, 1 close wins and 0 close losses, standardized schedule probability 0.695, expected wins 10.45, wins minus expectation -1.45. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 62.29 to 68.81 (+6.52); ordinal changes +3, SD changes -0.86, entropy -0.059, 80% width -2, TV=0.122. Record changes 9-4 to 9-5, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 62.30 to 68.38 (+6.08); ordinal changes +4, SD changes -0.74, entropy -0.055, 80% width -1, TV=0.125. Record changes 9-4 to 9-5, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 62.00 to 67.77 (+5.77); ordinal changes +2, SD changes -0.64, entropy -0.052, 80% width -2, TV=0.129. Record changes 9-4 to 9-5, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 61.58 to 67.12 (+5.53); ordinal changes +4, SD changes -0.57, entropy -0.049, 80% width -2, TV=0.133. Record changes 9-4 to 9-5, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 61.13 to 66.48 (+5.35); ordinal changes +4, SD changes -0.52, entropy -0.048, 80% width -2, TV=0.138. Record changes 9-4 to 9-5, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **9–5**, mean margin **+5.79**; close wins/losses 1/0, dominant wins 3. Frozen standardized schedule probability **0.674** (larger is easier), expected wins **11.02**, wins minus expectation **-2.02**. Reference: preseason_context_v1.2, ordinal 20, band strong, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 68.81 | 55 | 20.47 | 6.389 | 54–84 | 41–96 | 29–107 | 0.000/0.001/0.015 | +0.00 / +2.33 |
| 0.25 | 68.38 | 57 | 18.82 | 6.272 | 55–82 | 43–93 | 32–104 | 0.000/0.000/0.009 | -0.43 / +1.90 |
| 0.5 | 67.77 | 58 | 17.42 | 6.163 | 56–80 | 45–91 | 34–101 | 0.000/0.000/0.005 | -1.05 / +1.28 |
| 0.75 | 67.12 | 61 | 16.23 | 6.063 | 56–78 | 46–88 | 35–98 | 0.000/0.000/0.003 | -1.70 / +0.63 |
| 1 | 66.48 | 61 | 15.21 | 5.972 | 56–77 | 47–86 | 37–96 | 0.000/0.000/0.002 | -2.33 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401762522 / 2025-08-28 | away | South Florida (fbs) | 7–34 | -27 | 0.785 | 35.55/12.52 | -10.37 |
| 401760363 / 2025-09-06 | home | Eastern Washington (fcs) | 51–14 | +37 | 0.973 | 68.77/15.09 | +34.20 |
| 401760372 / 2025-09-20 | away | Air Force (fbs) | 49–37 | +12 | 0.709 | 105.41/10.90 | +9.50 |
| 401760379 / 2025-09-27 | home | App State (fbs) | 47–14 | +33 | 0.867 | 121.56/6.69 | +22.33 |
| 401757165 / 2025-10-04 | away | Notre Dame (fbs) | 7–28 | -21 | 0.232 | 4.65/3.02 | -24.15 |
| 401760387 / 2025-10-12 | home | New Mexico (fbs) | 41–25 | +16 | 0.961 | 92.89/13.02 | +11.65 |
| 401760393 / 2025-10-18 | home | UNLV (fbs) | 56–31 | +25 | 0.826 | 87.60/13.44 | +10.07 |
| 401760397 / 2025-10-25 | away | Nevada (fbs) | 24–3 | +21 | 0.843 | 122.11/7.12 | +16.34 |
| 401760402 / 2025-11-01 | home | Fresno State (fbs) | 7–30 | -23 | 0.864 | 91.27/14.24 | +11.21 |
| 401760410 / 2025-11-16 | away | San Diego State (fbs) | 7–17 | -10 | 0.857 | 72.42/15.50 | -0.26 |
| 401760416 / 2025-11-23 | home | Colorado State (fbs) | 49–21 | +28 | 0.874 | 124.29/6.82 | +23.67 |
| 401760422 / 2025-11-28 | away | Utah State (fbs) | 25–24 | +1 | 0.832 | 91.67/13.54 | +5.02 |
| 401777326 / 2025-12-06 | home | UNLV (fbs) | 38–21 | +17 | 0.826 | 87.60/13.44 | +10.07 |
| 401778302 / 2025-12-14 | neutral | Washington (fbs) | 10–38 | -28 | 0.569 | 18.83/8.48 | -13.52 |

Answers to the original questions at every candidate:

**How do winning and margins affect this strong prior-reference team?**

- At lambda=0, E[rank]=68.81, ordinal=55, SD=20.47, entropy=6.389 bits, central 80%=41-96; versus Record/Play the mean-rank changes are +0.00/+2.33. Actual evidence is 9-5, mean margin +5.79, 1 close wins and 0 close losses, standardized schedule probability 0.674, expected wins 11.02, wins minus expectation -2.02. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.25, E[rank]=68.38, ordinal=57, SD=18.82, entropy=6.272 bits, central 80%=43-93; versus Record/Play the mean-rank changes are -0.43/+1.90. Actual evidence is 9-5, mean margin +5.79, 1 close wins and 0 close losses, standardized schedule probability 0.674, expected wins 11.02, wins minus expectation -2.02. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.5, E[rank]=67.77, ordinal=58, SD=17.42, entropy=6.163 bits, central 80%=45-91; versus Record/Play the mean-rank changes are -1.05/+1.28. Actual evidence is 9-5, mean margin +5.79, 1 close wins and 0 close losses, standardized schedule probability 0.674, expected wins 11.02, wins minus expectation -2.02. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.75, E[rank]=67.12, ordinal=61, SD=16.23, entropy=6.063 bits, central 80%=46-88; versus Record/Play the mean-rank changes are -1.70/+0.63. Actual evidence is 9-5, mean margin +5.79, 1 close wins and 0 close losses, standardized schedule probability 0.674, expected wins 11.02, wins minus expectation -2.02. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=1, E[rank]=66.48, ordinal=61, SD=15.21, entropy=5.972 bits, central 80%=47-86; versus Record/Play the mean-rank changes are -2.33/+0.00. Actual evidence is 9-5, mean margin +5.79, 1 close wins and 0 close losses, standardized schedule probability 0.674, expected wins 11.02, wins minus expectation -2.02. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 62.29 to 68.81 (+6.52); ordinal changes +3, SD changes -0.86, entropy -0.059, 80% width -2, TV=0.122. Record changes 9-4 to 9-5, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 62.30 to 68.38 (+6.08); ordinal changes +4, SD changes -0.74, entropy -0.055, 80% width -1, TV=0.125. Record changes 9-4 to 9-5, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 62.00 to 67.77 (+5.77); ordinal changes +2, SD changes -0.64, entropy -0.052, 80% width -2, TV=0.129. Record changes 9-4 to 9-5, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 61.58 to 67.12 (+5.53); ordinal changes +4, SD changes -0.57, entropy -0.049, 80% width -2, TV=0.133. Record changes 9-4 to 9-5, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 61.13 to 66.48 (+5.35); ordinal changes +4, SD changes -0.52, entropy -0.048, 80% width -2, TV=0.138. Record changes 9-4 to 9-5, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 1 / 55 | +6.52 | +3 | -0.86 | 0.122 |
| 2025-prebowl → 2025-final | 0.25 | 1 / 55 | +6.08 | +4 | -0.74 | 0.125 |
| 2025-prebowl → 2025-final | 0.5 | 1 / 55 | +5.77 | +2 | -0.64 | 0.129 |
| 2025-prebowl → 2025-final | 0.75 | 1 / 55 | +5.53 | +4 | -0.57 | 0.133 |
| 2025-prebowl → 2025-final | 1 | 1 / 55 | +5.35 | +4 | -0.52 | 0.138 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2025 Buffalo (2084)

Frozen categories: middle, middle_schedule_pair.

### 2025-prebowl

Actual record **5–7**, mean margin **+0.50**; close wins/losses 3/3, dominant wins 1. Frozen standardized schedule probability **0.747** (larger is easier), expected wins **8.13**, wins minus expectation **-3.13**. Reference: preseason_context_v1.2, ordinal 65, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 119.27 | 116 | 11.11 | 5.340 | 113–127 | 105–132 | 92–135 | 0.000/0.000/0.000 | +0.00 / +1.12 |
| 0.25 | 118.96 | 117 | 10.37 | 5.277 | 113–126 | 105–131 | 94–134 | 0.000/0.000/0.000 | -0.32 / +0.80 |
| 0.5 | 118.64 | 117 | 9.76 | 5.212 | 113–126 | 106–130 | 95–133 | 0.000/0.000/0.000 | -0.63 / +0.49 |
| 0.75 | 118.37 | 115 | 9.24 | 5.146 | 113–125 | 106–129 | 96–133 | 0.000/0.000/0.000 | -0.91 / +0.21 |
| 1 | 118.16 | 114 | 8.78 | 5.081 | 113–124 | 107–128 | 97–132 | 0.000/0.000/0.000 | -1.12 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752793 / 2025-08-29 | away | Minnesota (fbs) | 10–23 | -13 | 0.276 | 72.30/16.24 | -17.69 |
| 401762794 / 2025-09-06 | home | Saint Francis (fcs) | 45–6 | +39 | 0.912 | 116.36/5.96 | +35.93 |
| 401762817 / 2025-09-13 | away | Kent State (fbs) | 31–28 | +3 | 0.854 | 131.75/4.14 | +6.57 |
| 401761613 / 2025-09-20 | home | Troy (fbs) | 17–21 | -4 | 0.676 | 104.78/11.02 | -2.24 |
| 401757170 / 2025-09-27 | home | UConn (fbs) | 17–20 | -3 | 0.645 | 85.06/14.68 | -8.15 |
| 401762831 / 2025-10-04 | home | Eastern Michigan (fbs) | 31–30 | +1 | 0.709 | 121.22/6.88 | +5.32 |
| 401762840 / 2025-10-18 | away | Massachusetts (fbs) | 28–21 | +7 | 0.795 | 135.03/1.19 | +9.35 |
| 401762844 / 2025-10-25 | home | Akron (fbs) | 16–24 | -8 | 0.776 | 124.58/6.24 | +7.48 |
| 401762849 / 2025-11-01 | away | Bowling Green (fbs) | 28–3 | +25 | 0.596 | 119.14/8.61 | -1.98 |
| 401762858 / 2025-11-13 | away | Central Michigan (fbs) | 19–38 | -19 | 0.697 | 108.64/10.62 | -7.01 |
| 401762865 / 2025-11-20 | home | Miami (OH) (fbs) | 20–37 | -17 | 0.615 | 96.20/12.42 | -5.06 |
| 401762870 / 2025-11-28 | home | Ohio (fbs) | 26–31 | -5 | 0.581 | 89.57/14.05 | -6.93 |

Answers to the original questions at every candidate:

**Does this middle prior-reference case show a schedule advantage amid uncertainty?**

- At lambda=0, E[rank]=119.27, ordinal=116, SD=11.11, entropy=5.340 bits, central 80%=105-132; versus Record/Play the mean-rank changes are +0.00/+1.12. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=63.60, SD=21.90, entropy=6.477, widths 50/80/95=33/59/83; this team's expected-rank gap is +55.68 and SD gap -10.79. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=118.96, ordinal=117, SD=10.37, entropy=5.277 bits, central 80%=105-131; versus Record/Play the mean-rank changes are -0.32/+0.80. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=60.02, SD=19.90, entropy=6.343, widths 50/80/95=30/54/77; this team's expected-rank gap is +58.94 and SD gap -9.52. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=118.64, ordinal=117, SD=9.76, entropy=5.212 bits, central 80%=106-130; versus Record/Play the mean-rank changes are -0.63/+0.49. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=58.09, SD=18.42, entropy=6.235, widths 50/80/95=27/50/71; this team's expected-rank gap is +60.55 and SD gap -8.66. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=118.37, ordinal=115, SD=9.24, entropy=5.146 bits, central 80%=106-129; versus Record/Play the mean-rank changes are -0.91/+0.21. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=57.05, SD=17.29, entropy=6.145, widths 50/80/95=25/46/67; this team's expected-rank gap is +61.32 and SD gap -8.05. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=118.16, ordinal=114, SD=8.78, entropy=5.081 bits, central 80%=107-128; versus Record/Play the mean-rank changes are -1.12/+0.00. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=56.47, SD=16.38, entropy=6.070, widths 50/80/95=24/44/64; this team's expected-rank gap is +61.68 and SD gap -7.60. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**How do identical middle-team records with different schedules compare across candidates?**

- At lambda=0, E[rank]=119.27, ordinal=116, SD=11.11, entropy=5.340 bits, central 80%=105-132; versus Record/Play the mean-rank changes are +0.00/+1.12. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=63.60, SD=21.90, entropy=6.477, widths 50/80/95=33/59/83; this team's expected-rank gap is +55.68 and SD gap -10.79. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=118.96, ordinal=117, SD=10.37, entropy=5.277 bits, central 80%=105-131; versus Record/Play the mean-rank changes are -0.32/+0.80. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=60.02, SD=19.90, entropy=6.343, widths 50/80/95=30/54/77; this team's expected-rank gap is +58.94 and SD gap -9.52. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=118.64, ordinal=117, SD=9.76, entropy=5.212 bits, central 80%=106-130; versus Record/Play the mean-rank changes are -0.63/+0.49. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=58.09, SD=18.42, entropy=6.235, widths 50/80/95=27/50/71; this team's expected-rank gap is +60.55 and SD gap -8.66. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=118.37, ordinal=115, SD=9.24, entropy=5.146 bits, central 80%=106-129; versus Record/Play the mean-rank changes are -0.91/+0.21. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=57.05, SD=17.29, entropy=6.145, widths 50/80/95=25/46/67; this team's expected-rank gap is +61.32 and SD gap -8.05. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=118.16, ordinal=114, SD=8.78, entropy=5.081 bits, central 80%=107-128; versus Record/Play the mean-rank changes are -1.12/+0.00. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=56.47, SD=16.38, entropy=6.070, widths 50/80/95=24/44/64; this team's expected-rank gap is +61.68 and SD gap -7.60. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 119.27 to 119.73 (+0.46); ordinal changes +1, SD changes -0.20, entropy -0.032, 80% width +0, TV=0.018. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 118.96 to 119.50 (+0.54); ordinal changes +0, SD changes -0.22, entropy -0.035, 80% width -1, TV=0.022. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 118.64 to 119.27 (+0.63); ordinal changes -2, SD changes -0.24, entropy -0.039, 80% width -1, TV=0.027. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 118.37 to 119.08 (+0.71); ordinal changes -1, SD changes -0.26, entropy -0.043, 80% width -1, TV=0.032. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 118.16 to 118.95 (+0.79); ordinal changes +0, SD changes -0.29, entropy -0.049, 80% width +0, TV=0.038. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **5–7**, mean margin **+0.50**; close wins/losses 3/3, dominant wins 1. Frozen standardized schedule probability **0.747** (larger is easier), expected wins **8.13**, wins minus expectation **-3.13**. Reference: preseason_context_v1.2, ordinal 65, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 119.73 | 117 | 10.91 | 5.308 | 114–128 | 105–132 | 93–135 | 0.000/0.000/0.000 | +0.00 / +0.79 |
| 0.25 | 119.50 | 117 | 10.15 | 5.242 | 114–127 | 106–131 | 95–134 | 0.000/0.000/0.000 | -0.24 / +0.55 |
| 0.5 | 119.27 | 115 | 9.52 | 5.173 | 114–126 | 107–130 | 96–134 | 0.000/0.000/0.000 | -0.47 / +0.32 |
| 0.75 | 119.08 | 114 | 8.97 | 5.103 | 114–125 | 107–129 | 98–133 | 0.000/0.000/0.000 | -0.66 / +0.13 |
| 1 | 118.95 | 114 | 8.49 | 5.032 | 114–125 | 108–129 | 99–132 | 0.000/0.000/0.000 | -0.79 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752793 / 2025-08-29 | away | Minnesota (fbs) | 10–23 | -13 | 0.276 | 72.80/15.52 | -17.90 |
| 401762794 / 2025-09-06 | home | Saint Francis (fcs) | 45–6 | +39 | 0.912 | 116.41/5.95 | +35.70 |
| 401762817 / 2025-09-13 | away | Kent State (fbs) | 31–28 | +3 | 0.854 | 131.92/4.02 | +6.34 |
| 401761613 / 2025-09-20 | home | Troy (fbs) | 17–21 | -4 | 0.676 | 106.56/10.23 | -1.99 |
| 401757170 / 2025-09-27 | home | UConn (fbs) | 17–20 | -3 | 0.645 | 92.07/13.68 | -6.56 |
| 401762831 / 2025-10-04 | home | Eastern Michigan (fbs) | 31–30 | +1 | 0.709 | 121.31/6.86 | +5.01 |
| 401762840 / 2025-10-18 | away | Massachusetts (fbs) | 28–21 | +7 | 0.795 | 135.03/1.19 | +8.99 |
| 401762844 / 2025-10-25 | home | Akron (fbs) | 16–24 | -8 | 0.776 | 124.92/6.10 | +7.33 |
| 401762849 / 2025-11-01 | away | Bowling Green (fbs) | 28–3 | +25 | 0.596 | 119.58/8.46 | -2.12 |
| 401762858 / 2025-11-13 | away | Central Michigan (fbs) | 19–38 | -19 | 0.697 | 111.37/9.51 | -6.30 |
| 401762865 / 2025-11-20 | home | Miami (OH) (fbs) | 20–37 | -17 | 0.615 | 98.73/11.55 | -4.66 |
| 401762870 / 2025-11-28 | home | Ohio (fbs) | 26–31 | -5 | 0.581 | 87.23/13.75 | -7.93 |

Answers to the original questions at every candidate:

**Does this middle prior-reference case show a schedule advantage amid uncertainty?**

- At lambda=0, E[rank]=119.73, ordinal=117, SD=10.91, entropy=5.308 bits, central 80%=105-132; versus Record/Play the mean-rank changes are +0.00/+0.79. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=65.57, SD=21.81, entropy=6.474, widths 50/80/95=33/59/83; this team's expected-rank gap is +54.16 and SD gap -10.90. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=119.50, ordinal=117, SD=10.15, entropy=5.242 bits, central 80%=106-131; versus Record/Play the mean-rank changes are -0.24/+0.55. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=61.79, SD=19.84, entropy=6.342, widths 50/80/95=30/53/76; this team's expected-rank gap is +57.71 and SD gap -9.69. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=119.27, ordinal=115, SD=9.52, entropy=5.173 bits, central 80%=107-130; versus Record/Play the mean-rank changes are -0.47/+0.32. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=59.66, SD=18.37, entropy=6.234, widths 50/80/95=27/49/71; this team's expected-rank gap is +59.61 and SD gap -8.85. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=119.08, ordinal=114, SD=8.97, entropy=5.103 bits, central 80%=107-129; versus Record/Play the mean-rank changes are -0.66/+0.13. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=58.43, SD=17.24, entropy=6.144, widths 50/80/95=25/46/67; this team's expected-rank gap is +60.65 and SD gap -8.26. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=118.95, ordinal=114, SD=8.49, entropy=5.032 bits, central 80%=108-129; versus Record/Play the mean-rank changes are -0.79/+0.00. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=57.68, SD=16.32, entropy=6.066, widths 50/80/95=24/44/64; this team's expected-rank gap is +61.27 and SD gap -7.83. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**How do identical middle-team records with different schedules compare across candidates?**

- At lambda=0, E[rank]=119.73, ordinal=117, SD=10.91, entropy=5.308 bits, central 80%=105-132; versus Record/Play the mean-rank changes are +0.00/+0.79. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=65.57, SD=21.81, entropy=6.474, widths 50/80/95=33/59/83; this team's expected-rank gap is +54.16 and SD gap -10.90. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=119.50, ordinal=117, SD=10.15, entropy=5.242 bits, central 80%=106-131; versus Record/Play the mean-rank changes are -0.24/+0.55. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=61.79, SD=19.84, entropy=6.342, widths 50/80/95=30/53/76; this team's expected-rank gap is +57.71 and SD gap -9.69. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=119.27, ordinal=115, SD=9.52, entropy=5.173 bits, central 80%=107-130; versus Record/Play the mean-rank changes are -0.47/+0.32. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=59.66, SD=18.37, entropy=6.234, widths 50/80/95=27/49/71; this team's expected-rank gap is +59.61 and SD gap -8.85. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=119.08, ordinal=114, SD=8.97, entropy=5.103 bits, central 80%=107-129; versus Record/Play the mean-rank changes are -0.66/+0.13. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=58.43, SD=17.24, entropy=6.144, widths 50/80/95=25/46/67; this team's expected-rank gap is +60.65 and SD gap -8.26. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=118.95, ordinal=114, SD=8.49, entropy=5.032 bits, central 80%=108-129; versus Record/Play the mean-rank changes are -0.79/+0.00. The frozen peer Kentucky is 5-7 with standardized schedule probability 0.438, E[rank]=57.68, SD=16.32, entropy=6.066, widths 50/80/95=24/44/64; this team's expected-rank gap is +61.27 and SD gap -7.83. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 119.27 to 119.73 (+0.46); ordinal changes +1, SD changes -0.20, entropy -0.032, 80% width +0, TV=0.018. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 118.96 to 119.50 (+0.54); ordinal changes +0, SD changes -0.22, entropy -0.035, 80% width -1, TV=0.022. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 118.64 to 119.27 (+0.63); ordinal changes -2, SD changes -0.24, entropy -0.039, 80% width -1, TV=0.027. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 118.37 to 119.08 (+0.71); ordinal changes -1, SD changes -0.26, entropy -0.043, 80% width -1, TV=0.032. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 118.16 to 118.95 (+0.79); ordinal changes +0, SD changes -0.29, entropy -0.049, 80% width +0, TV=0.038. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 0 / 55 | +0.46 | +1 | -0.20 | 0.018 |
| 2025-prebowl → 2025-final | 0.25 | 0 / 55 | +0.54 | +0 | -0.22 | 0.022 |
| 2025-prebowl → 2025-final | 0.5 | 0 / 55 | +0.63 | -2 | -0.24 | 0.027 |
| 2025-prebowl → 2025-final | 0.75 | 0 / 55 | +0.71 | -1 | -0.26 | 0.032 |
| 2025-prebowl → 2025-final | 1 | 0 / 55 | +0.79 | +0 | -0.29 | 0.038 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2025 Florida Atlantic (2226)

Frozen categories: weak.

### 2025-prebowl

Actual record **4–8**, mean margin **-6.83**; close wins/losses 1/1, dominant wins 1. Frozen standardized schedule probability **0.695** (larger is easier), expected wins **4.97**, wins minus expectation **-0.97**. Reference: preseason_context_v1.2, ordinal 120, band weak, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 110.03 | 109 | 15.03 | 5.842 | 101–121 | 90–127 | 74–133 | 0.000/0.000/0.000 | +0.00 / -5.30 |
| 0.25 | 111.60 | 110 | 13.19 | 5.672 | 104–121 | 94–127 | 81–132 | 0.000/0.000/0.000 | +1.56 / -3.73 |
| 0.5 | 113.15 | 107 | 11.59 | 5.499 | 107–121 | 98–127 | 86–131 | 0.000/0.000/0.000 | +3.12 / -2.18 |
| 0.75 | 114.40 | 108 | 10.30 | 5.342 | 108–122 | 101–126 | 91–131 | 0.000/0.000/0.000 | +4.36 / -0.93 |
| 1 | 115.33 | 107 | 9.29 | 5.202 | 110–122 | 103–126 | 94–131 | 0.000/0.000/0.000 | +5.30 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752797 / 2025-08-30 | away | Maryland (fbs) | 7–39 | -32 | 0.209 | 85.26/15.01 | -15.11 |
| 401762460 / 2025-09-06 | home | Florida A&M (fcs) | 56–14 | +42 | 0.812 | 115.09/6.15 | +34.63 |
| 401757233 / 2025-09-13 | away | Florida International (fbs) | 28–38 | -10 | 0.435 | 112.12/10.15 | -6.32 |
| 401762472 / 2025-09-27 | home | Memphis (fbs) | 26–55 | -29 | 0.358 | 61.63/16.08 | -14.69 |
| 401762477 / 2025-10-04 | away | Rice (fbs) | 27–21 | +6 | 0.401 | 124.47/6.35 | +0.31 |
| 401762482 / 2025-10-11 | home | UAB (fbs) | 53–33 | +20 | 0.617 | 125.28/6.44 | +7.17 |
| 401762490 / 2025-10-18 | away | South Florida (fbs) | 13–48 | -35 | 0.323 | 30.65/12.02 | -28.47 |
| 401762493 / 2025-10-25 | away | Navy (fbs) | 32–42 | -10 | 0.148 | 79.48/16.17 | -16.59 |
| 401762505 / 2025-11-08 | home | Tulsa (fbs) | 40–21 | +19 | 0.653 | 121.61/7.77 | +4.86 |
| 401762509 / 2025-11-15 | away | Tulane (fbs) | 24–35 | -11 | 0.272 | 63.29/15.58 | -20.57 |
| 401757175 / 2025-11-22 | home | UConn (fbs) | 45–48 | -3 | 0.413 | 85.06/14.68 | -8.89 |
| 401762519 / 2025-11-29 | home | East Carolina (fbs) | 3–42 | -39 | 0.328 | 55.15/15.36 | -16.23 |

Answers to the original questions at every candidate:

**Does the weak-team schedule/uncertainty reversal appear in this completed season?**

- At lambda=0, E[rank]=110.03, ordinal=109, SD=15.03, entropy=5.842 bits, central 80%=90-127; versus Record/Play the mean-rank changes are +0.00/-5.30. Actual evidence is 4-8, mean margin -6.83, 1 close wins and 1 close losses, standardized schedule probability 0.695, expected wins 4.97, wins minus expectation -0.97. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.25, E[rank]=111.60, ordinal=110, SD=13.19, entropy=5.672 bits, central 80%=94-127; versus Record/Play the mean-rank changes are +1.56/-3.73. Actual evidence is 4-8, mean margin -6.83, 1 close wins and 1 close losses, standardized schedule probability 0.695, expected wins 4.97, wins minus expectation -0.97. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.5, E[rank]=113.15, ordinal=107, SD=11.59, entropy=5.499 bits, central 80%=98-127; versus Record/Play the mean-rank changes are +3.12/-2.18. Actual evidence is 4-8, mean margin -6.83, 1 close wins and 1 close losses, standardized schedule probability 0.695, expected wins 4.97, wins minus expectation -0.97. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.75, E[rank]=114.40, ordinal=108, SD=10.30, entropy=5.342 bits, central 80%=101-126; versus Record/Play the mean-rank changes are +4.36/-0.93. Actual evidence is 4-8, mean margin -6.83, 1 close wins and 1 close losses, standardized schedule probability 0.695, expected wins 4.97, wins minus expectation -0.97. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=1, E[rank]=115.33, ordinal=107, SD=9.29, entropy=5.202 bits, central 80%=103-126; versus Record/Play the mean-rank changes are +5.30/+0.00. Actual evidence is 4-8, mean margin -6.83, 1 close wins and 1 close losses, standardized schedule probability 0.695, expected wins 4.97, wins minus expectation -0.97. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 110.03 to 110.85 (+0.82); ordinal changes +0, SD changes -0.26, entropy -0.031, 80% width +0, TV=0.023. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 111.60 to 112.63 (+1.04); ordinal changes -1, SD changes -0.30, entropy -0.038, 80% width +0, TV=0.033. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 113.15 to 114.32 (+1.17); ordinal changes +1, SD changes -0.31, entropy -0.044, 80% width -1, TV=0.042. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 114.40 to 115.68 (+1.28); ordinal changes +1, SD changes -0.32, entropy -0.049, 80% width -1, TV=0.052. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 115.33 to 116.71 (+1.38); ordinal changes +4, SD changes -0.32, entropy -0.054, 80% width -1, TV=0.062. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **4–8**, mean margin **-6.83**; close wins/losses 1/1, dominant wins 1. Frozen standardized schedule probability **0.695** (larger is easier), expected wins **4.97**, wins minus expectation **-0.97**. Reference: preseason_context_v1.2, ordinal 120, band weak, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 110.85 | 109 | 14.77 | 5.811 | 102–122 | 91–128 | 76–133 | 0.000/0.000/0.000 | +0.00 / -5.86 |
| 0.25 | 112.63 | 109 | 12.89 | 5.633 | 105–122 | 95–128 | 82–132 | 0.000/0.000/0.000 | +1.78 / -4.08 |
| 0.5 | 114.32 | 108 | 11.28 | 5.455 | 108–122 | 99–127 | 88–132 | 0.000/0.000/0.000 | +3.47 / -2.39 |
| 0.75 | 115.68 | 109 | 9.99 | 5.293 | 110–123 | 103–127 | 93–132 | 0.000/0.000/0.000 | +4.83 / -1.03 |
| 1 | 116.71 | 111 | 8.97 | 5.148 | 112–123 | 105–127 | 96–131 | 0.000/0.000/0.000 | +5.86 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752797 / 2025-08-30 | away | Maryland (fbs) | 7–39 | -32 | 0.209 | 85.01/15.11 | -15.73 |
| 401762460 / 2025-09-06 | home | Florida A&M (fcs) | 56–14 | +42 | 0.812 | 115.34/6.06 | +34.33 |
| 401757233 / 2025-09-13 | away | Florida International (fbs) | 28–38 | -10 | 0.435 | 117.14/8.64 | -4.60 |
| 401762472 / 2025-09-27 | home | Memphis (fbs) | 26–55 | -29 | 0.358 | 68.32/15.60 | -13.61 |
| 401762477 / 2025-10-04 | away | Rice (fbs) | 27–21 | +6 | 0.401 | 126.53/5.45 | +1.06 |
| 401762482 / 2025-10-11 | home | UAB (fbs) | 53–33 | +20 | 0.617 | 125.68/6.24 | +6.80 |
| 401762490 / 2025-10-18 | away | South Florida (fbs) | 13–48 | -35 | 0.323 | 35.55/12.52 | -27.68 |
| 401762493 / 2025-10-25 | away | Navy (fbs) | 32–42 | -10 | 0.148 | 73.48/15.63 | -18.64 |
| 401762505 / 2025-11-08 | home | Tulsa (fbs) | 40–21 | +19 | 0.653 | 121.53/7.90 | +4.19 |
| 401762509 / 2025-11-15 | away | Tulane (fbs) | 24–35 | -11 | 0.272 | 64.75/15.25 | -20.74 |
| 401757175 / 2025-11-22 | home | UConn (fbs) | 45–48 | -3 | 0.413 | 92.07/13.68 | -7.57 |
| 401762519 / 2025-11-29 | home | East Carolina (fbs) | 3–42 | -39 | 0.328 | 51.09/14.48 | -17.69 |

Answers to the original questions at every candidate:

**Does the weak-team schedule/uncertainty reversal appear in this completed season?**

- At lambda=0, E[rank]=110.85, ordinal=109, SD=14.77, entropy=5.811 bits, central 80%=91-128; versus Record/Play the mean-rank changes are +0.00/-5.86. Actual evidence is 4-8, mean margin -6.83, 1 close wins and 1 close losses, standardized schedule probability 0.695, expected wins 4.97, wins minus expectation -0.97. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.25, E[rank]=112.63, ordinal=109, SD=12.89, entropy=5.633 bits, central 80%=95-128; versus Record/Play the mean-rank changes are +1.78/-4.08. Actual evidence is 4-8, mean margin -6.83, 1 close wins and 1 close losses, standardized schedule probability 0.695, expected wins 4.97, wins minus expectation -0.97. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.5, E[rank]=114.32, ordinal=108, SD=11.28, entropy=5.455 bits, central 80%=99-127; versus Record/Play the mean-rank changes are +3.47/-2.39. Actual evidence is 4-8, mean margin -6.83, 1 close wins and 1 close losses, standardized schedule probability 0.695, expected wins 4.97, wins minus expectation -0.97. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.75, E[rank]=115.68, ordinal=109, SD=9.99, entropy=5.293 bits, central 80%=103-127; versus Record/Play the mean-rank changes are +4.83/-1.03. Actual evidence is 4-8, mean margin -6.83, 1 close wins and 1 close losses, standardized schedule probability 0.695, expected wins 4.97, wins minus expectation -0.97. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=1, E[rank]=116.71, ordinal=111, SD=8.97, entropy=5.148 bits, central 80%=105-127; versus Record/Play the mean-rank changes are +5.86/+0.00. Actual evidence is 4-8, mean margin -6.83, 1 close wins and 1 close losses, standardized schedule probability 0.695, expected wins 4.97, wins minus expectation -0.97. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 110.03 to 110.85 (+0.82); ordinal changes +0, SD changes -0.26, entropy -0.031, 80% width +0, TV=0.023. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 111.60 to 112.63 (+1.04); ordinal changes -1, SD changes -0.30, entropy -0.038, 80% width +0, TV=0.033. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 113.15 to 114.32 (+1.17); ordinal changes +1, SD changes -0.31, entropy -0.044, 80% width -1, TV=0.042. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 114.40 to 115.68 (+1.28); ordinal changes +1, SD changes -0.32, entropy -0.049, 80% width -1, TV=0.052. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 115.33 to 116.71 (+1.38); ordinal changes +4, SD changes -0.32, entropy -0.054, 80% width -1, TV=0.062. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 0 / 55 | +0.82 | +0 | -0.26 | 0.023 |
| 2025-prebowl → 2025-final | 0.25 | 0 / 55 | +1.04 | -1 | -0.30 | 0.033 |
| 2025-prebowl → 2025-final | 0.5 | 0 / 55 | +1.17 | +1 | -0.31 | 0.042 |
| 2025-prebowl → 2025-final | 0.75 | 0 / 55 | +1.28 | +1 | -0.32 | 0.052 |
| 2025-prebowl → 2025-final | 1 | 0 / 55 | +1.38 | +4 | -0.32 | 0.062 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2025 Iowa (2294)

Frozen categories: reference_boundary_12.

### 2025-prebowl

Actual record **8–4**, mean margin **+13.75**; close wins/losses 2/4, dominant wins 3. Frozen standardized schedule probability **0.464** (larger is easier), expected wins **7.86**, wins minus expectation **+0.14**. Reference: preseason_context_v1.2, ordinal 12, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 32.38 | 21 | 16.55 | 5.990 | 20–42 | 14–55 | 8–72 | 0.054/0.083/0.391 | +0.00 / +14.37 |
| 0.25 | 27.19 | 18 | 14.14 | 5.753 | 17–35 | 11–46 | 6–61 | 0.085/0.128/0.519 | -5.19 / +9.17 |
| 0.5 | 23.25 | 17 | 12.14 | 5.529 | 15–30 | 10–39 | 5–53 | 0.123/0.180/0.637 | -9.12 / +5.24 |
| 0.75 | 20.28 | 16 | 10.52 | 5.327 | 13–26 | 8–34 | 4–46 | 0.165/0.238/0.736 | -12.10 / +2.27 |
| 1 | 18.01 | 15 | 9.25 | 5.151 | 11–23 | 7–30 | 4–40 | 0.211/0.299/0.813 | -14.37 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752799 / 2025-08-30 | home | UAlbany (fcs) | 34–7 | +27 | 0.980 | 103.65/10.55 | +51.67 |
| 401752822 / 2025-09-06 | away | Iowa State (fbs) | 13–16 | -3 | 0.603 | 42.92/14.39 | +7.27 |
| 401752830 / 2025-09-13 | home | Massachusetts (fbs) | 47–7 | +40 | 0.973 | 135.03/1.19 | +39.16 |
| 401752844 / 2025-09-20 | away | Rutgers (fbs) | 38–28 | +10 | 0.588 | 66.40/16.88 | +12.82 |
| 401752853 / 2025-09-27 | home | Indiana (fbs) | 15–20 | -5 | 0.654 | 4.77/2.56 | -5.60 |
| 401752869 / 2025-10-11 | away | Wisconsin (fbs) | 37–0 | +37 | 0.594 | 65.19/17.45 | +12.54 |
| 401752875 / 2025-10-18 | home | Penn State (fbs) | 25–24 | +1 | 0.516 | 19.70/9.83 | +5.07 |
| 401752882 / 2025-10-25 | home | Minnesota (fbs) | 41–3 | +38 | 0.700 | 72.30/16.24 | +20.42 |
| 401752898 / 2025-11-08 | home | Oregon (fbs) | 16–18 | -2 | 0.429 | 5.23/3.33 | -5.21 |
| 401752907 / 2025-11-15 | away | USC (fbs) | 21–26 | -5 | 0.448 | 15.80/7.47 | -3.26 |
| 401752913 / 2025-11-22 | home | Michigan State (fbs) | 20–17 | +3 | 0.807 | 75.63/15.70 | +21.17 |
| 401752916 / 2025-11-28 | away | Nebraska (fbs) | 40–16 | +24 | 0.573 | 54.56/15.27 | +10.14 |

Answers to the original questions at every candidate:

**Does movement near this prior-reference boundary reflect substantial evidence or closely spaced teams?**

- At lambda=0, E[rank]=32.38, ordinal=21, SD=16.55, entropy=5.990 bits, central 80%=14-55; versus Record/Play the mean-rank changes are +0.00/+14.37. Latent top10/12/25 mass is 0.054/0.083/0.391. Adjacent expected-rank gaps are 0.148558681417839/2.4239048489586636. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=27.19, ordinal=18, SD=14.14, entropy=5.753 bits, central 80%=11-46; versus Record/Play the mean-rank changes are -5.19/+9.17. Latent top10/12/25 mass is 0.085/0.128/0.519. Adjacent expected-rank gaps are 0.0054187786113431/4.209313859533609. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=23.25, ordinal=17, SD=12.14, entropy=5.529 bits, central 80%=10-39; versus Record/Play the mean-rank changes are -9.12/+5.24. Latent top10/12/25 mass is 0.123/0.180/0.637. Adjacent expected-rank gaps are 3.509064486501547/4.044765264220711. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=20.28, ordinal=16, SD=10.52, entropy=5.327 bits, central 80%=8-34; versus Record/Play the mean-rank changes are -12.10/+2.27. Latent top10/12/25 mass is 0.165/0.238/0.736. Adjacent expected-rank gaps are 1.631442863501519/0.41020670497087863. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=18.01, ordinal=15, SD=9.25, entropy=5.151 bits, central 80%=7-30; versus Record/Play the mean-rank changes are -14.37/+0.00. Latent top10/12/25 mass is 0.211/0.299/0.813. Adjacent expected-rank gaps are 0.8091007962821948/1.621369174653136. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 32.38 to 25.75 (-6.62); ordinal changes -5, SD changes -2.48, entropy -0.255, 80% width -6, TV=0.172. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 27.19 to 22.25 (-4.93); ordinal changes -2, SD changes -2.01, entropy -0.237, 80% width -5, TV=0.153. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 23.25 to 19.58 (-3.67); ordinal changes -1, SD changes -1.62, entropy -0.217, 80% width -4, TV=0.134. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 20.28 to 17.60 (-2.68); ordinal changes -3, SD changes -1.30, entropy -0.193, 80% width -4, TV=0.112. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 18.01 to 16.11 (-1.90); ordinal changes -3, SD changes -1.03, entropy -0.169, 80% width -3, TV=0.090. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **9–4**, mean margin **+13.23**; close wins/losses 3/4, dominant wins 3. Frozen standardized schedule probability **0.459** (larger is easier), expected wins **8.49**, wins minus expectation **+0.51**. Reference: preseason_context_v1.2, ordinal 12, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 25.75 | 16 | 14.07 | 5.734 | 16–34 | 10–45 | 5–60 | 0.112/0.162/0.563 | +0.00 / +9.65 |
| 0.25 | 22.25 | 16 | 12.14 | 5.516 | 14–29 | 9–39 | 4–52 | 0.149/0.213/0.669 | -3.50 / +6.14 |
| 0.5 | 19.58 | 16 | 10.51 | 5.313 | 12–25 | 8–33 | 4–45 | 0.187/0.266/0.758 | -6.17 / +3.47 |
| 0.75 | 17.60 | 13 | 9.22 | 5.134 | 11–23 | 7–29 | 4–40 | 0.225/0.318/0.826 | -8.16 / +1.49 |
| 1 | 16.11 | 12 | 8.22 | 4.982 | 10–21 | 7–27 | 3–35 | 0.262/0.367/0.876 | -9.65 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752799 / 2025-08-30 | home | UAlbany (fcs) | 34–7 | +27 | 0.980 | 103.42/10.61 | +52.07 |
| 401752822 / 2025-09-06 | away | Iowa State (fbs) | 13–16 | -3 | 0.603 | 43.59/14.46 | +8.17 |
| 401752830 / 2025-09-13 | home | Massachusetts (fbs) | 47–7 | +40 | 0.973 | 135.03/1.19 | +39.50 |
| 401752844 / 2025-09-20 | away | Rutgers (fbs) | 38–28 | +10 | 0.588 | 65.23/16.85 | +13.18 |
| 401752853 / 2025-09-27 | home | Indiana (fbs) | 15–20 | -5 | 0.654 | 3.46/1.78 | -6.05 |
| 401752869 / 2025-10-11 | away | Wisconsin (fbs) | 37–0 | +37 | 0.594 | 65.67/17.44 | +13.28 |
| 401752875 / 2025-10-18 | home | Penn State (fbs) | 25–24 | +1 | 0.516 | 18.13/8.74 | +5.15 |
| 401752882 / 2025-10-25 | home | Minnesota (fbs) | 41–3 | +38 | 0.700 | 72.80/15.52 | +21.13 |
| 401752898 / 2025-11-08 | home | Oregon (fbs) | 16–18 | -2 | 0.429 | 5.29/3.20 | -4.30 |
| 401752907 / 2025-11-15 | away | USC (fbs) | 21–26 | -5 | 0.448 | 17.40/7.66 | -1.40 |
| 401752913 / 2025-11-22 | home | Michigan State (fbs) | 20–17 | +3 | 0.807 | 75.11/15.72 | +21.65 |
| 401752916 / 2025-11-28 | away | Nebraska (fbs) | 40–16 | +24 | 0.573 | 56.83/15.04 | +11.32 |
| 401778328 / 2025-12-31 | neutral | Vanderbilt (fbs) | 34–27 | +7 | 0.629 | 15.69/6.87 | +0.73 |

Answers to the original questions at every candidate:

**Does movement near this prior-reference boundary reflect substantial evidence or closely spaced teams?**

- At lambda=0, E[rank]=25.75, ordinal=16, SD=14.07, entropy=5.734 bits, central 80%=10-45; versus Record/Play the mean-rank changes are +0.00/+9.65. Latent top10/12/25 mass is 0.112/0.162/0.563. Adjacent expected-rank gaps are 3.9002607510190828/2.3332340497133472. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=22.25, ordinal=16, SD=12.14, entropy=5.516 bits, central 80%=9-39; versus Record/Play the mean-rank changes are -3.50/+6.14. Latent top10/12/25 mass is 0.149/0.213/0.669. Adjacent expected-rank gaps are 3.623290748151625/2.2464304545601372. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=19.58, ordinal=16, SD=10.51, entropy=5.313 bits, central 80%=8-33; versus Record/Play the mean-rank changes are -6.17/+3.47. Latent top10/12/25 mass is 0.187/0.266/0.758. Adjacent expected-rank gaps are 1.1289484395509604/2.472018860462363. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=17.60, ordinal=13, SD=9.22, entropy=5.134 bits, central 80%=7-29; versus Record/Play the mean-rank changes are -8.16/+1.49. Latent top10/12/25 mass is 0.225/0.318/0.826. Adjacent expected-rank gaps are 1.2330315814627326/1.2752942067612238. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=16.11, ordinal=12, SD=8.22, entropy=4.982 bits, central 80%=7-27; versus Record/Play the mean-rank changes are -9.65/+0.00. Latent top10/12/25 mass is 0.262/0.367/0.876. Adjacent expected-rank gaps are 0.4149513921916128/0.8720576005868388. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 32.38 to 25.75 (-6.62); ordinal changes -5, SD changes -2.48, entropy -0.255, 80% width -6, TV=0.172. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 27.19 to 22.25 (-4.93); ordinal changes -2, SD changes -2.01, entropy -0.237, 80% width -5, TV=0.153. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 23.25 to 19.58 (-3.67); ordinal changes -1, SD changes -1.62, entropy -0.217, 80% width -4, TV=0.134. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 20.28 to 17.60 (-2.68); ordinal changes -3, SD changes -1.30, entropy -0.193, 80% width -4, TV=0.112. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 18.01 to 16.11 (-1.90); ordinal changes -3, SD changes -1.03, entropy -0.169, 80% width -3, TV=0.090. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 1 / 55 | -6.62 | -5 | -2.48 | 0.172 |
| 2025-prebowl → 2025-final | 0.25 | 1 / 55 | -4.93 | -2 | -2.01 | 0.153 |
| 2025-prebowl → 2025-final | 0.5 | 1 / 55 | -3.67 | -1 | -1.62 | 0.134 |
| 2025-prebowl → 2025-final | 0.75 | 1 / 55 | -2.68 | -3 | -1.30 | 0.112 |
| 2025-prebowl → 2025-final | 1 | 1 / 55 | -1.90 | -3 | -1.03 | 0.090 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2025 Kentucky (96)

Frozen categories: middle_schedule_pair.

### 2025-prebowl

Actual record **5–7**, mean margin **-3.42**; close wins/losses 1/2, dominant wins 2. Frozen standardized schedule probability **0.438** (larger is easier), expected wins **4.47**, wins minus expectation **+0.53**. Reference: preseason_context_v1.2, ordinal 63, band middle, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 63.60 | 54 | 21.90 | 6.477 | 47–79 | 35–93 | 24–106 | 0.001/0.001/0.031 | +0.00 / +7.12 |
| 0.25 | 60.02 | 50 | 19.90 | 6.343 | 45–74 | 34–87 | 24–100 | 0.000/0.001/0.031 | -3.58 / +3.54 |
| 0.5 | 58.09 | 51 | 18.42 | 6.235 | 45–71 | 34–83 | 25–95 | 0.000/0.001/0.028 | -5.51 / +1.62 |
| 0.75 | 57.05 | 53 | 17.29 | 6.145 | 45–69 | 35–80 | 26–92 | 0.000/0.000/0.024 | -6.55 / +0.57 |
| 1 | 56.47 | 51 | 16.38 | 6.070 | 45–68 | 35–78 | 27–90 | 0.000/0.000/0.020 | -7.12 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752670 / 2025-08-30 | home | Toledo (fbs) | 24–16 | +8 | 0.587 | 52.03/15.25 | +1.27 |
| 401752686 / 2025-09-06 | home | Ole Miss (fbs) | 23–30 | -7 | 0.357 | 15.44/7.43 | -11.46 |
| 401752698 / 2025-09-13 | home | Eastern Michigan (fbs) | 48–23 | +25 | 0.711 | 121.22/6.88 | +22.65 |
| 401752722 / 2025-09-27 | away | South Carolina (fbs) | 13–35 | -22 | 0.156 | 40.47/13.58 | -8.28 |
| 401752728 / 2025-10-04 | away | Georgia (fbs) | 14–35 | -21 | 0.088 | 7.45/4.57 | -22.09 |
| 401752743 / 2025-10-18 | home | Texas (fbs) | 13–16 | -3 | 0.179 | 15.55/8.14 | -11.47 |
| 401752747 / 2025-10-25 | home | Tennessee (fbs) | 34–56 | -22 | 0.264 | 23.35/10.31 | -8.00 |
| 401752754 / 2025-11-01 | away | Auburn (fbs) | 10–3 | +7 | 0.357 | 35.24/13.16 | -9.97 |
| 401752760 / 2025-11-09 | home | Florida (fbs) | 38–7 | +31 | 0.330 | 44.13/14.28 | -0.95 |
| 401752767 / 2025-11-15 | home | Tennessee Tech (fcs) | 42–10 | +32 | 0.913 | 30.10/11.95 | +24.80 |
| 401752782 / 2025-11-22 | away | Vanderbilt (fbs) | 17–45 | -28 | 0.266 | 13.69/6.51 | -18.58 |
| 401752787 / 2025-11-29 | away | Louisville (fbs) | 0–41 | -41 | 0.260 | 38.84/14.40 | -8.86 |

Answers to the original questions at every candidate:

**How do identical middle-team records with different schedules compare across candidates?**

- At lambda=0, E[rank]=63.60, ordinal=54, SD=21.90, entropy=6.477 bits, central 80%=35-93; versus Record/Play the mean-rank changes are +0.00/+7.12. The frozen peer Buffalo is 5-7 with standardized schedule probability 0.747, E[rank]=119.27, SD=11.11, entropy=5.340, widths 50/80/95=15/28/44; this team's expected-rank gap is -55.68 and SD gap +10.79. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=60.02, ordinal=50, SD=19.90, entropy=6.343 bits, central 80%=34-87; versus Record/Play the mean-rank changes are -3.58/+3.54. The frozen peer Buffalo is 5-7 with standardized schedule probability 0.747, E[rank]=118.96, SD=10.37, entropy=5.277, widths 50/80/95=14/27/41; this team's expected-rank gap is -58.94 and SD gap +9.52. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=58.09, ordinal=51, SD=18.42, entropy=6.235 bits, central 80%=34-83; versus Record/Play the mean-rank changes are -5.51/+1.62. The frozen peer Buffalo is 5-7 with standardized schedule probability 0.747, E[rank]=118.64, SD=9.76, entropy=5.212, widths 50/80/95=14/25/39; this team's expected-rank gap is -60.55 and SD gap +8.66. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=57.05, ordinal=53, SD=17.29, entropy=6.145 bits, central 80%=35-80; versus Record/Play the mean-rank changes are -6.55/+0.57. The frozen peer Buffalo is 5-7 with standardized schedule probability 0.747, E[rank]=118.37, SD=9.24, entropy=5.146, widths 50/80/95=13/24/38; this team's expected-rank gap is -61.32 and SD gap +8.05. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=56.47, ordinal=51, SD=16.38, entropy=6.070 bits, central 80%=35-78; versus Record/Play the mean-rank changes are -7.12/+0.00. The frozen peer Buffalo is 5-7 with standardized schedule probability 0.747, E[rank]=118.16, SD=8.78, entropy=5.081, widths 50/80/95=12/22/36; this team's expected-rank gap is -61.68 and SD gap +7.60. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 63.60 to 65.57 (+1.98); ordinal changes -2, SD changes -0.10, entropy -0.003, 80% width +0, TV=0.036. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 60.02 to 61.79 (+1.77); ordinal changes -1, SD changes -0.05, entropy -0.001, 80% width -1, TV=0.036. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 58.09 to 59.66 (+1.57); ordinal changes +1, SD changes -0.05, entropy -0.001, 80% width -1, TV=0.034. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 57.05 to 58.43 (+1.38); ordinal changes -1, SD changes -0.05, entropy -0.002, 80% width +0, TV=0.032. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 56.47 to 57.68 (+1.20); ordinal changes -1, SD changes -0.06, entropy -0.003, 80% width +0, TV=0.029. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **5–7**, mean margin **-3.42**; close wins/losses 1/2, dominant wins 2. Frozen standardized schedule probability **0.438** (larger is easier), expected wins **4.47**, wins minus expectation **+0.53**. Reference: preseason_context_v1.2, ordinal 63, band middle, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 65.57 | 52 | 21.81 | 6.474 | 49–81 | 37–95 | 26–108 | 0.000/0.001/0.025 | +0.00 / +7.90 |
| 0.25 | 61.79 | 49 | 19.84 | 6.342 | 47–76 | 36–88 | 26–101 | 0.000/0.001/0.024 | -3.78 / +4.11 |
| 0.5 | 59.66 | 52 | 18.37 | 6.234 | 46–72 | 36–84 | 26–96 | 0.000/0.000/0.022 | -5.91 / +1.98 |
| 0.75 | 58.43 | 52 | 17.24 | 6.144 | 46–70 | 36–81 | 27–93 | 0.000/0.000/0.019 | -7.14 / +0.75 |
| 1 | 57.68 | 50 | 16.32 | 6.066 | 46–69 | 36–79 | 28–91 | 0.000/0.000/0.016 | -7.90 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752670 / 2025-08-30 | home | Toledo (fbs) | 24–16 | +8 | 0.587 | 52.31/14.52 | +1.08 |
| 401752686 / 2025-09-06 | home | Ole Miss (fbs) | 23–30 | -7 | 0.357 | 13.30/5.93 | -12.71 |
| 401752698 / 2025-09-13 | home | Eastern Michigan (fbs) | 48–23 | +25 | 0.711 | 121.31/6.86 | +22.43 |
| 401752722 / 2025-09-27 | away | South Carolina (fbs) | 13–35 | -22 | 0.156 | 42.20/13.77 | -8.04 |
| 401752728 / 2025-10-04 | away | Georgia (fbs) | 14–35 | -21 | 0.088 | 8.78/5.01 | -21.50 |
| 401752743 / 2025-10-18 | home | Texas (fbs) | 13–16 | -3 | 0.179 | 15.20/7.59 | -11.85 |
| 401752747 / 2025-10-25 | home | Tennessee (fbs) | 34–56 | -22 | 0.264 | 26.34/10.60 | -7.07 |
| 401752754 / 2025-11-01 | away | Auburn (fbs) | 10–3 | +7 | 0.357 | 37.47/13.16 | -9.52 |
| 401752760 / 2025-11-09 | home | Florida (fbs) | 38–7 | +31 | 0.330 | 45.45/14.33 | -0.85 |
| 401752767 / 2025-11-15 | home | Tennessee Tech (fcs) | 42–10 | +32 | 0.913 | 30.09/11.93 | +24.55 |
| 401752782 / 2025-11-22 | away | Vanderbilt (fbs) | 17–45 | -28 | 0.266 | 15.69/6.87 | -17.81 |
| 401752787 / 2025-11-29 | away | Louisville (fbs) | 0–41 | -41 | 0.260 | 38.01/13.71 | -9.37 |

Answers to the original questions at every candidate:

**How do identical middle-team records with different schedules compare across candidates?**

- At lambda=0, E[rank]=65.57, ordinal=52, SD=21.81, entropy=6.474 bits, central 80%=37-95; versus Record/Play the mean-rank changes are +0.00/+7.90. The frozen peer Buffalo is 5-7 with standardized schedule probability 0.747, E[rank]=119.73, SD=10.91, entropy=5.308, widths 50/80/95=15/28/43; this team's expected-rank gap is -54.16 and SD gap +10.90. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=61.79, ordinal=49, SD=19.84, entropy=6.342 bits, central 80%=36-88; versus Record/Play the mean-rank changes are -3.78/+4.11. The frozen peer Buffalo is 5-7 with standardized schedule probability 0.747, E[rank]=119.50, SD=10.15, entropy=5.242, widths 50/80/95=14/26/40; this team's expected-rank gap is -57.71 and SD gap +9.69. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=59.66, ordinal=52, SD=18.37, entropy=6.234 bits, central 80%=36-84; versus Record/Play the mean-rank changes are -5.91/+1.98. The frozen peer Buffalo is 5-7 with standardized schedule probability 0.747, E[rank]=119.27, SD=9.52, entropy=5.173, widths 50/80/95=13/24/39; this team's expected-rank gap is -59.61 and SD gap +8.85. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=58.43, ordinal=52, SD=17.24, entropy=6.144 bits, central 80%=36-81; versus Record/Play the mean-rank changes are -7.14/+0.75. The frozen peer Buffalo is 5-7 with standardized schedule probability 0.747, E[rank]=119.08, SD=8.97, entropy=5.103, widths 50/80/95=12/23/36; this team's expected-rank gap is -60.65 and SD gap +8.26. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=57.68, ordinal=50, SD=16.32, entropy=6.066 bits, central 80%=36-79; versus Record/Play the mean-rank changes are -7.90/+0.00. The frozen peer Buffalo is 5-7 with standardized schedule probability 0.747, E[rank]=118.95, SD=8.49, entropy=5.032, widths 50/80/95=12/22/34; this team's expected-rank gap is -61.27 and SD gap +7.83. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 63.60 to 65.57 (+1.98); ordinal changes -2, SD changes -0.10, entropy -0.003, 80% width +0, TV=0.036. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 60.02 to 61.79 (+1.77); ordinal changes -1, SD changes -0.05, entropy -0.001, 80% width -1, TV=0.036. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 58.09 to 59.66 (+1.57); ordinal changes +1, SD changes -0.05, entropy -0.001, 80% width -1, TV=0.034. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 57.05 to 58.43 (+1.38); ordinal changes -1, SD changes -0.05, entropy -0.002, 80% width +0, TV=0.032. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 56.47 to 57.68 (+1.20); ordinal changes -1, SD changes -0.06, entropy -0.003, 80% width +0, TV=0.029. Record changes 5-7 to 5-7, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 0 / 55 | +1.98 | -2 | -0.10 | 0.036 |
| 2025-prebowl → 2025-final | 0.25 | 0 / 55 | +1.77 | -1 | -0.05 | 0.036 |
| 2025-prebowl → 2025-final | 0.5 | 0 / 55 | +1.57 | +1 | -0.05 | 0.034 |
| 2025-prebowl → 2025-final | 0.75 | 0 / 55 | +1.38 | -1 | -0.05 | 0.032 |
| 2025-prebowl → 2025-final | 1 | 0 / 55 | +1.20 | -1 | -0.06 | 0.029 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2025 Navy (2426)

Frozen categories: excellent_record_easy_schedule.

### 2025-prebowl

Actual record **9–2**, mean margin **+5.55**; close wins/losses 3/0, dominant wins 1. Frozen standardized schedule probability **0.716** (larger is easier), expected wins **8.07**, wins minus expectation **+0.93**. Reference: preseason_context_v1.2, ordinal 43, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 39.53 | 30 | 21.69 | 6.399 | 23–54 | 13–70 | 6–87 | 0.069/0.092/0.299 | +0.00 / -42.17 |
| 0.25 | 55.68 | 43 | 21.59 | 6.457 | 39–71 | 28–85 | 17–98 | 0.007/0.010/0.080 | +16.16 / -26.01 |
| 0.5 | 67.29 | 59 | 20.09 | 6.360 | 53–82 | 40–93 | 28–104 | 0.000/0.001/0.017 | +27.76 / -14.41 |
| 0.75 | 75.66 | 66 | 18.18 | 6.213 | 64–89 | 51–98 | 37–108 | 0.000/0.000/0.003 | +36.13 / -6.04 |
| 1 | 81.69 | 73 | 16.36 | 6.059 | 71–93 | 60–102 | 46–111 | 0.000/0.000/0.001 | +42.17 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401762434 / 2025-08-30 | home | VMI (fcs) | 52–7 | +45 | 0.947 | 121.49/4.87 | +48.64 |
| 401762458 / 2025-09-06 | home | UAB (fbs) | 38–24 | +14 | 0.880 | 125.28/6.44 | +19.94 |
| 401762465 / 2025-09-14 | away | Tulsa (fbs) | 42–23 | +19 | 0.826 | 121.61/7.77 | +11.76 |
| 401762473 / 2025-09-27 | home | Rice (fbs) | 21–13 | +8 | 0.832 | 124.47/6.35 | +19.47 |
| 401760381 / 2025-10-04 | home | Air Force (fbs) | 34–31 | +3 | 0.698 | 105.16/10.97 | +10.97 |
| 401762484 / 2025-10-11 | away | Temple (fbs) | 32–31 | +1 | 0.785 | 106.22/10.59 | +5.08 |
| 401762493 / 2025-10-25 | home | Florida Atlantic (fbs) | 42–32 | +10 | 0.852 | 118.71/8.11 | +16.59 |
| 401762499 / 2025-11-01 | away | North Texas (fbs) | 17–31 | -14 | 0.752 | 43.81/14.15 | -12.54 |
| 401757166 / 2025-11-09 | away | Notre Dame (fbs) | 10–49 | -39 | 0.140 | 4.70/3.06 | -27.46 |
| 401762510 / 2025-11-15 | home | South Florida (fbs) | 41–38 | +3 | 0.776 | 30.65/12.02 | -10.32 |
| 401762516 / 2025-11-28 | away | Memphis (fbs) | 28–17 | +11 | 0.577 | 61.63/16.08 | -7.78 |

Answers to the original questions at every candidate:

**Does this excellent record receive disproportionate credit for an easier schedule?**

- At lambda=0, E[rank]=39.53, ordinal=30, SD=21.69, entropy=6.399 bits, central 80%=13-70; versus Record/Play the mean-rank changes are +0.00/-42.17. Actual evidence is 9-2, mean margin +5.55, 3 close wins and 0 close losses, standardized schedule probability 0.716, expected wins 8.07, wins minus expectation +0.93. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.25, E[rank]=55.68, ordinal=43, SD=21.59, entropy=6.457 bits, central 80%=28-85; versus Record/Play the mean-rank changes are +16.16/-26.01. Actual evidence is 9-2, mean margin +5.55, 3 close wins and 0 close losses, standardized schedule probability 0.716, expected wins 8.07, wins minus expectation +0.93. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.5, E[rank]=67.29, ordinal=59, SD=20.09, entropy=6.360 bits, central 80%=40-93; versus Record/Play the mean-rank changes are +27.76/-14.41. Actual evidence is 9-2, mean margin +5.55, 3 close wins and 0 close losses, standardized schedule probability 0.716, expected wins 8.07, wins minus expectation +0.93. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.75, E[rank]=75.66, ordinal=66, SD=18.18, entropy=6.213 bits, central 80%=51-98; versus Record/Play the mean-rank changes are +36.13/-6.04. Actual evidence is 9-2, mean margin +5.55, 3 close wins and 0 close losses, standardized schedule probability 0.716, expected wins 8.07, wins minus expectation +0.93. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=1, E[rank]=81.69, ordinal=73, SD=16.36, entropy=6.059 bits, central 80%=60-102; versus Record/Play the mean-rank changes are +42.17/+0.00. Actual evidence is 9-2, mean margin +5.55, 3 close wins and 0 close losses, standardized schedule probability 0.716, expected wins 8.07, wins minus expectation +0.93. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 39.53 to 31.91 (-7.62); ordinal changes -8, SD changes -3.13, entropy -0.253, 80% width -9, TV=0.149. Record changes 9-2 to 11-2, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 55.68 to 47.80 (-7.88); ordinal changes -2, SD changes -2.12, entropy -0.157, 80% width -5, TV=0.153. Record changes 9-2 to 11-2, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 67.29 to 59.65 (-7.63); ordinal changes -8, SD changes -1.32, entropy -0.097, 80% width -4, TV=0.159. Record changes 9-2 to 11-2, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 75.66 to 68.54 (-7.12); ordinal changes -4, SD changes -0.79, entropy -0.058, 80% width -1, TV=0.163. Record changes 9-2 to 11-2, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 81.69 to 75.14 (-6.55); ordinal changes -7, SD changes -0.55, entropy -0.044, 80% width -1, TV=0.167. Record changes 9-2 to 11-2, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **11–2**, mean margin **+6.46**; close wins/losses 4/0, dominant wins 1. Frozen standardized schedule probability **0.692** (larger is easier), expected wins **9.23**, wins minus expectation **+1.77**. Reference: preseason_context_v1.2, ordinal 43, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 31.91 | 22 | 18.56 | 6.146 | 18–43 | 10–58 | 4–75 | 0.108/0.143/0.422 | +0.00 / -43.24 |
| 0.25 | 47.80 | 41 | 19.47 | 6.300 | 33–61 | 23–75 | 14–88 | 0.011/0.017/0.127 | +15.90 / -27.34 |
| 0.5 | 59.65 | 51 | 18.77 | 6.263 | 46–73 | 35–84 | 25–95 | 0.001/0.001/0.029 | +27.75 / -15.49 |
| 0.75 | 68.54 | 62 | 17.39 | 6.155 | 57–81 | 45–91 | 33–100 | 0.000/0.000/0.005 | +36.63 / -6.60 |
| 1 | 75.14 | 66 | 15.81 | 6.016 | 65–86 | 54–95 | 42–104 | 0.000/0.000/0.001 | +43.24 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401762434 / 2025-08-30 | home | VMI (fcs) | 52–7 | +45 | 0.947 | 121.38/4.92 | +49.77 |
| 401762458 / 2025-09-06 | home | UAB (fbs) | 38–24 | +14 | 0.880 | 125.68/6.24 | +21.55 |
| 401762465 / 2025-09-14 | away | Tulsa (fbs) | 42–23 | +19 | 0.826 | 121.53/7.90 | +13.21 |
| 401762473 / 2025-09-27 | home | Rice (fbs) | 21–13 | +8 | 0.832 | 126.53/5.45 | +21.98 |
| 401760381 / 2025-10-04 | home | Air Force (fbs) | 34–31 | +3 | 0.698 | 105.41/10.90 | +12.66 |
| 401762484 / 2025-10-11 | away | Temple (fbs) | 32–31 | +1 | 0.785 | 104.97/10.92 | +6.23 |
| 401762493 / 2025-10-25 | home | Florida Atlantic (fbs) | 42–32 | +10 | 0.852 | 119.86/7.80 | +18.64 |
| 401762499 / 2025-11-01 | away | North Texas (fbs) | 17–31 | -14 | 0.752 | 44.12/13.68 | -10.85 |
| 401757166 / 2025-11-09 | away | Notre Dame (fbs) | 10–49 | -39 | 0.140 | 4.65/3.02 | -26.32 |
| 401762510 / 2025-11-15 | home | South Florida (fbs) | 41–38 | +3 | 0.776 | 35.55/12.52 | -7.17 |
| 401762516 / 2025-11-28 | away | Memphis (fbs) | 28–17 | +11 | 0.577 | 68.32/15.60 | -4.45 |
| 401762521 / 2025-12-13 | neutral | Army (fbs) | 17–16 | +1 | 0.615 | 83.09/13.93 | +2.54 |
| 401778333 / 2026-01-02 | neutral | Cincinnati (fbs) | 35–13 | +22 | 0.547 | 63.02/14.79 | -2.66 |

Answers to the original questions at every candidate:

**Does this excellent record receive disproportionate credit for an easier schedule?**

- At lambda=0, E[rank]=31.91, ordinal=22, SD=18.56, entropy=6.146 bits, central 80%=10-58; versus Record/Play the mean-rank changes are +0.00/-43.24. Actual evidence is 11-2, mean margin +6.46, 4 close wins and 0 close losses, standardized schedule probability 0.692, expected wins 9.23, wins minus expectation +1.77. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.25, E[rank]=47.80, ordinal=41, SD=19.47, entropy=6.300 bits, central 80%=23-75; versus Record/Play the mean-rank changes are +15.90/-27.34. Actual evidence is 11-2, mean margin +6.46, 4 close wins and 0 close losses, standardized schedule probability 0.692, expected wins 9.23, wins minus expectation +1.77. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.5, E[rank]=59.65, ordinal=51, SD=18.77, entropy=6.263 bits, central 80%=35-84; versus Record/Play the mean-rank changes are +27.75/-15.49. Actual evidence is 11-2, mean margin +6.46, 4 close wins and 0 close losses, standardized schedule probability 0.692, expected wins 9.23, wins minus expectation +1.77. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.75, E[rank]=68.54, ordinal=62, SD=17.39, entropy=6.155 bits, central 80%=45-91; versus Record/Play the mean-rank changes are +36.63/-6.60. Actual evidence is 11-2, mean margin +6.46, 4 close wins and 0 close losses, standardized schedule probability 0.692, expected wins 9.23, wins minus expectation +1.77. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=1, E[rank]=75.14, ordinal=66, SD=15.81, entropy=6.016 bits, central 80%=54-95; versus Record/Play the mean-rank changes are +43.24/+0.00. Actual evidence is 11-2, mean margin +6.46, 4 close wins and 0 close losses, standardized schedule probability 0.692, expected wins 9.23, wins minus expectation +1.77. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 39.53 to 31.91 (-7.62); ordinal changes -8, SD changes -3.13, entropy -0.253, 80% width -9, TV=0.149. Record changes 9-2 to 11-2, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 55.68 to 47.80 (-7.88); ordinal changes -2, SD changes -2.12, entropy -0.157, 80% width -5, TV=0.153. Record changes 9-2 to 11-2, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 67.29 to 59.65 (-7.63); ordinal changes -8, SD changes -1.32, entropy -0.097, 80% width -4, TV=0.159. Record changes 9-2 to 11-2, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 75.66 to 68.54 (-7.12); ordinal changes -4, SD changes -0.79, entropy -0.058, 80% width -1, TV=0.163. Record changes 9-2 to 11-2, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 81.69 to 75.14 (-6.55); ordinal changes -7, SD changes -0.55, entropy -0.044, 80% width -1, TV=0.167. Record changes 9-2 to 11-2, with 2 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 2 / 55 | -7.62 | -8 | -3.13 | 0.149 |
| 2025-prebowl → 2025-final | 0.25 | 2 / 55 | -7.88 | -2 | -2.12 | 0.153 |
| 2025-prebowl → 2025-final | 0.5 | 2 / 55 | -7.63 | -8 | -1.32 | 0.159 |
| 2025-prebowl → 2025-final | 0.75 | 2 / 55 | -7.12 | -4 | -0.79 | 0.163 |
| 2025-prebowl → 2025-final | 1 | 2 / 55 | -6.55 | -7 | -0.55 | 0.167 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2025 Purdue (2509)

Frozen categories: weak_schedule_pair.

### 2025-prebowl

Actual record **2–10**, mean margin **-13.08**; close wins/losses 0/3, dominant wins 1. Frozen standardized schedule probability **0.443** (larger is easier), expected wins **3.05**, wins minus expectation **-1.05**. Reference: preseason_context_v1.2, ordinal 115, band weak, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 107.12 | 105 | 19.40 | 6.154 | 96–122 | 80–130 | 61–134 | 0.000/0.000/0.000 | +0.00 / +15.42 |
| 0.25 | 100.76 | 99 | 18.33 | 6.174 | 89–114 | 76–123 | 59–130 | 0.000/0.000/0.000 | -6.36 / +9.06 |
| 0.5 | 96.55 | 92 | 16.98 | 6.090 | 86–109 | 73–117 | 59–125 | 0.000/0.000/0.000 | -10.57 / +4.86 |
| 0.75 | 93.73 | 90 | 15.84 | 5.997 | 84–105 | 72–113 | 59–120 | 0.000/0.000/0.000 | -13.39 / +2.03 |
| 1 | 91.69 | 84 | 14.92 | 5.915 | 82–102 | 72–110 | 59–117 | 0.000/0.000/0.000 | -15.42 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752801 / 2025-08-30 | home | Ball State (fbs) | 31–0 | +31 | 0.667 | 130.18/4.13 | +18.55 |
| 401752819 / 2025-09-06 | home | Southern Illinois (fcs) | 34–17 | +17 | 0.835 | 26.74/10.70 | +12.79 |
| 401752832 / 2025-09-13 | home | USC (fbs) | 17–33 | -16 | 0.144 | 15.80/7.47 | -20.31 |
| 401752848 / 2025-09-20 | away | Notre Dame (fbs) | 30–56 | -26 | 0.038 | 4.70/3.06 | -31.29 |
| 401752861 / 2025-10-04 | home | Illinois (fbs) | 27–43 | -16 | 0.203 | 33.83/12.19 | -14.26 |
| 401752864 / 2025-10-11 | away | Minnesota (fbs) | 20–27 | -7 | 0.144 | 72.30/16.24 | -10.45 |
| 401752874 / 2025-10-18 | away | Northwestern (fbs) | 0–19 | -19 | 0.301 | 64.31/16.15 | -12.48 |
| 401752884 / 2025-10-25 | home | Rutgers (fbs) | 24–27 | -3 | 0.235 | 66.40/16.88 | -5.68 |
| 401752890 / 2025-11-01 | away | Michigan (fbs) | 16–21 | -5 | 0.106 | 27.39/11.01 | -22.53 |
| 401752897 / 2025-11-08 | home | Ohio State (fbs) | 10–34 | -24 | 0.045 | 2.64/1.78 | -26.01 |
| 401752902 / 2025-11-16 | away | Washington (fbs) | 13–49 | -36 | 0.142 | 21.91/10.01 | -24.39 |
| 401752924 / 2025-11-29 | home | Indiana (fbs) | 3–56 | -53 | 0.191 | 4.77/2.56 | -24.98 |

Answers to the original questions at every candidate:

**Does the easier weak-team schedule reduce posterior SD and interval widths?**

- At lambda=0, E[rank]=107.12, ordinal=105, SD=19.40, entropy=6.154 bits, central 80%=80-130; versus Record/Play the mean-rank changes are +0.00/+15.42. The frozen peer UTEP is 2-10 with standardized schedule probability 0.708, E[rank]=130.76, SD=5.13, entropy=3.949, widths 50/80/95=8/13/19; this team's expected-rank gap is -23.64 and SD gap +14.28. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=100.76, ordinal=99, SD=18.33, entropy=6.174 bits, central 80%=76-123; versus Record/Play the mean-rank changes are -6.36/+9.06. The frozen peer UTEP is 2-10 with standardized schedule probability 0.708, E[rank]=129.83, SD=5.45, entropy=4.138, widths 50/80/95=8/13/21; this team's expected-rank gap is -29.07 and SD gap +12.88. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=96.55, ordinal=92, SD=16.98, entropy=6.090 bits, central 80%=73-117; versus Record/Play the mean-rank changes are -10.57/+4.86. The frozen peer UTEP is 2-10 with standardized schedule probability 0.708, E[rank]=128.88, SD=5.69, entropy=4.285, widths 50/80/95=8/15/22; this team's expected-rank gap is -32.33 and SD gap +11.29. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=93.73, ordinal=90, SD=15.84, entropy=5.997 bits, central 80%=72-113; versus Record/Play the mean-rank changes are -13.39/+2.03. The frozen peer UTEP is 2-10 with standardized schedule probability 0.708, E[rank]=127.94, SD=5.84, entropy=4.390, widths 50/80/95=8/16/23; this team's expected-rank gap is -34.21 and SD gap +9.99. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=91.69, ordinal=84, SD=14.92, entropy=5.915 bits, central 80%=72-110; versus Record/Play the mean-rank changes are -15.42/+0.00. The frozen peer UTEP is 2-10 with standardized schedule probability 0.708, E[rank]=127.04, SD=5.92, entropy=4.459, widths 50/80/95=8/16/24; this team's expected-rank gap is -35.34 and SD gap +9.00. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 107.12 to 106.02 (-1.10); ordinal changes -1, SD changes +0.46, entropy +0.043, 80% width +1, TV=0.023. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 100.76 to 99.55 (-1.21); ordinal changes -4, SD changes +0.28, entropy +0.027, 80% width +1, TV=0.027. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 96.55 to 95.39 (-1.16); ordinal changes -2, SD changes +0.18, entropy +0.017, 80% width +0, TV=0.028. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 93.73 to 92.67 (-1.06); ordinal changes -2, SD changes +0.12, entropy +0.013, 80% width +0, TV=0.027. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 91.69 to 90.76 (-0.94); ordinal changes -1, SD changes +0.10, entropy +0.010, 80% width +0, TV=0.025. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **2–10**, mean margin **-13.08**; close wins/losses 0/3, dominant wins 1. Frozen standardized schedule probability **0.443** (larger is easier), expected wins **3.05**, wins minus expectation **-1.05**. Reference: preseason_context_v1.2, ordinal 115, band weak, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 106.02 | 104 | 19.87 | 6.197 | 94–121 | 78–129 | 59–134 | 0.000/0.000/0.000 | +0.00 / +15.26 |
| 0.25 | 99.55 | 95 | 18.61 | 6.201 | 88–113 | 74–122 | 58–130 | 0.000/0.000/0.000 | -6.47 / +8.79 |
| 0.5 | 95.39 | 90 | 17.16 | 6.108 | 85–108 | 72–116 | 58–124 | 0.000/0.000/0.000 | -10.63 / +4.64 |
| 0.75 | 92.67 | 88 | 15.96 | 6.010 | 83–104 | 71–112 | 58–120 | 0.000/0.000/0.000 | -13.35 / +1.91 |
| 1 | 90.76 | 83 | 15.02 | 5.925 | 81–102 | 71–109 | 58–116 | 0.000/0.000/0.000 | -15.26 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401752801 / 2025-08-30 | home | Ball State (fbs) | 31–0 | +31 | 0.667 | 130.17/4.11 | +18.77 |
| 401752819 / 2025-09-06 | home | Southern Illinois (fcs) | 34–17 | +17 | 0.835 | 26.16/10.79 | +12.80 |
| 401752832 / 2025-09-13 | home | USC (fbs) | 17–33 | -16 | 0.144 | 17.40/7.66 | -19.46 |
| 401752848 / 2025-09-20 | away | Notre Dame (fbs) | 30–56 | -26 | 0.038 | 4.65/3.02 | -31.12 |
| 401752861 / 2025-10-04 | home | Illinois (fbs) | 27–43 | -16 | 0.203 | 31.55/10.98 | -14.66 |
| 401752864 / 2025-10-11 | away | Minnesota (fbs) | 20–27 | -7 | 0.144 | 72.80/15.52 | -10.04 |
| 401752874 / 2025-10-18 | away | Northwestern (fbs) | 0–19 | -19 | 0.301 | 60.58/15.42 | -13.14 |
| 401752884 / 2025-10-25 | home | Rutgers (fbs) | 24–27 | -3 | 0.235 | 65.23/16.85 | -5.69 |
| 401752890 / 2025-11-01 | away | Michigan (fbs) | 16–21 | -5 | 0.106 | 29.64/11.02 | -21.54 |
| 401752897 / 2025-11-08 | home | Ohio State (fbs) | 10–34 | -24 | 0.045 | 2.83/1.91 | -25.72 |
| 401752902 / 2025-11-16 | away | Washington (fbs) | 13–49 | -36 | 0.142 | 18.83/8.48 | -25.22 |
| 401752924 / 2025-11-29 | home | Indiana (fbs) | 3–56 | -53 | 0.191 | 3.46/1.78 | -25.41 |

Answers to the original questions at every candidate:

**Does the easier weak-team schedule reduce posterior SD and interval widths?**

- At lambda=0, E[rank]=106.02, ordinal=104, SD=19.87, entropy=6.197 bits, central 80%=78-129; versus Record/Play the mean-rank changes are +0.00/+15.26. The frozen peer UTEP is 2-10 with standardized schedule probability 0.708, E[rank]=130.63, SD=5.22, entropy=3.980, widths 50/80/95=7/13/20; this team's expected-rank gap is -24.61 and SD gap +14.65. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=99.55, ordinal=95, SD=18.61, entropy=6.201 bits, central 80%=74-122; versus Record/Play the mean-rank changes are -6.47/+8.79. The frozen peer UTEP is 2-10 with standardized schedule probability 0.708, E[rank]=129.80, SD=5.45, entropy=4.143, widths 50/80/95=8/13/21; this team's expected-rank gap is -30.25 and SD gap +13.16. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=95.39, ordinal=90, SD=17.16, entropy=6.108 bits, central 80%=72-116; versus Record/Play the mean-rank changes are -10.63/+4.64. The frozen peer UTEP is 2-10 with standardized schedule probability 0.708, E[rank]=128.99, SD=5.60, entropy=4.263, widths 50/80/95=8/14/22; this team's expected-rank gap is -33.60 and SD gap +11.56. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=92.67, ordinal=88, SD=15.96, entropy=6.010 bits, central 80%=71-112; versus Record/Play the mean-rank changes are -13.35/+1.91. The frozen peer UTEP is 2-10 with standardized schedule probability 0.708, E[rank]=128.24, SD=5.67, entropy=4.344, widths 50/80/95=8/15/22; this team's expected-rank gap is -35.57 and SD gap +10.29. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=90.76, ordinal=83, SD=15.02, entropy=5.925 bits, central 80%=71-109; versus Record/Play the mean-rank changes are -15.26/+0.00. The frozen peer UTEP is 2-10 with standardized schedule probability 0.708, E[rank]=127.54, SD=5.68, entropy=4.394, widths 50/80/95=9/15/23; this team's expected-rank gap is -36.79 and SD gap +9.33. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 107.12 to 106.02 (-1.10); ordinal changes -1, SD changes +0.46, entropy +0.043, 80% width +1, TV=0.023. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 100.76 to 99.55 (-1.21); ordinal changes -4, SD changes +0.28, entropy +0.027, 80% width +1, TV=0.027. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 96.55 to 95.39 (-1.16); ordinal changes -2, SD changes +0.18, entropy +0.017, 80% width +0, TV=0.028. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 93.73 to 92.67 (-1.06); ordinal changes -2, SD changes +0.12, entropy +0.013, 80% width +0, TV=0.027. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 91.69 to 90.76 (-0.94); ordinal changes -1, SD changes +0.10, entropy +0.010, 80% width +0, TV=0.025. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 0 / 55 | -1.10 | -1 | +0.46 | 0.023 |
| 2025-prebowl → 2025-final | 0.25 | 0 / 55 | -1.21 | -4 | +0.28 | 0.027 |
| 2025-prebowl → 2025-final | 0.5 | 0 / 55 | -1.16 | -2 | +0.18 | 0.028 |
| 2025-prebowl → 2025-final | 0.75 | 0 / 55 | -1.06 | -2 | +0.12 | 0.027 |
| 2025-prebowl → 2025-final | 1 | 0 / 55 | -0.94 | -1 | +0.10 | 0.025 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2025 SMU (2567)

Frozen categories: reference_boundary_25.

### 2025-prebowl

Actual record **8–4**, mean margin **+12.25**; close wins/losses 1/3, dominant wins 3. Frozen standardized schedule probability **0.519** (larger is easier), expected wins **7.49**, wins minus expectation **+0.51**. Reference: preseason_context_v1.2, ordinal 25, band strong, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 49.36 | 39 | 19.70 | 6.326 | 35–63 | 24–76 | 14–90 | 0.011/0.017/0.114 | +0.00 / +13.10 |
| 0.25 | 44.76 | 35 | 18.15 | 6.203 | 31–57 | 22–69 | 13–83 | 0.013/0.021/0.149 | -4.60 / +8.51 |
| 0.5 | 41.31 | 30 | 16.90 | 6.094 | 29–52 | 20–64 | 13–78 | 0.015/0.025/0.184 | -8.06 / +5.05 |
| 0.75 | 38.55 | 31 | 15.85 | 5.995 | 27–49 | 19–60 | 12–73 | 0.017/0.028/0.218 | -10.81 / +2.29 |
| 1 | 36.26 | 31 | 14.91 | 5.900 | 25–46 | 18–57 | 12–69 | 0.019/0.032/0.252 | -13.10 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401754521 / 2025-08-31 | home | East Texas A&M (fcs) | 42–13 | +29 | 0.966 | 97.56/11.78 | +45.60 |
| 401754525 / 2025-09-06 | home | Baylor (fbs) | 45–48 | -3 | 0.526 | 68.26/16.06 | +12.97 |
| 401754615 / 2025-09-13 | away | Missouri State (fbs) | 28–10 | +18 | 0.741 | 112.25/9.89 | +19.10 |
| 401754636 / 2025-09-20 | away | TCU (fbs) | 24–35 | -11 | 0.409 | 42.28/13.64 | -0.11 |
| 401754555 / 2025-10-04 | home | Syracuse (fbs) | 31–18 | +13 | 0.683 | 105.73/12.76 | +23.23 |
| 401754561 / 2025-10-11 | home | Stanford (fbs) | 34–10 | +24 | 0.846 | 97.84/12.95 | +20.72 |
| 401754565 / 2025-10-18 | away | Clemson (fbs) | 35–24 | +11 | 0.236 | 34.37/12.91 | -2.66 |
| 401752942 / 2025-10-25 | away | Wake Forest (fbs) | 12–13 | -1 | 0.688 | 68.66/15.58 | +6.80 |
| 401754581 / 2025-11-01 | home | Miami (fbs) | 26–20 | +6 | 0.540 | 10.53/5.64 | -8.00 |
| 401754584 / 2025-11-08 | away | Boston College (fbs) | 45–13 | +32 | 0.530 | 96.44/13.54 | +14.05 |
| 401754602 / 2025-11-22 | home | Louisville (fbs) | 38–6 | +32 | 0.582 | 38.84/14.40 | +5.00 |
| 401754609 / 2025-11-30 | away | California (fbs) | 35–38 | -3 | 0.749 | 95.13/13.63 | +13.67 |

Answers to the original questions at every candidate:

**Does movement near this prior-reference boundary reflect substantial evidence or closely spaced teams?**

- At lambda=0, E[rank]=49.36, ordinal=39, SD=19.70, entropy=6.326 bits, central 80%=24-76; versus Record/Play the mean-rank changes are +0.00/+13.10. Latent top10/12/25 mass is 0.011/0.017/0.114. Adjacent expected-rank gaps are 0.23575431783303458/1.5373042992951227. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=44.76, ordinal=35, SD=18.15, entropy=6.203 bits, central 80%=22-69; versus Record/Play the mean-rank changes are -4.60/+8.51. Latent top10/12/25 mass is 0.013/0.021/0.149. Adjacent expected-rank gaps are 0.590704409569696/1.1681883064333007. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=41.31, ordinal=30, SD=16.90, entropy=6.094 bits, central 80%=20-64; versus Record/Play the mean-rank changes are -8.06/+5.05. Latent top10/12/25 mass is 0.015/0.025/0.184. Adjacent expected-rank gaps are 1.529344084304853/0.34362847150162423. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=38.55, ordinal=31, SD=15.85, entropy=5.995 bits, central 80%=19-60; versus Record/Play the mean-rank changes are -10.81/+2.29. Latent top10/12/25 mass is 0.017/0.028/0.218. Adjacent expected-rank gaps are 0.5225616749452797/3.740497359705806. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=36.26, ordinal=31, SD=14.91, entropy=5.900 bits, central 80%=18-57; versus Record/Play the mean-rank changes are -13.10/+0.00. Latent top10/12/25 mass is 0.019/0.032/0.252. Adjacent expected-rank gaps are 0.12432108943608/4.826234854949554. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 49.36 to 41.85 (-7.51); ordinal changes -7, SD changes -1.05, entropy -0.093, 80% width -3, TV=0.148. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 44.76 to 39.00 (-5.76); ordinal changes -7, SD changes -1.10, entropy -0.103, 80% width -3, TV=0.126. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 41.31 to 36.86 (-4.45); ordinal changes -1, SD changes -1.13, entropy -0.110, 80% width -4, TV=0.107. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 38.55 to 35.16 (-3.39); ordinal changes -3, SD changes -1.13, entropy -0.116, 80% width -3, TV=0.088. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 36.26 to 33.76 (-2.49); ordinal changes -4, SD changes -1.11, entropy -0.119, 80% width -3, TV=0.071. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **9–4**, mean margin **+11.69**; close wins/losses 2/3, dominant wins 3. Frozen standardized schedule probability **0.523** (larger is easier), expected wins **8.18**, wins minus expectation **+0.82**. Reference: preseason_context_v1.2, ordinal 25, band strong, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 41.85 | 32 | 18.65 | 6.233 | 28–54 | 18–67 | 10–81 | 0.027/0.041/0.210 | +0.00 / +8.09 |
| 0.25 | 39.00 | 28 | 17.04 | 6.100 | 26–50 | 18–62 | 10–76 | 0.025/0.040/0.236 | -2.85 / +5.24 |
| 0.5 | 36.86 | 29 | 15.77 | 5.984 | 25–47 | 18–58 | 11–71 | 0.024/0.039/0.259 | -5.00 / +3.09 |
| 0.75 | 35.16 | 28 | 14.71 | 5.879 | 24–45 | 17–55 | 11–67 | 0.022/0.037/0.280 | -6.69 / +1.40 |
| 1 | 33.76 | 27 | 13.80 | 5.781 | 24–43 | 17–53 | 11–64 | 0.020/0.035/0.300 | -8.09 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401754521 / 2025-08-31 | home | East Texas A&M (fcs) | 42–13 | +29 | 0.966 | 97.70/11.74 | +45.95 |
| 401754525 / 2025-09-06 | home | Baylor (fbs) | 45–48 | -3 | 0.526 | 68.75/16.08 | +13.74 |
| 401754615 / 2025-09-13 | away | Missouri State (fbs) | 28–10 | +18 | 0.741 | 114.76/8.85 | +20.52 |
| 401754636 / 2025-09-20 | away | TCU (fbs) | 24–35 | -11 | 0.409 | 39.72/12.83 | -0.18 |
| 401754555 / 2025-10-04 | home | Syracuse (fbs) | 31–18 | +13 | 0.683 | 105.81/12.65 | +23.81 |
| 401754561 / 2025-10-11 | home | Stanford (fbs) | 34–10 | +24 | 0.846 | 96.93/13.07 | +21.05 |
| 401754565 / 2025-10-18 | away | Clemson (fbs) | 35–24 | +11 | 0.236 | 35.50/12.57 | -1.55 |
| 401752942 / 2025-10-25 | away | Wake Forest (fbs) | 12–13 | -1 | 0.688 | 62.14/15.06 | +5.84 |
| 401754581 / 2025-11-01 | home | Miami (fbs) | 26–20 | +6 | 0.540 | 8.07/3.85 | -9.25 |
| 401754584 / 2025-11-08 | away | Boston College (fbs) | 45–13 | +32 | 0.530 | 96.59/13.54 | +14.69 |
| 401754602 / 2025-11-22 | home | Louisville (fbs) | 38–6 | +32 | 0.582 | 38.01/13.71 | +5.46 |
| 401754609 / 2025-11-30 | away | California (fbs) | 35–38 | -3 | 0.749 | 94.66/13.12 | +14.11 |
| 401831583 / 2026-01-03 | neutral | Arizona (fbs) | 24–19 | +5 | 0.686 | 35.11/12.11 | +1.48 |

Answers to the original questions at every candidate:

**Does movement near this prior-reference boundary reflect substantial evidence or closely spaced teams?**

- At lambda=0, E[rank]=41.85, ordinal=32, SD=18.65, entropy=6.233 bits, central 80%=18-67; versus Record/Play the mean-rank changes are +0.00/+8.09. Latent top10/12/25 mass is 0.027/0.041/0.210. Adjacent expected-rank gaps are 0.9939521807317959/1.7751391596653718. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=39.00, ordinal=28, SD=17.04, entropy=6.100 bits, central 80%=18-62; versus Record/Play the mean-rank changes are -2.85/+5.24. Latent top10/12/25 mass is 0.025/0.040/0.236. Adjacent expected-rank gaps are 0.05580438202510152/0.4283635518967799. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=36.86, ordinal=29, SD=15.77, entropy=5.984 bits, central 80%=18-58; versus Record/Play the mean-rank changes are -5.00/+3.09. Latent top10/12/25 mass is 0.024/0.039/0.259. Adjacent expected-rank gaps are 0.9906826347155189/2.270176074753735. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=35.16, ordinal=28, SD=14.71, entropy=5.879 bits, central 80%=17-55; versus Record/Play the mean-rank changes are -6.69/+1.40. Latent top10/12/25 mass is 0.022/0.037/0.280. Adjacent expected-rank gaps are 0.7923636801381164/1.8397879480222272. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=33.76, ordinal=27, SD=13.80, entropy=5.781 bits, central 80%=17-53; versus Record/Play the mean-rank changes are -8.09/+0.00. Latent top10/12/25 mass is 0.020/0.035/0.300. Adjacent expected-rank gaps are 0.3977413202311837/1.214820953358121. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 49.36 to 41.85 (-7.51); ordinal changes -7, SD changes -1.05, entropy -0.093, 80% width -3, TV=0.148. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 44.76 to 39.00 (-5.76); ordinal changes -7, SD changes -1.10, entropy -0.103, 80% width -3, TV=0.126. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 41.31 to 36.86 (-4.45); ordinal changes -1, SD changes -1.13, entropy -0.110, 80% width -4, TV=0.107. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 38.55 to 35.16 (-3.39); ordinal changes -3, SD changes -1.13, entropy -0.116, 80% width -3, TV=0.088. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 36.26 to 33.76 (-2.49); ordinal changes -4, SD changes -1.11, entropy -0.119, 80% width -3, TV=0.071. Record changes 8-4 to 9-4, with 1 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 1 / 55 | -7.51 | -7 | -1.05 | 0.148 |
| 2025-prebowl → 2025-final | 0.25 | 1 / 55 | -5.76 | -7 | -1.10 | 0.126 |
| 2025-prebowl → 2025-final | 0.5 | 1 / 55 | -4.45 | -1 | -1.13 | 0.107 |
| 2025-prebowl → 2025-final | 0.75 | 1 / 55 | -3.39 | -3 | -1.13 | 0.088 |
| 2025-prebowl → 2025-final | 1 | 1 / 55 | -2.49 | -4 | -1.11 | 0.071 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2025 UTEP (2638)

Frozen categories: weak_schedule_pair.

### 2025-prebowl

Actual record **2–10**, mean margin **-7.08**; close wins/losses 0/3, dominant wins 0. Frozen standardized schedule probability **0.708** (larger is easier), expected wins **5.85**, wins minus expectation **-3.85**. Reference: preseason_context_v1.2, ordinal 114, band weak, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 130.76 | 135 | 5.13 | 3.949 | 128–135 | 124–136 | 118–136 | 0.000/0.000/0.000 | +0.00 / +3.72 |
| 0.25 | 129.83 | 133 | 5.45 | 4.138 | 127–134 | 123–135 | 116–136 | 0.000/0.000/0.000 | -0.92 / +2.80 |
| 0.5 | 128.88 | 132 | 5.69 | 4.285 | 126–133 | 121–135 | 115–136 | 0.000/0.000/0.000 | -1.88 / +1.84 |
| 0.75 | 127.94 | 130 | 5.84 | 4.390 | 125–132 | 120–135 | 114–136 | 0.000/0.000/0.000 | -2.82 / +0.90 |
| 1 | 127.04 | 130 | 5.92 | 4.459 | 124–131 | 119–134 | 113–136 | 0.000/0.000/0.000 | -3.72 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401757225 / 2025-08-30 | away | Utah State (fbs) | 16–28 | -12 | 0.443 | 88.72/14.50 | -18.89 |
| 401757231 / 2025-09-07 | home | UT Martin (fcs) | 42–17 | +25 | 0.836 | 73.11/14.46 | +15.17 |
| 401752706 / 2025-09-13 | away | Texas (fbs) | 10–27 | -17 | 0.045 | 15.55/8.14 | -35.93 |
| 401757267 / 2025-09-21 | home | UL Monroe (fbs) | 25–31 | -6 | 0.522 | 129.22/4.19 | +4.55 |
| 401757272 / 2025-09-28 | home | Louisiana Tech (fbs) | 11–30 | -19 | 0.555 | 95.93/13.04 | -10.86 |
| 401757277 / 2025-10-09 | home | Liberty (fbs) | 8–19 | -11 | 0.435 | 112.84/9.44 | -5.15 |
| 401757284 / 2025-10-15 | neutral | Sam Houston (fbs) | 35–17 | +18 | 0.369 | 130.83/3.32 | +2.66 |
| 401757289 / 2025-10-29 | away | Kennesaw State (fbs) | 20–33 | -13 | 0.640 | 103.88/11.83 | -14.75 |
| 401757297 / 2025-11-08 | home | Jacksonville State (fbs) | 27–30 | -3 | 0.509 | 112.22/9.39 | -5.45 |
| 401757304 / 2025-11-15 | away | Missouri State (fbs) | 24–38 | -14 | 0.389 | 112.25/9.89 | -11.63 |
| 401757309 / 2025-11-22 | home | New Mexico State (fbs) | 31–34 | -3 | 0.697 | 126.82/5.66 | +2.84 |
| 401757310 / 2025-11-29 | away | Delaware (fbs) | 31–61 | -30 | 0.408 | 120.38/8.09 | -7.51 |

Answers to the original questions at every candidate:

**Does the easier weak-team schedule reduce posterior SD and interval widths?**

- At lambda=0, E[rank]=130.76, ordinal=135, SD=5.13, entropy=3.949 bits, central 80%=124-136; versus Record/Play the mean-rank changes are +0.00/+3.72. The frozen peer Purdue is 2-10 with standardized schedule probability 0.443, E[rank]=107.12, SD=19.40, entropy=6.154, widths 50/80/95=27/51/74; this team's expected-rank gap is +23.64 and SD gap -14.28. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=129.83, ordinal=133, SD=5.45, entropy=4.138 bits, central 80%=123-135; versus Record/Play the mean-rank changes are -0.92/+2.80. The frozen peer Purdue is 2-10 with standardized schedule probability 0.443, E[rank]=100.76, SD=18.33, entropy=6.174, widths 50/80/95=26/48/72; this team's expected-rank gap is +29.07 and SD gap -12.88. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=128.88, ordinal=132, SD=5.69, entropy=4.285 bits, central 80%=121-135; versus Record/Play the mean-rank changes are -1.88/+1.84. The frozen peer Purdue is 2-10 with standardized schedule probability 0.443, E[rank]=96.55, SD=16.98, entropy=6.090, widths 50/80/95=24/45/67; this team's expected-rank gap is +32.33 and SD gap -11.29. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=127.94, ordinal=130, SD=5.84, entropy=4.390 bits, central 80%=120-135; versus Record/Play the mean-rank changes are -2.82/+0.90. The frozen peer Purdue is 2-10 with standardized schedule probability 0.443, E[rank]=93.73, SD=15.84, entropy=5.997, widths 50/80/95=22/42/62; this team's expected-rank gap is +34.21 and SD gap -9.99. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=127.04, ordinal=130, SD=5.92, entropy=4.459 bits, central 80%=119-134; versus Record/Play the mean-rank changes are -3.72/+0.00. The frozen peer Purdue is 2-10 with standardized schedule probability 0.443, E[rank]=91.69, SD=14.92, entropy=5.915, widths 50/80/95=21/39/59; this team's expected-rank gap is +35.34 and SD gap -9.00. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 130.76 to 130.63 (-0.12); ordinal changes +0, SD changes +0.09, entropy +0.030, 80% width +0, TV=0.010. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 129.83 to 129.80 (-0.03); ordinal changes +0, SD changes +0.00, entropy +0.005, 80% width +0, TV=0.003. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 128.88 to 128.99 (+0.11); ordinal changes +0, SD changes -0.09, entropy -0.022, 80% width -1, TV=0.007. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 127.94 to 128.24 (+0.30); ordinal changes +0, SD changes -0.18, entropy -0.046, 80% width -1, TV=0.020. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 127.04 to 127.54 (+0.51); ordinal changes +0, SD changes -0.24, entropy -0.065, 80% width -1, TV=0.035. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **2–10**, mean margin **-7.08**; close wins/losses 0/3, dominant wins 0. Frozen standardized schedule probability **0.708** (larger is easier), expected wins **5.85**, wins minus expectation **-3.85**. Reference: preseason_context_v1.2, ordinal 114, band weak, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 130.63 | 135 | 5.22 | 3.980 | 128–134 | 124–136 | 117–136 | 0.000/0.000/0.000 | +0.00 / +3.09 |
| 0.25 | 129.80 | 133 | 5.45 | 4.143 | 127–134 | 123–135 | 116–136 | 0.000/0.000/0.000 | -0.83 / +2.26 |
| 0.5 | 128.99 | 132 | 5.60 | 4.263 | 126–133 | 122–135 | 115–136 | 0.000/0.000/0.000 | -1.64 / +1.45 |
| 0.75 | 128.24 | 130 | 5.67 | 4.344 | 125–132 | 121–135 | 115–136 | 0.000/0.000/0.000 | -2.40 / +0.69 |
| 1 | 127.54 | 130 | 5.68 | 4.394 | 124–132 | 120–134 | 114–136 | 0.000/0.000/0.000 | -3.09 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401757225 / 2025-08-30 | away | Utah State (fbs) | 16–28 | -12 | 0.443 | 91.67/13.54 | -18.44 |
| 401757231 / 2025-09-07 | home | UT Martin (fcs) | 42–17 | +25 | 0.836 | 73.33/14.44 | +15.03 |
| 401752706 / 2025-09-13 | away | Texas (fbs) | 10–27 | -17 | 0.045 | 15.20/7.59 | -36.17 |
| 401757267 / 2025-09-21 | home | UL Monroe (fbs) | 25–31 | -6 | 0.522 | 129.21/4.20 | +4.26 |
| 401757272 / 2025-09-28 | home | Louisiana Tech (fbs) | 11–30 | -19 | 0.555 | 96.94/12.28 | -10.89 |
| 401757277 / 2025-10-09 | home | Liberty (fbs) | 8–19 | -11 | 0.435 | 113.62/9.33 | -5.06 |
| 401757284 / 2025-10-15 | neutral | Sam Houston (fbs) | 35–17 | +18 | 0.369 | 130.87/3.30 | +2.42 |
| 401757289 / 2025-10-29 | away | Kennesaw State (fbs) | 20–33 | -13 | 0.640 | 109.62/10.22 | -13.03 |
| 401757297 / 2025-11-08 | home | Jacksonville State (fbs) | 27–30 | -3 | 0.509 | 112.14/9.05 | -5.82 |
| 401757304 / 2025-11-15 | away | Missouri State (fbs) | 24–38 | -14 | 0.389 | 114.76/8.85 | -10.84 |
| 401757309 / 2025-11-22 | home | New Mexico State (fbs) | 31–34 | -3 | 0.697 | 127.31/5.46 | +2.91 |
| 401757310 / 2025-11-29 | away | Delaware (fbs) | 31–61 | -30 | 0.408 | 119.07/7.94 | -8.60 |

Answers to the original questions at every candidate:

**Does the easier weak-team schedule reduce posterior SD and interval widths?**

- At lambda=0, E[rank]=130.63, ordinal=135, SD=5.22, entropy=3.980 bits, central 80%=124-136; versus Record/Play the mean-rank changes are +0.00/+3.09. The frozen peer Purdue is 2-10 with standardized schedule probability 0.443, E[rank]=106.02, SD=19.87, entropy=6.197, widths 50/80/95=28/52/76; this team's expected-rank gap is +24.61 and SD gap -14.65. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=129.80, ordinal=133, SD=5.45, entropy=4.143 bits, central 80%=123-135; versus Record/Play the mean-rank changes are -0.83/+2.26. The frozen peer Purdue is 2-10 with standardized schedule probability 0.443, E[rank]=99.55, SD=18.61, entropy=6.201, widths 50/80/95=26/49/73; this team's expected-rank gap is +30.25 and SD gap -13.16. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=128.99, ordinal=132, SD=5.60, entropy=4.263 bits, central 80%=122-135; versus Record/Play the mean-rank changes are -1.64/+1.45. The frozen peer Purdue is 2-10 with standardized schedule probability 0.443, E[rank]=95.39, SD=17.16, entropy=6.108, widths 50/80/95=24/45/67; this team's expected-rank gap is +33.60 and SD gap -11.56. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=128.24, ordinal=130, SD=5.67, entropy=4.344 bits, central 80%=121-135; versus Record/Play the mean-rank changes are -2.40/+0.69. The frozen peer Purdue is 2-10 with standardized schedule probability 0.443, E[rank]=92.67, SD=15.96, entropy=6.010, widths 50/80/95=22/42/63; this team's expected-rank gap is +35.57 and SD gap -10.29. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=127.54, ordinal=130, SD=5.68, entropy=4.394 bits, central 80%=120-134; versus Record/Play the mean-rank changes are -3.09/+0.00. The frozen peer Purdue is 2-10 with standardized schedule probability 0.443, E[rank]=90.76, SD=15.02, entropy=5.925, widths 50/80/95=22/39/59; this team's expected-rank gap is +36.79 and SD gap -9.33. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 130.76 to 130.63 (-0.12); ordinal changes +0, SD changes +0.09, entropy +0.030, 80% width +0, TV=0.010. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 129.83 to 129.80 (-0.03); ordinal changes +0, SD changes +0.00, entropy +0.005, 80% width +0, TV=0.003. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 128.88 to 128.99 (+0.11); ordinal changes +0, SD changes -0.09, entropy -0.022, 80% width -1, TV=0.007. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 127.94 to 128.24 (+0.30); ordinal changes +0, SD changes -0.18, entropy -0.046, 80% width -1, TV=0.020. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 127.04 to 127.54 (+0.51); ordinal changes +0, SD changes -0.24, entropy -0.065, 80% width -1, TV=0.035. Record changes 2-10 to 2-10, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 0 / 55 | -0.12 | +0 | +0.09 | 0.010 |
| 2025-prebowl → 2025-final | 0.25 | 0 / 55 | -0.03 | +0 | +0.00 | 0.003 |
| 2025-prebowl → 2025-final | 0.5 | 0 / 55 | +0.11 | +0 | -0.09 | 0.007 |
| 2025-prebowl → 2025-final | 0.75 | 0 / 55 | +0.30 | +0 | -0.18 | 0.020 |
| 2025-prebowl → 2025-final | 1 | 0 / 55 | +0.51 | +0 | -0.24 | 0.035 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2025 Wisconsin (275)

Frozen categories: middling_record_difficult_schedule.

### 2025-prebowl

Actual record **4–8**, mean margin **-8.75**; close wins/losses 1/0, dominant wins 1. Frozen standardized schedule probability **0.398** (larger is easier), expected wins **5.48**, wins minus expectation **-1.48**. Reference: preseason_context_v1.2, ordinal 40, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 70.89 | 62 | 23.54 | 6.581 | 53–88 | 39–102 | 27–115 | 0.000/0.001/0.021 | +0.00 / +1.84 |
| 0.25 | 70.02 | 62 | 21.52 | 6.457 | 54–86 | 41–98 | 29–110 | 0.000/0.000/0.013 | -0.87 / +0.98 |
| 0.5 | 69.56 | 62 | 20.05 | 6.358 | 55–84 | 43–96 | 31–107 | 0.000/0.000/0.008 | -1.33 / +0.52 |
| 0.75 | 69.26 | 62 | 18.90 | 6.274 | 56–83 | 44–94 | 33–105 | 0.000/0.000/0.005 | -1.62 / +0.22 |
| 1 | 69.04 | 62 | 17.96 | 6.203 | 56–82 | 45–92 | 34–103 | 0.000/0.000/0.004 | -1.84 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401762791 / 2025-08-29 | home | Miami (OH) (fbs) | 17–0 | +17 | 0.759 | 96.20/12.42 | +11.63 |
| 401752820 / 2025-09-06 | home | Middle Tennessee (fbs) | 42–10 | +32 | 0.893 | 127.40/5.08 | +24.29 |
| 401752696 / 2025-09-13 | away | Alabama (fbs) | 14–38 | -24 | 0.188 | 11.28/5.52 | -21.12 |
| 401752845 / 2025-09-20 | home | Maryland (fbs) | 10–27 | -17 | 0.693 | 85.26/15.01 | +8.44 |
| 401752856 / 2025-10-04 | away | Michigan (fbs) | 10–24 | -14 | 0.342 | 27.39/11.01 | -14.18 |
| 401752869 / 2025-10-11 | home | Iowa (fbs) | 0–37 | -37 | 0.406 | 16.11/7.90 | -12.54 |
| 401752877 / 2025-10-18 | home | Ohio State (fbs) | 0–34 | -34 | 0.176 | 2.64/1.78 | -19.95 |
| 401752886 / 2025-10-25 | away | Oregon (fbs) | 7–21 | -14 | 0.180 | 5.23/3.33 | -24.56 |
| 401752899 / 2025-11-08 | home | Washington (fbs) | 13–10 | +3 | 0.540 | 21.91/10.01 | -10.06 |
| 401752901 / 2025-11-15 | away | Indiana (fbs) | 7–31 | -24 | 0.370 | 4.77/2.56 | -24.82 |
| 401752915 / 2025-11-23 | home | Illinois (fbs) | 27–10 | +17 | 0.511 | 33.83/12.19 | -5.64 |
| 401752918 / 2025-11-29 | away | Minnesota (fbs) | 7–17 | -10 | 0.423 | 72.30/16.24 | -1.31 |

Answers to the original questions at every candidate:

**Does opponent evidence explain this team's ordinary record?**

- At lambda=0, E[rank]=70.89, ordinal=62, SD=23.54, entropy=6.581 bits, central 80%=39-102; versus Record/Play the mean-rank changes are +0.00/+1.84. Actual evidence is 4-8, mean margin -8.75, 1 close wins and 0 close losses, standardized schedule probability 0.398, expected wins 5.48, wins minus expectation -1.48. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.25, E[rank]=70.02, ordinal=62, SD=21.52, entropy=6.457 bits, central 80%=41-98; versus Record/Play the mean-rank changes are -0.87/+0.98. Actual evidence is 4-8, mean margin -8.75, 1 close wins and 0 close losses, standardized schedule probability 0.398, expected wins 5.48, wins minus expectation -1.48. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.5, E[rank]=69.56, ordinal=62, SD=20.05, entropy=6.358 bits, central 80%=43-96; versus Record/Play the mean-rank changes are -1.33/+0.52. Actual evidence is 4-8, mean margin -8.75, 1 close wins and 0 close losses, standardized schedule probability 0.398, expected wins 5.48, wins minus expectation -1.48. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.75, E[rank]=69.26, ordinal=62, SD=18.90, entropy=6.274 bits, central 80%=44-94; versus Record/Play the mean-rank changes are -1.62/+0.22. Actual evidence is 4-8, mean margin -8.75, 1 close wins and 0 close losses, standardized schedule probability 0.398, expected wins 5.48, wins minus expectation -1.48. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=1, E[rank]=69.04, ordinal=62, SD=17.96, entropy=6.203 bits, central 80%=45-92; versus Record/Play the mean-rank changes are -1.84/+0.00. Actual evidence is 4-8, mean margin -8.75, 1 close wins and 0 close losses, standardized schedule probability 0.398, expected wins 5.48, wins minus expectation -1.48. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 70.89 to 69.18 (-1.71); ordinal changes -6, SD changes +0.20, entropy +0.011, 80% width +1, TV=0.031. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 70.02 to 69.00 (-1.02); ordinal changes -3, SD changes +0.11, entropy +0.007, 80% width +1, TV=0.020. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 69.56 to 69.11 (-0.45); ordinal changes +0, SD changes +0.06, entropy +0.004, 80% width +1, TV=0.010. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 69.26 to 69.32 (+0.06); ordinal changes +1, SD changes +0.02, entropy +0.002, 80% width +0, TV=0.001. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 69.04 to 69.55 (+0.51); ordinal changes +1, SD changes -0.01, entropy -0.001, 80% width +1, TV=0.011. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### 2025-final

Actual record **4–8**, mean margin **-8.75**; close wins/losses 1/0, dominant wins 1. Frozen standardized schedule probability **0.398** (larger is easier), expected wins **5.48**, wins minus expectation **-1.48**. Reference: preseason_context_v1.2, ordinal 40, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 69.18 | 56 | 23.74 | 6.592 | 51–87 | 37–101 | 25–114 | 0.001/0.001/0.025 | +0.00 / -0.38 |
| 0.25 | 69.00 | 59 | 21.63 | 6.464 | 53–85 | 40–98 | 29–110 | 0.000/0.000/0.015 | -0.18 / -0.55 |
| 0.5 | 69.11 | 62 | 20.11 | 6.362 | 55–84 | 42–96 | 31–107 | 0.000/0.000/0.009 | -0.07 / -0.45 |
| 0.75 | 69.32 | 63 | 18.92 | 6.276 | 56–83 | 44–94 | 33–105 | 0.000/0.000/0.005 | +0.14 / -0.24 |
| 1 | 69.55 | 63 | 17.95 | 6.202 | 57–82 | 45–93 | 35–103 | 0.000/0.000/0.003 | +0.38 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401762791 / 2025-08-29 | home | Miami (OH) (fbs) | 17–0 | +17 | 0.759 | 98.73/11.55 | +12.30 |
| 401752820 / 2025-09-06 | home | Middle Tennessee (fbs) | 42–10 | +32 | 0.893 | 127.78/4.93 | +24.38 |
| 401752696 / 2025-09-13 | away | Alabama (fbs) | 14–38 | -24 | 0.188 | 13.97/6.01 | -19.84 |
| 401752845 / 2025-09-20 | home | Maryland (fbs) | 10–27 | -17 | 0.693 | 85.01/15.11 | +8.24 |
| 401752856 / 2025-10-04 | away | Michigan (fbs) | 10–24 | -14 | 0.342 | 29.64/11.02 | -13.45 |
| 401752869 / 2025-10-11 | home | Iowa (fbs) | 0–37 | -37 | 0.406 | 14.71/7.09 | -13.28 |
| 401752877 / 2025-10-18 | home | Ohio State (fbs) | 0–34 | -34 | 0.176 | 2.83/1.91 | -19.89 |
| 401752886 / 2025-10-25 | away | Oregon (fbs) | 7–21 | -14 | 0.180 | 5.29/3.20 | -24.60 |
| 401752899 / 2025-11-08 | home | Washington (fbs) | 13–10 | +3 | 0.540 | 18.83/8.48 | -11.42 |
| 401752901 / 2025-11-15 | away | Indiana (fbs) | 7–31 | -24 | 0.370 | 3.46/1.78 | -25.73 |
| 401752915 / 2025-11-23 | home | Illinois (fbs) | 27–10 | +17 | 0.511 | 31.55/10.98 | -6.49 |
| 401752918 / 2025-11-29 | away | Minnesota (fbs) | 7–17 | -10 | 0.423 | 72.80/15.52 | -1.32 |

Answers to the original questions at every candidate:

**Does opponent evidence explain this team's ordinary record?**

- At lambda=0, E[rank]=69.18, ordinal=56, SD=23.74, entropy=6.592 bits, central 80%=37-101; versus Record/Play the mean-rank changes are +0.00/-0.38. Actual evidence is 4-8, mean margin -8.75, 1 close wins and 0 close losses, standardized schedule probability 0.398, expected wins 5.48, wins minus expectation -1.48. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.25, E[rank]=69.00, ordinal=59, SD=21.63, entropy=6.464 bits, central 80%=40-98; versus Record/Play the mean-rank changes are -0.18/-0.55. Actual evidence is 4-8, mean margin -8.75, 1 close wins and 0 close losses, standardized schedule probability 0.398, expected wins 5.48, wins minus expectation -1.48. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.5, E[rank]=69.11, ordinal=62, SD=20.11, entropy=6.362 bits, central 80%=42-96; versus Record/Play the mean-rank changes are -0.07/-0.45. Actual evidence is 4-8, mean margin -8.75, 1 close wins and 0 close losses, standardized schedule probability 0.398, expected wins 5.48, wins minus expectation -1.48. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=0.75, E[rank]=69.32, ordinal=63, SD=18.92, entropy=6.276 bits, central 80%=44-94; versus Record/Play the mean-rank changes are +0.14/-0.24. Actual evidence is 4-8, mean margin -8.75, 1 close wins and 0 close losses, standardized schedule probability 0.398, expected wins 5.48, wins minus expectation -1.48. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.
- At lambda=1, E[rank]=69.55, ordinal=63, SD=17.95, entropy=6.202 bits, central 80%=45-93; versus Record/Play the mean-rank changes are +0.38/+0.00. Actual evidence is 4-8, mean margin -8.75, 1 close wins and 0 close losses, standardized schedule probability 0.398, expected wins 5.48, wins minus expectation -1.48. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The primary expectation uses frozen preseason priors; original prospective availability remains unknown.

**How do uncertainty and postseason evidence change the pre-bowl interpretation?**

- At lambda=0, pre-bowl-to-final E[rank] moves 70.89 to 69.18 (-1.71); ordinal changes -6, SD changes +0.20, entropy +0.011, 80% width +1, TV=0.031. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.25, pre-bowl-to-final E[rank] moves 70.02 to 69.00 (-1.02); ordinal changes -3, SD changes +0.11, entropy +0.007, 80% width +1, TV=0.020. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.5, pre-bowl-to-final E[rank] moves 69.56 to 69.11 (-0.45); ordinal changes +0, SD changes +0.06, entropy +0.004, 80% width +1, TV=0.010. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=0.75, pre-bowl-to-final E[rank] moves 69.26 to 69.32 (+0.06); ordinal changes +1, SD changes +0.02, entropy +0.002, 80% width +0, TV=0.001. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.
- At lambda=1, pre-bowl-to-final E[rank] moves 69.04 to 69.55 (+0.51); ordinal changes +1, SD changes -0.01, entropy -0.001, 80% width +1, TV=0.011. Record changes 4-8 to 4-8, with 0 added own observations and 55 network observations. These are total network changes, including opponent updates; even unchanged own records can move, and own games do not supply an additive explanation.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2025-prebowl → 2025-final | 0 | 0 / 55 | -1.71 | -6 | +0.20 | 0.031 |
| 2025-prebowl → 2025-final | 0.25 | 0 / 55 | -1.02 | -3 | +0.11 | 0.020 |
| 2025-prebowl → 2025-final | 0.5 | 0 / 55 | -0.45 | +0 | +0.06 | 0.010 |
| 2025-prebowl → 2025-final | 0.75 | 0 / 55 | +0.06 | +1 | +0.02 | 0.001 |
| 2025-prebowl → 2025-final | 1 | 0 / 55 | +0.51 | +1 | -0.01 | 0.011 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Alabama (333)

Frozen categories: elite.

### 2026-09-08

Actual record **1–0**, mean margin **+38.00**; close wins/losses 0/0, dominant wins 1. Frozen standardized schedule probability **0.695** (larger is easier), expected wins **0.93**, wins minus expectation **+0.07**. Reference: retained_context_v1.4_posterior, ordinal 11, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 51.58 | 23 | 35.07 | 6.944 | 21–78 | 9–103 | 2–124 | 0.124/0.148/0.294 | +0.00 / +20.13 |
| 0.25 | 45.27 | 15 | 33.52 | 6.819 | 17–69 | 7–96 | 2–119 | 0.161/0.191/0.360 | -6.30 / +13.82 |
| 0.5 | 39.76 | 10 | 31.56 | 6.672 | 13–60 | 5–88 | 2–113 | 0.200/0.234/0.424 | -11.82 / +8.31 |
| 0.75 | 35.16 | 11 | 29.47 | 6.517 | 11–53 | 4–80 | 1–107 | 0.236/0.275/0.482 | -16.41 / +3.72 |
| 1 | 31.45 | 11 | 27.45 | 6.368 | 10–46 | 4–73 | 1–101 | 0.270/0.313/0.533 | -20.13 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856634 / 2026-09-05 | home | East Carolina (fbs) | 48–10 | +38 | 0.934 | 87.34/29.30 | +25.05 |

Answers to the original questions at every candidate:

**How do outcomes versus margins distinguish elite accomplishments?**

- At lambda=0, E[rank]=51.58, ordinal=23, SD=35.07, entropy=6.944 bits, central 80%=9-103; versus Record/Play the mean-rank changes are +0.00/+20.13. Actual evidence is 1-0, mean margin +38.00, 0 close wins and 0 close losses, standardized schedule probability 0.695, expected wins 0.93, wins minus expectation +0.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=45.27, ordinal=15, SD=33.52, entropy=6.819 bits, central 80%=7-96; versus Record/Play the mean-rank changes are -6.30/+13.82. Actual evidence is 1-0, mean margin +38.00, 0 close wins and 0 close losses, standardized schedule probability 0.695, expected wins 0.93, wins minus expectation +0.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=39.76, ordinal=10, SD=31.56, entropy=6.672 bits, central 80%=5-88; versus Record/Play the mean-rank changes are -11.82/+8.31. Actual evidence is 1-0, mean margin +38.00, 0 close wins and 0 close losses, standardized schedule probability 0.695, expected wins 0.93, wins minus expectation +0.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=35.16, ordinal=11, SD=29.47, entropy=6.517 bits, central 80%=4-80; versus Record/Play the mean-rank changes are -16.41/+3.72. Actual evidence is 1-0, mean margin +38.00, 0 close wins and 0 close losses, standardized schedule probability 0.695, expected wins 0.93, wins minus expectation +0.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=31.45, ordinal=11, SD=27.45, entropy=6.368 bits, central 80%=4-73; versus Record/Play the mean-rank changes are -20.13/+0.00. Actual evidence is 1-0, mean margin +38.00, 0 close wins and 0 close losses, standardized schedule probability 0.695, expected wins 0.93, wins minus expectation +0.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-09-27

Actual record **4–0**, mean margin **+27.75**; close wins/losses 0/0, dominant wins 3. Frozen standardized schedule probability **0.444** (larger is easier), expected wins **3.49**, wins minus expectation **+0.51**. Reference: retained_context_v1.4_posterior, ordinal 2, band elite, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 17.76 | 4 | 16.91 | 5.548 | 5–25 | 2–41 | 1–64 | 0.444/0.506/0.763 | +0.00 / +10.80 |
| 0.25 | 13.18 | 3 | 13.21 | 5.103 | 4–18 | 2–30 | 1–50 | 0.562/0.628/0.860 | -4.58 / +6.22 |
| 0.5 | 10.20 | 3 | 10.35 | 4.716 | 3–13 | 2–23 | 1–39 | 0.660/0.725/0.921 | -7.56 / +3.24 |
| 0.75 | 8.26 | 2 | 8.27 | 4.394 | 3–11 | 1–18 | 1–31 | 0.739/0.799/0.956 | -9.50 / +1.30 |
| 1 | 6.96 | 2 | 6.80 | 4.129 | 2–9 | 1–15 | 1–25 | 0.799/0.853/0.975 | -10.80 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856634 / 2026-09-05 | home | East Carolina (fbs) | 48–10 | +38 | 0.966 | 87.30/23.30 | +28.02 |
| 401856674 / 2026-09-12 | away | Kentucky (fbs) | 45–17 | +28 | 0.761 | 35.16/18.97 | +10.69 |
| 401856685 / 2026-09-19 | home | Florida State (fbs) | 50–36 | +14 | 0.887 | 42.48/22.21 | +18.67 |
| 401856696 / 2026-09-26 | home | South Carolina (fbs) | 49–18 | +31 | 0.871 | 36.01/19.36 | +17.29 |

Answers to the original questions at every candidate:

**How do outcomes versus margins distinguish elite accomplishments?**

- At lambda=0, E[rank]=17.76, ordinal=4, SD=16.91, entropy=5.548 bits, central 80%=2-41; versus Record/Play the mean-rank changes are +0.00/+10.80. Actual evidence is 4-0, mean margin +27.75, 0 close wins and 0 close losses, standardized schedule probability 0.444, expected wins 3.49, wins minus expectation +0.51. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=13.18, ordinal=3, SD=13.21, entropy=5.103 bits, central 80%=2-30; versus Record/Play the mean-rank changes are -4.58/+6.22. Actual evidence is 4-0, mean margin +27.75, 0 close wins and 0 close losses, standardized schedule probability 0.444, expected wins 3.49, wins minus expectation +0.51. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=10.20, ordinal=3, SD=10.35, entropy=4.716 bits, central 80%=2-23; versus Record/Play the mean-rank changes are -7.56/+3.24. Actual evidence is 4-0, mean margin +27.75, 0 close wins and 0 close losses, standardized schedule probability 0.444, expected wins 3.49, wins minus expectation +0.51. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=8.26, ordinal=2, SD=8.27, entropy=4.394 bits, central 80%=1-18; versus Record/Play the mean-rank changes are -9.50/+1.30. Actual evidence is 4-0, mean margin +27.75, 0 close wins and 0 close losses, standardized schedule probability 0.444, expected wins 3.49, wins minus expectation +0.51. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=6.96, ordinal=2, SD=6.80, entropy=4.129 bits, central 80%=1-15; versus Record/Play the mean-rank changes are -10.80/+0.00. Actual evidence is 4-0, mean margin +27.75, 0 close wins and 0 close losses, standardized schedule probability 0.444, expected wins 3.49, wins minus expectation +0.51. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-10-25T12-59-22.246522Z

Actual record **6–2**, mean margin **+17.00**; close wins/losses 0/1, dominant wins 4. Frozen standardized schedule probability **0.279** (larger is easier), expected wins **5.97**, wins minus expectation **+0.03**. Reference: retained_context_v1.4_posterior, ordinal 5, band elite, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 13.55 | 7 | 11.17 | 5.108 | 6–18 | 3–28 | 1–44 | 0.491/0.576/0.876 | +0.00 / +8.46 |
| 0.25 | 9.75 | 5 | 8.11 | 4.620 | 4–13 | 2–20 | 1–31 | 0.647/0.730/0.952 | -3.79 / +4.67 |
| 0.5 | 7.47 | 4 | 6.08 | 4.215 | 3–10 | 2–15 | 1–23 | 0.767/0.838/0.983 | -6.07 / +2.39 |
| 0.75 | 6.04 | 5 | 4.79 | 3.887 | 3–8 | 1–12 | 1–18 | 0.851/0.907/0.995 | -7.50 / +0.96 |
| 1 | 5.09 | 5 | 3.95 | 3.615 | 2–7 | 1–10 | 1–15 | 0.906/0.947/0.998 | -8.46 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856634 / 2026-09-05 | home | East Carolina (fbs) | 48–10 | +38 | 0.975 | 91.21/17.92 | +29.20 |
| 401856674 / 2026-09-12 | away | Kentucky (fbs) | 45–17 | +28 | 0.672 | 19.91/10.61 | +6.57 |
| 401856685 / 2026-09-19 | home | Florida State (fbs) | 50–36 | +14 | 0.848 | 27.04/14.92 | +15.51 |
| 401856696 / 2026-09-26 | home | South Carolina (fbs) | 49–18 | +31 | 0.826 | 23.26/12.93 | +14.15 |
| 401856707 / 2026-10-03 | away | Mississippi State (fbs) | 56–23 | +33 | 0.761 | 28.71/13.19 | +10.35 |
| 401856712 / 2026-10-10 | home | Georgia (fbs) | 17–20 | -3 | 0.547 | 4.15/3.10 | +1.68 |
| 401856723 / 2026-10-17 | away | Tennessee (fbs) | 10–26 | -16 | 0.524 | 10.70/6.67 | +0.90 |
| 401856727 / 2026-10-24 | home | Texas A&M (fbs) | 31–20 | +11 | 0.816 | 21.04/10.61 | +13.47 |

Answers to the original questions at every candidate:

**How do outcomes versus margins distinguish elite accomplishments?**

- At lambda=0, E[rank]=13.55, ordinal=7, SD=11.17, entropy=5.108 bits, central 80%=3-28; versus Record/Play the mean-rank changes are +0.00/+8.46. Actual evidence is 6-2, mean margin +17.00, 0 close wins and 1 close losses, standardized schedule probability 0.279, expected wins 5.97, wins minus expectation +0.03. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=9.75, ordinal=5, SD=8.11, entropy=4.620 bits, central 80%=2-20; versus Record/Play the mean-rank changes are -3.79/+4.67. Actual evidence is 6-2, mean margin +17.00, 0 close wins and 1 close losses, standardized schedule probability 0.279, expected wins 5.97, wins minus expectation +0.03. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=7.47, ordinal=4, SD=6.08, entropy=4.215 bits, central 80%=2-15; versus Record/Play the mean-rank changes are -6.07/+2.39. Actual evidence is 6-2, mean margin +17.00, 0 close wins and 1 close losses, standardized schedule probability 0.279, expected wins 5.97, wins minus expectation +0.03. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=6.04, ordinal=5, SD=4.79, entropy=3.887 bits, central 80%=1-12; versus Record/Play the mean-rank changes are -7.50/+0.96. Actual evidence is 6-2, mean margin +17.00, 0 close wins and 1 close losses, standardized schedule probability 0.279, expected wins 5.97, wins minus expectation +0.03. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=5.09, ordinal=5, SD=3.95, entropy=3.615 bits, central 80%=1-10; versus Record/Play the mean-rank changes are -8.46/+0.00. Actual evidence is 6-2, mean margin +17.00, 0 close wins and 1 close losses, standardized schedule probability 0.279, expected wins 5.97, wins minus expectation +0.03. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -33.82 | -19 | -18.15 | 0.473 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -32.09 | -12 | -20.30 | 0.503 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -29.56 | -7 | -21.22 | 0.521 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -26.90 | -9 | -21.20 | 0.533 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -24.49 | -9 | -20.65 | 0.541 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 4 / 445 | -4.21 | +3 | -5.75 | 0.135 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 4 / 445 | -3.42 | +2 | -5.10 | 0.124 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 4 / 445 | -2.73 | +1 | -4.27 | 0.115 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 4 / 445 | -2.22 | +3 | -3.48 | 0.112 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 4 / 445 | -1.87 | +3 | -2.85 | 0.113 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Auburn (2)

Frozen categories: middle_schedule_pair.

### 2026-09-08

Actual record **1–0**, mean margin **+1.00**; close wins/losses 1/0, dominant wins 0. Frozen standardized schedule probability **0.363** (larger is easier), expected wins **0.60**, wins minus expectation **+0.40**. Reference: retained_context_v1.4_posterior, ordinal 64, band middle, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 44.93 | 11 | 33.20 | 6.810 | 17–68 | 7–95 | 2–118 | 0.161/0.191/0.361 | +0.00 / -15.42 |
| 0.25 | 49.38 | 23 | 33.83 | 6.898 | 21–74 | 8–99 | 3–121 | 0.126/0.151/0.306 | +4.44 / -10.98 |
| 0.5 | 53.43 | 47 | 34.16 | 6.957 | 25–79 | 11–103 | 3–124 | 0.098/0.119/0.259 | +8.50 / -6.93 |
| 0.75 | 57.08 | 55 | 34.28 | 6.994 | 28–83 | 13–106 | 4–125 | 0.076/0.094/0.220 | +12.15 / -3.27 |
| 1 | 60.36 | 64 | 34.26 | 7.015 | 32–87 | 16–109 | 5–127 | 0.059/0.075/0.188 | +15.42 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856636 / 2026-09-05 | neutral | Baylor (fbs) | 17–16 | +1 | 0.604 | 42.78/26.57 | +4.59 |

Answers to the original questions at every candidate:

**For identical middle-team records, how do schedule and uncertainty differ across candidates?**

- At lambda=0, E[rank]=44.93, ordinal=11, SD=33.20, entropy=6.810 bits, central 80%=7-95; versus Record/Play the mean-rank changes are +0.00/-15.42. The frozen peer UConn is 1-0 with standardized schedule probability 0.980, E[rank]=66.24, SD=38.97, entropy=7.102, widths 50/80/95=67/109/131; this team's expected-rank gap is -21.30 and SD gap -5.76. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=49.38, ordinal=23, SD=33.83, entropy=6.898 bits, central 80%=8-99; versus Record/Play the mean-rank changes are +4.44/-10.98. The frozen peer UConn is 1-0 with standardized schedule probability 0.980, E[rank]=60.55, SD=37.87, entropy=7.067, widths 50/80/95=64/106/130; this team's expected-rank gap is -11.17 and SD gap -4.04. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=53.43, ordinal=47, SD=34.16, entropy=6.957 bits, central 80%=11-103; versus Record/Play the mean-rank changes are +8.50/-6.93. The frozen peer UConn is 1-0 with standardized schedule probability 0.980, E[rank]=56.41, SD=36.74, entropy=7.021, widths 50/80/95=61/101/127; this team's expected-rank gap is -2.98 and SD gap -2.58. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=57.08, ordinal=55, SD=34.28, entropy=6.994 bits, central 80%=13-106; versus Record/Play the mean-rank changes are +12.15/-3.27. The frozen peer UConn is 1-0 with standardized schedule probability 0.980, E[rank]=53.41, SD=35.75, entropy=6.977, widths 50/80/95=58/98/125; this team's expected-rank gap is +3.68 and SD gap -1.48. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=60.36, ordinal=64, SD=34.26, entropy=7.015 bits, central 80%=16-109; versus Record/Play the mean-rank changes are +15.42/+0.00. The frozen peer UConn is 1-0 with standardized schedule probability 0.980, E[rank]=51.19, SD=34.95, entropy=6.938, widths 50/80/95=57/96/123; this team's expected-rank gap is +9.16 and SD gap -0.69. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### 2026-09-27

Actual record **3–1**, mean margin **+9.25**; close wins/losses 2/1, dominant wins 1. Frozen standardized schedule probability **0.409** (larger is easier), expected wins **2.66**, wins minus expectation **+0.34**. Reference: retained_context_v1.4_posterior, ordinal 31, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 32.19 | 21 | 23.72 | 6.347 | 14–46 | 6–67 | 2–90 | 0.186/0.228/0.481 | +0.00 / +0.77 |
| 0.25 | 31.83 | 24 | 22.42 | 6.300 | 14–45 | 7–64 | 3–86 | 0.166/0.209/0.476 | -0.35 / +0.42 |
| 0.5 | 31.57 | 25 | 21.31 | 6.254 | 15–44 | 8–62 | 3–83 | 0.149/0.192/0.472 | -0.62 / +0.16 |
| 0.75 | 31.43 | 29 | 20.38 | 6.211 | 16–43 | 9–60 | 4–80 | 0.133/0.175/0.467 | -0.76 / +0.02 |
| 1 | 31.41 | 31 | 19.60 | 6.172 | 16–43 | 10–59 | 4–78 | 0.118/0.159/0.461 | -0.77 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856636 / 2026-09-05 | neutral | Baylor (fbs) | 17–16 | +1 | 0.621 | 34.21/19.02 | +4.95 |
| 401856671 / 2026-09-12 | home | Southern Miss (fbs) | 43–8 | +35 | 0.940 | 97.61/21.29 | +24.72 |
| 401856687 / 2026-09-19 | home | Florida (fbs) | 39–44 | -5 | 0.365 | 7.46/5.44 | -5.29 |
| 401856698 / 2026-09-26 | home | Vanderbilt (fbs) | 21–15 | +6 | 0.736 | 40.48/20.31 | +10.14 |

Answers to the original questions at every candidate:

**For identical middle-team records, how do schedule and uncertainty differ across candidates?**

- At lambda=0, E[rank]=32.19, ordinal=21, SD=23.72, entropy=6.347 bits, central 80%=6-67; versus Record/Play the mean-rank changes are +0.00/+0.77. The frozen peer UConn is 2-2 with standardized schedule probability 0.670, E[rank]=87.14, SD=29.28, entropy=6.819, widths 50/80/95=45/80/108; this team's expected-rank gap is -54.95 and SD gap -5.55. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=31.83, ordinal=24, SD=22.42, entropy=6.300 bits, central 80%=7-64; versus Record/Play the mean-rank changes are -0.35/+0.42. The frozen peer UConn is 2-2 with standardized schedule probability 0.670, E[rank]=79.14, SD=28.93, entropy=6.846, widths 50/80/95=45/79/108; this team's expected-rank gap is -47.31 and SD gap -6.51. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=31.57, ordinal=25, SD=21.31, entropy=6.254 bits, central 80%=8-62; versus Record/Play the mean-rank changes are -0.62/+0.16. The frozen peer UConn is 2-2 with standardized schedule probability 0.670, E[rank]=72.90, SD=28.14, entropy=6.825, widths 50/80/95=43/76/106; this team's expected-rank gap is -41.33 and SD gap -6.82. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=31.43, ordinal=29, SD=20.38, entropy=6.211 bits, central 80%=9-60; versus Record/Play the mean-rank changes are -0.76/+0.02. The frozen peer UConn is 2-2 with standardized schedule probability 0.670, E[rank]=68.06, SD=27.24, entropy=6.785, widths 50/80/95=41/74/103; this team's expected-rank gap is -36.63 and SD gap -6.86. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=31.41, ordinal=31, SD=19.60, entropy=6.172 bits, central 80%=10-59; versus Record/Play the mean-rank changes are -0.77/+0.00. The frozen peer UConn is 2-2 with standardized schedule probability 0.670, E[rank]=64.26, SD=26.38, entropy=6.741, widths 50/80/95=40/72/100; this team's expected-rank gap is -32.85 and SD gap -6.78. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### 2026-10-25T12-59-22.246522Z

Actual record **4–3**, mean margin **+1.29**; close wins/losses 3/1, dominant wins 1. Frozen standardized schedule probability **0.308** (larger is easier), expected wins **3.47**, wins minus expectation **+0.53**. Reference: retained_context_v1.4_posterior, ordinal 41, band middle, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 33.74 | 34 | 21.77 | 6.305 | 17–48 | 9–65 | 4–85 | 0.118/0.158/0.438 | +0.00 / +1.65 |
| 0.25 | 33.11 | 35 | 20.30 | 6.225 | 17–46 | 10–62 | 5–80 | 0.101/0.140/0.436 | -0.63 / +1.02 |
| 0.5 | 32.66 | 37 | 19.05 | 6.148 | 18–45 | 11–60 | 6–77 | 0.085/0.123/0.433 | -1.07 / +0.58 |
| 0.75 | 32.34 | 39 | 18.00 | 6.077 | 18–44 | 12–58 | 7–74 | 0.071/0.107/0.430 | -1.40 / +0.26 |
| 1 | 32.08 | 41 | 17.10 | 6.012 | 19–43 | 13–56 | 8–71 | 0.059/0.093/0.426 | -1.65 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856636 / 2026-09-05 | neutral | Baylor (fbs) | 17–16 | +1 | 0.568 | 29.89/14.58 | +2.62 |
| 401856671 / 2026-09-12 | home | Southern Miss (fbs) | 43–8 | +35 | 0.958 | 107.88/14.90 | +26.61 |
| 401856687 / 2026-09-19 | home | Florida (fbs) | 39–44 | -5 | 0.429 | 12.90/8.65 | -2.71 |
| 401856698 / 2026-09-26 | home | Vanderbilt (fbs) | 21–15 | +6 | 0.848 | 65.49/19.67 | +15.80 |
| 401856710 / 2026-10-03 | away | Tennessee (fbs) | 14–24 | -10 | 0.250 | 10.70/6.67 | -10.36 |
| 401856719 / 2026-10-17 | away | Georgia (fbs) | 27–24 | +3 | 0.147 | 4.15/3.10 | -15.75 |
| 401856728 / 2026-10-24 | home | LSU (fbs) | 17–38 | -21 | 0.268 | 4.44/2.81 | -9.17 |

Answers to the original questions at every candidate:

**For identical middle-team records, how do schedule and uncertainty differ across candidates?**

- At lambda=0, E[rank]=33.74, ordinal=34, SD=21.77, entropy=6.305 bits, central 80%=9-65; versus Record/Play the mean-rank changes are +0.00/+1.65. The frozen peer UConn is 4-3 with standardized schedule probability 0.697, E[rank]=90.60, SD=23.88, entropy=6.555, widths 50/80/95=35/63/90; this team's expected-rank gap is -56.86 and SD gap -2.10. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=33.11, ordinal=35, SD=20.30, entropy=6.225 bits, central 80%=10-62; versus Record/Play the mean-rank changes are -0.63/+1.02. The frozen peer UConn is 4-3 with standardized schedule probability 0.697, E[rank]=81.24, SD=23.54, entropy=6.572, widths 50/80/95=35/63/90; this team's expected-rank gap is -48.13 and SD gap -3.24. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=32.66, ordinal=37, SD=19.05, entropy=6.148 bits, central 80%=11-60; versus Record/Play the mean-rank changes are -1.07/+0.58. The frozen peer UConn is 4-3 with standardized schedule probability 0.697, E[rank]=73.93, SD=22.64, entropy=6.531, widths 50/80/95=34/60/87; this team's expected-rank gap is -41.27 and SD gap -3.59. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=32.34, ordinal=39, SD=18.00, entropy=6.077 bits, central 80%=12-58; versus Record/Play the mean-rank changes are -1.40/+0.26. The frozen peer UConn is 4-3 with standardized schedule probability 0.697, E[rank]=68.19, SD=21.60, entropy=6.469, widths 50/80/95=32/58/83; this team's expected-rank gap is -35.85 and SD gap -3.61. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=32.08, ordinal=41, SD=17.10, entropy=6.012 bits, central 80%=13-56; versus Record/Play the mean-rank changes are -1.65/+0.00. The frozen peer UConn is 4-3 with standardized schedule probability 0.697, E[rank]=63.60, SD=20.60, entropy=6.403, widths 50/80/95=30/55/81; this team's expected-rank gap is -31.52 and SD gap -3.50. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -12.75 | +10 | -9.48 | 0.180 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -17.54 | +1 | -11.41 | 0.244 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -21.86 | -22 | -12.84 | 0.303 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -25.66 | -26 | -13.89 | 0.356 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -28.94 | -33 | -14.66 | 0.402 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 3 / 445 | +1.55 | +13 | -1.95 | 0.082 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 3 / 445 | +1.27 | +11 | -2.11 | 0.084 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 3 / 445 | +1.09 | +12 | -2.26 | 0.087 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 3 / 445 | +0.91 | +10 | -2.39 | 0.090 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 3 / 445 | +0.67 | +10 | -2.50 | 0.091 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Central Michigan (2117)

Frozen categories: weak.

### 2026-09-08

Actual record **0–1**, mean margin **-31.00**; close wins/losses 0/0, dominant wins 0. Frozen standardized schedule probability **0.404** (larger is easier), expected wins **0.17**, wins minus expectation **-0.17**. Reference: retained_context_v1.4_posterior, ordinal 129, band weak, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 90.04 | 129 | 34.58 | 6.899 | 65–120 | 38–131 | 16–137 | 0.013/0.016/0.051 | +0.00 / -15.85 |
| 0.25 | 94.86 | 131 | 33.20 | 6.793 | 72–123 | 44–133 | 21–137 | 0.008/0.011/0.036 | +4.82 / -11.03 |
| 0.5 | 99.13 | 130 | 31.61 | 6.676 | 79–125 | 51–134 | 25–137 | 0.005/0.007/0.026 | +9.09 / -6.76 |
| 0.75 | 102.81 | 129 | 29.95 | 6.556 | 85–127 | 57–134 | 30–137 | 0.003/0.005/0.018 | +12.76 / -3.09 |
| 1 | 105.89 | 129 | 28.32 | 6.439 | 90–128 | 63–135 | 35–138 | 0.002/0.003/0.012 | +15.85 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401864498 / 2026-09-06 | away | New Mexico (fbs) | 7–38 | -31 | 0.170 | 59.59/29.93 | -16.96 |

Answers to the original questions at every candidate:

**Does weak-team uncertainty decrease on easier schedules, and how much does winning matter?**

- At lambda=0, E[rank]=90.04, ordinal=129, SD=34.58, entropy=6.899 bits, central 80%=38-131; versus Record/Play the mean-rank changes are +0.00/-15.85. Actual evidence is 0-1, mean margin -31.00, 0 close wins and 0 close losses, standardized schedule probability 0.404, expected wins 0.17, wins minus expectation -0.17. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=94.86, ordinal=131, SD=33.20, entropy=6.793 bits, central 80%=44-133; versus Record/Play the mean-rank changes are +4.82/-11.03. Actual evidence is 0-1, mean margin -31.00, 0 close wins and 0 close losses, standardized schedule probability 0.404, expected wins 0.17, wins minus expectation -0.17. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=99.13, ordinal=130, SD=31.61, entropy=6.676 bits, central 80%=51-134; versus Record/Play the mean-rank changes are +9.09/-6.76. Actual evidence is 0-1, mean margin -31.00, 0 close wins and 0 close losses, standardized schedule probability 0.404, expected wins 0.17, wins minus expectation -0.17. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=102.81, ordinal=129, SD=29.95, entropy=6.556 bits, central 80%=57-134; versus Record/Play the mean-rank changes are +12.76/-3.09. Actual evidence is 0-1, mean margin -31.00, 0 close wins and 0 close losses, standardized schedule probability 0.404, expected wins 0.17, wins minus expectation -0.17. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=105.89, ordinal=129, SD=28.32, entropy=6.439 bits, central 80%=63-135; versus Record/Play the mean-rank changes are +15.85/+0.00. Actual evidence is 0-1, mean margin -31.00, 0 close wins and 0 close losses, standardized schedule probability 0.404, expected wins 0.17, wins minus expectation -0.17. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-09-27

Actual record **2–2**, mean margin **-15.75**; close wins/losses 1/0, dominant wins 0. Frozen standardized schedule probability **0.529** (larger is easier), expected wins **1.39**, wins minus expectation **+0.61**. Reference: retained_context_v1.4_posterior, ordinal 124, band weak, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 71.60 | 82 | 31.34 | 6.951 | 47–96 | 29–114 | 14–128 | 0.016/0.021/0.081 | +0.00 / -40.75 |
| 0.25 | 86.56 | 101 | 29.09 | 6.813 | 66–110 | 44–123 | 26–132 | 0.003/0.004/0.024 | +14.96 / -25.80 |
| 0.5 | 98.32 | 116 | 25.37 | 6.561 | 83–118 | 62–128 | 38–135 | 0.000/0.001/0.006 | +26.72 / -14.03 |
| 0.75 | 106.65 | 123 | 21.49 | 6.277 | 95–123 | 77–131 | 54–136 | 0.000/0.000/0.001 | +35.06 / -5.70 |
| 1 | 112.35 | 124 | 18.22 | 6.015 | 103–126 | 87–132 | 68–137 | 0.000/0.000/0.000 | +40.75 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401864498 / 2026-09-06 | away | New Mexico (fbs) | 7–38 | -31 | 0.081 | 43.35/21.32 | -22.54 |
| 401866415 / 2026-09-12 | home | Colgate (fcs) | 13–10 | +3 | 0.705 | 30.65/17.77 | +8.91 |
| 401864508 / 2026-09-19 | home | Wyoming (fbs) | 24–10 | +14 | 0.595 | 113.43/15.29 | +3.93 |
| 401858238 / 2026-09-26 | away | Miami (fbs) | 3–52 | -49 | 0.014 | 6.15/5.02 | -33.84 |

Answers to the original questions at every candidate:

**Does weak-team uncertainty decrease on easier schedules, and how much does winning matter?**

- At lambda=0, E[rank]=71.60, ordinal=82, SD=31.34, entropy=6.951 bits, central 80%=29-114; versus Record/Play the mean-rank changes are +0.00/-40.75. Actual evidence is 2-2, mean margin -15.75, 1 close wins and 0 close losses, standardized schedule probability 0.529, expected wins 1.39, wins minus expectation +0.61. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=86.56, ordinal=101, SD=29.09, entropy=6.813 bits, central 80%=44-123; versus Record/Play the mean-rank changes are +14.96/-25.80. Actual evidence is 2-2, mean margin -15.75, 1 close wins and 0 close losses, standardized schedule probability 0.529, expected wins 1.39, wins minus expectation +0.61. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=98.32, ordinal=116, SD=25.37, entropy=6.561 bits, central 80%=62-128; versus Record/Play the mean-rank changes are +26.72/-14.03. Actual evidence is 2-2, mean margin -15.75, 1 close wins and 0 close losses, standardized schedule probability 0.529, expected wins 1.39, wins minus expectation +0.61. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=106.65, ordinal=123, SD=21.49, entropy=6.277 bits, central 80%=77-131; versus Record/Play the mean-rank changes are +35.06/-5.70. Actual evidence is 2-2, mean margin -15.75, 1 close wins and 0 close losses, standardized schedule probability 0.529, expected wins 1.39, wins minus expectation +0.61. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=112.35, ordinal=124, SD=18.22, entropy=6.015 bits, central 80%=87-132; versus Record/Play the mean-rank changes are +40.75/+0.00. Actual evidence is 2-2, mean margin -15.75, 1 close wins and 0 close losses, standardized schedule probability 0.529, expected wins 1.39, wins minus expectation +0.61. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-10-25T12-59-22.246522Z

Actual record **3–5**, mean margin **-8.00**; close wins/losses 1/2, dominant wins 0. Frozen standardized schedule probability **0.603** (larger is easier), expected wins **3.25**, wins minus expectation **-0.25**. Reference: retained_context_v1.4_posterior, ordinal 120, band weak, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 97.27 | 110 | 23.02 | 6.478 | 83–115 | 65–125 | 46–133 | 0.000/0.000/0.002 | +0.00 / -10.79 |
| 0.25 | 101.62 | 114 | 19.91 | 6.269 | 90–116 | 74–125 | 56–132 | 0.000/0.000/0.000 | +4.35 / -6.44 |
| 0.5 | 104.58 | 119 | 17.41 | 6.081 | 94–117 | 81–125 | 64–132 | 0.000/0.000/0.000 | +7.31 / -3.48 |
| 0.75 | 106.62 | 118 | 15.49 | 5.919 | 98–118 | 86–125 | 71–131 | 0.000/0.000/0.000 | +9.34 / -1.45 |
| 1 | 108.07 | 120 | 14.00 | 5.780 | 100–118 | 89–124 | 76–130 | 0.000/0.000/0.000 | +10.79 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401864498 / 2026-09-06 | away | New Mexico (fbs) | 7–38 | -31 | 0.059 | 32.80/15.57 | -24.00 |
| 401866415 / 2026-09-12 | home | Colgate (fcs) | 13–10 | +3 | 0.829 | 41.84/16.63 | +14.70 |
| 401864508 / 2026-09-19 | home | Wyoming (fbs) | 24–10 | +14 | 0.564 | 106.60/13.77 | +2.46 |
| 401858238 / 2026-09-26 | away | Miami (fbs) | 3–52 | -49 | 0.013 | 4.20/2.89 | -33.71 |
| 401866430 / 2026-10-03 | home | Akron (fbs) | 41–17 | +24 | 0.809 | 128.47/6.66 | +13.18 |
| 401866438 / 2026-10-10 | away | Ohio (fbs) | 14–31 | -17 | 0.227 | 81.54/18.79 | -11.46 |
| 401866443 / 2026-10-17 | home | Western Michigan (fbs) | 14–21 | -7 | 0.273 | 66.16/18.81 | -9.15 |
| 401866476 / 2026-10-24 | home | Miami (OH) (fbs) | 27–28 | -1 | 0.481 | 97.00/16.63 | -0.71 |

Answers to the original questions at every candidate:

**Does weak-team uncertainty decrease on easier schedules, and how much does winning matter?**

- At lambda=0, E[rank]=97.27, ordinal=110, SD=23.02, entropy=6.478 bits, central 80%=65-125; versus Record/Play the mean-rank changes are +0.00/-10.79. Actual evidence is 3-5, mean margin -8.00, 1 close wins and 2 close losses, standardized schedule probability 0.603, expected wins 3.25, wins minus expectation -0.25. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=101.62, ordinal=114, SD=19.91, entropy=6.269 bits, central 80%=74-125; versus Record/Play the mean-rank changes are +4.35/-6.44. Actual evidence is 3-5, mean margin -8.00, 1 close wins and 2 close losses, standardized schedule probability 0.603, expected wins 3.25, wins minus expectation -0.25. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=104.58, ordinal=119, SD=17.41, entropy=6.081 bits, central 80%=81-125; versus Record/Play the mean-rank changes are +7.31/-3.48. Actual evidence is 3-5, mean margin -8.00, 1 close wins and 2 close losses, standardized schedule probability 0.603, expected wins 3.25, wins minus expectation -0.25. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=106.62, ordinal=118, SD=15.49, entropy=5.919 bits, central 80%=86-125; versus Record/Play the mean-rank changes are +9.34/-1.45. Actual evidence is 3-5, mean margin -8.00, 1 close wins and 2 close losses, standardized schedule probability 0.603, expected wins 3.25, wins minus expectation -0.25. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=108.07, ordinal=120, SD=14.00, entropy=5.780 bits, central 80%=89-124; versus Record/Play the mean-rank changes are +10.79/+0.00. Actual evidence is 3-5, mean margin -8.00, 1 close wins and 2 close losses, standardized schedule probability 0.603, expected wins 3.25, wins minus expectation -0.25. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -18.45 | -47 | -3.24 | 0.251 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -8.31 | -30 | -4.11 | 0.184 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -0.81 | -14 | -6.24 | 0.168 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | +3.85 | -6 | -8.45 | 0.183 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | +6.46 | -5 | -10.09 | 0.192 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 4 / 445 | +25.68 | +28 | -8.32 | 0.362 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 4 / 445 | +15.07 | +13 | -9.18 | 0.248 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 4 / 445 | +6.26 | +3 | -7.96 | 0.178 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 4 / 445 | -0.04 | -5 | -6.01 | 0.179 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 4 / 445 | -4.29 | -4 | -4.22 | 0.226 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Georgia (61)

Frozen categories: boundary_12.

### 2026-09-08

Actual record **1–0**, mean margin **+60.00**; close wins/losses 0/0, dominant wins 1. Frozen standardized schedule probability **0.977** (larger is easier), expected wins **1.00**, wins minus expectation **+0.00**. Reference: retained_context_v1.4_posterior, ordinal 32, band strong, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 67.80 | 79 | 39.35 | 7.107 | 34–101 | 14–123 | 4–134 | 0.075/0.090/0.188 | +0.00 / +25.48 |
| 0.25 | 59.32 | 52 | 37.89 | 7.057 | 26–90 | 10–115 | 3–131 | 0.101/0.121/0.243 | -8.48 / +17.00 |
| 0.5 | 52.39 | 41 | 36.01 | 6.964 | 21–79 | 8–106 | 2–127 | 0.128/0.153/0.296 | -15.40 / +10.07 |
| 0.75 | 46.81 | 36 | 34.08 | 6.854 | 17–71 | 7–98 | 2–121 | 0.156/0.185/0.347 | -20.98 / +4.49 |
| 1 | 42.32 | 32 | 32.30 | 6.740 | 15–64 | 6–91 | 2–116 | 0.183/0.216/0.395 | -25.48 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856658 / 2026-09-05 | home | Tennessee State (fcs) | 63–3 | +60 | 0.998 | 82.47/28.50 | +50.38 |

Answers to the original questions at every candidate:

**Does this boundary crossing reflect meaningful evidence or close teams?**

- At lambda=0, E[rank]=67.80, ordinal=79, SD=39.35, entropy=7.107 bits, central 80%=14-123; versus Record/Play the mean-rank changes are +0.00/+25.48. Latent top10/12/25 mass is 0.075/0.090/0.188. Adjacent expected-rank gaps are 0.06734154837340611/0.02277654623935632. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=59.32, ordinal=52, SD=37.89, entropy=7.057 bits, central 80%=10-115; versus Record/Play the mean-rank changes are -8.48/+17.00. Latent top10/12/25 mass is 0.101/0.121/0.243. Adjacent expected-rank gaps are 0.2884524594322002/1.16147595820361. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=52.39, ordinal=41, SD=36.01, entropy=6.964 bits, central 80%=8-106; versus Record/Play the mean-rank changes are -15.40/+10.07. Latent top10/12/25 mass is 0.128/0.153/0.296. Adjacent expected-rank gaps are 0.39542840913516386/0.11569688717835191. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=46.81, ordinal=36, SD=34.08, entropy=6.854 bits, central 80%=7-98; versus Record/Play the mean-rank changes are -20.98/+4.49. Latent top10/12/25 mass is 0.156/0.185/0.347. Adjacent expected-rank gaps are 0.7907656874312678/0.39034936244543417. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=42.32, ordinal=32, SD=32.30, entropy=6.740 bits, central 80%=6-91; versus Record/Play the mean-rank changes are -25.48/+0.00. Latent top10/12/25 mass is 0.183/0.216/0.395. Adjacent expected-rank gaps are 0.06518943912313802/0.21169777858989391. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

### 2026-09-27

Actual record **4–0**, mean margin **+41.50**; close wins/losses 0/0, dominant wins 4. Frozen standardized schedule probability **0.638** (larger is easier), expected wins **3.73**, wins minus expectation **+0.27**. Reference: retained_context_v1.4_posterior, ordinal 4, band elite, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 23.37 | 9 | 21.40 | 5.947 | 7–34 | 3–55 | 1–80 | 0.354/0.408/0.655 | +0.00 / +16.27 |
| 0.25 | 15.33 | 6 | 15.73 | 5.325 | 4–20 | 2–37 | 1–60 | 0.517/0.580/0.815 | -8.04 / +8.23 |
| 0.5 | 10.99 | 5 | 11.65 | 4.823 | 3–14 | 2–25 | 1–45 | 0.641/0.705/0.904 | -12.38 / +3.90 |
| 0.75 | 8.56 | 4 | 8.96 | 4.444 | 3–11 | 1–19 | 1–34 | 0.730/0.790/0.949 | -14.80 / +1.47 |
| 1 | 7.09 | 4 | 7.20 | 4.156 | 2–9 | 1–15 | 1–26 | 0.793/0.847/0.972 | -16.27 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856658 / 2026-09-05 | home | Tennessee State (fcs) | 63–3 | +60 | 0.999 | 96.86/20.16 | +54.58 |
| 401856673 / 2026-09-12 | home | Western Kentucky (fbs) | 70–20 | +50 | 0.990 | 115.03/16.32 | +35.69 |
| 401856686 / 2026-09-19 | away | Arkansas (fbs) | 45–17 | +28 | 0.899 | 63.71/26.34 | +18.94 |
| 401856700 / 2026-09-26 | home | Oklahoma (fbs) | 41–13 | +28 | 0.838 | 22.24/13.56 | +15.08 |

Answers to the original questions at every candidate:

**Does this boundary crossing reflect meaningful evidence or close teams?**

- At lambda=0, E[rank]=23.37, ordinal=9, SD=21.40, entropy=5.947 bits, central 80%=3-55; versus Record/Play the mean-rank changes are +0.00/+16.27. Latent top10/12/25 mass is 0.354/0.408/0.655. Adjacent expected-rank gaps are 0.2488567651500624/0.7849453011651661. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=15.33, ordinal=6, SD=15.73, entropy=5.325 bits, central 80%=2-37; versus Record/Play the mean-rank changes are -8.04/+8.23. Latent top10/12/25 mass is 0.517/0.580/0.815. Adjacent expected-rank gaps are 0.4819062696320504/1.6466435504840522. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=10.99, ordinal=5, SD=11.65, entropy=4.823 bits, central 80%=2-25; versus Record/Play the mean-rank changes are -12.38/+3.90. Latent top10/12/25 mass is 0.641/0.705/0.904. Adjacent expected-rank gaps are 0.18325309491306285/1.2589111866512184. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=8.56, ordinal=4, SD=8.96, entropy=4.444 bits, central 80%=1-19; versus Record/Play the mean-rank changes are -14.80/+1.47. Latent top10/12/25 mass is 0.730/0.790/0.949. Adjacent expected-rank gaps are 0.20500894929643998/0.4061549093264354. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=7.09, ordinal=4, SD=7.20, entropy=4.156 bits, central 80%=1-15; versus Record/Play the mean-rank changes are -16.27/+0.00. Latent top10/12/25 mass is 0.793/0.847/0.972. Adjacent expected-rank gaps are 0.09279689836611027/0.3809591588866308. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

### 2026-10-25T12-59-22.246522Z

Actual record **6–1**, mean margin **+27.14**; close wins/losses 1/1, dominant wins 4. Frozen standardized schedule probability **0.487** (larger is easier), expected wins **5.97**, wins minus expectation **+0.03**. Reference: retained_context_v1.4_posterior, ordinal 6, band elite, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 16.89 | 12 | 15.01 | 5.461 | 6–23 | 3–38 | 1–58 | 0.430/0.502/0.792 | +0.00 / +10.38 |
| 0.25 | 11.91 | 10 | 10.89 | 4.940 | 4–16 | 2–25 | 1–43 | 0.574/0.653/0.903 | -4.98 / +5.39 |
| 0.5 | 9.15 | 7 | 8.14 | 4.539 | 4–12 | 2–19 | 1–31 | 0.683/0.759/0.956 | -7.74 / +2.64 |
| 0.75 | 7.54 | 7 | 6.43 | 4.240 | 3–10 | 1–15 | 1–24 | 0.762/0.832/0.981 | -9.35 / +1.03 |
| 1 | 6.51 | 6 | 5.35 | 4.011 | 3–9 | 1–13 | 1–20 | 0.819/0.881/0.991 | -10.38 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856658 / 2026-09-05 | home | Tennessee State (fcs) | 63–3 | +60 | 0.999 | 88.86/15.21 | +51.80 |
| 401856673 / 2026-09-12 | home | Western Kentucky (fbs) | 70–20 | +50 | 0.987 | 109.13/15.11 | +33.98 |
| 401856686 / 2026-09-19 | away | Arkansas (fbs) | 45–17 | +28 | 0.892 | 58.84/21.73 | +17.95 |
| 401856700 / 2026-09-26 | home | Oklahoma (fbs) | 41–13 | +28 | 0.825 | 19.36/10.38 | +14.01 |
| 401856705 / 2026-10-03 | home | Vanderbilt (fbs) | 38–14 | +24 | 0.957 | 65.49/19.67 | +25.24 |
| 401856712 / 2026-10-10 | away | Alabama (fbs) | 20–17 | +3 | 0.453 | 5.67/3.41 | -1.68 |
| 401856719 / 2026-10-17 | home | Auburn (fbs) | 24–27 | -3 | 0.853 | 23.62/13.00 | +15.75 |

Answers to the original questions at every candidate:

**Does this boundary crossing reflect meaningful evidence or close teams?**

- At lambda=0, E[rank]=16.89, ordinal=12, SD=15.01, entropy=5.461 bits, central 80%=3-38; versus Record/Play the mean-rank changes are +0.00/+10.38. Latent top10/12/25 mass is 0.430/0.502/0.792. Adjacent expected-rank gaps are 0.1590162836901854/0.013539916373922267. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=11.91, ordinal=10, SD=10.89, entropy=4.940 bits, central 80%=2-25; versus Record/Play the mean-rank changes are -4.98/+5.39. Latent top10/12/25 mass is 0.574/0.653/0.903. Adjacent expected-rank gaps are 0.19456823013668156/0.2880943262495492. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=9.15, ordinal=7, SD=8.14, entropy=4.539 bits, central 80%=2-19; versus Record/Play the mean-rank changes are -7.74/+2.64. Latent top10/12/25 mass is 0.683/0.759/0.956. Adjacent expected-rank gaps are 1.071679267943484/0.11096624019288726. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=7.54, ordinal=7, SD=6.43, entropy=4.240 bits, central 80%=1-15; versus Record/Play the mean-rank changes are -9.35/+1.03. Latent top10/12/25 mass is 0.762/0.832/0.981. Adjacent expected-rank gaps are 0.3419317356826497/1.920950337473755. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=6.51, ordinal=6, SD=5.35, entropy=4.011 bits, central 80%=1-13; versus Record/Play the mean-rank changes are -10.38/+0.00. Latent top10/12/25 mass is 0.819/0.881/0.991. Adjacent expected-rank gaps are 1.4270448037118335/0.07123459330294413. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -44.43 | -70 | -17.96 | 0.512 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -43.99 | -46 | -22.16 | 0.575 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -41.40 | -36 | -24.36 | 0.613 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -38.25 | -32 | -25.12 | 0.631 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -35.23 | -28 | -25.10 | 0.638 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 3 / 445 | -6.48 | +3 | -6.38 | 0.138 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 3 / 445 | -3.42 | +4 | -4.84 | 0.101 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 3 / 445 | -1.84 | +2 | -3.51 | 0.082 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 3 / 445 | -1.02 | +3 | -2.54 | 0.072 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 3 / 445 | -0.58 | +2 | -1.85 | 0.067 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Kentucky (96)

Frozen categories: strong.

### 2026-09-08

Actual record **1–0**, mean margin **+32.00**; close wins/losses 0/0, dominant wins 1. Frozen standardized schedule probability **0.921** (larger is easier), expected wins **0.94**, wins minus expectation **+0.06**. Reference: retained_context_v1.4_posterior, ordinal 58, band middle, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 64.96 | 57 | 38.60 | 7.096 | 32–97 | 13–120 | 4–133 | 0.081/0.097/0.201 | +0.00 / +7.52 |
| 0.25 | 63.05 | 61 | 37.65 | 7.081 | 31–94 | 13–117 | 4–132 | 0.080/0.097/0.205 | -1.92 / +5.61 |
| 0.5 | 61.09 | 61 | 36.76 | 7.061 | 30–90 | 13–114 | 4–130 | 0.081/0.098/0.211 | -3.87 / +3.65 |
| 0.75 | 59.20 | 61 | 35.92 | 7.038 | 29–87 | 12–111 | 4–129 | 0.083/0.100/0.219 | -5.76 / +1.76 |
| 1 | 57.44 | 58 | 35.13 | 7.012 | 28–84 | 12–108 | 4–127 | 0.085/0.103/0.227 | -7.52 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856659 / 2026-09-05 | home | Youngstown State (fcs) | 45–13 | +32 | 0.940 | 45.89/27.93 | +28.31 |

Answers to the original questions at every candidate:

**Does a strong team's evaluation depend on wins or dominant margins?**

- At lambda=0, E[rank]=64.96, ordinal=57, SD=38.60, entropy=7.096 bits, central 80%=13-120; versus Record/Play the mean-rank changes are +0.00/+7.52. Actual evidence is 1-0, mean margin +32.00, 0 close wins and 0 close losses, standardized schedule probability 0.921, expected wins 0.94, wins minus expectation +0.06. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=63.05, ordinal=61, SD=37.65, entropy=7.081 bits, central 80%=13-117; versus Record/Play the mean-rank changes are -1.92/+5.61. Actual evidence is 1-0, mean margin +32.00, 0 close wins and 0 close losses, standardized schedule probability 0.921, expected wins 0.94, wins minus expectation +0.06. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=61.09, ordinal=61, SD=36.76, entropy=7.061 bits, central 80%=13-114; versus Record/Play the mean-rank changes are -3.87/+3.65. Actual evidence is 1-0, mean margin +32.00, 0 close wins and 0 close losses, standardized schedule probability 0.921, expected wins 0.94, wins minus expectation +0.06. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=59.20, ordinal=61, SD=35.92, entropy=7.038 bits, central 80%=12-111; versus Record/Play the mean-rank changes are -5.76/+1.76. Actual evidence is 1-0, mean margin +32.00, 0 close wins and 0 close losses, standardized schedule probability 0.921, expected wins 0.94, wins minus expectation +0.06. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=57.44, ordinal=58, SD=35.13, entropy=7.012 bits, central 80%=12-108; versus Record/Play the mean-rank changes are -7.52/+0.00. Actual evidence is 1-0, mean margin +32.00, 0 close wins and 0 close losses, standardized schedule probability 0.921, expected wins 0.94, wins minus expectation +0.06. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-09-27

Actual record **3–1**, mean margin **+9.50**; close wins/losses 0/0, dominant wins 1. Frozen standardized schedule probability **0.452** (larger is easier), expected wins **2.31**, wins minus expectation **+0.69**. Reference: retained_context_v1.4_posterior, ordinal 29, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 35.77 | 29 | 26.02 | 6.493 | 15–52 | 7–74 | 2–97 | 0.166/0.204/0.434 | +0.00 / +4.62 |
| 0.25 | 34.97 | 28 | 24.42 | 6.434 | 16–50 | 8–70 | 3–93 | 0.151/0.189/0.431 | -0.81 / +3.82 |
| 0.5 | 33.66 | 30 | 22.82 | 6.356 | 16–47 | 8–66 | 3–88 | 0.144/0.182/0.440 | -2.11 / +2.51 |
| 0.75 | 32.33 | 32 | 21.34 | 6.274 | 16–45 | 8–63 | 3–83 | 0.139/0.179/0.454 | -3.44 / +1.18 |
| 1 | 31.15 | 29 | 20.04 | 6.195 | 16–43 | 9–60 | 3–79 | 0.134/0.175/0.468 | -4.62 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856659 / 2026-09-05 | home | Youngstown State (fcs) | 45–13 | +32 | 0.925 | 13.57/11.86 | +22.62 |
| 401856674 / 2026-09-12 | home | Alabama (fbs) | 17–45 | -28 | 0.239 | 6.82/4.95 | -10.69 |
| 401856694 / 2026-09-19 | away | Texas A&M (fbs) | 31–21 | +10 | 0.282 | 19.81/13.19 | -9.31 |
| 401864576 / 2026-09-26 | home | South Alabama (fbs) | 45–21 | +24 | 0.865 | 88.35/22.09 | +17.84 |

Answers to the original questions at every candidate:

**Does a strong team's evaluation depend on wins or dominant margins?**

- At lambda=0, E[rank]=35.77, ordinal=29, SD=26.02, entropy=6.493 bits, central 80%=7-74; versus Record/Play the mean-rank changes are +0.00/+4.62. Actual evidence is 3-1, mean margin +9.50, 0 close wins and 0 close losses, standardized schedule probability 0.452, expected wins 2.31, wins minus expectation +0.69. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=34.97, ordinal=28, SD=24.42, entropy=6.434 bits, central 80%=8-70; versus Record/Play the mean-rank changes are -0.81/+3.82. Actual evidence is 3-1, mean margin +9.50, 0 close wins and 0 close losses, standardized schedule probability 0.452, expected wins 2.31, wins minus expectation +0.69. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=33.66, ordinal=30, SD=22.82, entropy=6.356 bits, central 80%=8-66; versus Record/Play the mean-rank changes are -2.11/+2.51. Actual evidence is 3-1, mean margin +9.50, 0 close wins and 0 close losses, standardized schedule probability 0.452, expected wins 2.31, wins minus expectation +0.69. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=32.33, ordinal=32, SD=21.34, entropy=6.274 bits, central 80%=8-63; versus Record/Play the mean-rank changes are -3.44/+1.18. Actual evidence is 3-1, mean margin +9.50, 0 close wins and 0 close losses, standardized schedule probability 0.452, expected wins 2.31, wins minus expectation +0.69. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=31.15, ordinal=29, SD=20.04, entropy=6.195 bits, central 80%=9-60; versus Record/Play the mean-rank changes are -4.62/+0.00. Actual evidence is 3-1, mean margin +9.50, 0 close wins and 0 close losses, standardized schedule probability 0.452, expected wins 2.31, wins minus expectation +0.69. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-10-25T12-59-22.246522Z

Actual record **7–1**, mean margin **+8.12**; close wins/losses 2/0, dominant wins 1. Frozen standardized schedule probability **0.350** (larger is easier), expected wins **4.68**, wins minus expectation **+2.32**. Reference: retained_context_v1.4_posterior, ordinal 20, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 9.06 | 2 | 8.32 | 4.530 | 3–12 | 2–19 | 1–32 | 0.692/0.765/0.954 | +0.00 / -7.36 |
| 0.25 | 10.83 | 7 | 9.11 | 4.777 | 4–14 | 2–22 | 1–36 | 0.599/0.684/0.933 | +1.77 / -5.59 |
| 0.5 | 12.70 | 11 | 9.71 | 4.969 | 6–17 | 3–25 | 1–39 | 0.499/0.594/0.908 | +3.63 / -3.72 |
| 0.75 | 14.59 | 17 | 10.12 | 5.107 | 8–19 | 4–27 | 2–41 | 0.401/0.500/0.879 | +5.53 / -1.83 |
| 1 | 16.42 | 20 | 10.37 | 5.196 | 9–21 | 5–30 | 2–43 | 0.310/0.409/0.845 | +7.36 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856659 / 2026-09-05 | home | Youngstown State (fcs) | 45–13 | +32 | 0.944 | 6.83/5.38 | +23.56 |
| 401856674 / 2026-09-12 | home | Alabama (fbs) | 17–45 | -28 | 0.328 | 5.67/3.41 | -6.57 |
| 401856694 / 2026-09-19 | away | Texas A&M (fbs) | 31–21 | +10 | 0.436 | 21.04/10.61 | -2.52 |
| 401864576 / 2026-09-26 | home | South Alabama (fbs) | 45–21 | +24 | 0.931 | 88.52/17.87 | +22.61 |
| 401856709 / 2026-10-03 | away | South Carolina (fbs) | 35–34 | +1 | 0.457 | 23.26/12.93 | -1.73 |
| 401856715 / 2026-10-10 | home | LSU (fbs) | 35–34 | +1 | 0.300 | 4.44/2.81 | -7.71 |
| 401856722 / 2026-10-17 | away | Oklahoma (fbs) | 21–13 | +8 | 0.413 | 19.36/10.38 | -3.44 |
| 401856729 / 2026-10-24 | home | Vanderbilt (fbs) | 34–17 | +17 | 0.871 | 65.49/19.67 | +17.19 |

Answers to the original questions at every candidate:

**Does a strong team's evaluation depend on wins or dominant margins?**

- At lambda=0, E[rank]=9.06, ordinal=2, SD=8.32, entropy=4.530 bits, central 80%=2-19; versus Record/Play the mean-rank changes are +0.00/-7.36. Actual evidence is 7-1, mean margin +8.12, 2 close wins and 0 close losses, standardized schedule probability 0.350, expected wins 4.68, wins minus expectation +2.32. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=10.83, ordinal=7, SD=9.11, entropy=4.777 bits, central 80%=2-22; versus Record/Play the mean-rank changes are +1.77/-5.59. Actual evidence is 7-1, mean margin +8.12, 2 close wins and 0 close losses, standardized schedule probability 0.350, expected wins 4.68, wins minus expectation +2.32. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=12.70, ordinal=11, SD=9.71, entropy=4.969 bits, central 80%=3-25; versus Record/Play the mean-rank changes are +3.63/-3.72. Actual evidence is 7-1, mean margin +8.12, 2 close wins and 0 close losses, standardized schedule probability 0.350, expected wins 4.68, wins minus expectation +2.32. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=14.59, ordinal=17, SD=10.12, entropy=5.107 bits, central 80%=4-27; versus Record/Play the mean-rank changes are +5.53/-1.83. Actual evidence is 7-1, mean margin +8.12, 2 close wins and 0 close losses, standardized schedule probability 0.350, expected wins 4.68, wins minus expectation +2.32. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=16.42, ordinal=20, SD=10.37, entropy=5.196 bits, central 80%=5-30; versus Record/Play the mean-rank changes are +7.36/+0.00. Actual evidence is 7-1, mean margin +8.12, 2 close wins and 0 close losses, standardized schedule probability 0.350, expected wins 4.68, wins minus expectation +2.32. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -29.19 | -28 | -12.58 | 0.345 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -28.08 | -33 | -13.23 | 0.346 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -27.42 | -31 | -13.94 | 0.353 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -26.87 | -29 | -14.58 | 0.361 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -26.29 | -29 | -15.10 | 0.368 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 4 / 445 | -26.71 | -27 | -17.69 | 0.584 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 4 / 445 | -24.14 | -21 | -15.31 | 0.546 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 4 / 445 | -20.97 | -19 | -13.11 | 0.495 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 4 / 445 | -17.74 | -15 | -11.22 | 0.439 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 4 / 445 | -14.73 | -9 | -9.67 | 0.383 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 LSU (99)

Frozen categories: boundary_10.

### 2026-09-08

Actual record **1–0**, mean margin **+41.00**; close wins/losses 0/0, dominant wins 1. Frozen standardized schedule probability **0.502** (larger is easier), expected wins **0.89**, wins minus expectation **+0.11**. Reference: retained_context_v1.4_posterior, ordinal 1, band elite, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 42.89 | 8 | 32.74 | 6.761 | 15–65 | 6–92 | 2–117 | 0.178/0.209/0.387 | +0.00 / +21.04 |
| 0.25 | 36.69 | 1 | 30.94 | 6.573 | 11–56 | 4–84 | 1–111 | 0.237/0.275/0.471 | -6.21 / +14.84 |
| 0.5 | 30.90 | 1 | 28.54 | 6.340 | 8–46 | 3–74 | 1–104 | 0.301/0.345/0.555 | -11.99 / +9.05 |
| 0.75 | 25.90 | 1 | 25.79 | 6.086 | 7–37 | 3–64 | 1–96 | 0.366/0.415/0.633 | -16.99 / +4.05 |
| 1 | 21.85 | 1 | 23.01 | 5.833 | 5–30 | 2–55 | 1–87 | 0.427/0.479/0.699 | -21.04 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856660 / 2026-09-05 | home | Clemson (fbs) | 51–10 | +41 | 0.886 | 54.42/29.14 | +19.70 |

Answers to the original questions at every candidate:

**Does this boundary crossing reflect meaningful evidence or close teams?**

- At lambda=0, E[rank]=42.89, ordinal=8, SD=32.74, entropy=6.761 bits, central 80%=6-92; versus Record/Play the mean-rank changes are +0.00/+21.04. Latent top10/12/25 mass is 0.178/0.209/0.387. Adjacent expected-rank gaps are 0.11072491296908282/0.13558104270044424. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=36.69, ordinal=1, SD=30.94, entropy=6.573 bits, central 80%=4-84; versus Record/Play the mean-rank changes are -6.21/+14.84. Latent top10/12/25 mass is 0.237/0.275/0.471. Adjacent expected-rank gaps are None/1.231285444579612. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=30.90, ordinal=1, SD=28.54, entropy=6.340 bits, central 80%=3-74; versus Record/Play the mean-rank changes are -11.99/+9.05. Latent top10/12/25 mass is 0.301/0.345/0.555. Adjacent expected-rank gaps are None/4.176837337887125. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=25.90, ordinal=1, SD=25.79, entropy=6.086 bits, central 80%=3-64; versus Record/Play the mean-rank changes are -16.99/+4.05. Latent top10/12/25 mass is 0.366/0.415/0.633. Adjacent expected-rank gaps are None/5.366312119204498. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=21.85, ordinal=1, SD=23.01, entropy=5.833 bits, central 80%=2-55; versus Record/Play the mean-rank changes are -21.04/+0.00. Latent top10/12/25 mass is 0.427/0.479/0.699. Adjacent expected-rank gaps are None/4.874868761068555. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

### 2026-09-27

Actual record **3–1**, mean margin **+23.25**; close wins/losses 0/0, dominant wins 3. Frozen standardized schedule probability **0.348** (larger is easier), expected wins **3.17**, wins minus expectation **-0.17**. Reference: retained_context_v1.4_posterior, ordinal 3, band elite, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 26.40 | 11 | 21.95 | 6.110 | 10–38 | 4–58 | 1–83 | 0.276/0.329/0.595 | +0.00 / +19.41 |
| 0.25 | 17.75 | 9 | 16.80 | 5.546 | 6–24 | 3–41 | 1–64 | 0.437/0.503/0.770 | -8.65 / +10.76 |
| 0.5 | 12.25 | 6 | 12.36 | 4.991 | 4–16 | 2–28 | 1–48 | 0.589/0.658/0.884 | -14.15 / +5.25 |
| 0.75 | 8.97 | 5 | 9.12 | 4.518 | 3–12 | 1–20 | 1–34 | 0.710/0.774/0.945 | -17.43 / +1.97 |
| 1 | 7.00 | 3 | 6.94 | 4.136 | 2–9 | 1–15 | 1–26 | 0.798/0.853/0.975 | -19.41 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856660 / 2026-09-05 | home | Clemson (fbs) | 51–10 | +41 | 0.886 | 37.01/18.90 | +18.30 |
| 401867796 / 2026-09-12 | home | Louisiana Tech (fbs) | 45–14 | +31 | 0.951 | 71.24/25.62 | +25.26 |
| 401856688 / 2026-09-19 | away | Ole Miss (fbs) | 24–32 | -8 | 0.554 | 13.83/11.34 | +2.16 |
| 401856702 / 2026-09-26 | home | Texas A&M (fbs) | 35–6 | +29 | 0.780 | 19.81/13.19 | +11.96 |

Answers to the original questions at every candidate:

**Does this boundary crossing reflect meaningful evidence or close teams?**

- At lambda=0, E[rank]=26.40, ordinal=11, SD=21.95, entropy=6.110 bits, central 80%=4-58; versus Record/Play the mean-rank changes are +0.00/+19.41. Latent top10/12/25 mass is 0.276/0.329/0.595. Adjacent expected-rank gaps are 2.2524579929383997/0.05418968627985521. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=17.75, ordinal=9, SD=16.80, entropy=5.546 bits, central 80%=3-41; versus Record/Play the mean-rank changes are -8.65/+10.76. Latent top10/12/25 mass is 0.437/0.503/0.770. Adjacent expected-rank gaps are 0.0325663664619853/2.458131069757414. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=12.25, ordinal=6, SD=12.36, entropy=4.991 bits, central 80%=2-28; versus Record/Play the mean-rank changes are -14.15/+5.25. Latent top10/12/25 mass is 0.589/0.658/0.884. Adjacent expected-rank gaps are 1.2589111866512184/1.0479566285639326. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=8.97, ordinal=5, SD=9.12, entropy=4.518 bits, central 80%=1-20; versus Record/Play the mean-rank changes are -17.43/+1.97. Latent top10/12/25 mass is 0.710/0.774/0.945. Adjacent expected-rank gaps are 0.4061549093264354/0.22127160506172672. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=7.00, ordinal=3, SD=6.94, entropy=4.136 bits, central 80%=1-15; versus Record/Play the mean-rank changes are -19.41/+0.00. Latent top10/12/25 mass is 0.798/0.853/0.975. Adjacent expected-rank gaps are 0.04246544445418188/0.09279689836611027. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

### 2026-10-25T12-59-22.246522Z

Actual record **6–2**, mean margin **+24.12**; close wins/losses 0/1, dominant wins 5. Frozen standardized schedule probability **0.378** (larger is easier), expected wins **6.49**, wins minus expectation **-0.49**. Reference: retained_context_v1.4_posterior, ordinal 1, band elite, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 16.46 | 10 | 13.35 | 5.395 | 7–22 | 3–35 | 1–52 | 0.405/0.483/0.806 | +0.00 / +12.33 |
| 0.25 | 9.83 | 6 | 8.50 | 4.644 | 4–13 | 2–20 | 1–33 | 0.649/0.728/0.946 | -6.62 / +5.70 |
| 0.5 | 6.71 | 3 | 5.70 | 4.062 | 3–9 | 1–14 | 1–21 | 0.810/0.871/0.987 | -9.75 / +2.58 |
| 0.75 | 5.09 | 1 | 4.20 | 3.629 | 2–7 | 1–10 | 1–16 | 0.900/0.941/0.997 | -11.37 / +0.96 |
| 1 | 4.13 | 1 | 3.32 | 3.291 | 2–6 | 1–8 | 1–13 | 0.948/0.973/0.999 | -12.33 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856660 / 2026-09-05 | home | Clemson (fbs) | 51–10 | +41 | 0.928 | 42.41/16.72 | +21.26 |
| 401867796 / 2026-09-12 | home | Louisiana Tech (fbs) | 45–14 | +31 | 0.951 | 60.17/19.02 | +24.12 |
| 401856688 / 2026-09-19 | away | Ole Miss (fbs) | 24–32 | -8 | 0.457 | 6.32/4.71 | -1.51 |
| 401856702 / 2026-09-26 | home | Texas A&M (fbs) | 35–6 | +29 | 0.837 | 21.04/10.61 | +14.62 |
| 401856706 / 2026-10-03 | home | McNeese (fcs) | 63–14 | +49 | 0.999 | 88.21/18.27 | +51.87 |
| 401856715 / 2026-10-10 | away | Kentucky (fbs) | 34–35 | -1 | 0.700 | 19.91/10.61 | +7.71 |
| 401856720 / 2026-10-17 | home | Mississippi State (fbs) | 31–0 | +31 | 0.885 | 28.71/13.19 | +17.74 |
| 401856728 / 2026-10-24 | away | Auburn (fbs) | 38–17 | +21 | 0.732 | 23.62/13.00 | +9.17 |

Answers to the original questions at every candidate:

**Does this boundary crossing reflect meaningful evidence or close teams?**

- At lambda=0, E[rank]=16.46, ordinal=10, SD=13.35, entropy=5.395 bits, central 80%=3-35; versus Record/Play the mean-rank changes are +0.00/+12.33. Latent top10/12/25 mass is 0.405/0.483/0.806. Adjacent expected-rank gaps are 1.280700594735471/0.2720388671786438. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=9.83, ordinal=6, SD=8.50, entropy=4.644 bits, central 80%=2-20; versus Record/Play the mean-rank changes are -6.62/+5.70. Latent top10/12/25 mass is 0.649/0.728/0.946. Adjacent expected-rank gaps are 0.07777397228178806/0.997238096897286. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=6.71, ordinal=3, SD=5.70, entropy=4.062 bits, central 80%=1-14; versus Record/Play the mean-rank changes are -9.75/+2.58. Latent top10/12/25 mass is 0.810/0.871/0.987. Adjacent expected-rank gaps are 0.3779148326664652/0.7653239891855028. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=5.09, ordinal=1, SD=4.20, entropy=3.629 bits, central 80%=1-10; versus Record/Play the mean-rank changes are -11.37/+0.96. Latent top10/12/25 mass is 0.900/0.941/0.997. Adjacent expected-rank gaps are None/0.2072701203335683. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=4.13, ordinal=1, SD=3.32, entropy=3.291 bits, central 80%=1-8; versus Record/Play the mean-rank changes are -12.33/+0.00. Latent top10/12/25 mass is 0.948/0.973/0.999. Adjacent expected-rank gaps are None/0.584638850487722. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -16.49 | +3 | -10.79 | 0.227 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -18.93 | +8 | -14.15 | 0.299 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -18.65 | +5 | -16.19 | 0.343 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -16.93 | +4 | -16.68 | 0.364 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -14.85 | +2 | -16.07 | 0.374 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 4 / 445 | -9.95 | -1 | -8.60 | 0.212 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 4 / 445 | -7.92 | -3 | -8.30 | 0.229 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 4 / 445 | -5.54 | -3 | -6.65 | 0.220 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 4 / 445 | -3.88 | -4 | -4.92 | 0.209 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 4 / 445 | -2.87 | -2 | -3.62 | 0.200 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Liberty (2335)

Frozen categories: excellent_record_easy_schedule.

### 2026-09-08

Actual record **0–1**, mean margin **-7.00**; close wins/losses 0/1, dominant wins 0. Frozen standardized schedule probability **0.374** (larger is easier), expected wins **0.30**, wins minus expectation **-0.30**. Reference: retained_context_v1.4_posterior, ordinal 77, band middle, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 83.46 | 119 | 36.09 | 7.003 | 55–115 | 31–129 | 12–136 | 0.019/0.025/0.075 | +0.00 / +15.55 |
| 0.25 | 78.62 | 98 | 35.96 | 7.044 | 50–109 | 27–126 | 11–135 | 0.024/0.031/0.090 | -4.84 / +10.71 |
| 0.5 | 74.46 | 90 | 35.59 | 7.059 | 46–104 | 25–123 | 10–134 | 0.028/0.036/0.105 | -9.00 / +6.55 |
| 0.75 | 70.92 | 84 | 35.10 | 7.057 | 43–99 | 23–119 | 9–132 | 0.033/0.042/0.119 | -12.54 / +3.01 |
| 1 | 67.91 | 77 | 34.58 | 7.045 | 40–95 | 21–116 | 8–131 | 0.037/0.047/0.132 | -15.55 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401869960 / 2026-09-05 | away | James Madison (fbs) | 13–20 | -7 | 0.297 | 54.51/29.04 | -9.67 |

Answers to the original questions at every candidate:

**Does this excellent record receive disproportionate easier-schedule credit?**

- At lambda=0, E[rank]=83.46, ordinal=119, SD=36.09, entropy=7.003 bits, central 80%=31-129; versus Record/Play the mean-rank changes are +0.00/+15.55. Actual evidence is 0-1, mean margin -7.00, 0 close wins and 1 close losses, standardized schedule probability 0.374, expected wins 0.30, wins minus expectation -0.30. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=78.62, ordinal=98, SD=35.96, entropy=7.044 bits, central 80%=27-126; versus Record/Play the mean-rank changes are -4.84/+10.71. Actual evidence is 0-1, mean margin -7.00, 0 close wins and 1 close losses, standardized schedule probability 0.374, expected wins 0.30, wins minus expectation -0.30. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=74.46, ordinal=90, SD=35.59, entropy=7.059 bits, central 80%=25-123; versus Record/Play the mean-rank changes are -9.00/+6.55. Actual evidence is 0-1, mean margin -7.00, 0 close wins and 1 close losses, standardized schedule probability 0.374, expected wins 0.30, wins minus expectation -0.30. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=70.92, ordinal=84, SD=35.10, entropy=7.057 bits, central 80%=23-119; versus Record/Play the mean-rank changes are -12.54/+3.01. Actual evidence is 0-1, mean margin -7.00, 0 close wins and 1 close losses, standardized schedule probability 0.374, expected wins 0.30, wins minus expectation -0.30. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=67.91, ordinal=77, SD=34.58, entropy=7.045 bits, central 80%=21-116; versus Record/Play the mean-rank changes are -15.55/+0.00. Actual evidence is 0-1, mean margin -7.00, 0 close wins and 1 close losses, standardized schedule probability 0.374, expected wins 0.30, wins minus expectation -0.30. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-09-27

Actual record **3–1**, mean margin **+11.75**; close wins/losses 1/1, dominant wins 1. Frozen standardized schedule probability **0.650** (larger is easier), expected wins **2.69**, wins minus expectation **+0.31**. Reference: retained_context_v1.4_posterior, ordinal 62, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 60.39 | 66 | 32.19 | 6.959 | 34–85 | 18–106 | 7–122 | 0.046/0.059/0.165 | +0.00 / +9.05 |
| 0.25 | 57.86 | 63 | 30.47 | 6.891 | 33–81 | 19–101 | 7–117 | 0.042/0.054/0.166 | -2.53 / +6.52 |
| 0.5 | 55.48 | 64 | 28.88 | 6.820 | 32–77 | 19–96 | 8–113 | 0.038/0.050/0.168 | -4.91 / +4.14 |
| 0.75 | 53.30 | 64 | 27.46 | 6.749 | 31–73 | 19–92 | 8–109 | 0.036/0.047/0.170 | -7.09 / +1.95 |
| 1 | 51.34 | 62 | 26.21 | 6.681 | 30–70 | 19–88 | 9–106 | 0.033/0.045/0.174 | -9.05 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401869960 / 2026-09-05 | away | James Madison (fbs) | 13–20 | -7 | 0.198 | 25.75/16.85 | -13.94 |
| 401868187 / 2026-09-12 | home | Gardner-Webb (fcs) | 24–23 | +1 | 0.910 | 27.21/17.49 | +22.35 |
| 401866422 / 2026-09-19 | home | Ball State (fbs) | 51–15 | +36 | 0.945 | 128.32/8.66 | +25.25 |
| 401869941 / 2026-09-24 | away | Coastal Carolina (fbs) | 34–17 | +17 | 0.637 | 92.31/21.93 | +5.74 |

Answers to the original questions at every candidate:

**Does this excellent record receive disproportionate easier-schedule credit?**

- At lambda=0, E[rank]=60.39, ordinal=66, SD=32.19, entropy=6.959 bits, central 80%=18-106; versus Record/Play the mean-rank changes are +0.00/+9.05. Actual evidence is 3-1, mean margin +11.75, 1 close wins and 1 close losses, standardized schedule probability 0.650, expected wins 2.69, wins minus expectation +0.31. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=57.86, ordinal=63, SD=30.47, entropy=6.891 bits, central 80%=19-101; versus Record/Play the mean-rank changes are -2.53/+6.52. Actual evidence is 3-1, mean margin +11.75, 1 close wins and 1 close losses, standardized schedule probability 0.650, expected wins 2.69, wins minus expectation +0.31. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=55.48, ordinal=64, SD=28.88, entropy=6.820 bits, central 80%=19-96; versus Record/Play the mean-rank changes are -4.91/+4.14. Actual evidence is 3-1, mean margin +11.75, 1 close wins and 1 close losses, standardized schedule probability 0.650, expected wins 2.69, wins minus expectation +0.31. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=53.30, ordinal=64, SD=27.46, entropy=6.749 bits, central 80%=19-92; versus Record/Play the mean-rank changes are -7.09/+1.95. Actual evidence is 3-1, mean margin +11.75, 1 close wins and 1 close losses, standardized schedule probability 0.650, expected wins 2.69, wins minus expectation +0.31. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=51.34, ordinal=62, SD=26.21, entropy=6.681 bits, central 80%=19-88; versus Record/Play the mean-rank changes are -9.05/+0.00. Actual evidence is 3-1, mean margin +11.75, 1 close wins and 1 close losses, standardized schedule probability 0.650, expected wins 2.69, wins minus expectation +0.31. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-10-25T12-59-22.246522Z

Actual record **6–1**, mean margin **+17.57**; close wins/losses 1/1, dominant wins 3. Frozen standardized schedule probability **0.702** (larger is easier), expected wins **5.60**, wins minus expectation **+0.40**. Reference: retained_context_v1.4_posterior, ordinal 43, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 46.94 | 47 | 27.00 | 6.697 | 25–66 | 13–85 | 5–104 | 0.072/0.092/0.252 | +0.00 / +11.05 |
| 0.25 | 43.73 | 48 | 24.29 | 6.557 | 25–60 | 14–78 | 6–97 | 0.063/0.082/0.259 | -3.21 / +7.83 |
| 0.5 | 40.83 | 46 | 21.93 | 6.414 | 24–55 | 15–72 | 6–90 | 0.056/0.076/0.270 | -6.11 / +4.93 |
| 0.75 | 38.21 | 45 | 19.86 | 6.271 | 24–51 | 15–66 | 7–84 | 0.052/0.071/0.286 | -8.73 / +2.32 |
| 1 | 35.89 | 43 | 18.03 | 6.129 | 23–47 | 15–61 | 7–78 | 0.049/0.069/0.305 | -11.05 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401869960 / 2026-09-05 | away | James Madison (fbs) | 13–20 | -7 | 0.310 | 28.88/14.66 | -7.79 |
| 401868187 / 2026-09-12 | home | Gardner-Webb (fcs) | 24–23 | +1 | 0.984 | 50.47/18.22 | +33.58 |
| 401866422 / 2026-09-19 | home | Ball State (fbs) | 51–15 | +36 | 0.962 | 123.97/9.05 | +27.27 |
| 401869941 / 2026-09-24 | away | Coastal Carolina (fbs) | 34–17 | +17 | 0.772 | 96.57/16.61 | +11.43 |
| 401871050 / 2026-10-02 | away | Delaware (fbs) | 30–14 | +16 | 0.780 | 97.93/16.79 | +11.86 |
| 401870766 / 2026-10-08 | home | Sam Houston (fbs) | 35–3 | +32 | 0.955 | 121.84/11.09 | +26.47 |
| 401871056 / 2026-10-22 | away | Kennesaw State (fbs) | 34–6 | +28 | 0.833 | 107.33/14.93 | +14.82 |

Answers to the original questions at every candidate:

**Does this excellent record receive disproportionate easier-schedule credit?**

- At lambda=0, E[rank]=46.94, ordinal=47, SD=27.00, entropy=6.697 bits, central 80%=13-85; versus Record/Play the mean-rank changes are +0.00/+11.05. Actual evidence is 6-1, mean margin +17.57, 1 close wins and 1 close losses, standardized schedule probability 0.702, expected wins 5.60, wins minus expectation +0.40. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=43.73, ordinal=48, SD=24.29, entropy=6.557 bits, central 80%=14-78; versus Record/Play the mean-rank changes are -3.21/+7.83. Actual evidence is 6-1, mean margin +17.57, 1 close wins and 1 close losses, standardized schedule probability 0.702, expected wins 5.60, wins minus expectation +0.40. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=40.83, ordinal=46, SD=21.93, entropy=6.414 bits, central 80%=15-72; versus Record/Play the mean-rank changes are -6.11/+4.93. Actual evidence is 6-1, mean margin +17.57, 1 close wins and 1 close losses, standardized schedule probability 0.702, expected wins 5.60, wins minus expectation +0.40. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=38.21, ordinal=45, SD=19.86, entropy=6.271 bits, central 80%=15-66; versus Record/Play the mean-rank changes are -8.73/+2.32. Actual evidence is 6-1, mean margin +17.57, 1 close wins and 1 close losses, standardized schedule probability 0.702, expected wins 5.60, wins minus expectation +0.40. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=35.89, ordinal=43, SD=18.03, entropy=6.129 bits, central 80%=15-61; versus Record/Play the mean-rank changes are -11.05/+0.00. Actual evidence is 6-1, mean margin +17.57, 1 close wins and 1 close losses, standardized schedule probability 0.702, expected wins 5.60, wins minus expectation +0.40. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -23.07 | -53 | -3.90 | 0.272 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -20.76 | -35 | -5.49 | 0.252 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -18.98 | -26 | -6.71 | 0.238 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -17.62 | -20 | -7.64 | 0.229 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -16.57 | -15 | -8.37 | 0.227 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 3 / 445 | -13.45 | -19 | -5.19 | 0.179 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 3 / 445 | -14.13 | -15 | -6.19 | 0.205 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 3 / 445 | -14.66 | -18 | -6.95 | 0.229 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 3 / 445 | -15.09 | -19 | -7.60 | 0.253 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 3 / 445 | -15.45 | -19 | -8.18 | 0.275 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Massachusetts (113)

Frozen categories: weak_schedule_pair.

### 2026-09-08

Actual record **1–0**, mean margin **+16.00**; close wins/losses 0/0, dominant wins 0. Frozen standardized schedule probability **0.590** (larger is easier), expected wins **0.21**, wins minus expectation **+0.79**. Reference: retained_context_v1.4_posterior, ordinal 27, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 43.33 | 10 | 32.66 | 6.770 | 16–66 | 6–93 | 2–117 | 0.172/0.203/0.379 | +0.00 / +3.45 |
| 0.25 | 42.38 | 9 | 32.15 | 6.744 | 15–64 | 6–91 | 2–115 | 0.175/0.207/0.388 | -0.95 / +2.50 |
| 0.5 | 41.49 | 16 | 31.66 | 6.719 | 15–62 | 6–89 | 2–114 | 0.178/0.211/0.396 | -1.84 / +1.60 |
| 0.75 | 40.66 | 22 | 31.18 | 6.694 | 15–61 | 6–88 | 2–113 | 0.182/0.215/0.404 | -2.67 / +0.77 |
| 1 | 39.89 | 27 | 30.72 | 6.670 | 14–60 | 6–86 | 2–111 | 0.185/0.219/0.412 | -3.45 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858423 / 2026-09-03 | away | Rutgers (fbs) | 37–21 | +16 | 0.207 | 89.67/30.79 | -14.88 |

Answers to the original questions at every candidate:

**For identical weak-team records, does the easier schedule narrow uncertainty?**

- At lambda=0, E[rank]=43.33, ordinal=10, SD=32.66, entropy=6.770 bits, central 80%=6-93; versus Record/Play the mean-rank changes are +0.00/+3.45. The frozen peer Stanford is 1-1 with standardized schedule probability 0.469, E[rank]=65.68, SD=34.79, entropy=7.044, widths 50/80/95=57/96/123; this team's expected-rank gap is -22.34 and SD gap -2.13. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=42.38, ordinal=9, SD=32.15, entropy=6.744 bits, central 80%=6-91; versus Record/Play the mean-rank changes are -0.95/+2.50. The frozen peer Stanford is 1-1 with standardized schedule probability 0.469, E[rank]=74.42, SD=33.94, entropy=7.020, widths 50/80/95=56/94/121; this team's expected-rank gap is -32.04 and SD gap -1.78. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=41.49, ordinal=16, SD=31.66, entropy=6.719 bits, central 80%=6-89; versus Record/Play the mean-rank changes are -1.84/+1.60. The frozen peer Stanford is 1-1 with standardized schedule probability 0.469, E[rank]=81.76, SD=32.57, entropy=6.951, widths 50/80/95=52/89/116; this team's expected-rank gap is -40.27 and SD gap -0.91. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=40.66, ordinal=22, SD=31.18, entropy=6.694 bits, central 80%=6-88; versus Record/Play the mean-rank changes are -2.67/+0.77. The frozen peer Stanford is 1-1 with standardized schedule probability 0.469, E[rank]=87.85, SD=30.99, entropy=6.859, widths 50/80/95=48/85/112; this team's expected-rank gap is -47.19 and SD gap +0.19. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=39.89, ordinal=27, SD=30.72, entropy=6.670 bits, central 80%=6-86; versus Record/Play the mean-rank changes are -3.45/+0.00. The frozen peer Stanford is 1-1 with standardized schedule probability 0.469, E[rank]=92.91, SD=29.35, entropy=6.757, widths 50/80/95=45/80/107; this team's expected-rank gap is -53.02 and SD gap +1.37. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### 2026-09-27

Actual record **4–0**, mean margin **+23.50**; close wins/losses 0/0, dominant wins 1. Frozen standardized schedule probability **0.826** (larger is easier), expected wins **2.86**, wins minus expectation **+1.14**. Reference: retained_context_v1.4_posterior, ordinal 52, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 41.49 | 35 | 30.21 | 6.698 | 16–62 | 6–86 | 2–109 | 0.165/0.196/0.379 | +0.00 / -6.38 |
| 0.25 | 43.78 | 41 | 29.25 | 6.725 | 20–64 | 9–87 | 3–107 | 0.125/0.151/0.326 | +2.29 / -4.09 |
| 0.5 | 45.41 | 47 | 28.23 | 6.719 | 23–65 | 11–86 | 3–106 | 0.095/0.117/0.285 | +3.92 / -2.46 |
| 0.75 | 46.74 | 48 | 27.27 | 6.697 | 26–66 | 14–86 | 4–105 | 0.072/0.090/0.249 | +5.25 / -1.13 |
| 1 | 47.87 | 52 | 26.41 | 6.668 | 27–67 | 16–86 | 6–104 | 0.054/0.068/0.218 | +6.38 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858423 / 2026-09-03 | away | Rutgers (fbs) | 37–21 | +16 | 0.263 | 74.37/24.86 | -10.80 |
| 401866420 / 2026-09-12 | home | Sacred Heart (fcs) | 37–10 | +27 | 0.929 | 69.56/22.39 | +24.75 |
| 401866424 / 2026-09-19 | home | Stonehill (fcs) | 36–14 | +22 | 0.944 | 76.22/21.03 | +26.58 |
| 401866429 / 2026-09-27 | away | Sacramento State (fbs) | 35–6 | +29 | 0.725 | 129.43/7.40 | +9.44 |

Answers to the original questions at every candidate:

**For identical weak-team records, does the easier schedule narrow uncertainty?**

- At lambda=0, E[rank]=41.49, ordinal=35, SD=30.21, entropy=6.698 bits, central 80%=6-86; versus Record/Play the mean-rank changes are +0.00/-6.38. The frozen peer Stanford is 2-2 with standardized schedule probability 0.442, E[rank]=59.79, SD=30.50, entropy=6.898, widths 50/80/95=49/83/111; this team's expected-rank gap is -18.30 and SD gap -0.29. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=43.78, ordinal=41, SD=29.25, entropy=6.725 bits, central 80%=9-87; versus Record/Play the mean-rank changes are +2.29/-4.09. The frozen peer Stanford is 2-2 with standardized schedule probability 0.442, E[rank]=70.69, SD=29.50, entropy=6.875, widths 50/80/95=46/80/108; this team's expected-rank gap is -26.91 and SD gap -0.25. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=45.41, ordinal=47, SD=28.23, entropy=6.719 bits, central 80%=11-86; versus Record/Play the mean-rank changes are +3.92/-2.46. The frozen peer Stanford is 2-2 with standardized schedule probability 0.442, E[rank]=79.28, SD=27.88, entropy=6.793, widths 50/80/95=43/76/103; this team's expected-rank gap is -33.87 and SD gap +0.35. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=46.74, ordinal=48, SD=27.27, entropy=6.697 bits, central 80%=14-86; versus Record/Play the mean-rank changes are +5.25/-1.13. The frozen peer Stanford is 2-2 with standardized schedule probability 0.442, E[rank]=86.16, SD=26.05, entropy=6.687, widths 50/80/95=39/71/99; this team's expected-rank gap is -39.42 and SD gap +1.22. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=47.87, ordinal=52, SD=26.41, entropy=6.668 bits, central 80%=16-86; versus Record/Play the mean-rank changes are +6.38/+0.00. The frozen peer Stanford is 2-2 with standardized schedule probability 0.442, E[rank]=91.80, SD=24.18, entropy=6.568, widths 50/80/95=35/64/93; this team's expected-rank gap is -43.92 and SD gap +2.23. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### 2026-10-25T12-59-22.246522Z

Actual record **4–4**, mean margin **+5.38**; close wins/losses 0/2, dominant wins 1. Frozen standardized schedule probability **0.769** (larger is easier), expected wins **3.87**, wins minus expectation **+0.13**. Reference: retained_context_v1.4_posterior, ordinal 112, band weak, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 105.25 | 119 | 20.25 | 6.252 | 93–121 | 77–129 | 57–135 | 0.000/0.000/0.001 | +0.00 / +3.70 |
| 0.25 | 103.75 | 117 | 19.35 | 6.219 | 92–118 | 77–126 | 59–133 | 0.000/0.000/0.000 | -1.50 / +2.19 |
| 0.5 | 102.75 | 117 | 18.45 | 6.169 | 92–116 | 78–124 | 60–131 | 0.000/0.000/0.000 | -2.50 / +1.20 |
| 0.75 | 102.06 | 114 | 17.61 | 6.112 | 92–115 | 78–123 | 61–130 | 0.000/0.000/0.000 | -3.19 / +0.50 |
| 1 | 101.56 | 112 | 16.84 | 6.055 | 91–114 | 79–121 | 63–128 | 0.000/0.000/0.000 | -3.70 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858423 / 2026-09-03 | away | Rutgers (fbs) | 37–21 | +16 | 0.108 | 70.63/21.26 | -19.54 |
| 401866420 / 2026-09-12 | home | Sacred Heart (fcs) | 37–10 | +27 | 0.934 | 88.88/15.79 | +23.48 |
| 401866424 / 2026-09-19 | home | Stonehill (fcs) | 36–14 | +22 | 0.900 | 76.83/16.87 | +19.77 |
| 401866429 / 2026-09-27 | away | Sacramento State (fbs) | 35–6 | +29 | 0.628 | 133.98/3.37 | +4.94 |
| 401866433 / 2026-10-03 | home | Eastern Michigan (fbs) | 14–38 | -24 | 0.487 | 114.47/11.48 | -0.48 |
| 401866440 / 2026-10-10 | home | Miami (OH) (fbs) | 19–20 | -1 | 0.337 | 97.00/16.63 | -6.52 |
| 401866442 / 2026-10-17 | away | Buffalo (fbs) | 18–25 | -7 | 0.375 | 117.64/11.75 | -5.09 |
| 401861965 / 2026-10-23 | away | UConn (fbs) | 16–35 | -19 | 0.099 | 68.46/19.90 | -20.08 |

Answers to the original questions at every candidate:

**For identical weak-team records, does the easier schedule narrow uncertainty?**

- At lambda=0, E[rank]=105.25, ordinal=119, SD=20.25, entropy=6.252 bits, central 80%=77-129; versus Record/Play the mean-rank changes are +0.00/+3.70. The frozen peer Stanford is 4-4 with standardized schedule probability 0.438, E[rank]=55.49, SD=25.71, entropy=6.672, widths 50/80/95=40/70/96; this team's expected-rank gap is +49.76 and SD gap -5.46. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=103.75, ordinal=117, SD=19.35, entropy=6.219 bits, central 80%=77-126; versus Record/Play the mean-rank changes are -1.50/+2.19. The frozen peer Stanford is 4-4 with standardized schedule probability 0.438, E[rank]=74.70, SD=25.29, entropy=6.672, widths 50/80/95=39/69/95; this team's expected-rank gap is +29.05 and SD gap -5.93. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=102.75, ordinal=117, SD=18.45, entropy=6.169 bits, central 80%=78-124; versus Record/Play the mean-rank changes are -2.50/+1.20. The frozen peer Stanford is 4-4 with standardized schedule probability 0.438, E[rank]=89.14, SD=22.32, entropy=6.476, widths 50/80/95=32/59/88; this team's expected-rank gap is +13.62 and SD gap -3.87. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=102.06, ordinal=114, SD=17.61, entropy=6.112 bits, central 80%=78-123; versus Record/Play the mean-rank changes are -3.19/+0.50. The frozen peer Stanford is 4-4 with standardized schedule probability 0.438, E[rank]=99.04, SD=18.91, entropy=6.223, widths 50/80/95=26/49/74; this team's expected-rank gap is +3.02 and SD gap -1.30. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=101.56, ordinal=112, SD=16.84, entropy=6.055 bits, central 80%=79-121; versus Record/Play the mean-rank changes are -3.70/+0.00. The frozen peer Stanford is 4-4 with standardized schedule probability 0.438, E[rank]=105.76, SD=16.16, entropy=5.992, widths 50/80/95=23/42/64; this team's expected-rank gap is -4.20 and SD gap +0.68. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -1.84 | +25 | -2.45 | 0.035 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | +1.40 | +32 | -2.90 | 0.086 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | +3.92 | +31 | -3.43 | 0.135 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | +6.08 | +26 | -3.91 | 0.178 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | +7.99 | +25 | -4.31 | 0.215 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 4 / 445 | +63.76 | +84 | -9.96 | 0.751 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 4 / 445 | +59.97 | +76 | -9.90 | 0.745 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 4 / 445 | +57.34 | +70 | -9.77 | 0.746 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 4 / 445 | +55.32 | +66 | -9.66 | 0.750 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 4 / 445 | +53.68 | +60 | -9.57 | 0.754 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Memphis (235)

Frozen categories: narrow_win_weaker_opponent.

### 2026-09-08

Actual record **2–0**, mean margin **+12.00**; close wins/losses 1/0, dominant wins 0. Frozen standardized schedule probability **0.583** (larger is easier), expected wins **1.36**, wins minus expectation **+0.64**. Reference: retained_context_v1.4_posterior, ordinal 40, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 38.11 | 2 | 29.37 | 6.606 | 14–57 | 5–82 | 2–106 | 0.194/0.229/0.427 | +0.00 / -9.14 |
| 0.25 | 40.89 | 6 | 29.46 | 6.676 | 16–60 | 7–85 | 2–107 | 0.158/0.190/0.379 | +2.78 / -6.36 |
| 0.5 | 43.31 | 18 | 29.39 | 6.722 | 19–63 | 8–86 | 3–108 | 0.129/0.158/0.338 | +5.20 / -3.94 |
| 0.75 | 45.41 | 31 | 29.22 | 6.751 | 21–66 | 10–88 | 3–109 | 0.106/0.132/0.304 | +7.30 / -1.84 |
| 1 | 47.25 | 40 | 29.00 | 6.767 | 24–68 | 12–89 | 4–109 | 0.088/0.111/0.274 | +9.14 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401862693 / 2026-08-30 | away | UNLV (fbs) | 27–21 | +6 | 0.547 | 65.92/27.89 | +1.99 |
| 401862695 / 2026-09-05 | home | Arkansas State (fbs) | 42–24 | +18 | 0.816 | 92.60/27.66 | +15.86 |

Answers to the original questions at every candidate:

**How much credit does a narrow win against a weaker opponent receive?**

- At lambda=0, E[rank]=38.11, ordinal=2, SD=29.37, entropy=6.606 bits, central 80%=5-82; versus Record/Play the mean-rank changes are +0.00/-9.14. Actual evidence is 2-0, mean margin +12.00, 1 close wins and 0 close losses, standardized schedule probability 0.583, expected wins 1.36, wins minus expectation +0.64. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=40.89, ordinal=6, SD=29.46, entropy=6.676 bits, central 80%=7-85; versus Record/Play the mean-rank changes are +2.78/-6.36. Actual evidence is 2-0, mean margin +12.00, 1 close wins and 0 close losses, standardized schedule probability 0.583, expected wins 1.36, wins minus expectation +0.64. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=43.31, ordinal=18, SD=29.39, entropy=6.722 bits, central 80%=8-86; versus Record/Play the mean-rank changes are +5.20/-3.94. Actual evidence is 2-0, mean margin +12.00, 1 close wins and 0 close losses, standardized schedule probability 0.583, expected wins 1.36, wins minus expectation +0.64. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=45.41, ordinal=31, SD=29.22, entropy=6.751 bits, central 80%=10-88; versus Record/Play the mean-rank changes are +7.30/-1.84. Actual evidence is 2-0, mean margin +12.00, 1 close wins and 0 close losses, standardized schedule probability 0.583, expected wins 1.36, wins minus expectation +0.64. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=47.25, ordinal=40, SD=29.00, entropy=6.767 bits, central 80%=12-89; versus Record/Play the mean-rank changes are +9.14/+0.00. Actual evidence is 2-0, mean margin +12.00, 1 close wins and 0 close losses, standardized schedule probability 0.583, expected wins 1.36, wins minus expectation +0.64. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-09-27

Actual record **3–1**, mean margin **+7.50**; close wins/losses 1/0, dominant wins 0. Frozen standardized schedule probability **0.608** (larger is easier), expected wins **2.49**, wins minus expectation **+0.51**. Reference: retained_context_v1.4_posterior, ordinal 76, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 49.58 | 48 | 29.49 | 6.806 | 26–70 | 13–92 | 5–112 | 0.074/0.094/0.248 | +0.00 / -14.77 |
| 0.25 | 55.13 | 56 | 28.81 | 6.820 | 32–76 | 18–96 | 8–113 | 0.040/0.052/0.172 | +5.55 / -9.22 |
| 0.5 | 59.21 | 69 | 27.84 | 6.793 | 37–80 | 23–98 | 11–114 | 0.021/0.029/0.121 | +9.63 / -5.14 |
| 0.75 | 62.18 | 71 | 26.82 | 6.751 | 41–82 | 27–99 | 15–114 | 0.011/0.017/0.087 | +12.60 / -2.17 |
| 1 | 64.35 | 76 | 25.87 | 6.705 | 44–84 | 30–99 | 18–113 | 0.006/0.010/0.065 | +14.77 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401862693 / 2026-08-30 | away | UNLV (fbs) | 27–21 | +6 | 0.546 | 80.01/24.00 | +1.86 |
| 401862695 / 2026-09-05 | home | Arkansas State (fbs) | 42–24 | +18 | 0.778 | 95.51/21.33 | +12.62 |
| 401860881 / 2026-09-12 | away | Boise State (fbs) | 20–38 | -18 | 0.182 | 24.45/16.05 | -14.73 |
| 401862772 / 2026-09-19 | home | UT Martin (fcs) | 45–21 | +24 | 0.982 | 70.87/22.58 | +34.51 |

Answers to the original questions at every candidate:

**How much credit does a narrow win against a weaker opponent receive?**

- At lambda=0, E[rank]=49.58, ordinal=48, SD=29.49, entropy=6.806 bits, central 80%=13-92; versus Record/Play the mean-rank changes are +0.00/-14.77. Actual evidence is 3-1, mean margin +7.50, 1 close wins and 0 close losses, standardized schedule probability 0.608, expected wins 2.49, wins minus expectation +0.51. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=55.13, ordinal=56, SD=28.81, entropy=6.820 bits, central 80%=18-96; versus Record/Play the mean-rank changes are +5.55/-9.22. Actual evidence is 3-1, mean margin +7.50, 1 close wins and 0 close losses, standardized schedule probability 0.608, expected wins 2.49, wins minus expectation +0.51. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=59.21, ordinal=69, SD=27.84, entropy=6.793 bits, central 80%=23-98; versus Record/Play the mean-rank changes are +9.63/-5.14. Actual evidence is 3-1, mean margin +7.50, 1 close wins and 0 close losses, standardized schedule probability 0.608, expected wins 2.49, wins minus expectation +0.51. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=62.18, ordinal=71, SD=26.82, entropy=6.751 bits, central 80%=27-99; versus Record/Play the mean-rank changes are +12.60/-2.17. Actual evidence is 3-1, mean margin +7.50, 1 close wins and 0 close losses, standardized schedule probability 0.608, expected wins 2.49, wins minus expectation +0.51. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=64.35, ordinal=76, SD=25.87, entropy=6.705 bits, central 80%=30-99; versus Record/Play the mean-rank changes are +14.77/+0.00. Actual evidence is 3-1, mean margin +7.50, 1 close wins and 0 close losses, standardized schedule probability 0.608, expected wins 2.49, wins minus expectation +0.51. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-10-25T12-59-22.246522Z

Actual record **6–2**, mean margin **+12.38**; close wins/losses 2/0, dominant wins 1. Frozen standardized schedule probability **0.649** (larger is easier), expected wins **5.70**, wins minus expectation **+0.30**. Reference: retained_context_v1.4_posterior, ordinal 67, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 47.58 | 49 | 24.23 | 6.583 | 29–64 | 17–81 | 8–98 | 0.041/0.056/0.203 | +0.00 / -5.89 |
| 0.25 | 50.11 | 53 | 22.78 | 6.514 | 33–66 | 22–82 | 12–97 | 0.020/0.029/0.146 | +2.53 / -3.37 |
| 0.5 | 51.71 | 62 | 21.51 | 6.440 | 35–67 | 25–81 | 15–96 | 0.010/0.015/0.108 | +4.13 / -1.77 |
| 0.75 | 52.76 | 65 | 20.46 | 6.370 | 37–67 | 27–81 | 18–95 | 0.005/0.008/0.081 | +5.18 / -0.71 |
| 1 | 53.48 | 67 | 19.58 | 6.308 | 39–67 | 29–80 | 20–94 | 0.002/0.004/0.062 | +5.89 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401862693 / 2026-08-30 | away | UNLV (fbs) | 27–21 | +6 | 0.571 | 74.98/19.61 | +2.75 |
| 401862695 / 2026-09-05 | home | Arkansas State (fbs) | 42–24 | +18 | 0.800 | 89.60/16.93 | +12.93 |
| 401860881 / 2026-09-12 | away | Boise State (fbs) | 20–38 | -18 | 0.187 | 21.12/11.54 | -13.64 |
| 401862772 / 2026-09-19 | home | UT Martin (fcs) | 45–21 | +24 | 0.989 | 70.49/18.43 | +36.27 |
| 401862787 / 2026-10-03 | away | Charlotte (fbs) | 59–8 | +51 | 0.920 | 129.09/7.17 | +21.02 |
| 401862798 / 2026-10-10 | home | UAB (fbs) | 38–13 | +25 | 0.850 | 99.95/16.18 | +16.06 |
| 401862801 / 2026-10-16 | away | Tulane (fbs) | 17–31 | -14 | 0.580 | 76.24/19.43 | +3.08 |
| 401862807 / 2026-10-22 | home | East Carolina (fbs) | 41–34 | +7 | 0.807 | 91.21/17.92 | +13.43 |

Answers to the original questions at every candidate:

**How much credit does a narrow win against a weaker opponent receive?**

- At lambda=0, E[rank]=47.58, ordinal=49, SD=24.23, entropy=6.583 bits, central 80%=17-81; versus Record/Play the mean-rank changes are +0.00/-5.89. Removing the frozen 401862807 observation changes this team's mean rank by +4.44, ordinal by +5, SD by +1.46, TV=0.070. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=0.25, E[rank]=50.11, ordinal=53, SD=22.78, entropy=6.514 bits, central 80%=22-82; versus Record/Play the mean-rank changes are +2.53/-3.37. Removing the frozen 401862807 observation changes this team's mean rank by +1.51, ordinal by +3, SD by +1.18, TV=0.031. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=0.5, E[rank]=51.71, ordinal=62, SD=21.51, entropy=6.440 bits, central 80%=25-81; versus Record/Play the mean-rank changes are +4.13/-1.77. Removing the frozen 401862807 observation changes this team's mean rank by -0.63, ordinal by -2, SD by +0.97, TV=0.031. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=0.75, E[rank]=52.76, ordinal=65, SD=20.46, entropy=6.370 bits, central 80%=27-81; versus Record/Play the mean-rank changes are +5.18/-0.71. Removing the frozen 401862807 observation changes this team's mean rank by -2.26, ordinal by -4, SD by +0.78, TV=0.054. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=1, E[rank]=53.48, ordinal=67, SD=19.58, entropy=6.308 bits, central 80%=29-80; versus Record/Play the mean-rank changes are +5.89/+0.00. Removing the frozen 401862807 observation changes this team's mean rank by -3.56, ordinal by -7, SD by +0.60, TV=0.080. This measures the total network response to removing one observation, not an additive contribution.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 2 / 359 | +11.47 | +46 | +0.12 | 0.180 |
| 2026-09-08 → 2026-09-27 | 0.25 | 2 / 359 | +14.24 | +50 | -0.65 | 0.214 |
| 2026-09-08 → 2026-09-27 | 0.5 | 2 / 359 | +15.90 | +51 | -1.56 | 0.237 |
| 2026-09-08 → 2026-09-27 | 0.75 | 2 / 359 | +16.76 | +40 | -2.40 | 0.252 |
| 2026-09-08 → 2026-09-27 | 1 | 2 / 359 | +17.10 | +36 | -3.13 | 0.263 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 4 / 445 | -2.00 | +1 | -5.26 | 0.114 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 4 / 445 | -5.02 | -3 | -6.04 | 0.139 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 4 / 445 | -7.50 | -7 | -6.32 | 0.168 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 4 / 445 | -9.41 | -6 | -6.36 | 0.197 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 4 / 445 | -10.87 | -9 | -6.29 | 0.223 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Michigan (130)

Frozen categories: competitive_elite_loss.

### 2026-09-08

Actual record **1–0**, mean margin **+1.00**; close wins/losses 1/0, dominant wins 0. Frozen standardized schedule probability **0.462** (larger is easier), expected wins **0.74**, wins minus expectation **+0.26**. Reference: retained_context_v1.4_posterior, ordinal 83, band middle, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 50.97 | 19 | 34.92 | 6.934 | 21–77 | 8–103 | 2–124 | 0.127/0.151/0.299 | +0.00 / -19.48 |
| 0.25 | 56.80 | 44 | 35.35 | 7.009 | 27–84 | 11–108 | 3–126 | 0.091/0.110/0.237 | +5.83 / -13.65 |
| 0.5 | 61.98 | 65 | 35.35 | 7.045 | 33–90 | 15–112 | 5–129 | 0.065/0.080/0.187 | +11.01 / -8.46 |
| 0.75 | 66.52 | 74 | 35.10 | 7.055 | 38–94 | 19–115 | 7–130 | 0.046/0.058/0.149 | +15.54 / -3.93 |
| 1 | 70.45 | 83 | 34.71 | 7.049 | 43–98 | 23–118 | 9–131 | 0.033/0.042/0.119 | +19.48 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858428 / 2026-09-05 | home | Western Michigan (fbs) | 13–12 | +1 | 0.744 | 48.29/28.40 | +11.52 |

Answers to the original questions at every candidate:

**How does a competitive loss to an excellent opponent affect each candidate?**

- At lambda=0, E[rank]=50.97, ordinal=19, SD=34.92, entropy=6.934 bits, central 80%=8-103; versus Record/Play the mean-rank changes are +0.00/-19.48. Actual evidence is 1-0, mean margin +1.00, 1 close wins and 0 close losses, standardized schedule probability 0.462, expected wins 0.74, wins minus expectation +0.26. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=56.80, ordinal=44, SD=35.35, entropy=7.009 bits, central 80%=11-108; versus Record/Play the mean-rank changes are +5.83/-13.65. Actual evidence is 1-0, mean margin +1.00, 1 close wins and 0 close losses, standardized schedule probability 0.462, expected wins 0.74, wins minus expectation +0.26. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=61.98, ordinal=65, SD=35.35, entropy=7.045 bits, central 80%=15-112; versus Record/Play the mean-rank changes are +11.01/-8.46. Actual evidence is 1-0, mean margin +1.00, 1 close wins and 0 close losses, standardized schedule probability 0.462, expected wins 0.74, wins minus expectation +0.26. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=66.52, ordinal=74, SD=35.10, entropy=7.055 bits, central 80%=19-115; versus Record/Play the mean-rank changes are +15.54/-3.93. Actual evidence is 1-0, mean margin +1.00, 1 close wins and 0 close losses, standardized schedule probability 0.462, expected wins 0.74, wins minus expectation +0.26. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=70.45, ordinal=83, SD=34.71, entropy=7.049 bits, central 80%=23-118; versus Record/Play the mean-rank changes are +19.48/+0.00. Actual evidence is 1-0, mean margin +1.00, 1 close wins and 0 close losses, standardized schedule probability 0.462, expected wins 0.74, wins minus expectation +0.26. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-09-27

Actual record **3–1**, mean margin **+10.50**; close wins/losses 2/1, dominant wins 1. Frozen standardized schedule probability **0.491** (larger is easier), expected wins **2.93**, wins minus expectation **+0.07**. Reference: retained_context_v1.4_posterior, ordinal 41, band middle, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 38.45 | 33 | 26.44 | 6.562 | 17–55 | 8–77 | 3–100 | 0.137/0.170/0.387 | +0.00 / -1.75 |
| 0.25 | 38.69 | 33 | 25.28 | 6.532 | 18–55 | 9–75 | 3–97 | 0.117/0.149/0.368 | +0.24 / -1.51 |
| 0.5 | 39.12 | 38 | 24.26 | 6.503 | 20–55 | 11–74 | 4–94 | 0.098/0.128/0.347 | +0.67 / -1.08 |
| 0.75 | 39.65 | 40 | 23.35 | 6.473 | 21–55 | 12–73 | 5–92 | 0.081/0.108/0.324 | +1.20 / -0.56 |
| 1 | 40.20 | 41 | 22.52 | 6.440 | 23–55 | 13–72 | 6–91 | 0.066/0.091/0.302 | +1.75 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858428 / 2026-09-05 | home | Western Michigan (fbs) | 13–12 | +1 | 0.851 | 63.37/24.19 | +16.77 |
| 401856679 / 2026-09-12 | home | Oklahoma (fbs) | 17–10 | +7 | 0.604 | 22.24/13.56 | +4.33 |
| 401858450 / 2026-09-19 | home | UTEP (fbs) | 52–17 | +35 | 0.984 | 125.63/10.64 | +33.66 |
| 401858463 / 2026-09-26 | home | Iowa (fbs) | 19–20 | -1 | 0.486 | 13.51/10.12 | -0.47 |

Answers to the original questions at every candidate:

**How does a competitive loss to an excellent opponent affect each candidate?**

- At lambda=0, E[rank]=38.45, ordinal=33, SD=26.44, entropy=6.562 bits, central 80%=8-77; versus Record/Play the mean-rank changes are +0.00/-1.75. Actual evidence is 3-1, mean margin +10.50, 2 close wins and 1 close losses, standardized schedule probability 0.491, expected wins 2.93, wins minus expectation +0.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=38.69, ordinal=33, SD=25.28, entropy=6.532 bits, central 80%=9-75; versus Record/Play the mean-rank changes are +0.24/-1.51. Actual evidence is 3-1, mean margin +10.50, 2 close wins and 1 close losses, standardized schedule probability 0.491, expected wins 2.93, wins minus expectation +0.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=39.12, ordinal=38, SD=24.26, entropy=6.503 bits, central 80%=11-74; versus Record/Play the mean-rank changes are +0.67/-1.08. Actual evidence is 3-1, mean margin +10.50, 2 close wins and 1 close losses, standardized schedule probability 0.491, expected wins 2.93, wins minus expectation +0.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=39.65, ordinal=40, SD=23.35, entropy=6.473 bits, central 80%=12-73; versus Record/Play the mean-rank changes are +1.20/-0.56. Actual evidence is 3-1, mean margin +10.50, 2 close wins and 1 close losses, standardized schedule probability 0.491, expected wins 2.93, wins minus expectation +0.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=40.20, ordinal=41, SD=22.52, entropy=6.440 bits, central 80%=13-72; versus Record/Play the mean-rank changes are +1.75/+0.00. Actual evidence is 3-1, mean margin +10.50, 2 close wins and 1 close losses, standardized schedule probability 0.491, expected wins 2.93, wins minus expectation +0.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-10-25T12-59-22.246522Z

Actual record **4–3**, mean margin **+4.57**; close wins/losses 3/3, dominant wins 1. Frozen standardized schedule probability **0.399** (larger is easier), expected wins **4.26**, wins minus expectation **-0.26**. Reference: retained_context_v1.4_posterior, ordinal 49, band middle, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 41.15 | 40 | 22.84 | 6.451 | 23–56 | 14–74 | 7–93 | 0.053/0.075/0.289 | +0.00 / +1.43 |
| 0.25 | 40.25 | 44 | 21.08 | 6.352 | 24–54 | 15–70 | 8–88 | 0.042/0.063/0.281 | -0.90 / +0.53 |
| 0.5 | 39.83 | 44 | 19.77 | 6.272 | 25–53 | 16–67 | 9–84 | 0.033/0.052/0.269 | -1.32 / +0.11 |
| 0.75 | 39.70 | 47 | 18.76 | 6.207 | 25–52 | 17–65 | 10–81 | 0.026/0.043/0.257 | -1.45 / -0.02 |
| 1 | 39.72 | 49 | 17.95 | 6.153 | 26–52 | 18–64 | 11–79 | 0.020/0.035/0.243 | -1.43 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858428 / 2026-09-05 | home | Western Michigan (fbs) | 13–12 | +1 | 0.840 | 66.16/18.81 | +15.27 |
| 401856679 / 2026-09-12 | home | Oklahoma (fbs) | 17–10 | +7 | 0.511 | 19.36/10.38 | +0.47 |
| 401858450 / 2026-09-19 | home | UTEP (fbs) | 52–17 | +35 | 0.988 | 131.45/5.32 | +34.53 |
| 401858463 / 2026-09-26 | home | Iowa (fbs) | 19–20 | -1 | 0.484 | 17.58/10.17 | -0.58 |
| 401858474 / 2026-10-03 | away | Minnesota (fbs) | 14–20 | -6 | 0.594 | 45.26/18.19 | +3.60 |
| 401858490 / 2026-10-17 | home | Penn State (fbs) | 21–18 | +3 | 0.587 | 25.74/12.51 | +3.46 |
| 401858496 / 2026-10-24 | home | Indiana (fbs) | 6–13 | -7 | 0.260 | 4.85/3.44 | -9.55 |

Answers to the original questions at every candidate:

**How does a competitive loss to an excellent opponent affect each candidate?**

- At lambda=0, E[rank]=41.15, ordinal=40, SD=22.84, entropy=6.451 bits, central 80%=14-74; versus Record/Play the mean-rank changes are +0.00/+1.43. Removing the frozen 401858496 observation changes this team's mean rank by -4.04, ordinal by -4, SD by +0.07, TV=0.081. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=0.25, E[rank]=40.25, ordinal=44, SD=21.08, entropy=6.352 bits, central 80%=15-70; versus Record/Play the mean-rank changes are -0.90/+0.53. Removing the frozen 401858496 observation changes this team's mean rank by -1.71, ordinal by -4, SD by +0.60, TV=0.043. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=0.5, E[rank]=39.83, ordinal=44, SD=19.77, entropy=6.272 bits, central 80%=16-67; versus Record/Play the mean-rank changes are -1.32/+0.11. Removing the frozen 401858496 observation changes this team's mean rank by +0.03, ordinal by +0, SD by +0.87, TV=0.026. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=0.75, E[rank]=39.70, ordinal=47, SD=18.76, entropy=6.207 bits, central 80%=17-65; versus Record/Play the mean-rank changes are -1.45/-0.02. Removing the frozen 401858496 observation changes this team's mean rank by +1.36, ordinal by +0, SD by +1.00, TV=0.030. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=1, E[rank]=39.72, ordinal=49, SD=17.95, entropy=6.153 bits, central 80%=18-64; versus Record/Play the mean-rank changes are -1.43/+0.00. Removing the frozen 401858496 observation changes this team's mean rank by +2.42, ordinal by +2, SD by +1.04, TV=0.049. This measures the total network response to removing one observation, not an additive contribution.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -12.52 | +14 | -8.47 | 0.172 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -18.10 | -11 | -10.07 | 0.240 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -22.86 | -27 | -11.09 | 0.302 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -26.87 | -34 | -11.75 | 0.358 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -30.25 | -42 | -12.19 | 0.406 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 3 / 445 | +2.70 | +7 | -3.61 | 0.128 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 3 / 445 | +1.56 | +11 | -4.20 | 0.131 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 3 / 445 | +0.71 | +6 | -4.50 | 0.131 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 3 / 445 | +0.05 | +7 | -4.59 | 0.129 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 3 / 445 | -0.48 | +8 | -4.57 | 0.126 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Mississippi State (344)

Frozen categories: middling_record_difficult_schedule.

### 2026-09-08

Actual record **1–0**, mean margin **+49.00**; close wins/losses 0/0, dominant wins 1. Frozen standardized schedule probability **0.911** (larger is easier), expected wins **0.97**, wins minus expectation **+0.03**. Reference: retained_context_v1.4_posterior, ordinal 18, band strong, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 61.75 | 44 | 37.57 | 7.071 | 29–92 | 12–115 | 3–131 | 0.088/0.105/0.217 | +0.00 / +27.22 |
| 0.25 | 52.91 | 35 | 35.87 | 6.969 | 22–80 | 9–106 | 2–126 | 0.122/0.145/0.287 | -8.84 / +18.37 |
| 0.5 | 45.47 | 23 | 33.54 | 6.823 | 17–69 | 7–97 | 2–119 | 0.158/0.187/0.357 | -16.29 / +10.93 |
| 0.75 | 39.41 | 19 | 31.00 | 6.659 | 14–59 | 5–87 | 2–112 | 0.194/0.229/0.422 | -22.35 / +4.87 |
| 1 | 34.54 | 18 | 28.52 | 6.492 | 12–51 | 5–77 | 1–105 | 0.229/0.269/0.481 | -27.22 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856662 / 2026-09-05 | home | UL Monroe (fbs) | 62–13 | +49 | 0.969 | 126.47/12.08 | +30.62 |

Answers to the original questions at every candidate:

**Does opponent evidence explain this team's ordinary record?**

- At lambda=0, E[rank]=61.75, ordinal=44, SD=37.57, entropy=7.071 bits, central 80%=12-115; versus Record/Play the mean-rank changes are +0.00/+27.22. Actual evidence is 1-0, mean margin +49.00, 0 close wins and 0 close losses, standardized schedule probability 0.911, expected wins 0.97, wins minus expectation +0.03. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=52.91, ordinal=35, SD=35.87, entropy=6.969 bits, central 80%=9-106; versus Record/Play the mean-rank changes are -8.84/+18.37. Actual evidence is 1-0, mean margin +49.00, 0 close wins and 0 close losses, standardized schedule probability 0.911, expected wins 0.97, wins minus expectation +0.03. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=45.47, ordinal=23, SD=33.54, entropy=6.823 bits, central 80%=7-97; versus Record/Play the mean-rank changes are -16.29/+10.93. Actual evidence is 1-0, mean margin +49.00, 0 close wins and 0 close losses, standardized schedule probability 0.911, expected wins 0.97, wins minus expectation +0.03. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=39.41, ordinal=19, SD=31.00, entropy=6.659 bits, central 80%=5-87; versus Record/Play the mean-rank changes are -22.35/+4.87. Actual evidence is 1-0, mean margin +49.00, 0 close wins and 0 close losses, standardized schedule probability 0.911, expected wins 0.97, wins minus expectation +0.03. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=34.54, ordinal=18, SD=28.52, entropy=6.492 bits, central 80%=5-77; versus Record/Play the mean-rank changes are -27.22/+0.00. Actual evidence is 1-0, mean margin +49.00, 0 close wins and 0 close losses, standardized schedule probability 0.911, expected wins 0.97, wins minus expectation +0.03. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-09-27

Actual record **4–0**, mean margin **+22.00**; close wins/losses 2/0, dominant wins 1. Frozen standardized schedule probability **0.450** (larger is easier), expected wins **2.93**, wins minus expectation **+1.07**. Reference: retained_context_v1.4_posterior, ordinal 10, band elite, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 14.65 | 3 | 15.14 | 5.257 | 4–19 | 2–35 | 1–58 | 0.535/0.599/0.829 | +0.00 / +1.98 |
| 0.25 | 13.98 | 4 | 13.96 | 5.191 | 4–19 | 2–32 | 1–53 | 0.539/0.607/0.845 | -0.67 / +1.31 |
| 0.5 | 13.42 | 8 | 12.94 | 5.131 | 4–18 | 2–30 | 1–50 | 0.543/0.613/0.859 | -1.23 / +0.75 |
| 0.75 | 12.99 | 10 | 12.07 | 5.080 | 5–17 | 2–29 | 1–46 | 0.546/0.619/0.872 | -1.66 / +0.32 |
| 1 | 12.67 | 10 | 11.35 | 5.037 | 5–17 | 2–27 | 1–44 | 0.546/0.622/0.882 | -1.98 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856662 / 2026-09-05 | home | UL Monroe (fbs) | 62–13 | +49 | 0.991 | 130.60/6.70 | +36.41 |
| 401856677 / 2026-09-12 | away | Minnesota (fbs) | 38–13 | +25 | 0.640 | 38.88/20.31 | +5.55 |
| 401856691 / 2026-09-19 | away | South Carolina (fbs) | 41–34 | +7 | 0.622 | 36.01/19.36 | +4.80 |
| 401856703 / 2026-09-26 | home | Missouri (fbs) | 31–24 | +7 | 0.678 | 26.93/17.45 | +7.52 |

Answers to the original questions at every candidate:

**Does opponent evidence explain this team's ordinary record?**

- At lambda=0, E[rank]=14.65, ordinal=3, SD=15.14, entropy=5.257 bits, central 80%=2-35; versus Record/Play the mean-rank changes are +0.00/+1.98. Actual evidence is 4-0, mean margin +22.00, 2 close wins and 0 close losses, standardized schedule probability 0.450, expected wins 2.93, wins minus expectation +1.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=13.98, ordinal=4, SD=13.96, entropy=5.191 bits, central 80%=2-32; versus Record/Play the mean-rank changes are -0.67/+1.31. Actual evidence is 4-0, mean margin +22.00, 2 close wins and 0 close losses, standardized schedule probability 0.450, expected wins 2.93, wins minus expectation +1.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=13.42, ordinal=8, SD=12.94, entropy=5.131 bits, central 80%=2-30; versus Record/Play the mean-rank changes are -1.23/+0.75. Actual evidence is 4-0, mean margin +22.00, 2 close wins and 0 close losses, standardized schedule probability 0.450, expected wins 2.93, wins minus expectation +1.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=12.99, ordinal=10, SD=12.07, entropy=5.080 bits, central 80%=2-29; versus Record/Play the mean-rank changes are -1.66/+0.32. Actual evidence is 4-0, mean margin +22.00, 2 close wins and 0 close losses, standardized schedule probability 0.450, expected wins 2.93, wins minus expectation +1.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=12.67, ordinal=10, SD=11.35, entropy=5.037 bits, central 80%=2-27; versus Record/Play the mean-rank changes are -1.98/+0.00. Actual evidence is 4-0, mean margin +22.00, 2 close wins and 0 close losses, standardized schedule probability 0.450, expected wins 2.93, wins minus expectation +1.07. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-10-25T12-59-22.246522Z

Actual record **4–3**, mean margin **+2.29**; close wins/losses 2/0, dominant wins 1. Frozen standardized schedule probability **0.304** (larger is easier), expected wins **3.14**, wins minus expectation **+0.86**. Reference: retained_context_v1.4_posterior, ordinal 35, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 27.12 | 26 | 18.70 | 6.030 | 13–37 | 7–54 | 3–74 | 0.177/0.231/0.564 | +0.00 / -1.52 |
| 0.25 | 27.60 | 28 | 17.62 | 5.989 | 15–37 | 9–52 | 4–71 | 0.143/0.193/0.541 | +0.48 / -1.04 |
| 0.5 | 28.03 | 30 | 16.50 | 5.929 | 16–37 | 10–51 | 5–69 | 0.111/0.157/0.517 | +0.91 / -0.61 |
| 0.75 | 28.38 | 33 | 15.40 | 5.855 | 17–37 | 11–49 | 6–66 | 0.085/0.125/0.495 | +1.26 / -0.26 |
| 1 | 28.64 | 35 | 14.37 | 5.773 | 18–36 | 13–47 | 7–64 | 0.063/0.099/0.474 | +1.52 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856662 / 2026-09-05 | home | UL Monroe (fbs) | 62–13 | +49 | 0.985 | 129.48/6.52 | +32.86 |
| 401856677 / 2026-09-12 | away | Minnesota (fbs) | 38–13 | +25 | 0.557 | 45.26/18.19 | +2.13 |
| 401856691 / 2026-09-19 | away | South Carolina (fbs) | 41–34 | +7 | 0.357 | 23.26/12.93 | -5.70 |
| 401856703 / 2026-09-26 | home | Missouri (fbs) | 31–24 | +7 | 0.419 | 16.30/10.73 | -3.10 |
| 401856707 / 2026-10-03 | home | Alabama (fbs) | 23–56 | -33 | 0.239 | 5.67/3.41 | -10.35 |
| 401856720 / 2026-10-17 | away | LSU (fbs) | 0–31 | -31 | 0.115 | 4.44/2.81 | -17.74 |
| 401856730 / 2026-10-24 | home | Oklahoma (fbs) | 15–23 | -8 | 0.469 | 19.36/10.38 | -1.14 |

Answers to the original questions at every candidate:

**Does opponent evidence explain this team's ordinary record?**

- At lambda=0, E[rank]=27.12, ordinal=26, SD=18.70, entropy=6.030 bits, central 80%=7-54; versus Record/Play the mean-rank changes are +0.00/-1.52. Actual evidence is 4-3, mean margin +2.29, 2 close wins and 0 close losses, standardized schedule probability 0.304, expected wins 3.14, wins minus expectation +0.86. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=27.60, ordinal=28, SD=17.62, entropy=5.989 bits, central 80%=9-52; versus Record/Play the mean-rank changes are +0.48/-1.04. Actual evidence is 4-3, mean margin +2.29, 2 close wins and 0 close losses, standardized schedule probability 0.304, expected wins 3.14, wins minus expectation +0.86. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=28.03, ordinal=30, SD=16.50, entropy=5.929 bits, central 80%=10-51; versus Record/Play the mean-rank changes are +0.91/-0.61. Actual evidence is 4-3, mean margin +2.29, 2 close wins and 0 close losses, standardized schedule probability 0.304, expected wins 3.14, wins minus expectation +0.86. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=28.38, ordinal=33, SD=15.40, entropy=5.855 bits, central 80%=11-49; versus Record/Play the mean-rank changes are +1.26/-0.26. Actual evidence is 4-3, mean margin +2.29, 2 close wins and 0 close losses, standardized schedule probability 0.304, expected wins 3.14, wins minus expectation +0.86. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=28.64, ordinal=35, SD=14.37, entropy=5.773 bits, central 80%=13-47; versus Record/Play the mean-rank changes are +1.52/+0.00. Actual evidence is 4-3, mean margin +2.29, 2 close wins and 0 close losses, standardized schedule probability 0.304, expected wins 3.14, wins minus expectation +0.86. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -47.10 | -41 | -22.43 | 0.613 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -38.93 | -31 | -21.91 | 0.557 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -32.04 | -15 | -20.60 | 0.505 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -26.42 | -9 | -18.93 | 0.456 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -21.87 | -8 | -17.17 | 0.411 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 3 / 445 | +12.47 | +23 | +3.56 | 0.368 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 3 / 445 | +13.62 | +24 | +3.66 | 0.416 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 3 / 445 | +14.61 | +22 | +3.56 | 0.463 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 3 / 445 | +15.39 | +23 | +3.33 | 0.506 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 3 / 445 | +15.97 | +25 | +3.02 | 0.545 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Missouri (142)

Frozen categories: boundary_25.

### 2026-09-08

Actual record **1–0**, mean margin **+40.00**; close wins/losses 0/0, dominant wins 1. Frozen standardized schedule probability **0.951** (larger is easier), expected wins **0.99**, wins minus expectation **+0.01**. Reference: retained_context_v1.4_posterior, ordinal 62, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 66.24 | 63 | 38.97 | 7.102 | 33–99 | 13–121 | 4–134 | 0.078/0.094/0.195 | +0.00 / +7.58 |
| 0.25 | 63.42 | 63 | 38.11 | 7.087 | 31–95 | 13–118 | 4–133 | 0.082/0.099/0.207 | -2.82 / +4.76 |
| 0.5 | 61.35 | 63 | 37.37 | 7.069 | 29–91 | 12–115 | 3–131 | 0.085/0.102/0.216 | -4.88 / +2.69 |
| 0.75 | 59.82 | 64 | 36.77 | 7.053 | 28–89 | 12–113 | 3–130 | 0.087/0.105/0.223 | -6.42 / +1.16 |
| 1 | 58.66 | 62 | 36.28 | 7.038 | 28–87 | 12–111 | 3–129 | 0.088/0.107/0.228 | -7.58 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856663 / 2026-09-04 | home | Arkansas-Pine Bluff (fcs) | 54–14 | +40 | 0.987 | 64.58/32.52 | +40.24 |

Answers to the original questions at every candidate:

**Does this boundary crossing reflect meaningful evidence or close teams?**

- At lambda=0, E[rank]=66.24, ordinal=63, SD=38.97, entropy=7.102 bits, central 80%=13-121; versus Record/Play the mean-rank changes are +0.00/+7.58. Latent top10/12/25 mass is 0.078/0.094/0.195. Adjacent expected-rank gaps are 0.0/1.4210854715202004e-14. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=63.42, ordinal=63, SD=38.11, entropy=7.087 bits, central 80%=13-118; versus Record/Play the mean-rank changes are -2.82/+4.76. Latent top10/12/25 mass is 0.082/0.099/0.207. Adjacent expected-rank gaps are 0.08403004394585878/0.16321774566391412. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=61.35, ordinal=63, SD=37.37, entropy=7.069 bits, central 80%=12-115; versus Record/Play the mean-rank changes are -4.88/+2.69. Latent top10/12/25 mass is 0.085/0.102/0.216. Adjacent expected-rank gaps are 0.08709429502889066/0.3692630724273158. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=59.82, ordinal=64, SD=36.77, entropy=7.053 bits, central 80%=12-113; versus Record/Play the mean-rank changes are -6.42/+1.16. Latent top10/12/25 mass is 0.087/0.105/0.223. Adjacent expected-rank gaps are 0.247609530827944/1.8665736001334068. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=58.66, ordinal=62, SD=36.28, entropy=7.038 bits, central 80%=12-111; versus Record/Play the mean-rank changes are -7.58/+0.00. Latent top10/12/25 mass is 0.088/0.107/0.228. Adjacent expected-rank gaps are 0.25573808896375283/1.2978960875769587. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

### 2026-09-27

Actual record **3–1**, mean margin **+15.00**; close wins/losses 0/1, dominant wins 1. Frozen standardized schedule probability **0.568** (larger is easier), expected wins **2.92**, wins minus expectation **+0.08**. Reference: retained_context_v1.4_posterior, ordinal 42, band middle, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 47.86 | 43 | 29.60 | 6.790 | 24–69 | 12–91 | 4–111 | 0.086/0.109/0.273 | +0.00 / +7.39 |
| 0.25 | 45.74 | 48 | 27.74 | 6.710 | 23–65 | 12–86 | 4–106 | 0.082/0.105/0.279 | -2.11 / +5.28 |
| 0.5 | 43.76 | 42 | 26.16 | 6.634 | 23–61 | 12–81 | 5–101 | 0.081/0.104/0.289 | -4.09 / +3.29 |
| 0.75 | 41.99 | 43 | 24.83 | 6.563 | 22–58 | 12–77 | 5–97 | 0.080/0.105/0.300 | -5.86 / +1.53 |
| 1 | 40.47 | 42 | 23.69 | 6.499 | 22–56 | 12–74 | 5–93 | 0.080/0.105/0.311 | -7.39 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856663 / 2026-09-04 | home | Arkansas-Pine Bluff (fcs) | 54–14 | +40 | 0.994 | 76.64/27.98 | +43.32 |
| 401856678 / 2026-09-12 | away | Kansas (fbs) | 38–21 | +17 | 0.685 | 62.79/25.61 | +7.82 |
| 401856689 / 2026-09-19 | home | Troy (fbs) | 27–17 | +10 | 0.914 | 94.41/21.76 | +22.16 |
| 401856703 / 2026-09-26 | away | Mississippi State (fbs) | 24–31 | -7 | 0.322 | 16.77/11.29 | -7.52 |

Answers to the original questions at every candidate:

**Does this boundary crossing reflect meaningful evidence or close teams?**

- At lambda=0, E[rank]=47.86, ordinal=43, SD=29.60, entropy=6.790 bits, central 80%=12-91; versus Record/Play the mean-rank changes are +0.00/+7.39. Latent top10/12/25 mass is 0.086/0.109/0.273. Adjacent expected-rank gaps are 0.22230699064670034/0.9118822340828814. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=45.74, ordinal=48, SD=27.74, entropy=6.710 bits, central 80%=12-86; versus Record/Play the mean-rank changes are -2.11/+5.28. Latent top10/12/25 mass is 0.082/0.105/0.279. Adjacent expected-rank gaps are 0.00932252380336962/0.7312195748456389. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=43.76, ordinal=42, SD=26.16, entropy=6.634 bits, central 80%=12-81; versus Record/Play the mean-rank changes are -4.09/+3.29. Latent top10/12/25 mass is 0.081/0.104/0.289. Adjacent expected-rank gaps are 1.6280026735126683/0.3366666915011223. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=41.99, ordinal=43, SD=24.83, entropy=6.563 bits, central 80%=12-77; versus Record/Play the mean-rank changes are -5.86/+1.53. Latent top10/12/25 mass is 0.080/0.105/0.300. Adjacent expected-rank gaps are 1.699376639329138/0.7216786407130158. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=40.47, ordinal=42, SD=23.69, entropy=6.499 bits, central 80%=12-74; versus Record/Play the mean-rank changes are -7.39/+0.00. Latent top10/12/25 mass is 0.080/0.105/0.311. Adjacent expected-rank gaps are 0.2644542113857682/0.293940868277069. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

### 2026-10-25T12-59-22.246522Z

Actual record **5–2**, mean margin **+11.57**; close wins/losses 0/1, dominant wins 2. Frozen standardized schedule probability **0.412** (larger is easier), expected wins **4.68**, wins minus expectation **+0.32**. Reference: retained_context_v1.4_posterior, ordinal 26, band strong, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 26.88 | 25 | 19.84 | 6.077 | 12–38 | 6–55 | 2–76 | 0.216/0.270/0.573 | +0.00 / +6.17 |
| 0.25 | 24.80 | 23 | 18.16 | 5.950 | 11–34 | 6–51 | 2–70 | 0.230/0.289/0.614 | -2.08 / +4.09 |
| 0.5 | 23.16 | 25 | 16.81 | 5.840 | 11–32 | 6–47 | 2–66 | 0.244/0.308/0.650 | -3.72 / +2.44 |
| 0.75 | 21.83 | 25 | 15.68 | 5.743 | 10–30 | 5–44 | 2–62 | 0.256/0.324/0.682 | -5.05 / +1.12 |
| 1 | 20.71 | 26 | 14.69 | 5.654 | 10–28 | 5–41 | 2–58 | 0.267/0.340/0.710 | -6.17 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401856663 / 2026-09-04 | home | Arkansas-Pine Bluff (fcs) | 54–14 | +40 | 0.997 | 69.48/21.53 | +44.19 |
| 401856678 / 2026-09-12 | away | Kansas (fbs) | 38–21 | +17 | 0.698 | 44.93/17.77 | +7.85 |
| 401856689 / 2026-09-19 | home | Troy (fbs) | 27–17 | +10 | 0.970 | 108.05/15.42 | +29.11 |
| 401856703 / 2026-09-26 | away | Mississippi State (fbs) | 24–31 | -7 | 0.581 | 28.71/13.19 | +3.10 |
| 401856708 / 2026-10-03 | home | Florida (fbs) | 45–17 | +28 | 0.528 | 12.90/8.65 | +1.15 |
| 401856716 / 2026-10-10 | home | Texas A&M (fbs) | 27–6 | +21 | 0.651 | 21.04/10.61 | +6.07 |
| 401856721 / 2026-10-17 | away | Ole Miss (fbs) | 3–31 | -28 | 0.256 | 6.32/4.71 | -10.05 |

Answers to the original questions at every candidate:

**Does this boundary crossing reflect meaningful evidence or close teams?**

- At lambda=0, E[rank]=26.88, ordinal=25, SD=19.84, entropy=6.077 bits, central 80%=6-55; versus Record/Play the mean-rank changes are +0.00/+6.17. Latent top10/12/25 mass is 0.216/0.270/0.573. Adjacent expected-rank gaps are 0.5828230077199059/0.2410444679954935. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.25, E[rank]=24.80, ordinal=23, SD=18.16, entropy=5.950 bits, central 80%=6-51; versus Record/Play the mean-rank changes are -2.08/+4.09. Latent top10/12/25 mass is 0.230/0.289/0.614. Adjacent expected-rank gaps are 0.647650670384607/0.06591965571125513. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.5, E[rank]=23.16, ordinal=25, SD=16.81, entropy=5.840 bits, central 80%=6-47; versus Record/Play the mean-rank changes are -3.72/+2.44. Latent top10/12/25 mass is 0.244/0.308/0.650. Adjacent expected-rank gaps are 0.17286606992815834/1.3474435032239747. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=0.75, E[rank]=21.83, ordinal=25, SD=15.68, entropy=5.743 bits, central 80%=5-44; versus Record/Play the mean-rank changes are -5.05/+1.12. Latent top10/12/25 mass is 0.256/0.324/0.682. Adjacent expected-rank gaps are 0.39215694536506973/0.9633233850080707. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.
- At lambda=1, E[rank]=20.71, ordinal=26, SD=14.69, entropy=5.654 bits, central 80%=5-41; versus Record/Play the mean-rank changes are -6.17/+0.00. Latent top10/12/25 mass is 0.267/0.340/0.710. Adjacent expected-rank gaps are 0.3695526157259863/0.16486162462198095. Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -18.38 | -20 | -9.36 | 0.229 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -17.68 | -15 | -10.37 | 0.234 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -17.59 | -21 | -11.21 | 0.243 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -17.83 | -21 | -11.94 | 0.255 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -18.19 | -20 | -12.59 | 0.267 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 3 / 445 | -20.98 | -18 | -9.77 | 0.321 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 3 / 445 | -20.94 | -25 | -9.58 | 0.349 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 3 / 445 | -20.60 | -17 | -9.35 | 0.369 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 3 / 445 | -20.16 | -18 | -9.15 | 0.386 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 3 / 445 | -19.75 | -16 | -9.00 | 0.400 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Oregon (2483)

Frozen categories: dominant_win_overmatched_opponent.

### 2026-09-08

Actual record **1–0**, mean margin **+7.00**; close wins/losses 1/0, dominant wins 0. Frozen standardized schedule probability **0.392** (larger is easier), expected wins **0.84**, wins minus expectation **+0.16**. Reference: retained_context_v1.4_posterior, ordinal 56, band middle, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 47.34 | 15 | 33.92 | 6.864 | 18–71 | 7–98 | 2–121 | 0.146/0.174/0.335 | +0.00 / -9.30 |
| 0.25 | 50.12 | 25 | 34.04 | 6.912 | 21–75 | 9–100 | 3–122 | 0.122/0.147/0.299 | +2.79 / -6.51 |
| 0.5 | 52.58 | 43 | 34.04 | 6.945 | 24–78 | 10–102 | 3–123 | 0.102/0.124/0.268 | +5.24 / -4.06 |
| 0.75 | 54.73 | 51 | 33.97 | 6.967 | 26–80 | 12–104 | 4–124 | 0.086/0.106/0.242 | +7.40 / -1.90 |
| 1 | 56.63 | 56 | 33.85 | 6.982 | 28–82 | 13–105 | 4–125 | 0.074/0.092/0.219 | +9.30 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858433 / 2026-09-05 | home | Boise State (fbs) | 34–27 | +7 | 0.843 | 37.71/25.46 | +16.59 |

Answers to the original questions at every candidate:

**Does one dominant win against an overmatched opponent drive the ranking?**

- At lambda=0, E[rank]=47.34, ordinal=15, SD=33.92, entropy=6.864 bits, central 80%=7-98; versus Record/Play the mean-rank changes are +0.00/-9.30. Actual evidence is 1-0, mean margin +7.00, 1 close wins and 0 close losses, standardized schedule probability 0.392, expected wins 0.84, wins minus expectation +0.16. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=50.12, ordinal=25, SD=34.04, entropy=6.912 bits, central 80%=9-100; versus Record/Play the mean-rank changes are +2.79/-6.51. Actual evidence is 1-0, mean margin +7.00, 1 close wins and 0 close losses, standardized schedule probability 0.392, expected wins 0.84, wins minus expectation +0.16. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=52.58, ordinal=43, SD=34.04, entropy=6.945 bits, central 80%=10-102; versus Record/Play the mean-rank changes are +5.24/-4.06. Actual evidence is 1-0, mean margin +7.00, 1 close wins and 0 close losses, standardized schedule probability 0.392, expected wins 0.84, wins minus expectation +0.16. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=54.73, ordinal=51, SD=33.97, entropy=6.967 bits, central 80%=12-104; versus Record/Play the mean-rank changes are +7.40/-1.90. Actual evidence is 1-0, mean margin +7.00, 1 close wins and 0 close losses, standardized schedule probability 0.392, expected wins 0.84, wins minus expectation +0.16. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=56.63, ordinal=56, SD=33.85, entropy=6.982 bits, central 80%=13-105; versus Record/Play the mean-rank changes are +9.30/+0.00. Actual evidence is 1-0, mean margin +7.00, 1 close wins and 0 close losses, standardized schedule probability 0.392, expected wins 0.84, wins minus expectation +0.16. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-09-27

Actual record **3–1**, mean margin **+24.25**; close wins/losses 1/0, dominant wins 1. Frozen standardized schedule probability **0.430** (larger is easier), expected wins **3.25**, wins minus expectation **-0.25**. Reference: retained_context_v1.4_posterior, ordinal 17, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 33.07 | 24 | 25.22 | 6.404 | 13–48 | 6–70 | 2–94 | 0.201/0.243/0.480 | +0.00 / +13.78 |
| 0.25 | 28.11 | 18 | 22.35 | 6.183 | 11–41 | 5–61 | 2–84 | 0.245/0.297/0.563 | -4.95 / +8.83 |
| 0.5 | 24.41 | 17 | 19.97 | 5.981 | 9–35 | 4–53 | 2–75 | 0.287/0.347/0.635 | -8.66 / +5.12 |
| 0.75 | 21.54 | 16 | 17.96 | 5.795 | 8–30 | 4–48 | 2–68 | 0.325/0.394/0.697 | -11.52 / +2.26 |
| 1 | 19.29 | 17 | 16.24 | 5.625 | 8–26 | 4–43 | 1–62 | 0.359/0.436/0.749 | -13.78 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858433 / 2026-09-05 | home | Boise State (fbs) | 34–27 | +7 | 0.819 | 24.45/16.05 | +14.24 |
| 401856782 / 2026-09-12 | away | Oklahoma State (fbs) | 31–39 | -8 | 0.768 | 36.27/22.21 | +11.34 |
| 401858455 / 2026-09-19 | home | Portland State (fcs) | 84–0 | +84 | 0.999 | 89.81/20.51 | +52.12 |
| 401858469 / 2026-09-26 | away | USC (fbs) | 41–27 | +14 | 0.660 | 20.19/13.15 | +6.32 |

Answers to the original questions at every candidate:

**Does one dominant win against an overmatched opponent drive the ranking?**

- At lambda=0, E[rank]=33.07, ordinal=24, SD=25.22, entropy=6.404 bits, central 80%=6-70; versus Record/Play the mean-rank changes are +0.00/+13.78. Actual evidence is 3-1, mean margin +24.25, 1 close wins and 0 close losses, standardized schedule probability 0.430, expected wins 3.25, wins minus expectation -0.25. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=28.11, ordinal=18, SD=22.35, entropy=6.183 bits, central 80%=5-61; versus Record/Play the mean-rank changes are -4.95/+8.83. Actual evidence is 3-1, mean margin +24.25, 1 close wins and 0 close losses, standardized schedule probability 0.430, expected wins 3.25, wins minus expectation -0.25. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=24.41, ordinal=17, SD=19.97, entropy=5.981 bits, central 80%=4-53; versus Record/Play the mean-rank changes are -8.66/+5.12. Actual evidence is 3-1, mean margin +24.25, 1 close wins and 0 close losses, standardized schedule probability 0.430, expected wins 3.25, wins minus expectation -0.25. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=21.54, ordinal=16, SD=17.96, entropy=5.795 bits, central 80%=4-48; versus Record/Play the mean-rank changes are -11.52/+2.26. Actual evidence is 3-1, mean margin +24.25, 1 close wins and 0 close losses, standardized schedule probability 0.430, expected wins 3.25, wins minus expectation -0.25. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=19.29, ordinal=17, SD=16.24, entropy=5.625 bits, central 80%=4-43; versus Record/Play the mean-rank changes are -13.78/+0.00. Actual evidence is 3-1, mean margin +24.25, 1 close wins and 0 close losses, standardized schedule probability 0.430, expected wins 3.25, wins minus expectation -0.25. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-10-25T12-59-22.246522Z

Actual record **4–3**, mean margin **+12.43**; close wins/losses 2/1, dominant wins 1. Frozen standardized schedule probability **0.325** (larger is easier), expected wins **5.04**, wins minus expectation **-1.04**. Reference: retained_context_v1.4_posterior, ordinal 24, band strong, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 38.08 | 38 | 22.08 | 6.389 | 21–52 | 12–69 | 6–89 | 0.073/0.102/0.339 | +0.00 / +17.97 |
| 0.25 | 31.79 | 33 | 19.17 | 6.152 | 17–44 | 10–59 | 5–77 | 0.104/0.145/0.451 | -6.29 / +11.67 |
| 0.5 | 26.94 | 28 | 16.81 | 5.923 | 14–37 | 9–51 | 4–67 | 0.140/0.196/0.562 | -11.14 / +6.82 |
| 0.75 | 23.12 | 27 | 14.77 | 5.695 | 12–31 | 8–45 | 4–60 | 0.180/0.252/0.664 | -14.97 / +3.00 |
| 1 | 20.12 | 24 | 12.93 | 5.471 | 11–25 | 7–39 | 3–53 | 0.221/0.309/0.751 | -17.97 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858433 / 2026-09-05 | home | Boise State (fbs) | 34–27 | +7 | 0.762 | 21.12/11.54 | +11.03 |
| 401856782 / 2026-09-12 | away | Oklahoma State (fbs) | 31–39 | -8 | 0.580 | 18.21/11.32 | +3.10 |
| 401858455 / 2026-09-19 | home | Portland State (fcs) | 84–0 | +84 | 0.999 | 81.93/17.46 | +49.14 |
| 401858469 / 2026-09-26 | away | USC (fbs) | 41–27 | +14 | 0.648 | 22.93/11.49 | +5.74 |
| 401858484 / 2026-10-10 | home | UCLA (fbs) | 15–17 | -2 | 0.642 | 11.61/7.07 | +5.54 |
| 401858492 / 2026-10-18 | home | Nebraska (fbs) | 17–14 | +3 | 0.647 | 11.90/7.30 | +5.72 |
| 401858495 / 2026-10-24 | away | Illinois (fbs) | 10–21 | -11 | 0.764 | 39.01/17.45 | +10.77 |

Answers to the original questions at every candidate:

**Does one dominant win against an overmatched opponent drive the ranking?**

- At lambda=0, E[rank]=38.08, ordinal=38, SD=22.08, entropy=6.389 bits, central 80%=12-69; versus Record/Play the mean-rank changes are +0.00/+17.97. Removing the frozen 401858455 observation changes this team's mean rank by +0.17, ordinal by +0, SD by +0.12, TV=0.003. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=0.25, E[rank]=31.79, ordinal=33, SD=19.17, entropy=6.152 bits, central 80%=10-59; versus Record/Play the mean-rank changes are -6.29/+11.67. Removing the frozen 401858455 observation changes this team's mean rank by +3.09, ordinal by +4, SD by +1.03, TV=0.065. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=0.5, E[rank]=26.94, ordinal=28, SD=16.81, entropy=5.923 bits, central 80%=9-51; versus Record/Play the mean-rank changes are -11.14/+6.82. Removing the frozen 401858455 observation changes this team's mean rank by +5.01, ordinal by +7, SD by +1.68, TV=0.126. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=0.75, E[rank]=23.12, ordinal=27, SD=14.77, entropy=5.695 bits, central 80%=8-45; versus Record/Play the mean-rank changes are -14.97/+3.00. Removing the frozen 401858455 observation changes this team's mean rank by +6.31, ordinal by +6, SD by +2.25, TV=0.181. This measures the total network response to removing one observation, not an additive contribution.
- At lambda=1, E[rank]=20.12, ordinal=24, SD=12.93, entropy=5.471 bits, central 80%=7-39; versus Record/Play the mean-rank changes are -17.97/+0.00. Removing the frozen 401858455 observation changes this team's mean rank by +7.15, ordinal by +7, SD by +2.79, TV=0.228. This measures the total network response to removing one observation, not an additive contribution.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -14.27 | +9 | -8.71 | 0.184 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -22.01 | -7 | -11.69 | 0.288 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -28.17 | -26 | -14.07 | 0.379 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -33.19 | -35 | -16.01 | 0.460 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -37.35 | -39 | -17.61 | 0.532 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 3 / 445 | +5.02 | +14 | -3.14 | 0.166 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 3 / 445 | +3.68 | +15 | -3.18 | 0.171 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 3 / 445 | +2.54 | +11 | -3.16 | 0.172 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 3 / 445 | +1.57 | +11 | -3.19 | 0.169 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 3 / 445 | +0.83 | +7 | -3.31 | 0.168 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 Stanford (24)

Frozen categories: endpoint_disagreement, weak_schedule_pair.

### 2026-09-08

Actual record **1–1**, mean margin **-14.50**; close wins/losses 0/0, dominant wins 0. Frozen standardized schedule probability **0.469** (larger is easier), expected wins **0.71**, wins minus expectation **+0.29**. Reference: retained_context_v1.4_posterior, ordinal 116, band weak, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 65.68 | 61 | 34.79 | 7.044 | 37–93 | 19–114 | 7–129 | 0.042/0.054/0.150 | +0.00 / -27.23 |
| 0.25 | 74.42 | 92 | 33.94 | 7.020 | 47–102 | 27–120 | 12–132 | 0.020/0.026/0.090 | +8.75 / -18.49 |
| 0.5 | 81.76 | 104 | 32.57 | 6.951 | 57–108 | 35–123 | 18–133 | 0.009/0.012/0.053 | +16.08 / -11.15 |
| 0.75 | 87.85 | 112 | 30.99 | 6.859 | 66–113 | 42–126 | 23–134 | 0.004/0.006/0.031 | +22.17 / -5.06 |
| 1 | 92.91 | 116 | 29.35 | 6.757 | 73–117 | 49–128 | 29–135 | 0.002/0.003/0.018 | +27.23 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858201 / 2026-08-29 | home | Hawai'i (fbs) | 37–27 | +10 | 0.631 | 101.12/22.94 | +5.85 |
| 401858206 / 2026-09-05 | home | Miami (fbs) | 6–45 | -39 | 0.079 | 9.22/9.16 | -22.10 |

Answers to the original questions at every candidate:

**Which games and uncertainty explain the Quality of Record/Play disagreement?**

- At lambda=0, E[rank]=65.68, ordinal=61, SD=34.79, entropy=7.044 bits, central 80%=19-114; versus Record/Play the mean-rank changes are +0.00/-27.23. The frozen peer Massachusetts is 1-0 with standardized schedule probability 0.590, E[rank]=43.33, SD=32.66, entropy=6.770, widths 50/80/95=51/88/116; this team's expected-rank gap is +22.34 and SD gap +2.13. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=74.42, ordinal=92, SD=33.94, entropy=7.020 bits, central 80%=27-120; versus Record/Play the mean-rank changes are +8.75/-18.49. The frozen peer Massachusetts is 1-0 with standardized schedule probability 0.590, E[rank]=42.38, SD=32.15, entropy=6.744, widths 50/80/95=50/86/114; this team's expected-rank gap is +32.04 and SD gap +1.78. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=81.76, ordinal=104, SD=32.57, entropy=6.951 bits, central 80%=35-123; versus Record/Play the mean-rank changes are +16.08/-11.15. The frozen peer Massachusetts is 1-0 with standardized schedule probability 0.590, E[rank]=41.49, SD=31.66, entropy=6.719, widths 50/80/95=48/84/113; this team's expected-rank gap is +40.27 and SD gap +0.91. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=87.85, ordinal=112, SD=30.99, entropy=6.859 bits, central 80%=42-126; versus Record/Play the mean-rank changes are +22.17/-5.06. The frozen peer Massachusetts is 1-0 with standardized schedule probability 0.590, E[rank]=40.66, SD=31.18, entropy=6.694, widths 50/80/95=47/83/112; this team's expected-rank gap is +47.19 and SD gap -0.19. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=92.91, ordinal=116, SD=29.35, entropy=6.757 bits, central 80%=49-128; versus Record/Play the mean-rank changes are +27.23/+0.00. The frozen peer Massachusetts is 1-0 with standardized schedule probability 0.590, E[rank]=39.89, SD=30.72, entropy=6.670, widths 50/80/95=47/81/110; this team's expected-rank gap is +53.02 and SD gap -1.37. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**For identical weak-team records, does the easier schedule narrow uncertainty?**

- At lambda=0, E[rank]=65.68, ordinal=61, SD=34.79, entropy=7.044 bits, central 80%=19-114; versus Record/Play the mean-rank changes are +0.00/-27.23. The frozen peer Massachusetts is 1-0 with standardized schedule probability 0.590, E[rank]=43.33, SD=32.66, entropy=6.770, widths 50/80/95=51/88/116; this team's expected-rank gap is +22.34 and SD gap +2.13. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=74.42, ordinal=92, SD=33.94, entropy=7.020 bits, central 80%=27-120; versus Record/Play the mean-rank changes are +8.75/-18.49. The frozen peer Massachusetts is 1-0 with standardized schedule probability 0.590, E[rank]=42.38, SD=32.15, entropy=6.744, widths 50/80/95=50/86/114; this team's expected-rank gap is +32.04 and SD gap +1.78. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=81.76, ordinal=104, SD=32.57, entropy=6.951 bits, central 80%=35-123; versus Record/Play the mean-rank changes are +16.08/-11.15. The frozen peer Massachusetts is 1-0 with standardized schedule probability 0.590, E[rank]=41.49, SD=31.66, entropy=6.719, widths 50/80/95=48/84/113; this team's expected-rank gap is +40.27 and SD gap +0.91. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=87.85, ordinal=112, SD=30.99, entropy=6.859 bits, central 80%=42-126; versus Record/Play the mean-rank changes are +22.17/-5.06. The frozen peer Massachusetts is 1-0 with standardized schedule probability 0.590, E[rank]=40.66, SD=31.18, entropy=6.694, widths 50/80/95=47/83/112; this team's expected-rank gap is +47.19 and SD gap -0.19. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=92.91, ordinal=116, SD=29.35, entropy=6.757 bits, central 80%=49-128; versus Record/Play the mean-rank changes are +27.23/+0.00. The frozen peer Massachusetts is 1-0 with standardized schedule probability 0.590, E[rank]=39.89, SD=30.72, entropy=6.670, widths 50/80/95=47/81/110; this team's expected-rank gap is +53.02 and SD gap -1.37. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### 2026-09-27

Actual record **2–2**, mean margin **-12.50**; close wins/losses 1/0, dominant wins 0. Frozen standardized schedule probability **0.442** (larger is easier), expected wins **1.30**, wins minus expectation **+0.70**. Reference: retained_context_v1.4_posterior, ordinal 109, band weak, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 59.79 | 65 | 30.50 | 6.898 | 35–83 | 21–103 | 10–120 | 0.030/0.040/0.147 | +0.00 / -32.01 |
| 0.25 | 70.69 | 81 | 29.50 | 6.875 | 48–93 | 31–110 | 17–124 | 0.008/0.012/0.065 | +10.90 / -21.11 |
| 0.5 | 79.28 | 92 | 27.88 | 6.793 | 59–101 | 40–115 | 25–127 | 0.002/0.003/0.028 | +19.49 / -12.52 |
| 0.75 | 86.16 | 99 | 26.05 | 6.687 | 68–106 | 49–119 | 31–129 | 0.000/0.001/0.011 | +26.37 / -5.63 |
| 1 | 91.80 | 109 | 24.18 | 6.568 | 76–110 | 58–121 | 38–130 | 0.000/0.000/0.005 | +32.01 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858201 / 2026-08-29 | home | Hawai'i (fbs) | 37–27 | +10 | 0.733 | 112.50/15.64 | +10.09 |
| 401858206 / 2026-09-05 | home | Miami (fbs) | 6–45 | -39 | 0.062 | 6.15/5.02 | -23.11 |
| 401858231 / 2026-09-19 | away | Duke (fbs) | 7–35 | -28 | 0.082 | 24.32/15.95 | -22.37 |
| 401858240 / 2026-09-27 | home | Georgia Tech (fbs) | 34–27 | +7 | 0.424 | 70.13/23.71 | -3.09 |

Answers to the original questions at every candidate:

**Which games and uncertainty explain the Quality of Record/Play disagreement?**

- At lambda=0, E[rank]=59.79, ordinal=65, SD=30.50, entropy=6.898 bits, central 80%=21-103; versus Record/Play the mean-rank changes are +0.00/-32.01. The frozen peer Massachusetts is 4-0 with standardized schedule probability 0.826, E[rank]=41.49, SD=30.21, entropy=6.698, widths 50/80/95=47/81/108; this team's expected-rank gap is +18.30 and SD gap +0.29. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=70.69, ordinal=81, SD=29.50, entropy=6.875 bits, central 80%=31-110; versus Record/Play the mean-rank changes are +10.90/-21.11. The frozen peer Massachusetts is 4-0 with standardized schedule probability 0.826, E[rank]=43.78, SD=29.25, entropy=6.725, widths 50/80/95=45/79/105; this team's expected-rank gap is +26.91 and SD gap +0.25. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=79.28, ordinal=92, SD=27.88, entropy=6.793 bits, central 80%=40-115; versus Record/Play the mean-rank changes are +19.49/-12.52. The frozen peer Massachusetts is 4-0 with standardized schedule probability 0.826, E[rank]=45.41, SD=28.23, entropy=6.719, widths 50/80/95=43/76/104; this team's expected-rank gap is +33.87 and SD gap -0.35. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=86.16, ordinal=99, SD=26.05, entropy=6.687 bits, central 80%=49-119; versus Record/Play the mean-rank changes are +26.37/-5.63. The frozen peer Massachusetts is 4-0 with standardized schedule probability 0.826, E[rank]=46.74, SD=27.27, entropy=6.697, widths 50/80/95=41/73/102; this team's expected-rank gap is +39.42 and SD gap -1.22. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=91.80, ordinal=109, SD=24.18, entropy=6.568 bits, central 80%=58-121; versus Record/Play the mean-rank changes are +32.01/+0.00. The frozen peer Massachusetts is 4-0 with standardized schedule probability 0.826, E[rank]=47.87, SD=26.41, entropy=6.668, widths 50/80/95=41/71/99; this team's expected-rank gap is +43.92 and SD gap -2.23. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**For identical weak-team records, does the easier schedule narrow uncertainty?**

- At lambda=0, E[rank]=59.79, ordinal=65, SD=30.50, entropy=6.898 bits, central 80%=21-103; versus Record/Play the mean-rank changes are +0.00/-32.01. The frozen peer Massachusetts is 4-0 with standardized schedule probability 0.826, E[rank]=41.49, SD=30.21, entropy=6.698, widths 50/80/95=47/81/108; this team's expected-rank gap is +18.30 and SD gap +0.29. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=70.69, ordinal=81, SD=29.50, entropy=6.875 bits, central 80%=31-110; versus Record/Play the mean-rank changes are +10.90/-21.11. The frozen peer Massachusetts is 4-0 with standardized schedule probability 0.826, E[rank]=43.78, SD=29.25, entropy=6.725, widths 50/80/95=45/79/105; this team's expected-rank gap is +26.91 and SD gap +0.25. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=79.28, ordinal=92, SD=27.88, entropy=6.793 bits, central 80%=40-115; versus Record/Play the mean-rank changes are +19.49/-12.52. The frozen peer Massachusetts is 4-0 with standardized schedule probability 0.826, E[rank]=45.41, SD=28.23, entropy=6.719, widths 50/80/95=43/76/104; this team's expected-rank gap is +33.87 and SD gap -0.35. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=86.16, ordinal=99, SD=26.05, entropy=6.687 bits, central 80%=49-119; versus Record/Play the mean-rank changes are +26.37/-5.63. The frozen peer Massachusetts is 4-0 with standardized schedule probability 0.826, E[rank]=46.74, SD=27.27, entropy=6.697, widths 50/80/95=41/73/102; this team's expected-rank gap is +39.42 and SD gap -1.22. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=91.80, ordinal=109, SD=24.18, entropy=6.568 bits, central 80%=58-121; versus Record/Play the mean-rank changes are +32.01/+0.00. The frozen peer Massachusetts is 4-0 with standardized schedule probability 0.826, E[rank]=47.87, SD=26.41, entropy=6.668, widths 50/80/95=41/71/99; this team's expected-rank gap is +43.92 and SD gap -2.23. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### 2026-10-25T12-59-22.246522Z

Actual record **4–4**, mean margin **-15.62**; close wins/losses 2/0, dominant wins 0. Frozen standardized schedule probability **0.438** (larger is easier), expected wins **2.30**, wins minus expectation **+1.70**. Reference: retained_context_v1.4_posterior, ordinal 116, band weak, schedule tercile hard.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 55.49 | 59 | 25.71 | 6.672 | 35–74 | 23–92 | 14–109 | 0.013/0.020/0.124 | +0.00 / -50.27 |
| 0.25 | 74.70 | 88 | 25.29 | 6.672 | 56–94 | 39–107 | 26–120 | 0.001/0.001/0.022 | +19.21 / -31.06 |
| 0.5 | 89.14 | 99 | 22.32 | 6.476 | 75–106 | 58–116 | 39–126 | 0.000/0.000/0.003 | +33.65 / -16.62 |
| 0.75 | 99.04 | 111 | 18.91 | 6.223 | 88–113 | 74–122 | 56–129 | 0.000/0.000/0.000 | +43.55 / -6.72 |
| 1 | 105.76 | 116 | 16.16 | 5.992 | 96–118 | 84–125 | 69–132 | 0.000/0.000/0.000 | +50.27 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401858201 / 2026-08-29 | home | Hawai'i (fbs) | 37–27 | +10 | 0.578 | 104.41/14.15 | +3.03 |
| 401858206 / 2026-09-05 | home | Miami (fbs) | 6–45 | -39 | 0.036 | 4.20/2.89 | -26.52 |
| 401858231 / 2026-09-19 | away | Duke (fbs) | 7–35 | -28 | 0.075 | 34.88/15.95 | -22.19 |
| 401858240 / 2026-09-27 | home | Georgia Tech (fbs) | 34–27 | +7 | 0.360 | 75.05/19.62 | -5.51 |
| 401858251 / 2026-10-03 | away | Wake Forest (fbs) | 3–57 | -54 | 0.034 | 16.65/9.42 | -27.85 |
| 401858257 / 2026-10-10 | away | Notre Dame (fbs) | 7–58 | -51 | 0.015 | 4.54/3.15 | -32.64 |
| 401858262 / 2026-10-17 | home | Elon (fcs) | 34–7 | +27 | 0.960 | 82.44/16.87 | +27.33 |
| 401858272 / 2026-10-24 | home | NC State (fbs) | 31–28 | +3 | 0.243 | 54.74/18.74 | -10.67 |

Answers to the original questions at every candidate:

**Which games and uncertainty explain the Quality of Record/Play disagreement?**

- At lambda=0, E[rank]=55.49, ordinal=59, SD=25.71, entropy=6.672 bits, central 80%=23-92; versus Record/Play the mean-rank changes are +0.00/-50.27. The frozen peer Massachusetts is 4-4 with standardized schedule probability 0.769, E[rank]=105.25, SD=20.25, entropy=6.252, widths 50/80/95=29/53/79; this team's expected-rank gap is -49.76 and SD gap +5.46. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=74.70, ordinal=88, SD=25.29, entropy=6.672 bits, central 80%=39-107; versus Record/Play the mean-rank changes are +19.21/-31.06. The frozen peer Massachusetts is 4-4 with standardized schedule probability 0.769, E[rank]=103.75, SD=19.35, entropy=6.219, widths 50/80/95=27/50/75; this team's expected-rank gap is -29.05 and SD gap +5.93. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=89.14, ordinal=99, SD=22.32, entropy=6.476 bits, central 80%=58-116; versus Record/Play the mean-rank changes are +33.65/-16.62. The frozen peer Massachusetts is 4-4 with standardized schedule probability 0.769, E[rank]=102.75, SD=18.45, entropy=6.169, widths 50/80/95=25/47/72; this team's expected-rank gap is -13.62 and SD gap +3.87. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=99.04, ordinal=111, SD=18.91, entropy=6.223 bits, central 80%=74-122; versus Record/Play the mean-rank changes are +43.55/-6.72. The frozen peer Massachusetts is 4-4 with standardized schedule probability 0.769, E[rank]=102.06, SD=17.61, entropy=6.112, widths 50/80/95=24/46/70; this team's expected-rank gap is -3.02 and SD gap +1.30. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=105.76, ordinal=116, SD=16.16, entropy=5.992 bits, central 80%=84-125; versus Record/Play the mean-rank changes are +50.27/+0.00. The frozen peer Massachusetts is 4-4 with standardized schedule probability 0.769, E[rank]=101.56, SD=16.84, entropy=6.055, widths 50/80/95=24/43/66; this team's expected-rank gap is +4.20 and SD gap -0.68. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

**For identical weak-team records, does the easier schedule narrow uncertainty?**

- At lambda=0, E[rank]=55.49, ordinal=59, SD=25.71, entropy=6.672 bits, central 80%=23-92; versus Record/Play the mean-rank changes are +0.00/-50.27. The frozen peer Massachusetts is 4-4 with standardized schedule probability 0.769, E[rank]=105.25, SD=20.25, entropy=6.252, widths 50/80/95=29/53/79; this team's expected-rank gap is -49.76 and SD gap +5.46. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=74.70, ordinal=88, SD=25.29, entropy=6.672 bits, central 80%=39-107; versus Record/Play the mean-rank changes are +19.21/-31.06. The frozen peer Massachusetts is 4-4 with standardized schedule probability 0.769, E[rank]=103.75, SD=19.35, entropy=6.219, widths 50/80/95=27/50/75; this team's expected-rank gap is -29.05 and SD gap +5.93. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=89.14, ordinal=99, SD=22.32, entropy=6.476 bits, central 80%=58-116; versus Record/Play the mean-rank changes are +33.65/-16.62. The frozen peer Massachusetts is 4-4 with standardized schedule probability 0.769, E[rank]=102.75, SD=18.45, entropy=6.169, widths 50/80/95=25/47/72; this team's expected-rank gap is -13.62 and SD gap +3.87. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=99.04, ordinal=111, SD=18.91, entropy=6.223 bits, central 80%=74-122; versus Record/Play the mean-rank changes are +43.55/-6.72. The frozen peer Massachusetts is 4-4 with standardized schedule probability 0.769, E[rank]=102.06, SD=17.61, entropy=6.112, widths 50/80/95=24/46/70; this team's expected-rank gap is -3.02 and SD gap +1.30. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=105.76, ordinal=116, SD=16.16, entropy=5.992 bits, central 80%=84-125; versus Record/Play the mean-rank changes are +50.27/+0.00. The frozen peer Massachusetts is 4-4 with standardized schedule probability 0.769, E[rank]=101.56, SD=16.84, entropy=6.055, widths 50/80/95=24/43/66; this team's expected-rank gap is +4.20 and SD gap -0.68. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 2 / 359 | -5.89 | +4 | -4.30 | 0.104 |
| 2026-09-08 → 2026-09-27 | 0.25 | 2 / 359 | -3.73 | -11 | -4.43 | 0.108 |
| 2026-09-08 → 2026-09-27 | 0.5 | 2 / 359 | -2.48 | -12 | -4.69 | 0.117 |
| 2026-09-08 → 2026-09-27 | 0.75 | 2 / 359 | -1.68 | -13 | -4.94 | 0.127 |
| 2026-09-08 → 2026-09-27 | 1 | 2 / 359 | -1.11 | -7 | -5.17 | 0.136 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 4 / 445 | -4.30 | -6 | -4.79 | 0.111 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 4 / 445 | +4.00 | +7 | -4.22 | 0.107 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 4 / 445 | +9.86 | +7 | -5.56 | 0.178 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 4 / 445 | +12.88 | +12 | -7.13 | 0.233 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 4 / 445 | +13.96 | +7 | -8.02 | 0.272 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 UConn (41)

Frozen categories: middle_schedule_pair.

### 2026-09-08

Actual record **1–0**, mean margin **+49.00**; close wins/losses 0/0, dominant wins 1. Frozen standardized schedule probability **0.980** (larger is easier), expected wins **0.98**, wins minus expectation **+0.02**. Reference: retained_context_v1.4_posterior, ordinal 44, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 66.24 | 69 | 38.97 | 7.102 | 33–99 | 13–121 | 4–134 | 0.078/0.094/0.195 | +0.00 / +15.04 |
| 0.25 | 60.55 | 54 | 37.87 | 7.067 | 28–91 | 11–116 | 3–132 | 0.094/0.113/0.230 | -5.69 / +9.35 |
| 0.5 | 56.41 | 49 | 36.74 | 7.021 | 25–85 | 10–110 | 3–129 | 0.107/0.128/0.258 | -9.83 / +5.21 |
| 0.75 | 53.41 | 47 | 35.75 | 6.977 | 23–80 | 9–106 | 3–127 | 0.117/0.140/0.280 | -12.83 / +2.21 |
| 1 | 51.19 | 44 | 34.95 | 6.938 | 21–77 | 8–103 | 2–124 | 0.126/0.150/0.298 | -15.04 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401861962 / 2026-09-05 | home | Lafayette (fcs) | 56–7 | +49 | 0.979 | 89.60/29.59 | +38.67 |

Answers to the original questions at every candidate:

**For identical middle-team records, how do schedule and uncertainty differ across candidates?**

- At lambda=0, E[rank]=66.24, ordinal=69, SD=38.97, entropy=7.102 bits, central 80%=13-121; versus Record/Play the mean-rank changes are +0.00/+15.04. The frozen peer Auburn is 1-0 with standardized schedule probability 0.363, E[rank]=44.93, SD=33.20, entropy=6.810, widths 50/80/95=52/89/117; this team's expected-rank gap is +21.30 and SD gap +5.76. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=60.55, ordinal=54, SD=37.87, entropy=7.067 bits, central 80%=11-116; versus Record/Play the mean-rank changes are -5.69/+9.35. The frozen peer Auburn is 1-0 with standardized schedule probability 0.363, E[rank]=49.38, SD=33.83, entropy=6.898, widths 50/80/95=54/92/119; this team's expected-rank gap is +11.17 and SD gap +4.04. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=56.41, ordinal=49, SD=36.74, entropy=7.021 bits, central 80%=10-110; versus Record/Play the mean-rank changes are -9.83/+5.21. The frozen peer Auburn is 1-0 with standardized schedule probability 0.363, E[rank]=53.43, SD=34.16, entropy=6.957, widths 50/80/95=55/93/122; this team's expected-rank gap is +2.98 and SD gap +2.58. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=53.41, ordinal=47, SD=35.75, entropy=6.977 bits, central 80%=9-106; versus Record/Play the mean-rank changes are -12.83/+2.21. The frozen peer Auburn is 1-0 with standardized schedule probability 0.363, E[rank]=57.08, SD=34.28, entropy=6.994, widths 50/80/95=56/94/122; this team's expected-rank gap is -3.68 and SD gap +1.48. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=51.19, ordinal=44, SD=34.95, entropy=6.938 bits, central 80%=8-103; versus Record/Play the mean-rank changes are -15.04/+0.00. The frozen peer Auburn is 1-0 with standardized schedule probability 0.363, E[rank]=60.36, SD=34.26, entropy=7.015, widths 50/80/95=56/94/123; this team's expected-rank gap is -9.16 and SD gap +0.69. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### 2026-09-27

Actual record **2–2**, mean margin **+12.50**; close wins/losses 0/1, dominant wins 2. Frozen standardized schedule probability **0.670** (larger is easier), expected wins **2.56**, wins minus expectation **-0.56**. Reference: retained_context_v1.4_posterior, ordinal 75, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 87.14 | 103 | 29.28 | 6.819 | 66–110 | 45–124 | 26–133 | 0.003/0.005/0.024 | +0.00 / +22.88 |
| 0.25 | 79.14 | 94 | 28.93 | 6.846 | 58–102 | 39–117 | 22–129 | 0.005/0.007/0.036 | -8.00 / +14.88 |
| 0.5 | 72.90 | 87 | 28.14 | 6.825 | 52–94 | 35–110 | 19–124 | 0.007/0.010/0.049 | -14.24 / +8.64 |
| 0.75 | 68.06 | 81 | 27.24 | 6.785 | 48–88 | 32–105 | 17–119 | 0.009/0.012/0.062 | -19.08 / +3.79 |
| 1 | 64.26 | 75 | 26.38 | 6.741 | 45–84 | 29–100 | 16–115 | 0.011/0.015/0.074 | -22.88 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401861962 / 2026-09-05 | home | Lafayette (fcs) | 56–7 | +49 | 0.988 | 92.15/20.45 | +38.38 |
| 401858444 / 2026-09-12 | home | Maryland (fbs) | 14–38 | -24 | 0.451 | 53.47/24.15 | -1.99 |
| 401861963 / 2026-09-19 | away | Southern Miss (fbs) | 48–20 | +28 | 0.602 | 97.61/21.29 | +4.24 |
| 401861970 / 2026-09-26 | away | Miami (OH) (fbs) | 21–24 | -3 | 0.513 | 85.23/24.02 | +0.53 |

Answers to the original questions at every candidate:

**For identical middle-team records, how do schedule and uncertainty differ across candidates?**

- At lambda=0, E[rank]=87.14, ordinal=103, SD=29.28, entropy=6.819 bits, central 80%=45-124; versus Record/Play the mean-rank changes are +0.00/+22.88. The frozen peer Auburn is 3-1 with standardized schedule probability 0.409, E[rank]=32.19, SD=23.72, entropy=6.347, widths 50/80/95=33/62/89; this team's expected-rank gap is +54.95 and SD gap +5.55. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=79.14, ordinal=94, SD=28.93, entropy=6.846 bits, central 80%=39-117; versus Record/Play the mean-rank changes are -8.00/+14.88. The frozen peer Auburn is 3-1 with standardized schedule probability 0.409, E[rank]=31.83, SD=22.42, entropy=6.300, widths 50/80/95=32/58/84; this team's expected-rank gap is +47.31 and SD gap +6.51. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=72.90, ordinal=87, SD=28.14, entropy=6.825 bits, central 80%=35-110; versus Record/Play the mean-rank changes are -14.24/+8.64. The frozen peer Auburn is 3-1 with standardized schedule probability 0.409, E[rank]=31.57, SD=21.31, entropy=6.254, widths 50/80/95=30/55/81; this team's expected-rank gap is +41.33 and SD gap +6.82. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=68.06, ordinal=81, SD=27.24, entropy=6.785 bits, central 80%=32-105; versus Record/Play the mean-rank changes are -19.08/+3.79. The frozen peer Auburn is 3-1 with standardized schedule probability 0.409, E[rank]=31.43, SD=20.38, entropy=6.211, widths 50/80/95=28/52/77; this team's expected-rank gap is +36.63 and SD gap +6.86. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=64.26, ordinal=75, SD=26.38, entropy=6.741 bits, central 80%=29-100; versus Record/Play the mean-rank changes are -22.88/+0.00. The frozen peer Auburn is 3-1 with standardized schedule probability 0.409, E[rank]=31.41, SD=19.60, entropy=6.172, widths 50/80/95=28/50/75; this team's expected-rank gap is +32.85 and SD gap +6.78. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### 2026-10-25T12-59-22.246522Z

Actual record **4–3**, mean margin **+11.71**; close wins/losses 0/2, dominant wins 2. Frozen standardized schedule probability **0.697** (larger is easier), expected wins **4.85**, wins minus expectation **-0.85**. Reference: retained_context_v1.4_posterior, ordinal 74, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 90.60 | 103 | 23.88 | 6.555 | 75–109 | 57–119 | 39–128 | 0.000/0.001/0.006 | +0.00 / +27.00 |
| 0.25 | 81.24 | 93 | 23.54 | 6.572 | 65–99 | 49–111 | 33–122 | 0.001/0.001/0.010 | -9.36 / +17.64 |
| 0.5 | 73.93 | 85 | 22.64 | 6.531 | 58–91 | 44–103 | 29–115 | 0.001/0.001/0.015 | -16.67 / +10.33 |
| 0.75 | 68.19 | 79 | 21.60 | 6.469 | 53–84 | 40–97 | 27–109 | 0.001/0.002/0.022 | -22.41 / +4.59 |
| 1 | 63.60 | 74 | 20.60 | 6.403 | 49–78 | 37–91 | 24–104 | 0.002/0.003/0.029 | -27.00 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401861962 / 2026-09-05 | home | Lafayette (fcs) | 56–7 | +49 | 0.990 | 86.55/15.79 | +37.21 |
| 401858444 / 2026-09-12 | home | Maryland (fbs) | 14–38 | -24 | 0.578 | 68.23/21.61 | +3.08 |
| 401861963 / 2026-09-19 | away | Southern Miss (fbs) | 48–20 | +28 | 0.709 | 107.88/14.90 | +8.47 |
| 401861970 / 2026-09-26 | away | Miami (OH) (fbs) | 21–24 | -3 | 0.622 | 97.00/16.63 | +4.79 |
| 401858252 / 2026-10-03 | home | Syracuse (fbs) | 41–42 | -1 | 0.445 | 48.58/17.78 | -2.14 |
| 401861964 / 2026-10-10 | away | Temple (fbs) | 34–20 | +14 | 0.609 | 95.21/17.85 | +4.31 |
| 401861965 / 2026-10-23 | home | Massachusetts (fbs) | 35–16 | +19 | 0.901 | 121.03/11.85 | +20.08 |

Answers to the original questions at every candidate:

**For identical middle-team records, how do schedule and uncertainty differ across candidates?**

- At lambda=0, E[rank]=90.60, ordinal=103, SD=23.88, entropy=6.555 bits, central 80%=57-119; versus Record/Play the mean-rank changes are +0.00/+27.00. The frozen peer Auburn is 4-3 with standardized schedule probability 0.308, E[rank]=33.74, SD=21.77, entropy=6.305, widths 50/80/95=32/57/82; this team's expected-rank gap is +56.86 and SD gap +2.10. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.25, E[rank]=81.24, ordinal=93, SD=23.54, entropy=6.572 bits, central 80%=49-111; versus Record/Play the mean-rank changes are -9.36/+17.64. The frozen peer Auburn is 4-3 with standardized schedule probability 0.308, E[rank]=33.11, SD=20.30, entropy=6.225, widths 50/80/95=30/53/76; this team's expected-rank gap is +48.13 and SD gap +3.24. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.5, E[rank]=73.93, ordinal=85, SD=22.64, entropy=6.531 bits, central 80%=44-103; versus Record/Play the mean-rank changes are -16.67/+10.33. The frozen peer Auburn is 4-3 with standardized schedule probability 0.308, E[rank]=32.66, SD=19.05, entropy=6.148, widths 50/80/95=28/50/72; this team's expected-rank gap is +41.27 and SD gap +3.59. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=0.75, E[rank]=68.19, ordinal=79, SD=21.60, entropy=6.469 bits, central 80%=40-97; versus Record/Play the mean-rank changes are -22.41/+4.59. The frozen peer Auburn is 4-3 with standardized schedule probability 0.308, E[rank]=32.34, SD=18.00, entropy=6.077, widths 50/80/95=27/47/68; this team's expected-rank gap is +35.85 and SD gap +3.61. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.
- At lambda=1, E[rank]=63.60, ordinal=74, SD=20.60, entropy=6.403 bits, central 80%=37-91; versus Record/Play the mean-rank changes are -27.00/+0.00. The frozen peer Auburn is 4-3 with standardized schedule probability 0.308, E[rank]=32.08, SD=17.10, entropy=6.012, widths 50/80/95=25/44/64; this team's expected-rank gap is +31.52 and SD gap +3.50. Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | +20.90 | +34 | -9.69 | 0.268 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | +18.60 | +40 | -8.94 | 0.272 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | +16.49 | +38 | -8.60 | 0.274 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | +14.65 | +34 | -8.51 | 0.274 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | +13.07 | +31 | -8.57 | 0.274 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 3 / 445 | +3.46 | +0 | -5.40 | 0.125 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 3 / 445 | +2.10 | -1 | -5.39 | 0.124 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 3 / 445 | +1.03 | -2 | -5.50 | 0.127 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 3 / 445 | +0.14 | -2 | -5.64 | 0.134 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 3 / 445 | -0.66 | -1 | -5.78 | 0.140 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.

## 2026 UTSA (2636)

Frozen categories: middle.

### 2026-09-08

Actual record **1–0**, mean margin **+29.00**; close wins/losses 0/0, dominant wins 1. Frozen standardized schedule probability **0.948** (larger is easier), expected wins **0.94**, wins minus expectation **+0.06**. Reference: retained_context_v1.4_posterior, ordinal 81, band middle, schedule tercile easy.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 66.24 | 67 | 38.97 | 7.102 | 33–99 | 13–121 | 4–134 | 0.078/0.094/0.195 | +0.00 / -3.16 |
| 0.25 | 67.40 | 74 | 38.35 | 7.101 | 35–99 | 15–121 | 4–134 | 0.069/0.083/0.179 | +1.16 / -2.00 |
| 0.5 | 68.26 | 78 | 37.84 | 7.099 | 36–100 | 16–121 | 5–133 | 0.061/0.074/0.167 | +2.02 / -1.14 |
| 0.75 | 68.90 | 80 | 37.41 | 7.095 | 37–100 | 18–121 | 5–133 | 0.056/0.068/0.157 | +2.67 / -0.50 |
| 1 | 69.40 | 81 | 37.04 | 7.090 | 38–100 | 19–121 | 6–133 | 0.051/0.063/0.149 | +3.16 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401862700 / 2026-09-05 | home | UT Rio Grande Valley (fcs) | 45–16 | +29 | 0.942 | 62.55/32.31 | +30.02 |

Answers to the original questions at every candidate:

**Is any easier-schedule advantage visible amid middle-team uncertainty?**

- At lambda=0, E[rank]=66.24, ordinal=67, SD=38.97, entropy=7.102 bits, central 80%=13-121; versus Record/Play the mean-rank changes are +0.00/-3.16. Actual evidence is 1-0, mean margin +29.00, 0 close wins and 0 close losses, standardized schedule probability 0.948, expected wins 0.94, wins minus expectation +0.06. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=67.40, ordinal=74, SD=38.35, entropy=7.101 bits, central 80%=15-121; versus Record/Play the mean-rank changes are +1.16/-2.00. Actual evidence is 1-0, mean margin +29.00, 0 close wins and 0 close losses, standardized schedule probability 0.948, expected wins 0.94, wins minus expectation +0.06. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=68.26, ordinal=78, SD=37.84, entropy=7.099 bits, central 80%=16-121; versus Record/Play the mean-rank changes are +2.02/-1.14. Actual evidence is 1-0, mean margin +29.00, 0 close wins and 0 close losses, standardized schedule probability 0.948, expected wins 0.94, wins minus expectation +0.06. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=68.90, ordinal=80, SD=37.41, entropy=7.095 bits, central 80%=18-121; versus Record/Play the mean-rank changes are +2.67/-0.50. Actual evidence is 1-0, mean margin +29.00, 0 close wins and 0 close losses, standardized schedule probability 0.948, expected wins 0.94, wins minus expectation +0.06. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=69.40, ordinal=81, SD=37.04, entropy=7.090 bits, central 80%=19-121; versus Record/Play the mean-rank changes are +3.16/+0.00. Actual evidence is 1-0, mean margin +29.00, 0 close wins and 0 close losses, standardized schedule probability 0.948, expected wins 0.94, wins minus expectation +0.06. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-09-27

Actual record **3–1**, mean margin **+6.00**; close wins/losses 1/0, dominant wins 1. Frozen standardized schedule probability **0.533** (larger is easier), expected wins **2.26**, wins minus expectation **+0.74**. Reference: retained_context_v1.4_posterior, ordinal 67, band middle, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 41.50 | 36 | 28.77 | 6.675 | 18–61 | 8–84 | 3–106 | 0.135/0.165/0.361 | +0.00 / -12.14 |
| 0.25 | 46.47 | 49 | 28.03 | 6.726 | 24–66 | 12–87 | 5–107 | 0.078/0.100/0.271 | +4.98 / -7.16 |
| 0.5 | 49.85 | 54 | 26.97 | 6.714 | 28–69 | 17–88 | 7–107 | 0.046/0.061/0.208 | +8.35 / -3.79 |
| 0.75 | 52.12 | 59 | 25.92 | 6.677 | 32–71 | 20–88 | 10–106 | 0.027/0.038/0.165 | +10.62 / -1.52 |
| 1 | 53.64 | 67 | 24.95 | 6.634 | 34–71 | 23–88 | 13–105 | 0.017/0.025/0.134 | +12.14 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401862700 / 2026-09-05 | home | UT Rio Grande Valley (fcs) | 45–16 | +29 | 0.973 | 58.45/26.08 | +32.48 |
| 401860884 / 2026-09-12 | away | Texas State (fbs) | 31–26 | +5 | 0.487 | 65.72/25.33 | -0.53 |
| 401856693 / 2026-09-20 | away | Texas (fbs) | 6–30 | -24 | 0.066 | 5.49/4.33 | -22.74 |
| 401860896 / 2026-09-26 | home | Colorado State (fbs) | 59–45 | +14 | 0.735 | 82.70/23.08 | +10.34 |

Answers to the original questions at every candidate:

**Is any easier-schedule advantage visible amid middle-team uncertainty?**

- At lambda=0, E[rank]=41.50, ordinal=36, SD=28.77, entropy=6.675 bits, central 80%=8-84; versus Record/Play the mean-rank changes are +0.00/-12.14. Actual evidence is 3-1, mean margin +6.00, 1 close wins and 0 close losses, standardized schedule probability 0.533, expected wins 2.26, wins minus expectation +0.74. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=46.47, ordinal=49, SD=28.03, entropy=6.726 bits, central 80%=12-87; versus Record/Play the mean-rank changes are +4.98/-7.16. Actual evidence is 3-1, mean margin +6.00, 1 close wins and 0 close losses, standardized schedule probability 0.533, expected wins 2.26, wins minus expectation +0.74. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=49.85, ordinal=54, SD=26.97, entropy=6.714 bits, central 80%=17-88; versus Record/Play the mean-rank changes are +8.35/-3.79. Actual evidence is 3-1, mean margin +6.00, 1 close wins and 0 close losses, standardized schedule probability 0.533, expected wins 2.26, wins minus expectation +0.74. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=52.12, ordinal=59, SD=25.92, entropy=6.677 bits, central 80%=20-88; versus Record/Play the mean-rank changes are +10.62/-1.52. Actual evidence is 3-1, mean margin +6.00, 1 close wins and 0 close losses, standardized schedule probability 0.533, expected wins 2.26, wins minus expectation +0.74. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=53.64, ordinal=67, SD=24.95, entropy=6.634 bits, central 80%=23-88; versus Record/Play the mean-rank changes are +12.14/+0.00. Actual evidence is 3-1, mean margin +6.00, 1 close wins and 0 close losses, standardized schedule probability 0.533, expected wins 2.26, wins minus expectation +0.74. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### 2026-10-25T12-59-22.246522Z

Actual record **7–1**, mean margin **+7.50**; close wins/losses 4/0, dominant wins 1. Frozen standardized schedule probability **0.579** (larger is easier), expected wins **5.05**, wins minus expectation **+1.95**. Reference: retained_context_v1.4_posterior, ordinal 65, band middle, schedule tercile middle.

| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 0 | 22.65 | 19 | 17.92 | 5.870 | 9–32 | 4–48 | 1–68 | 0.296/0.354/0.653 | +0.00 / -30.74 |
| 0.25 | 33.02 | 34 | 20.14 | 6.241 | 18–45 | 10–62 | 4–80 | 0.115/0.150/0.416 | +10.37 / -20.37 |
| 0.5 | 41.77 | 48 | 20.47 | 6.338 | 26–55 | 17–71 | 9–86 | 0.035/0.051/0.235 | +19.11 / -11.63 |
| 0.75 | 48.36 | 57 | 20.12 | 6.337 | 33–62 | 24–76 | 15–90 | 0.010/0.015/0.126 | +25.71 / -5.03 |
| 1 | 53.39 | 65 | 19.60 | 6.311 | 38–67 | 28–80 | 20–93 | 0.002/0.005/0.067 | +30.74 / +0.00 |

| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |
|---|---|---|---:|---:|---:|---|---:|
| 401862700 / 2026-09-05 | home | UT Rio Grande Valley (fcs) | 45–16 | +29 | 0.988 | 68.71/18.76 | +35.46 |
| 401860884 / 2026-09-12 | away | Texas State (fbs) | 31–26 | +5 | 0.413 | 53.92/19.48 | -3.41 |
| 401856693 / 2026-09-20 | away | Texas (fbs) | 6–30 | -24 | 0.068 | 6.26/4.30 | -22.04 |
| 401860896 / 2026-09-26 | home | Colorado State (fbs) | 59–45 | +14 | 0.825 | 96.74/17.44 | +14.51 |
| 401862788 / 2026-10-03 | away | Rice (fbs) | 16–14 | +2 | 0.804 | 111.42/13.13 | +13.07 |
| 401862794 / 2026-10-08 | home | South Florida (fbs) | 31–24 | +7 | 0.610 | 59.46/19.92 | +4.32 |
| 401862804 / 2026-10-18 | home | Navy (fbs) | 38–34 | +4 | 0.780 | 88.04/17.92 | +11.91 |
| 401862811 / 2026-10-25 | away | Tulane (fbs) | 42–19 | +23 | 0.564 | 76.24/19.43 | +2.46 |

Answers to the original questions at every candidate:

**Is any easier-schedule advantage visible amid middle-team uncertainty?**

- At lambda=0, E[rank]=22.65, ordinal=19, SD=17.92, entropy=5.870 bits, central 80%=4-48; versus Record/Play the mean-rank changes are +0.00/-30.74. Actual evidence is 7-1, mean margin +7.50, 4 close wins and 0 close losses, standardized schedule probability 0.579, expected wins 5.05, wins minus expectation +1.95. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.25, E[rank]=33.02, ordinal=34, SD=20.14, entropy=6.241 bits, central 80%=10-62; versus Record/Play the mean-rank changes are +10.37/-20.37. Actual evidence is 7-1, mean margin +7.50, 4 close wins and 0 close losses, standardized schedule probability 0.579, expected wins 5.05, wins minus expectation +1.95. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.5, E[rank]=41.77, ordinal=48, SD=20.47, entropy=6.338 bits, central 80%=17-71; versus Record/Play the mean-rank changes are +19.11/-11.63. Actual evidence is 7-1, mean margin +7.50, 4 close wins and 0 close losses, standardized schedule probability 0.579, expected wins 5.05, wins minus expectation +1.95. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=0.75, E[rank]=48.36, ordinal=57, SD=20.12, entropy=6.337 bits, central 80%=24-76; versus Record/Play the mean-rank changes are +25.71/-5.03. Actual evidence is 7-1, mean margin +7.50, 4 close wins and 0 close losses, standardized schedule probability 0.579, expected wins 5.05, wins minus expectation +1.95. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.
- At lambda=1, E[rank]=53.39, ordinal=65, SD=19.60, entropy=6.311 bits, central 80%=28-80; versus Record/Play the mean-rank changes are +30.74/+0.00. Actual evidence is 7-1, mean margin +7.50, 4 close wins and 0 close losses, standardized schedule probability 0.579, expected wins 5.05, wins minus expectation +1.95. These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. The expectation is retrospective and reuses these games.

### Evolution across available snapshots

| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |
|---|---:|---|---:|---:|---:|---:|
| 2026-09-08 → 2026-09-27 | 0 | 3 / 359 | -24.74 | -31 | -10.20 | 0.286 |
| 2026-09-08 → 2026-09-27 | 0.25 | 3 / 359 | -20.93 | -25 | -10.33 | 0.264 |
| 2026-09-08 → 2026-09-27 | 0.5 | 3 / 359 | -18.41 | -24 | -10.87 | 0.267 |
| 2026-09-08 → 2026-09-27 | 0.75 | 3 / 359 | -16.79 | -21 | -11.49 | 0.277 |
| 2026-09-08 → 2026-09-27 | 1 | 3 / 359 | -15.76 | -14 | -12.09 | 0.289 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0 | 4 / 445 | -18.84 | -17 | -10.84 | 0.302 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.25 | 4 / 445 | -13.45 | -15 | -7.88 | 0.215 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.5 | 4 / 445 | -8.08 | -6 | -6.51 | 0.157 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 0.75 | 4 / 445 | -3.76 | -2 | -5.80 | 0.138 |
| 2026-09-27 → 2026-10-25T12-59-22.246522Z | 1 | 4 / 445 | -0.24 | -2 | -5.35 | 0.138 |

These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.
