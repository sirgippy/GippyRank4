# Issue 136 browser evidence

These PNGs were rendered from the retained 2026 full-posterior production artifacts with
`node scripts/capture-issue-136.cjs` against the repository's Playwright
Chromium server. The script routes this worktree's `site/` files directly and
uses local neutral 60×40 image placeholders to preserve opponent-logo geometry.
The browser timezone is fixed to America/Chicago for reproducible dates.

| Capture | What to inspect |
| --- | --- |
| [Georgia desktop](georgia-desktop.png) | Schedule-specific help, completed games, separate weekly belief rows, a future matchup, and the opened Oklahoma retrospective distribution. |
| [Georgia mobile](georgia-mobile.png) | The same schedule at 390 px, with the distribution open and no horizontal overflow. |
| [Georgia vs Tennessee State](georgia-offscale-intervals.png) | The central 50% lies beyond ±40, so the chart uses a wider view of the labeled central intervals, expected margin, and actual margin. |
| [Georgia vs Tennessee State mobile](georgia-offscale-intervals-mobile.png) | The expanded interval view and facts at 390 px. |
| [LSU vs McNeese](lsu-future-intervals.png) | A future forecast with 71% of mass outside ±40 uses the same wider interval view. |
| [Massachusetts at Rutgers](massachusetts-surprising.png) | An away win by 16 despite a substantial expected Rutgers margin under the selected snapshot's full posterior. |
| [Colorado vs Weber State](colorado-near-expectation.png) | The observed result sits close to the full-posterior expected margin. |
| [Historical Georgia](georgia-historical.png) | The Sep 26 selected snapshot ends the belief sequence; Oklahoma remains future and earlier games use historical expectations. |
| [Massachusetts season](massachusetts-season.png) | A broad-rank-uncertainty team on mobile, with distinct completed and future rows. |

Each capture reported 0 px document overflow at its viewport width.
