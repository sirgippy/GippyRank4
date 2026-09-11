# r/CFB computer-ballot export

The rankings page can download the currently selected published Top 25 as the
JSON accepted by the r/CFB Poll ballot editor.  The download is generated in
the browser; it never authenticates to, edits, or submits anything on the poll
site.  A selection with fewer than 25 rated teams, or without a mapped Top-25
team, stays disabled and explains why it cannot be exported.

## Mapping provenance

[`data/reference/redditcfb_team_handles.csv`](../data/reference/redditcfb_team_handles.csv)
is the durable `cfbd_team_id,team_name,redditcfb_handle` reference.  It is
checked in so handles are not derived from display names.  The handles were
verified against the RedditCFB team namespace on 2026-09-08.  The production
2026 publication currently has 138 FBS identities, all 138 mapped, with no
duplicate IDs or handles.  The audit records these manually resolved aliases:
South Florida → `usf`, Western Kentucky → `wku`, East Carolina → `ecu`, App
State → `appalachianstate`, Florida Atlantic → `fau`, UConn → `connecticut`,
Florida International → `fiu`, UL Monroe → `ulm`, Sam Houston →
`samhoustonstate`, and Massachusetts → `umass`.

The build exposes the mapping and audit under `manifest.json`'s `redditcfb`
object.  Missing mappings remain visible in the audit, and the browser fails
closed for an export that would contain one.

## Rationale templates

The overall rationale is deterministic presentation text.  Its template is:

```text
Generated from GippyRank4, a probabilistic college-football ranking model
that estimates underlying team quality from game performance and expresses
uncertainty rather than treating rank as perfectly known. This ballot uses the
{season} {snapshot label} {family} {prior when Predictive} rankings.
{family-specific semantic note}

Explore the rankings and uncertainty at: {configured canonical site URL}
```

Predictive uses this semantic note:

```text
Predictive estimates current underlying team quality using preseason
information plus games.
```

Performance uses this note:

```text
Performance asks what quality is implied by games played, using Context
estimates to interpret opponent quality. Performance is not standings,
strength of record, or postseason deservingness.
```

Predictive team entries use:

```text
GippyRank expected rank: {expected rank to one decimal}. Central 80% interval:
{low}–{high}. Top-25 probability: {published percentage}.
```

Performance team entries use the same published values with this explicit
prefix:

```text
GippyRank Performance-equivalent expected rank: {expected rank to one decimal}.
Central 80% interval: {low}–{high}. Top-25 probability: {published percentage}.
```

The values come from the selected snapshot's published ranking rows.  No PMFs
are recomputed in JavaScript, no football facts are invented, and the exported
entry order is the displayed expected-rank order.  The JSON has exactly 25
entries with ranks 1–25, `poll_type: "computer"`, and canonical handles.

For example, a downloaded filename is
`gippyrank-2026-week-2-predictive-context.json`; the Performance equivalent
ends in `-performance.json`.  Historical published snapshots use their own
ranking rows, uncertainty values, and snapshot label.
