# Team logo integration

GippyRank uses the RedditCFB CDN directly for team marks. The source of truth
for the browser URL is `team_logos.url_template` in
`site/publish_config.json`; the default is:

```text
https://cdn.redditcfb.com/60x40/cfb/{handle}.png
```

The checked-in mapping in `src/gippyrank/team_logos.py` preserves the CFBD
team ID and display name while adding the verified RedditCFB handle. The
exporter publishes one manifest-level `team_logos.handles` map keyed by stable
team ID, so ranking rows and schedule entries do not repeat presentation
handles. The manifest audit covers both the published FBS ranking set and
every team identity rendered by team-season pages, including schedule
opponents. Unknown identities are omitted from the map instead of being
guessed from a display name.

The static site uses the shared manifest template on the rankings table,
ranking-detail view, team-season header, and schedule opponent cards. Logo
images are direct, lazy CDN loads in fixed aspect-ratio boxes. They are
decorative (`alt=""` and `aria-hidden`) when the adjacent team name is
present; a failed image removes only the image and leaves the team name and
data usable. The uncertainty chart continues to expose its accessible text
alternative, so a logo is never the only team identifier.

The current 2026 FBS publication set contains 138 active teams, all with a
verified handle. The team-season pages render 238 distinct team identities;
235 have verified handles and three are explicitly listed in the manifest
audit as text-only fallbacks: Houston Christian, Southeast Missouri State,
and East Texas A&M. The explicit namespace aliases are South Florida (`usf`),
Western Kentucky (`wku`), East Carolina (`ecu`), App State
(`appalachianstate`), Florida Atlantic (`fau`), UConn (`connecticut`),
Florida International (`fiu`), UL Monroe (`ulm`), Sam Houston
(`samhoustonstate`), and Massachusetts (`umass`).

This repository has no Django poll, ballot editor, or generated weekly-summary
image endpoint; those issue surfaces do not exist in the static GippyRank
architecture and were not invented as part of this integration.
