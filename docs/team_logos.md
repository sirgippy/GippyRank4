# Team logo integration

GippyRank uses the RedditCFB CDN directly for team marks. The source of truth
for the browser URL is `team_logos.url_template` in
`site/publish_config.json`; the default is:

```text
https://cdn.redditcfb.com/60x40/cfb/{handle}.png
```

The checked-in mapping in `src/gippyrank/team_logos.py` preserves the CFBD
team ID and display name while adding the verified RedditCFB handle. Exported
ranking rows carry `logo_handle`; team-season rows carry both the focal
team's handle and `opponent_logo_handle`. The exporter records the active-team
audit in `site/data/manifest.json` and leaves an unknown identity unmarked
instead of guessing a URL.

The static site uses the shared manifest template on the rankings table,
ranking-detail view, team-season header, and schedule opponent cards. Logo
images are direct, lazy CDN loads in fixed aspect-ratio boxes. They are
decorative (`alt=""` and `aria-hidden`) when the adjacent team name is
present; a failed image removes only the image and leaves the team name and
data usable. The uncertainty chart continues to expose its accessible text
alternative, so a logo is never the only team identifier.

The current 2026 FBS publication set contains 138 active teams, all with a
verified handle. The explicit namespace aliases are South Florida (`usf`),
Western Kentucky (`wku`), East Carolina (`ecu`), App State
(`appalachianstate`), Florida Atlantic (`fau`), UConn (`connecticut`),
Florida International (`fiu`), UL Monroe (`ulm`), Sam Houston
(`samhoustonstate`), and Massachusetts (`umass`).

This repository has no Django poll, ballot editor, or generated weekly-summary
image endpoint; those issue surfaces do not exist in the static GippyRank
architecture and were not invented as part of this integration.
