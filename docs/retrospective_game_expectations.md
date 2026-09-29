# Retrospective completed-game expectations

The version `2.0` `retrospective_game_expectations` object in a predictive
snapshot's `team_seasons.json` compares each included result with the selected
snapshot's **full posterior** beliefs about its participants. It uses the
posterior PMFs already computed for that publication, including evidence from
the evaluated game. It is a hindsight comparison, not a pregame forecast or
an attribution of a ranking change to one game.

For an included game `g`, the distribution is the Historical Likelihood V1
margin mixture under the two ordinary posterior rank PMFs from the selected
snapshot. The Historical Likelihood supplies the same site, FBS/FCS pairing,
orientation, and Student-t response as future-game predictions. The two PMFs
are combined as a product of approximate loopy-BP marginals; this does not
claim an exact joint Bayesian posterior. No per-game exclusion or BP replay
runs during snapshot generation. Historical publications use their own frozen
posterior and included-game boundary, so later results cannot enter their
expectations.

Every record is canonically home-oriented: `home points - away points`. It
contains the integer actual margin, exact mixture mean and median, central
50%, 80%, and 95% intervals, and the observed-margin lower and upper one-sided
tails. The continuous mixture makes the lower tail the observed percentile and
the upper tail its complement. The browser reverses the orientation for an away
focal team and uses the tail in the observed favorable or unfavorable direction.
The fixed-scale 40-bin `display_distribution` spans -40 to +40 points and
records off-axis mass separately. Exact scalar summaries always come from the
mixture, never the display bins.

The artifact's `source_snapshot_id`, `included_game_ids`, cutoff, likelihood
version and pinned SHA-256, a canonical `posterior_pmfs_sha256`, the machine-readable
`conditioning: selected_snapshot_full_posterior_including_game`, inference
implementation, and
per-record provenance bind it to the
selected publication. The source artifact contains every included modeled
game, including FCS/FCS evidence. Static publication validates this full source
and exports only records with an FBS participant as
`retrospective_game_expectations_site_projection`, with explicit
`site_projection_version: 1.0`, `published_game_ids`, and
`coverage: fbs_team_schedules`. A Performance page
uses the paired Context artifact and preserves its Context source provenance.

Retained publications can be refreshed with
`uv run python scripts/backfill_retrospective_game_expectations.py`. This reads
frozen included-game rows, verifies the saved prior and posterior PMF hashes,
validates team and rank coverage and PMF normalization, checks the pinned Historical
Likelihood V1 bytes, and uses the stored posterior PMFs. It does not regenerate
posterior beliefs or rankings. The script
also refreshes paired Performance artifacts before static site export.

The serialized hindsight artifact excludes wall-clock runtime so repeated
forced backfills produce the same bytes. Runtime is printed by the backfill
command. Retained snapshots preserve their original `source_retrieved_at`
value. Where older FBS and FCS responses arrived at different times, the
metadata marks that field `legacy_first_response` and records the later
availability as `combined_source_available_at`. Their historical effective
cutoff and included evidence remain frozen. New snapshots use the later
response for both source availability and cutoff clamping. Two early live
publications have no constituent retrieval times; they carry
`legacy_unverified_availability` and leave combined availability unset.
Their included games remain the only durable result evidence.
