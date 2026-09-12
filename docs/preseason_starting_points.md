# Preseason starting points

The public site calls the Predictive prior a **Preseason starting point**.
“Prior” remains the mathematical and internal model term, but readers should
first understand what the object does:

```text
preseason information → starting probability distribution
starting distribution + game evidence → current probability distribution
```

GippyRank does not begin by assigning a team one certain rank. It begins with
probabilities over plausible rank states. Results update that distribution; a
current expected rank is not a preseason ordinal rank plus a fixed number of
points.

## Context and History

Both Predictive families use the same historical foundation and the same game
evidence/update framework after kickoff.

| Information before kickoff | Context | History |
| --- | --- | --- |
| Previous program performance | Yes | Yes |
| Recruiting | Yes | No |
| Roster talent | Yes | No |
| Returning production | Yes | No |
| Coach tenure | Yes | No |
| Selected-season game evidence | Added after preseason | Added after preseason |

Context asks what we knew about this particular team before kickoff. History
asks what we would believe from the program’s track record alone. History uses
previous program performance across recent and longer historical windows; it
does not use current roster talent, recruiting, returning production, or coach
tenure.

The site intentionally describes input families rather than claiming that an
individual feature caused an exact number of rank spots. The production
Context model has correlated, regularized predictors and probabilistic
uncertainty, so the public page shows the inputs that shape the starting
distribution without inventing feature attribution.

## Team-page comparison

For a Predictive team page, the manifest contains an explicit build-time
reference from the selected snapshot to the published preseason snapshot for
the same season and prior family. The browser loads the exact lazy PMFs for:

```text
published preseason Context → selected Context snapshot
published preseason History → selected History snapshot
```

The selected side is always the exact snapshot in the URL, including an older
historical snapshot. The browser does not choose the earliest file, parse a
snapshot ID, reconstruct a PMF, or substitute today’s latest ranking. Both
distributions use one shared rank axis so changes in location, shape, and
uncertainty remain visible.

Preseason pages show the selected starting distribution and its input-family
explanation instead of a redundant Preseason → Preseason comparison.
Performance pages do not have a Predictive preseason prior and therefore do
not fabricate this comparison.

This presentation change does not alter Context C 1.2, History H 1.1,
Posterior V1, game likelihoods, uncertainty calculations, or published ranking
values.
