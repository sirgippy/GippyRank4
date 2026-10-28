"""Phase 4 synthesis of the fixed synthetic, population and diagnostic evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gippyrank.research.issue_242_quality_of_results import MARGIN_WEIGHTS

WEEK9 = "2026-10-25T12-59-22.246522Z"


def table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers),
            *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
        ]
    )


def figures(output: Path, sensitivity: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for ax, profile in zip(axes, ("elite", "weak"), strict=True):
        rows = [r for r in sensitivity if r["profile"] == profile]
        ax.plot(
            [r["margin_weight"] for r in rows],
            [r["outcome_boundary_benefit"] for r in rows],
            "o-",
            label="Loss by 3 → win by 3",
        )
        ax.plot(
            [r["margin_weight"] for r in rows],
            [r["within_win_margin_benefit"] for r in rows],
            "s-",
            label="Win by 10 → win by 35",
        )
        ax.set(
            title=rows[0]["focal_team_name"],
            xlabel="Margin weight λ",
            xticks=MARGIN_WEIGHTS,
        )
        ax.axhline(0, color="gray", lw=0.7)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Improvement in expected rank")
    axes[1].legend(fontsize=8)
    fig.savefig(
        output / "outcome_sensitivity.png",
        dpi=160,
        metadata={"Software": "GippyRank4 issue 267"},
    )
    plt.close(fig)


def write_report(
    root: Path,
    output: Path,
    cases: list,
    evolution: list,
    focused: list,
    diagnostics: list,
    sensitivity: list,
) -> None:
    figures(output, sensitivity)
    execution = json.loads((output / "execution.json").read_text())
    origin = json.loads((output / "provenance.json").read_text())
    gates = json.loads((output / "endpoint_parity.json").read_text())
    by_case = {(r["snapshot_id"], r["team_id"], r["margin_weight"]): r for r in cases}

    def case(sid, t, w=0.25):
        return by_case[sid, t, w]

    text = []

    def add(value):
        text.append(value.strip())

    add("""# Phase 4: real-season Quality of Results, cases and controlled evidence changes

The frozen experiment is complete: **40 score replacements and 25 game removals converged**, and all **27 team seasons / 70 snapshot-case observations** have been evaluated at all five weights. The 32 baseline/reference/order solves were verified and consumed without rerunning inference or rewriting their artifacts. The intermediate candidates retain opponent and margin information while making a narrow win materially more valuable than Quality of Play does. That supports the proposed family, but it does not establish schedule equivalence or uniquely identify a defensible weight.

The evidence is sufficient to proceed to a **separate methodology-selection discussion**, provided that discussion explicitly accepts the observed tradeoffs and the remaining schedule/uncertainty limits. This report does not select or activate a production methodology. No empirical schedule correction, new weight, refitted prior or likelihood, or production publication was introduced.

## Scope and evidence levels

The authoritative inputs are the [frozen issue 263 protocol](../../../docs/issue_263_real_season_protocol.md), its two case inventories, and the [completed issue 265 population study](../issue_265_real_seasons/report.md). The new diagnostics concern the entire frozen **2026 Week 9** network: **975 observations, 266 variables including FCS, and 138 FBS marginal distributions**. A removal leaves 974 observations. The 2025 pre-bowl/final and 2026 Weeks 2/5/9 cases use their original snapshot-specific priors and networks.

**Numerically established:** convergence and PMF validity, endpoint equality, four-state evidence responses, and single-observation removal responses. These are controlled changes to model evidence, not causal effects of real-world scheduling decisions.

**Descriptive:** actual records/margins, frozen opponent-aware reference expectations, real-season schedule associations, matched-record comparisons, boundary positions, and uncertainty patterns. Schedules, teams, snapshots and comparisons are dependent; selection emphasizes specific questions rather than representative sampling.

**Interpretation after observing results:** whether these tradeoffs satisfy the intended greater importance of winning while retaining margin/opponent information, and whether a later selection can be defended. Human polls, CFP selection and predictive-ranking agreement supply no ground-truth standard here.""")
    add(f"""## Numerical execution and provenance

All 65 diagnostics converged under deterministic cold-start BP: maximum 500 iterations, tolerance 1e-9, damping .35; original Context priors, Historical Likelihood V1, existing rematch factors and Performance prior stripping. There were no relaxed settings or failed cells. Iterations range **{min(r["iterations"] for r in diagnostics)}–{max(r["iterations"] for r in diagnostics)}**; largest terminal message delta **{max(r["max_message_delta"] for r in diagnostics):.3g}**. Inference/checkpoint execution took **{execution["elapsed_seconds"]:.1f} seconds**, summed solve time **{sum(r["runtime_seconds"] for r in diagnostics):.1f} seconds**, slowest solve **{max(r["runtime_seconds"] for r in diagnostics):.2f} seconds**; process high-water RSS **{max(r["process_peak_rss_bytes"] for r in diagnostics) / 1024**3:.2f} GiB**. RSS is a cumulative process maximum, not memory attributable to an individual game. The frozen three-hour, 600-second-per-solve and 4-GiB bounds were respected.

Every complete-network Context PMF and prior-stripped FBS PMF is retained and validated for population, support, finiteness, nonnegativity and normalization. Evidence counts and raw/grouped factor counts are checked against the exact perturbed network. All four λ=0 winning-margin comparisons pass: maximum FBS PMF difference **{max(g["max_absolute_pmf_difference"] for g in gates.values()):.3g}**, maximum expected-rank difference **{max(g["max_expected_rank_difference"] for g in gates.values()):.3g}**; the three artificial wins are identical within the frozen 1e-8/N×1e-8 tolerances. Full-network Context parity also passes.

Original baseline revision: `{origin["baseline"]["execution_revision"]}`; original fingerprint: `{origin["baseline"]["fingerprint"]}`. New diagnostic execution revision: `{origin["execution_revision"]}`. Exact source/input/implementation fingerprints, original and replacement dated observations, team/game populations, numerical configuration and resource measurements accompany every checkpoint. The [manifest](manifest.json) hashes every retained output and records the original baseline verification identity. [Execution instructions](../../../docs/issue_267_real_season_diagnostics.md) describe deterministic replay and the separate analysis fingerprint.""")
    add("""## Controlled outcome-boundary versus within-win sensitivity

Each hypothetical score replaces one actual observation, retaining its ID, opponent, site, date and all other evidence. Opponent points are `max(actual_opponent_points,3)`, and focal points add the frozen hypothetical margin, then restore home/away orientation. These scenarios are evidence probes, not football predictions.

Positive benefits improve expected rank. Outcome benefit is `E[rank at −3] − E[rank at +3]`; margin benefit is `E[rank at +10] − E[rank at +35]`. The difference is outcome minus margin benefit. No ratio or universal points-per-win exchange rate is implied.""")
    add(
        table(
            [
                "Profile / focal team",
                "λ",
                "Outcome benefit",
                "Within-win benefit",
                "Difference",
                "Ordinal outcome / margin benefit",
            ],
            [
                [
                    r["profile"] + " / " + r["focal_team_name"],
                    f"{r['margin_weight']:g}",
                    f"{r['outcome_boundary_benefit']:.3f}",
                    f"{r['within_win_margin_benefit']:.3f}",
                    f"{r['outcome_minus_margin_benefit']:+.3f}",
                    f"{r['outcome_boundary_ordinal_benefit']} / {r['within_win_ordinal_benefit']}",
                ]
                for r in sensitivity
            ],
        )
    )
    add(
        "![Outcome and margin sensitivity in the two frozen games](outcome_sensitivity.png)"
    )
    add("""At λ=0 the response is entirely outcome based and within-win margin sensitivity vanishes. At .25 the additional outcome credit is substantial, while positive within-win sensitivity demonstrates that scoring margins remain informative. Higher weights increasingly emphasize a decisive win over an already positive narrow margin. At .50 the outcome benefit still exceeds the within-win benefit in both games; at .75 that ordering reverses in both (only .010 ranks in Alabama). Alabama’s within-win benefit peaks at .75 and falls slightly at 1 (.654 to .640), so it is not monotone in margin weight. This reflects a changed posterior location near the best-rank boundary and the full network response, rather than any failure of the endpoint or convergence gates. Central Michigan’s +3 instead of −3 increases its .25 SD from 19.84 to 22.32 and entropy from 6.262 to 6.472 as its mean moves away from the weak end of bounded support; a better result need not narrow a posterior. The elite and weak response magnitudes differ because their posterior locations, uncertainties, opponents and connected evidence differ; this experiment does not isolate an intrinsic strength interaction. The table, rather than an assumed monotone exchange rate, is the direct answer for these two games.

Issue 252's separate approximately-even and weaker-opponent examples gave .25 outcome/margin benefits of 8.40/1.69 and 5.44/1.73; λ=1 gave 1.50/4.91 and 1.30/5.45. The current elite/weak extremes extend that evidence without rerunning those simulations. These are four particular contexts, not universal calibration. The complete four-state means, ordinal ranks, SD/entropy/50/80/95% intervals and latent top-set masses are in [outcome_sensitivity.csv](outcome_sensitivity.csv); opponent and all other FBS responses versus the actual baseline remain separate in [game_diagnostics.csv](game_diagnostics.csv) and [network_changes.csv.gz](network_changes.csv.gz).""")
    add("## Single-observation removals and network propagation")
    removals = [r for r in focused if r["kind"] == "removal" and r["role"] == "focal"]
    add(
        table(
            [
                "Frozen focal / profile",
                "λ",
                "ΔE[rank]",
                "Δordinal",
                "ΔSD",
                "Δentropy",
                "TV",
                "W1",
            ],
            [
                [
                    r["team_name"] + " / " + r["profile"],
                    f"{r['margin_weight']:g}",
                    f"{r['delta_expected_rank']:+.3f}",
                    f"{r['delta_ordinal_rank']:+d}",
                    f"{r['delta_rank_sd']:+.3f}",
                    f"{r['delta_entropy_bits']:+.3f}",
                    f"{r['total_variation']:.3f}",
                    f"{r['wasserstein_rank']:.3f}",
                ]
                for r in removals
            ],
        )
    )
    add("""Positive rank changes mean worse after removing the observation. Negative changes mean the remaining evidence evaluates the team better. This sign can differ across λ: an observed loss may be positive margin evidence relative to the opponent, while still being a loss under the binary factor. Removal is not the same operation as replacing a loss with a win. A blowout removal can strongly affect margin candidates while having less effect under Record. These are coherent consequences of which evidence each candidate retains, not numerical nonconvergence. Specifically, removing Michigan’s 6–13 loss to Indiana improves its Record mean by 4.040 but worsens its Play mean by 2.419; the sign crosses near .50 (+.026, TV .026). Removing Memphis’s 41–34 win over East Carolina worsens Record by 4.440 but improves Play by 3.555; the sign has already crossed at .50 (−.627). These changes expose actual outcome-versus-margin conflicts at intermediate weights. Oregon’s 84–0 Portland State win improves its .25 evaluation by a total removal response of 3.094 places, versus .172 under Record and 7.153 under Play. Even the conspicuous blowout therefore does not dominate every candidate’s entire season evaluation.

Each removed observation decrements exactly the two participants' eligible-game counts; all other counts and original priors remain fixed. The engine rebuilds grouped factors. The frozen Week 9 removal games have no second meeting to remove; regression coverage also checks retaining a legitimate rematch and the existing zero-game uniform rule. Full PMFs, credible endpoint/width changes, opponent responses, and every rated FBS change are retained. Oregon’s opponent Portland State is FCS: its corresponding response uses the full Context posterior on FCS support, explicitly labeled separately from FBS prior-stripped Performance. It has no FBS ordinal or FBS top-set status; these fields are inapplicable. **Do not sum these removals:** propagation overlaps through opponents, shared opponents and cycles, so they are total network responses rather than independent game contributions.""")
    add(
        table(
            [
                "Game / λ=.25 removal",
                "Focal ΔE",
                "Opponent ΔE",
                "Other max absolute ΔE (team)",
                "Other mean absolute ΔE",
                "Top10 / 12 / 25 entries",
            ],
            [
                [
                    r["game_id"],
                    f"{next(f['delta_expected_rank'] for f in focused if f['perturbation_key'] == r['perturbation_key'] and f['margin_weight'] == 0.25 and f['role'] == 'focal'):+.3f}",
                    f"{next(f['delta_expected_rank'] for f in focused if f['perturbation_key'] == r['perturbation_key'] and f['margin_weight'] == 0.25 and f['role'] == 'opponent'):+.3f}",
                    f"{r['other_max_absolute_expected_rank_change']:.3f} ({r['other_max_change_team']})",
                    f"{r['other_mean_absolute_expected_rank_change']:.4f}",
                    " / ".join(
                        ", ".join(r[f"top{k}_entry_names"]) or "none"
                        for k in (10, 12, 25)
                    ),
                ]
                for r in diagnostics
                if r["kind"] == "removal" and r["margin_weight"] == 0.25
            ],
        )
    )
    changed = [
        r
        for r in diagnostics
        if any(r[f"top{k}_entries"] or r[f"top{k}_exits"] for k in (10, 12, 25))
    ]
    add(
        f"""Across the 65 probes, **{len(changed)}** alter at least one of the top10/12/25 sets. The largest other-team absolute mean response is **{max(r["other_max_absolute_expected_rank_change"] for r in diagnostics):.3f}** ranks. All 136 FBS nonparticipant rows per FBS–FBS probe, and all 137 for the FCS-opponent probe, are retained, including negligible changes; no significance threshold deletes rows. “Above tolerance” counts use only the frozen numerical mean tolerance, not a substantive or statistical threshold. [diagnostic_summary.csv](diagnostic_summary.csv) includes all entries/exits, union movements, and maximum/mean spread; [network_changes.csv.gz](network_changes.csv.gz) adds adjacent mean gaps, latent mass and uncertainty for every team."""
    )
    add("## Priority case: Stanford, 2026 Week 9")
    stanford = case(WEEK9, "24")
    add(f"""Stanford's **4–4** record has mean margin **−15.625**, standardized schedule probability **.438**, and only **{stanford["expected_wins"]:.2f}** retrospective expected wins. Four wins exceed that reference by **{stanford["wins_minus_expectation"]:.2f}**, but that expectation reuses the same games. Its four losses are by **39, 28, 54 and 51**, whereas the wins are by **10, 7, 27 and 3**. Two narrow wins, Georgia Tech and NC State, have retrospective integrated win chances about .360 and .243. The Hawai'i win has chance .578; the Elon game is explicitly FCS, with chance .960. Large losses drive the negative mean margin without reversing the actual outcomes of the other four games.

At λ=.25 its expected rank is **{stanford["expected_rank"]:.2f}**, ordinal **{stanford["ordinal_rank"]}**, versus Play **{case(WEEK9, "24", 1)["expected_rank"]:.2f} / {case(WEEK9, "24", 1)["ordinal_rank"]}** and Record **{case(WEEK9, "24", 0)["expected_rank"]:.2f} / {case(WEEK9, "24", 0)["ordinal_rank"]}**. Thus .25 gives **31.06 mean-rank places** more credit than Play while retaining a materially worse evaluation than Record. SD is **{stanford["rank_sd"]:.2f}**, entropy **{stanford["entropy_bits"]:.3f}** bits, and the 80% interval **{stanford["low_80"]}–{stanford["high_80"]}**; this is a diffuse evaluation, not a claim that Stanford is precisely the 88th-best team.

The difference is understandable as a tradeoff: outcomes against a demanding opponent network retain value despite very poor loss margins. It is also a serious defensibility test for low weights. A reader who wants catastrophic losses to dominate season evaluation will reject some of this credit. Calling it excessive solely because it disagrees with Play would assume the conclusion. None of Stanford's games was selected for removal, so this study cannot identify additive shares of its 31.06-place difference. Its complete opponent distributions, margins and all five posterior summaries are preserved in the frozen case appendix.""")
    add("## Matched records, schedules and uncertainty")
    pairs = [
        ("2026 Week 9 middle", WEEK9, "2", "41"),
        ("2026 Week 9 weak", WEEK9, "24", "113"),
        ("2025 pre-bowl middle", "2025-prebowl", "96", "2084"),
        ("2025 final middle", "2025-final", "96", "2084"),
        ("2025 pre-bowl weak", "2025-prebowl", "2509", "2638"),
        ("2025 final weak", "2025-final", "2509", "2638"),
    ]
    add(
        "The first member in each pair has the harder frozen standardized schedule. Larger schedule probability means easier; the reference team and reference construction differ across seasons. Tables retain every λ. Same records do not equalize opponent quality, expectations, loss margins or reference strength."
    )
    add(
        table(
            [
                "Pair / harder vs easier",
                "Record (hard / easy)",
                "Schedule p (hard / easy)",
                "Wins minus expectation (hard / easy)",
            ],
            [
                [
                    label
                    + ": "
                    + case(sid, a)["team_name"]
                    + " / "
                    + case(sid, b)["team_name"],
                    f"{case(sid, a)['wins']}-{case(sid, a)['losses']} / {case(sid, b)['wins']}-{case(sid, b)['losses']}",
                    f"{case(sid, a)['schedule_reference_mean_win_probability']:.3f} / {case(sid, b)['schedule_reference_mean_win_probability']:.3f}",
                    f"{case(sid, a)['wins_minus_expectation']:+.2f} / {case(sid, b)['wins_minus_expectation']:+.2f}",
                ]
                for label, sid, a, b in pairs
            ],
        )
    )
    add(
        table(
            [
                "Pair",
                "λ",
                "E hard / easy",
                "Easy−hard E gap",
                "SD hard / easy",
                "Entropy hard / easy",
                "Widths50/80/95 hard / easy",
            ],
            [
                [
                    label,
                    f"{w:g}",
                    f"{case(sid, a, w)['expected_rank']:.2f} / {case(sid, b, w)['expected_rank']:.2f}",
                    f"{case(sid, b, w)['expected_rank'] - case(sid, a, w)['expected_rank']:+.2f}",
                    f"{case(sid, a, w)['rank_sd']:.2f} / {case(sid, b, w)['rank_sd']:.2f}",
                    f"{case(sid, a, w)['entropy_bits']:.3f} / {case(sid, b, w)['entropy_bits']:.3f}",
                    " / ".join(
                        "/".join(
                            str(case(sid, t, w)[f"width_{c}"]) for c in (50, 80, 95)
                        )
                        for t in (a, b)
                    ),
                ]
                for label, sid, a, b in pairs
                for w in MARGIN_WEIGHTS
            ],
        )
    )
    add("""**Auburn versus UConn:** both are 4–3, yet Auburn's schedule probability .308 versus .697 and wins above expectation +.53 versus −.85 accompany a large Auburn advantage at every candidate. At .25 the expected-rank gap is **48.13** and ordinal positions are **35 versus 93**. UConn's +11.71 mean margin exceeds Auburn's +1.29, so raw margin alone would misread the opponent-aware accomplishment. This supports retained difficult-schedule recognition, without estimating a causal schedule effect.

**Stanford versus Massachusetts:** Play reverses their ordering, placing easier-schedule Massachusetts ahead by 4.20 expected ranks (101.56 versus 105.76); .75 still puts Stanford ahead by 3.02, so this concern is not hidden by the .25 emphasis. Their uncertainty ordering also reverses at Play: Stanford SD 16.16 versus Massachusetts 16.84. Both are 4–4, but schedule .438 versus .769 and retrospective excess wins +1.70 versus +.13 distinguish the records. At .25 Stanford leads by **29.05 expected ranks**, with SD **25.29 versus 19.35**, entropy **6.672 versus 6.219**, and 80% widths **69 versus 50**. Thus the easier team's posterior is narrower, matching the direction of the synthetic weak uncertainty concern, even though the harder team ranks substantially better. Massachusetts is much nearer the lower boundary of the bounded rank support; its four wins and +5.38 margin also differ from Stanford's distribution of results. Those observations help explain why narrowness is not automatically greater information or fairness; they do not isolate or resolve the mechanism of the uncertainty reversal. The weak Central Michigan perturbation supplies a direct outcome/margin sensitivity measurement, but changing its one game does not control its entire schedule.

**Kentucky/Buffalo:** the 2025 5–7 records correspond to primary schedule probabilities **.438/.747** and wins versus preseason expectation **+.53/−3.13**. At .25 final expected ranks are **61.79/119.50**. This is coherent with very different opponent-aware accomplishments, not a requirement to reward schedule difficulty independently of outcomes. Neither team adds an own postseason game, yet their ranks move as 55 network observations arrive: final-minus-pre-bowl .25 changes are **+1.77/+0.54**. This demonstrates why own-game-only narratives are incomplete.

**Purdue/UTEP:** both are 2–10, with schedule probabilities **.443/.708**, preseason excess wins **−1.05/−3.85**, and final .25 expected ranks **99.55/129.80**. SDs **18.61/5.45**, entropies **6.201/4.143** and 80% widths **49/13** again make the easier team narrower. The very low UTEP distribution on bounded support and worse shortfall from preseason expectations accompany the gap. Neither team has an own new postseason game; their .25 network responses are **−1.21/−.03**. The matched-record evidence supports distinct accomplishment evaluations, while the cause and desirability of the uncertainty contrast remain unresolved.""")
    add("## Priority historical case: 2025 Navy")
    navy = case("2025-final", "2426")
    add(f"""Navy finishes **11–2**, mean margin **+6.46**, with primary schedule probability **.692**, preseason expected wins **{navy["expected_wins"]:.2f}**, and excess wins **+1.77**. Its opponents include many weak teams, and its North Texas/Notre Dame losses are by 14/39. Four wins are by seven or fewer: Air Force +3, Temple +1, South Florida +3 and Army +1. The final Cincinnati win is by 22. Navy's excellent record survives narrow games rather than merely reflecting large margins against overmatched opponents.

At .25 its final expected rank is **{navy["expected_rank"]:.2f}**, ordinal **41**, compared with Play **{case("2025-final", "2426", 1)["expected_rank"]:.2f} / 66**, a **27.34-place** mean-rank difference. Its .25 SD **19.47** and 80% interval **{navy["low_80"]}–{navy["high_80"]}** retain substantial uncertainty. The easier schedule keeps it far from a top-ranked position despite 11 wins. Its late Army/Cincinnati wins and the rest of the network move .25 expected rank from **55.68 to 47.80**. This is a defensible outcome-focused reading if close wins matter substantially, but readers must accept that several narrow wins can outweigh a poor loss margin. It does not validate a general easy-schedule reward or make .25 uniquely correct.""")
    add("## Every frozen case, including unremarkable and contrary examples")
    add(
        "The table gives each selected team at its latest snapshot; all 70 observations, original categories/questions, full game evidence and all five uncertainty profiles appear in [case_studies.md](case_studies.md). [case_results.csv](case_results.csv) contains 350 candidate rows and explicit per-question answers; [case_evolution.csv](case_evolution.csv) contains all 215 transitions. Selection facts, observations, candidate results and interpretations occupy separate fields. Early snapshot categories retain their Week 9 selection meaning, even when the selected diagnostic game had not yet occurred."
    )
    latest = sorted(
        [
            r
            for r in cases
            if r["margin_weight"] == 0.25 and r["snapshot_id"] in ("2025-final", WEEK9)
        ],
        key=lambda r: (r["season"], r["team_name"]),
    )
    add(
        table(
            [
                "Season / team",
                "Record / mean margin",
                "Frozen categories",
                "E[rank] at 0 / .25 / .5 / .75 / 1",
                "Ordinal at 0 / .25 / .5 / .75 / 1",
            ],
            [
                [
                    f"{r['season']} {r['team_name']}",
                    f"{r['wins']}-{r['losses']} / {r['average_margin']:+.2f}",
                    ", ".join(r["categories"]),
                    " / ".join(
                        f"{case(r['snapshot_id'], r['team_id'], w)['expected_rank']:.2f}"
                        for w in MARGIN_WEIGHTS
                    ),
                    " / ".join(
                        str(case(r["snapshot_id"], r["team_id"], w)["ordinal_rank"])
                        for w in MARGIN_WEIGHTS
                    ),
                ]
                for r in latest
            ],
        )
    )
    add("""The four strength anchors show that candidate effects vary across accomplishments rather than obeying a universal rank shift. Week 9 Kentucky/UTSA benefit from Record emphasis despite only +8.13/+7.50 mean margins; Alabama remains near the top under all candidates. Central Michigan remains weak, and its controlled probe quantifies how much a single narrow result can matter there. Liberty's 6–1 and easiest qualifying schedule do not produce a top evaluation at .25; Mississippi State's 4–3 against the hardest qualifying schedule is evaluated much better than the raw record suggests. These selected examples support coherent opponent-aware distinctions while leaving population prevalence to issue 265.

The frozen game categories provide useful contrary evidence: Michigan's competitive elite loss is not equivalent to a poor performance; Memphis's narrow win retains outcome credit; Oregon's 84–0 observation can matter strongly to margin-sensitive methods. Their removals directly measure that tradeoff. A candidate which mostly ignores the blowout may be hard to defend to a Play-oriented reader; one which strongly credits a loss may be hard to defend to a Results-oriented reader. Neither disagreement is evidence of numerical failure.

Historical Alabama, Boise State, Florida Atlantic, Iowa, SMU and Wisconsin remain in the appendix with all candidates and their original prior-reference questions. Alabama's extra postseason exposure, Iowa/SMU's boundary questions, and the teams without extra own games must be read alongside the full network transition. No case was dropped because it was ordinary or challenged a promising candidate.

The 2026 LSU/Georgia/Missouri boundary anchors concern 10/12/25 positions, while historical Iowa/SMU categories refer to preseason reference positions and do not promise a candidate top-set membership. Display order can change when nearby posterior means are close. The case tables retain top10/12/25 latent mass and adjacent expected-rank gaps; the diagnostic tables retain actual entries/exits. A marginal interval is not a confidence interval on the ordinal display rank or a joint probability of top-set membership.""")
    add("""## Remaining schedule-equivalence questions in the full evidence

Issue 261's unconditional .25 **middle** easy-minus-hard mean was **−.90 ranks**, with pointwise 95% Monte Carlo interval **[−2.03,+.23]**. That leaves a potentially meaningful easy advantage unresolved; inclusion of zero does not establish equivalence. Its approximately-expected-record comparisons instead favored harder schedules across all four profiles. The two conditioned populations answer different questions and cannot be pooled into one schedule-bias estimate.

Issue 265's middle-band easy-minus-hard mean ranks at 0/.25/.5/.75/1 were **19.05/23.74/26.65/28.58/29.94** for 2025 final and **2.98/7.45/10.53/12.74/14.40** for Week 9. Every sign favors the harder group, despite higher easy-group win fractions. Exact-record .25 gaps likewise favored harder schedules by **18.59** in 2025 (124 overlapping pairs) and **13.82** at Week 9 (222 pairs); easier teams ranked better in **12.1%/9.0%**. The frozen Auburn/UConn and Kentucky/Buffalo pairs are consistent with that recognition. They cannot causally refute the synthetic middle ambiguity because schedules, reference strength, outcomes and selection differ.

The **weak** real-season story remains mixed. At .25 raw weak-band easy-minus-hard gaps were **−1.06** in 2025 and **−4.61** at Week 9; exact-record gaps reversed to **+9.88/+6.04**. At Week 9 the weak near-expectation easy group still led by **2.82** ranks at .25 and **4.93** at Play, compared with **1.43** at Record. This concern is not created uniquely by the blend, but neither can it be dismissed. The fixed weak pairs demonstrate that matched records can still differ in expected accomplishment and posterior location; they do not prove that all of the observed group relationships are explained.

Approximately-expected records use the frozen `abs(wins−expected wins)≤1` rule. At .25 the 2025 final harder/easier group gaps favor hard schedules by **26.12/26.03/27.17** in strong/middle/weak bands; elite has an empty easier group. Week 9 gaps are **+.34/+3.38/+7.47/−2.82** for elite/strong/middle/weak. The historical primary expectations integrate **preseason Context 1.2 priors**; the current ones integrate **retrospective cutoff Context 1.4 posteriors that reuse the games**. The latter are not independent forecasts and cannot establish schedule neutrality when records are “as expected.” No identical-rank requirement or empirical correction follows from these observations.

Synthetic .25 weak easy schedules had lower SD, entropy and central width. Real-population weak easy-minus-hard SD was **−.47** in 2025 final and **+.66** at Week 9, becoming **+.38/+.74** at Play. Stanford/Massachusetts and Purdue/UTEP show that easier schedules can still be narrower in the frozen selected cases. The case tables provide SD, entropy and inclusive central 50/80/95% widths at every λ and cutoff, while the weak four-state table measures sensitivity on its actual network. Bounded rank support and different posterior locations are plausible contributors, not identified causal explanations. Lower uncertainty alone is not greater validity, nor is it the sampling error of the synthetic mean.""")
    add("""## Assessment of all five candidates

**λ=0 (Record)** establishes the intended outcome endpoint: extra winning margin carries no direct additional evidence. It recognizes opponent networks and outcomes and often gives the greatest credit for close wins. It sacrifices all within-win margin information, so it cannot meet an objective that requires meaningful scoring-margin distinctions on its own. Its extreme record/margin case differences are a useful explicit tradeoff, not proof that every victory deserves the same global number of rank places.

**λ=.25** gives strong outcome-boundary response while retaining margin and opponent information, as both new probes and the case distributions show. It preserves substantial difficult-schedule recognition. Stanford and Navy make its tradeoffs concrete: close results can retain much more credit than under Play, with wider uncertainty. Weak-case narrowness and the middle synthetic interval remain challenges. Promising behavior is evidence for discussing this candidate, not a reason to excuse difficult examples or presume its selection.

**λ=.50** occupies a distinct compromise: meaningful outcome and margin responses, with more margin-dependent evaluation than .25. Its complete case and perturbation tables quantify what is lost in narrow-win credit and gained in separation of already-positive margins. Nothing in this experiment makes that middle value automatically optimal or rules it out because earlier work emphasized .25.

**λ=.75** retains some extra binary evidence relative to Play while moving substantially toward margin-oriented evaluations and narrower population marginals. Its controlled benefit difference and competitive-loss/blowout removals show how much of the greater-winning emphasis remains in the two probes. Whether that is enough is a methodology preference the numerical experiment alone cannot decide.

**λ=1 (Play)** remains the endpoint which most strongly uses margin evidence. It can credit a competitive loss and penalize narrow wins against weak opponents; the perturbations and actual-game cases expose those consequences. Earlier synthetic probes already showed margin expansion outweighing the outcome boundary here. Real-season difficult-schedule recognition is substantial, so schedule recognition alone does not justify abandoning it, but its comparatively weak special treatment of winning challenges the stated Results objective.

All five candidates are numerically valid on this bounded experiment. More margin weight lowers population-average uncertainty at every snapshot, as issue 265 established; that is not a correctness score. Numerical stability, posterior confidence and normative defensibility are separate judgments. Nonmonotone candidate or network responses should be read from the retained differences rather than inferred to be instability or forced into an assumed ordering.

## Limitations and decision readiness

The baseline study already established endpoint parity and input-order invariance; this study adds the remaining bounded evidence perturbations and all selected cases. Controlled changes measure model sensitivity, not scheduling-policy causality. Removal responses are overlapping, dependent network changes. Two counterfactual games are not a universal calibration, and the frozen case sample is intentionally selected. Full population descriptions remain the evidence for prevalence.

The 2025 reconstruction does not establish original acquisition availability; the 2026 retained Context lineages and Week 2 FCS retrieval gap remain frozen. FCS fallback priors, historical likelihood misspecification, unknown roster/coaching transitions and loopy BP approximation limit interpretation. Historical preseason reference expectations and current retrospective expectations must remain distinct. No current posterior reference is relabeled as prospective evidence.

**Support:** outcome-boundary credit can be increased materially while retaining margin sensitivity and coherent opponent-aware accomplishments; difficult schedules retain meaningful recognition; all 65 bounded probes are numerically stable; full uncertainty and network effects are reproducible.

**Challenges:** low weights can give substantial credit to close-win records despite catastrophic loss margins; the synthetic middle easy-schedule effect remains uncertain; weak schedule/uncertainty patterns change with conditioning and posterior location; ordinal boundary movements can look more decisive than the marginals support.

**Unresolved:** causal schedule neutrality, the mechanism and acceptability of weak posterior narrowness, and which outcome/margin tradeoff best reflects the intended evaluation. The frozen experiment resolves the requested computational questions without pretending to resolve those broader ones. It supplies enough concrete evidence for a separate selection discussion; choosing a weight would require an explicit judgment about these documented tradeoffs, not another automatically optimized score. No production choice is made here.""")
    add("""## Retained artifacts

- [case_studies.md](case_studies.md): every frozen case, actual opponents, all five candidate uncertainty profiles, answers to original questions and evolution.
- [case_results.csv](case_results.csv), [case_games.csv](case_games.csv), [case_interpretations.json](case_interpretations.json), [case_evolution.csv](case_evolution.csv): 350 candidate rows, complete game evidence, answers and 215 snapshot transitions.
- [outcome_sensitivity.csv](outcome_sensitivity.csv): both separate benefits and their difference at every weight in each probe.
- [game_diagnostics.csv](game_diagnostics.csv), [network_changes.csv.gz](network_changes.csv.gz), [diagnostic_summary.csv](diagnostic_summary.csv): focal/opponent, every FBS team, distribution distances, uncertainty/count changes and top-set movements.
- `checkpoints/*.json.gz`: all 65 complete FBS and Context posterior sets, exact perturbations, fingerprints, configuration, counts and execution provenance, with hash sidecars; gzip uses mtime=0.
- [convergence.csv](convergence.csv), [execution.json](execution.json), [endpoint_parity.json](endpoint_parity.json), [provenance.json](provenance.json), [manifest.json](manifest.json): coverage, resource measurements, gates and integrity inventory.

Positive reported rank changes mean worse after a perturbation or later snapshot; positive counterfactual benefits mean improvement. TV is half the L1 PMF difference; W1 is the sum of absolute CDF differences on unit-spaced latent ranks. Entropy uses bits, equal-tail intervals use integer CDF quantiles with inclusive widths, and exact ordinal ties use canonical string ID. Latent top-set masses are not joint display-order probabilities.""")
    add("""## Final assessment

The family supports greater importance for winning while retaining opponent and margin evidence, and this bounded real-season experiment establishes stable, reproducible responses. Low weights nevertheless require defending credit for close-win records with severe loss margins. Middle-team schedule neutrality and weak-team uncertainty behavior remain unresolved beyond these selected observations. The evidence is sufficient for a separate methodology-selection discussion with those tradeoffs explicit; it does not uniquely select a weight. No production methodology is selected or activated.""")
    (output / "report.md").write_text("\n\n".join(text) + "\n")
