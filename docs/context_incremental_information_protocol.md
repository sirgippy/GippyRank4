# Context incremental-information benchmark protocol

## Question and scope

Measure whether repaired production Context 1.4 adds predictive information about the final FBS rank distribution after rolling History 1.1 and cutoff-safe games are available. This is an exploratory development benchmark for seasons 2022–2025. Those seasons have already informed the research sequence, so held-out folds are not an untouched confirmation set. The benchmark does not authorize a production change.

The seven checkpoints are the retained actual-date weekly cutoffs in `data/processed/posterior_backtest/{season}_rolling.json`. Checkpoints 1 (preseason), 2 (September), and 7 (December) are reproduced first. Only after their evidence, baseline-score, and rounded pilot-score gates pass are checkpoints 3–6 run as bounded exploratory diagnostics.

## Priors, outcomes, and inputs

- Reconstruct the 528 fitted Context priors from `context_v1_4_candidate/development_model_inputs.json`, `context_db_repair_172/historical_models/`, and `historical_repaired_features` plus `production_pmf`. Replace the obsolete transfer-impact fields with the repaired feature boundary before calling `production_pmf`.
- Preserve all six retained native cold starts. Do not replace them with another History or Context artifact. Keep their raw Context covariates missing while allowing them to contribute PMFs and scores.
- Read History priors from arm `HH` in `context_history_crossover/hybrid_prior_results.csv`. Validate the serialized PMF hash before normalizing as the production `Team` constructor does.
- Use the fixed V1 likelihood at `data/processed/posterior/historical_likelihood_v1.json` and the retained cutoff-safe game populations. Exclude games whose final-score availability is within 48 hours of a cutoff, and use the retained FBS/FCS fallback rules.
- Verify the repair, retained-evidence, cutoff, likelihood, raw-game, and target-corpus hashes. Record included game IDs and canonical game-row hashes for every season/checkpoint.
- Inference accepts no outcome targets. The target corpus is hash-checked and loaded only after the checkpoint 1/2/7 target-free inference panel has been built. Persist the resulting target PMFs in a separate scoring file.

Inference is identical for History and Context: `max_iterations=500`, `tolerance=1e-9`, `damping=0.35`. Fail on population/evidence mismatch or nonconvergence. Normalize serialized PMFs as production does. The retained study contains 534 team-seasons across four seasons: 528 fitted feature rows and six native cold starts; 403 team-seasons are outer evaluations in 2023–2025.

## Frozen-message sensitivity arms

For each fitted team, extract the model-specific log message as `log(posterior) - log(own_prior)` on shared positive support. Normalize products with stable log-sum-exp arithmetic. Compute these arms:

| Arm | Own-team prior | Frozen message |
|---|---|---|
| HH | History | History |
| CH | Context | History |
| HC | History | Context |
| CC | Context | Context |

Require HH and CC to reconstruct their original posteriors within `1e-10`. Do not add epsilon or smooth zero support. A posterior with mass outside its own prior support is an error. These are frozen-message sensitivity analyses, not new converged joint posteriors or causal opponent attributions.

## Residual probes

At each checkpoint, evaluate outer folds separately: train on 2022/evaluate on 2023; train on 2022–2023/evaluate on 2024; train on 2022–2024/evaluate on 2025. No evaluated-season target, preprocessing statistic, or model-selection result enters that fold's fit. Every checkpoint is reported independently; do not pool repeated checkpoints as independent team-seasons.

For rank `r` among `N`, let `z(r)=logit((r-0.5)/N)`, `qH` be the History posterior and `D(r)=log(pC(r))-log(pH(r))` from Context and History priors. Center `log(qH)` and D across supported rank bins. Fit:

```text
q_probe(r) ∝ exp(log(qH(r)) + a * centered_log_qH(r)
                 + z(r) * [b0 + b'X] + optional w * centered_D(r))
```

The History calibration control uses `mean_lag1_z`, `lag2_z_mean`, `lag3_z_mean`, `long_run_z_mean`, History posterior expected-rank percentile, and History posterior rank-SD/population. The existing-signal extension adds only D. The raw-feature extension adds all `MODEL_FEATURES - H_FEATURES` together, including repaired DB sum and coverage, and does not include D.

Fit preprocessing on training rows only: median imputation, then training means and population standard deviations, a scale floor of `1e-6`, and missingness indicators. Missing raw Context covariates remain missing for native cold starts. Minimize mean target-PMF cross-entropy plus `0.5 * lambda * ||coefficients||^2`, including the intercept. Bound `a` to `[-0.75, 3]`; other coefficients are unbounded. Start at zero and use L-BFGS-B with analytic gradients, `maxiter=2000`, `ftol=1e-13`, and `gtol=1e-8`.

Choose lambda from `[0.01, 0.1, 1]`. One training season uses `0.1`. Otherwise use forward-chained inner validation over earlier seasons only, weighting each validation team-season equally; ties within `1e-12` choose the stronger regularization. Record fold membership, all candidate losses, preprocessing, optimizer details, and coefficients.

## Evidence and scoring

For every team/checkpoint, store full Context and History prior/posterior PMFs, the four crossed PMFs, usable prediction-time features, games played, and exact `KL(History posterior || History prior)`. KL uses only positive posterior support; positive posterior mass outside prior support fails. Do not smooth.

Store per-team and aggregate NLL, CRPS, expected-rank error, central 80% interval width, and target mass for the interval. The retained `study_context_history_crossover.score_pmf` contract floors forecast probabilities at `1e-15` for reported NLL; nested lambda selection uses exact target cross-entropy on supported ranks with no smoothing. Report by checkpoint, season, and fitted/cold-start status. Compare each Context addition to the same History calibration control. Describe each checkpoint's result as positive, negative, or mixed/inconclusive based on the three outer-season deltas; this is descriptive and does not establish a production decision.

## Parity gates and provenance

Before checkpoints 3–6, reproduce retained rolling History 1.1 scores for checkpoints 1, 2, and 7 within `1e-8`, verify all repaired Context 1.4 prior PMFs against the retained `historical_priors` within `1e-12`, and reproduce the rounded pilot mean-NLL table within `1e-5`. Also reproduce December crossed-arm mean NLLs HH 3.83089, CH 3.85330, HC 3.85739, CC 3.88793 and September existing-signal deltas for 2023/2024/2025 of -0.01218/-0.00631/-0.00493 within `1e-5`. The retained crossover `CC` arm and posterior score table are Context 1.3, as confirmed by the crossover builder's `CONTEXT_1_3_FEATURES` path; they are recorded as a legacy comparison and are not a valid exact score target for repaired Context 1.4. Do not change the protocol to force parity. Stop and investigate any failed gate before extension.

The exploratory pilot protocol SHA-256 was `327d2d301d59668898dff89046e617ddfc5f0d36dd42e282d2fecd71b0d67101`; it is provenance only. This checked-in protocol and the output manifest define the reproducible implementation for this benchmark.

## CLI

Run from the repository root with explicit external input roots:

```bash
uv run python scripts/build_context_incremental_information.py \
  --input-root /path/to/repair-input-root \
  --targets /path/to/team_season_rank_distributions.csv \
  --raw-games /path/to/cfbd/games
```

Use `--pilot-only` to stop after the parity-gated 1/2/7 reproduction. The default runs all seven checkpoints and writes the target-free inference panel, separate scoring targets, portable scores, evidence, report, and SHA-256 manifest under `data/processed/context_incremental_information/`.
