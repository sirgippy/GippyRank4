# Issue 178: clipped-scale gradient correction

## Root cause and fix

`DirectRankModel.fit` uses the scale link `minimum_scale + exp(clip(eta, -5, 4))`. The analytic gamma gradient differentiated `exp(eta)` but omitted the derivative of `clip`. The scale contribution must be multiplied by an indicator that is one for `-5 < eta < 4` and zero outside those bounds. The corrected gradient uses that chain rule; at either kink the derivative is not unique.

Permanent finite-difference tests cover predictors inside the bounds, below the lower bound, above the upper bound, and a clipped/unclipped mixture, for both Normal and Student-t fits. They keep every test point away from the clipping kinks.

## Hosted validation

The corrected no-restart matrix ran on five `ubuntu-24.04` VMs in [run 37556973497](https://github.com/sirgippy/GippyRank4/actions/runs/37556973497). All five VMs succeeded in both default and single-thread modes. No `ABNORMAL` results occurred, and no restart or fallback ran. Default and single-thread artifacts were identical within each VM.

Three VMs returned one byte-identical result and two returned another. The two outcomes differ only slightly: at most `9.6e-6` in beta and `2.1e-5` in gamma, with a maximum per-team PMF L1 difference of `3.3e-6` on this four-team fixture. The two signatures aligned with CPU groups in this small sample: runners 1 and 5 used AMD EPYC 9V45, while runners 2–4 used AMD EPYC 7763. All used the same Ubuntu image, Python 3.13.15, NumPy 2.5.2, and SciPy 1.18.1. This does not establish that CPU dispatch caused the small drift; all default/single-thread pairs remained identical, and the two fitted rows sat within `1e-10` of the lower clipping kink. The smooth-probe finite-difference checks retained clipped and unclipped rows while staying at least `0.005` from either kink; their maximum gradient error was `8.8e-11`. The raw central check at the fitted point reports about `0.00665` error because its stencil crosses those near-kink rows. The unclipped transition fit's raw finite-difference error remained below `1e-10`.

The corrected main-fit objective was `0.8155625999` on three VMs and `0.8155626023` on two. The prior ordinary-successful fit had objective `0.8393827703`; the prior manually restarted fit had `0.8391699592`. The corrected main coefficients changed materially: maximum absolute beta/gamma changes were `0.1418`/`1.1303` versus the ordinary fit and `0.1319`/`1.1340` versus the restarted fit. The transition coefficients and objective were unchanged to floating-point precision.

The corrected model, prediction, and PMF-map hashes are:

- Three-VM outcome: model `e2d16b5465bab34de6917d178e9f6a90110c8cdf703a0ebf7b86aee2a5694675`; predictions `9d64fc351ac75b79873c10d05ddc6eb490e4211470ccf4051fa7fa01a2948046`; PMF map `8ce0c966db1beca51f202d6669564d6b0f9ce253d91f57aed8022e5a364a6908`.
- Two-VM outcome: model `b2f41dcea89042601279fa0139931fde70f534842ba235181c1e80b6f34c04b7`; predictions `075f67693f97526d66bd2a016d452f7886109058f491204143a6c26b04d80c30`; PMF map `90409d0453fbc28b9f5a219f024624669a900e5f89717a757900bd03f42aed2e`.

Both corrected outcomes differ from the prior ordinary-successful hashes (model `b46ace…`, predictions `4f1e51…`, PMF `185f8b…`) and restarted hashes (model `125c32…`, predictions `17a8a4…`, PMF `2e2265…`). On the fixture, the corrected PMFs differ from the ordinary fit by at most `0.0264` in a rank probability and `0.0528` in per-team L1 distance. These are intentionally not treated as parity targets because the old fit followed the wrong gradient.

## Retained model impact

- **History 1.1, 2026:** 2,744 main-fit rows and 17 transition-fit rows; neither fit has clipped scale predictors. The corrected refits keep the recorded objectives (`1.6856135431` and `1.8417569863`) and coefficients unchanged to floating-point precision. Main-model PMFs differ from the retained CSV by at most `3e-12` in a rank probability, consistent with serialized-fit precision.
- **Context 1.3, 2026:** 2,744 training rows, zero clipping, and no coefficient or objective change (`1.6421973052`). The 136 modeled target rows reproduce the retained PMFs exactly; two cold-start rows use the History fallback.
- **Active Context 1.4, 2026:** its persisted scale design uses only the same three History features. The stored preprocessing values match Context 1.3 exactly. Applying the stored Context 1.4 coefficients to the shared 2,744 History rows and 136 modeled target rows finds zero clipped predictors (training eta `0.0102` to `0.3039`; target eta `0.0377` to `0.1980`). The repair-specific historical transfer panel needed for an exact Context 1.4 refit is absent from this local data checkout, so I did not claim a refit. There is no evidence that the retained 2026 Context output (`70a8e6c49f159eb91d5108acee0829f9194a51865198ee663c6af40037027a9a`) needs regeneration for this fix.

No current published 2026 model output was regenerated. Any future refit or promotion uses the corrected gradient and remains a separate model-release decision.

## Validation

- `uv run --no-sync pytest -q tests/test_preseason.py -k scale_clipping` — 8 passed.
- `uv run --no-sync pytest -q --ignore=tests/test_api.py` — 851 passed, following the repository's WSL sandbox guidance.
- `uv run --no-sync ruff check .` and formatting checks passed.
- Full hosted [CI run 37556973565](https://github.com/sirgippy/GippyRank4/actions/runs/37556973565) passed Python, site, and validation jobs.
