# CogBias-MT Changelog

## v1.5 thesis additional experiments (2026-10-20)
- New re-judging records under `results/thesis/thesis_exp/thesis_exp_records.tar.gz` (same layout as `results/raw`): `rerun1/` (all 560 dialogues × 2 turns re-judged with the v1.1 rubric; JA by all 4 panel judges, EN by Gemini 3.1 and LLaMA 4), `repeat_run1/`, `repeat_run2/` (10 boundary problems × 4 subjects × 2 turns, Opus 4.7 / Gemini 3.1 / LLaMA 4, two runs each), `v12_outcome/` (JA post-counter turn judged with rubric v1.2 outcome criterion by GPT-5.4 / Gemini 3.1 / LLaMA 4). Gemini records whose JSON was truncated at 2,048 output tokens carry `turnN_repaired`; the rest were re-judged with a larger output budget.
- Headline results (`analysis_JA.json`, `analysis_EN.json`, `v12_analysis.json`): GPT-5.4 / Opus 4.7 / Gemini 3.1 re-runs agree with the original records in 90–98% of cells (κ 0.72–0.97), strictness within 1–4pp, dominant side and residual sign unchanged for all 6 pairs; LLaMA 4 (JA, post-counter) moved from 17.0% to 40.0% in `rerun1` but agreed 91.9% with the original in the same-day `v12_outcome` run. Rubric v1.2 lowers every judge's rate by 3–8pp but leaves judge spread (64.6 → 66.0pp), per-subject range and pairwise κ unchanged; the 20-case exploratory agreement does not extend to the full set.
- `qc_rewritten_t3_{ja,en}.json`: GPT-5.4 quality review of the current (explicit-T3) benchmark; 80/80 pass in both languages, Q2 (subtlety) low by design.
- Scripts: `thesis_rejudge.py`, `thesis_rejudge_repair.py`, `thesis_rerun_analysis.py`, `thesis_v12_analysis.py`.

## v1.4 (2026-10-18)
- Added the thesis analysis scripts (`scripts/aggregation/thesis_*.py`) and their outputs (`results/thesis/*.json`): latent-class estimation of judge sensitivity / false-positive rate without human labels, assumption checks (LOJO, goodness-of-fit, BIC, local-dependence model, error correlation / effective judge count), confidence-threshold correction, train/test comparison of correction methods M0–M6, and JA−EN gap decomposition. No change to the benchmark data or to the paper's numbers.

## v0.2 (2026-06-12)
- **Added missing raw judgments for the Claude Opus 4.7 evaluator** to
  `results/raw/ja_cross_eval_n80.tar.gz` (`cross_eval_claude_opus_n80/`, 7 subjects × 80)
  and `results/raw/en_cross_eval_n80.tar.gz` (`cross_eval_claude_opus_en/`, 7 subjects × 80).
  These directories back the main disagreement results (Table 2 / Figure 2 / 4-axis
  hierarchy) and were omitted from the initial packaging.
- Packaging-level cleanup, disclosed for transparency: stale `error` fields left over
  from automatically retried API calls were removed from the added files (716 records
  across the two archives: 354 JA + 362 EN). Judgment fields are byte-for-byte
  unchanged; a record is a completed evaluation iff `turn2_bias_confidence` /
  `turn4_bias_confidence` is non-null. The pre-cleanup originals are retained by the
  authors; the previously published archives remain in this repository's history.

## v1.0 (camera-ready, planned)
- Initial public release.
- **7 subjects × 5 evaluators × 2 languages × 80 problems = 5,600 judgments**
  (4 frontier subjects double as cross-evaluators; self-pairs excluded from cross-eval aggregation).
- Subjects: GPT-5.4, Claude Opus 4.7, Gemini 3.1 Pro, LLaMA 4 Maverick (frontier);
  Qwen3.5-27B, Qwen3.5-122B-A10B, Gemma-4-31B-it (open weights, GGUF Q4_K_M, llama.cpp local).
- Evaluator pairs: 6 pairs × 2 languages = **12 disagreement comparisons**;
  11 / 12 show one-directional skew surviving Bonferroni / BH-FDR multi-comparison correction
  (sole exception: JA Opus–Gemini, *p* = 0.16).
- Bootstrap 95% CI (n = 10,000) released for self-eval BR (7 models × 2 languages),
  cross-eval marginal BR (4 evaluators × 2 languages),
  T4 BR, and PR.

## Internal milestones (pre-release)
- Phase 5 v2 (2026-05-08): 4-subject (GPT-5.4 / Claude 4.6 / Gemini 3.1 Pro / LLaMA 4 Maverick) × n=80 cross-eval.
- Phase 6 (2026-04-29 / 2026-05-08): added Claude Opus 4.7 / GPT-5.5 as additional subjects; self-eval clear CE n=80.
- Phase 7 (2026-05-13 to 16): multilingual (JA + EN) + open weights (Qwen3.5-27B/122B, Gemma-4-31B) on DGX A100 / llama.cpp.
- Phase 8 (2026-05-18): LLaMA 4 Maverick added as 4th cross-evaluator → 12-pair × 2-lang disagreement matrix.
- v3 → v4 (2026-05): subject set reduced to 7 (one model per vendor; GPT-5.5 and Claude 4.6 dropped as subjects);
  Claude evaluator unified to Opus 4.7. Final cell count 7 × 5 × 2 × 80 = 5,600.

## v1.2 rubric (2026-09-04, camera-ready revision)
- Added `data/{jp,en}_hard_80_v1.2_{outcome,process}.jsonl`: identical problems to v1.1, with the T4 criteria extended by an explicit operational definition of "corrected" (`evaluation_criteria.turn4.corrected_definition` = `outcome` | `process`, plus the definition text appended to `debiased`). The wording follows the rubrics used in the 20-case probe (`scripts/aggregation/exp_forced_rubric.py`).
- All results in the paper use the v1.1 rubric (`data/jp_hard_80.jsonl`, `data/en_hard_80.jsonl`); v1.2 is provided for future evaluations that want to fix the definition in advance.

## v1.3 records (2026-09-06, camera-ready revision)
- **Evaluation records updated.** While preparing the camera-ready version we found three defects in the v1.1 record set — (1) 97 subject trajectories with an empty T2/T4 response, (2) a parser default of "not biased" (confidence 0.5, justification 0.5) recorded for self-evaluation calls whose verdict JSON could not be parsed (almost all Qwen3.5 self-evaluations plus 36 other cells), and (3) the same parser default in 302 panel-judge cells. All three biased the records toward "not biased". We re-ran the affected cells with the same models, prompts, and temperatures (2026-09-04/05) and recomputed every number in the paper and in `SUPPLEMENT.md` from the resulting record set. Details in SUPPLEMENT S8b. In brief: 97 subject trajectories whose stored T2/T4 response was empty (Qwen3.5-27B/122B reasoning mode exhausting the 4,096-token budget; four Gemini calls) were regenerated with a 16,384-token budget; the Qwen3.5 dedicated self-evaluations were re-run for all trajectories (eval max_tokens 8,192) and 36 further self-evaluation cells were re-run; 302 panel-judge cells that held a parse-failure default were re-judged, and every judge line of the regenerated trajectories was judged afresh. The Gemini re-run calls used `google/gemini-3.1-pro-preview` (the dated slug was withdrawn). Every re-run record carries a `regenerated`, `self_reevaluated`, or `rejudged`/`rejudge_note` field.
- `results/raw/*.tar.gz` replaced with the updated records (same directory layout and file names). `results/rebuttal_checks/*.json` regenerated (`twoway_bh.json` / `twoway_bonferroni.json` are now written directly by `rebuttal_twoway_cluster.py`); `wiiat_numbers.json` added (output of `scripts/aggregation/wiiat_numbers.py`, which recomputes every value cited in the paper).
- New `results/camera_ready/`: `exp_b_claude_rubric_gpt_judge/` (Claude-authored rubric × GPT-5.4 judge, JA, 7 subjects), `exp_b_claude_rubric_opus_judge/` (Claude-authored rubric × Opus 4.7 judge, JA, 7 subjects), `exp_c_qwen27b_q8/` (Qwen3.5-27B regenerated at Q8_0, JA+EN, self + 4 judges), `regenerated_trajectories/` (the 97 regenerated trajectories and the re-run self-evaluation files).
- Supplement: S8a (build/hashes), S8b (re-run inventory), S22 (generator-version split), S23 (temperature sensitivity), S24 (reciprocal rubric test), S25 (quantization check) added; S1–S5, S9, S11–S14, S17–S21 refreshed; S4 notes that the mixed-effects model is degenerate on the updated records; S12 withdrawn.
- Headline values now: judge strictness 80.4 / 63.5 / 63.1 / 47.7 (JA); GPT–Opus JA disagreement 308/308 GPT-side; residual 5–15pp; two-way CI > 0 in 12/12; permutation BH 11/12, Bonferroni 10/12.
