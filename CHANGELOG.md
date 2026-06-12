# CogBias-MT Changelog

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
