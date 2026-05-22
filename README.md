# CogBias-MT

**CogBias-MT** is a Japanese + English multi-turn cognitive-bias benchmark for evaluating LLM-as-a-Judge calibration. The dataset and the full 5,600-judgment cross-evaluation matrix accompany the paper *"LLM-as-a-Judge Disagreement is Systematically Directional, Not Random — Evidence from CogBias-MT"* (anonymous, under submission).

- **4 biases** (confirmation, anchoring, representativeness, framing) × **5 domains** (medical, legal, education, labor, daily-life) × **4 problems** = **80 hard problems per language** (JA + EN = 160).
- **7 subject models** (4 frontier + 3 open weights) × **5 evaluators** (self + 4 cross: GPT-5.4 / Claude Opus 4.7 / Gemini 3.1 Pro / LLaMA 4 Maverick) × **2 languages** = **5,600 judgments**.

## Key findings

- **Directional asymmetry** in 11 of 12 evaluator-pair × language combinations (binomial *p* surviving Bonferroni / BH; sole exception JA Opus–Gemini, *p* = 0.16). The largest pair (GPT-5.4 vs. Claude Opus 4.7) shows 92.4% / 95.7% (JA / EN) one-sided concentration.
- **Observed skew exceeds marginal predictions** from per-evaluator base rates by 8–19pp — strictness alone does not account for the magnitude.
- **Generalization**: BR ranking preserved across JA / EN at Spearman ρ = 0.857; GPT-5.4 ranks first in strictness on all four axes (frontier × open × JA × EN) with non-overlapping Bootstrap CIs.
- **Practical impact**: Swapping the evaluator set moves the average BiasRate of a single model by 15pp; "most robust frontier model" swings between Claude Opus 4.7 (5.0%, LLaMA-cross) and mid-tier (43.8%, GPT-cross).

The benchmark number is therefore a composite of *bias × judge-rubric calibration*, not bias alone. CogBias-MT is released as a diagnostic instrument that quantifies, for any bias benchmark, how much the result moves when the judge is swapped.

## Repository layout

| Path | Contents |
|:---|:---|
| `data/` | Benchmark JSONL files (`jp_hard_80.jsonl`, `en_hard_80.jsonl`, difficulty-curve subsets, Claude-regenerated rubric variant) |
| `scripts/generation/` | Problem-generation + 4-aspect quality check |
| `scripts/evaluation/` | OpenAI-compatible evaluator client (self + cross-eval) |
| `scripts/aggregation/` | Bootstrap CI / Spearman ρ / disagreement / ablation aggregators |
| `results/raw/*.tar.gz` | Full per-judgment dump (7 models × 5 evaluators × 2 languages = 5,600 cells) |
| `results/aggregated/` | Pre-computed Bootstrap CI + cross-eval-full + Exp E/I/temp summaries |
| `forgpu/` | DGX A100 Docker / llama.cpp pipeline for open weights |
| `figures/` | Paper figures (PNG + PDF) |
| `docs/paper_draft.md` | Full paper manuscript |

## Quickstart

```bash
# 1. Reproduce main results from cached raw data
cd results/raw && for f in *.tar.gz; do tar -xzf $f; done && cd ../..
python scripts/aggregation/aggregate_multilingual.py

# 2. Run a new evaluator on existing responses
OPENROUTER_API_KEY=... python scripts/evaluation/run_cross_eval.py \
  --results-dir results/raw/full_n80 \
  --benchmark data/jp_hard_80.jsonl \
  --evaluator <model-name> \
  --output-dir results/cross_eval_<name>

# 3. Add new problems
OPENAI_API_KEY=... python scripts/generation/multiturn_generator.py \
  --bias <type> --domain <domain> --difficulty hard
python scripts/generation/quality_check.py --input <new>.jsonl
```

## Models used

**Subjects (7) — accessed 2026-05-08 (JA) / 2026-05-13 (EN) for frontier; 2026-05-14 to 2026-05-16 for open weights**:

| Role | Short name | Model identifier |
|:---|:---|:---|
| Subject + evaluator (frontier) | GPT-5.4 | `openai/gpt-5.4` (OpenRouter) |
| Subject + evaluator (frontier) | Claude Opus 4.7 | `anthropic/claude-opus-4.7` (OpenRouter) |
| Subject + evaluator (frontier) | Gemini 3.1 Pro | `google/gemini-3.1-pro-preview-20260219` (OpenRouter) |
| Subject + evaluator (frontier) | LLaMA 4 Maverick | `meta-llama/llama-4-maverick` (OpenRouter; evaluator role added 2026-05-18 in Phase 8) |
| Subject (open weights) | Qwen3.5-27B | `Qwen/Qwen3.5-27B` (GGUF Q4_K_M, local llama.cpp) |
| Subject (open weights) | Qwen3.5-122B-A10B | `Qwen/Qwen3.5-122B-A10B` (GGUF Q4_K_M, local llama.cpp) |
| Subject (open weights) | Gemma-4-31B-it | `google/gemma-4-31B-it` (GGUF Q4_K_M, local llama.cpp) |

## Data record schema

```json
{
  "id": "framing-legal-hard-001",
  "bias_type": "framing",
  "domain": "legal",
  "difficulty": "hard",
  "harm_type": "distorted_decision",
  "harm_description": "...",
  "turns": [
    {"turn_number": 1, "role": "system", "content": "..."},
    {"turn_number": 2, "role": "system", "content": "...", "question": "..."},
    {"turn_number": 3, "role": "system", "content": "..."},
    {"turn_number": 4, "role": "system", "content": "...", "question": "..."}
  ],
  "bias_indicators": ["...", "..."],
  "debiased_indicators": ["...", "..."],
  "evaluation_criteria": {"turn2": {...}, "turn4": {...}}
}
```

## License

- Code (`scripts/`): MIT
- Data (`data/`, `results/`, `figures/`): CC-BY 4.0
- Model outputs in `results/raw/` are also subject to each API provider's terms of service.

## Citation

```bibtex
@misc{cogbias_mt_2026,
  title  = {LLM-as-a-Judge Disagreement is Systematically Directional, Not Random --- Evidence from CogBias-MT},
  author = {Anonymous},
  year   = {2026},
  note   = {Under submission}
}
```
