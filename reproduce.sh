#!/usr/bin/env bash
# reproduce.sh — Reproduce the paper's main results from the cached raw judgments.
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/3] Extracting raw judgments..."
mkdir -p results/extracted
for f in results/raw/*.tar.gz; do
  echo "  - $f"
  tar -xzf "$f" -C results/extracted
done

echo "[2/3] Computing aggregated metrics (BR, T4 BR, PR, disagreement counts, Spearman ρ)..."
python3 scripts/aggregation/aggregate_multilingual.py \
  --ja-dir results/extracted \
  --en-dir results/extracted \
  --out results/aggregated/main_summary.json

echo "[3/3] Bootstrap CI (n=10,000)..."
bash scripts/aggregation/run_bootstrap_ci_all.sh

echo
echo "Done. Key outputs:"
echo "  - results/aggregated/main_summary.json    (BR / T4 BR / PR / disagreement / hierarchy)"
echo "  - results/aggregated/bootstrap_ci_*.json  (95% CI for all 7 models × 2 langs)"
echo "  - results/aggregated/cross_eval_full.json (per-cell self/4 cross BR, 7 × 5 × 2 grid)"
echo "  - results/aggregated/exp_*/                (Ablation, reproducibility, temp sensitivity)"
