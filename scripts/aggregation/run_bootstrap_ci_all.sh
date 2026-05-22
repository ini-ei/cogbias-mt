#!/bin/bash
# run_bootstrap_ci_all.sh — JA/EN × frontier/open の 4 セットで Bootstrap CI 計算。
#
# 出力:
#   results/phase5_v2/bootstrap_ci_n80.json            (JA frontier)
#   results/phase5_v2/bootstrap_ci_n80_open.json       (JA open)
#   results/phase5_v2_en/bootstrap_ci_n80_en.json      (EN frontier)
#   results/phase5_v2_en/bootstrap_ci_n80_en_open.json (EN open)
#
# 使い方:
#   bash scripts/run_bootstrap_ci_all.sh

set -uo pipefail

FRONTIER_MODELS="gpt-5 gpt-5.5 claude-4-sonnet claude-opus-4.7 gemini-3.1-pro llama-4-maverick"
OPEN_MODELS="qwen3.5-27b gemma-4-31b qwen3.5-122b"
N_BOOT=10000

run() {
    local results_dir="$1"
    local output="$2"
    shift 2
    local models="$@"
    echo ""
    echo "=========================================="
    echo "Bootstrap CI: $results_dir"
    echo "  models: $models"
    echo "  output: $output"
    echo "=========================================="
    .venv/bin/python scripts/phase5_bootstrap_ci.py \
        --results-dir "$results_dir" \
        --models $models \
        --n-boot $N_BOOT \
        --output "$output"
}

# JA frontier
run "results/phase5_v2/full_n80" \
    "results/phase5_v2/bootstrap_ci_n80.json" \
    $FRONTIER_MODELS

# JA open
run "results/phase5_v2/full_n80_open" \
    "results/phase5_v2/bootstrap_ci_n80_open.json" \
    $OPEN_MODELS

# EN frontier
run "results/phase5_v2_en/full_n80_en" \
    "results/phase5_v2_en/bootstrap_ci_n80_en.json" \
    $FRONTIER_MODELS

# EN open
run "results/phase5_v2_en/full_n80_en_open" \
    "results/phase5_v2_en/bootstrap_ci_n80_en_open.json" \
    $OPEN_MODELS

echo ""
echo "=========================================="
echo "Bootstrap CI computation done."
echo "=========================================="
ls -la results/phase5_v2/bootstrap_ci_n80*.json results/phase5_v2_en/bootstrap_ci_n80*.json 2>/dev/null
