#!/bin/bash
# run_openmodels_pipeline.sh (llama.cpp版) — GGUF モデルを llama-server で serve して評価。
#
# 動作:
#   各モデルごとに:
#     1) llama-server -hf <repo>:<quant> をバックグラウンド起動 (port 8000)
#     2) /health まで待機
#     3) OPENAI_BASE_URL=http://localhost:8000/v1 でクライアント切替
#     4) phase5_benchmark_validation.py を JA/EN で実行
#     5) llama-server 停止

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOGDIR="${PIPELINE_LOGDIR:-logs/phase7_open}"
mkdir -p "$LOGDIR"

# GGUF モデル (ggml-org / unsloth から)
MODELS="${MODELS:-qwen3.5-27b gemma-4-31b qwen3.5-122b}"
LANGS="${LANGS:-ja en}"
PORT="${PORT:-8000}"

# モデル簡易名 → HF GGUF repo (latest stable quant)
# Q4_K_M は質と速度のバランス点。重い場合は Q3_K_M に下げる選択肢あり。
get_hf_repo() {
    case "$1" in
        qwen3.5-27b)   echo "unsloth/Qwen3.5-27B-GGUF:Q4_K_M" ;;
        qwen3.5-122b)  echo "unsloth/Qwen3.5-122B-A10B-GGUF:Q4_K_M" ;;
        qwen3.5-35b)   echo "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" ;;
        gemma-4-31b)   echo "ggml-org/gemma-4-31B-it-GGUF:Q4_K_M" ;;
        gemma-4-26b)   echo "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M" ;;
        *) echo "" ;;
    esac
}

# モデルサイズに応じた --n-gpu-layers (-ngl)。-1 = 全レイヤ GPU。
# A100 80GB なら 27B/31B は 1 GPU でも乗るが、layer split で複数 GPU 並列化
# (tensor-split 自動配分)。
get_ngl() { echo -1; }

# context 長
get_ctx_size() { echo 16384; }

get_bench() {
    case "$1" in
        ja) echo "results/phase5_v2/hard_all_80_clear_ce.jsonl" ;;
        en) echo "results/phase5_v2_en/hard_all_80_en_clear_ce.jsonl" ;;
        *) echo "" ;;
    esac
}

get_outdir() {
    case "$1" in
        ja) echo "results/phase5_v2/full_n80_open" ;;
        en) echo "results/phase5_v2_en/full_n80_en_open" ;;
        *) echo "" ;;
    esac
}

wait_server_ready() {
    local port="$1"
    local timeout="${2:-1800}"  # 30 分 (重いモデル DL + load)
    local elapsed=0
    echo "[wait] /health on :$port (timeout ${timeout}s)"
    while [ $elapsed -lt "$timeout" ]; do
        if curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "[wait] llama-server ready after ${elapsed}s"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "[wait] TIMEOUT after ${timeout}s"
    return 1
}

echo "=========================================="
echo "Phase 7 open-model pipeline (llama.cpp / GGUF)"
echo "  MODELS : $MODELS"
echo "  LANGS  : $LANGS"
echo "  PORT   : $PORT"
echo "=========================================="

for MODEL in $MODELS; do
    HF_REPO="$(get_hf_repo "$MODEL")"
    NGL="$(get_ngl "$MODEL")"
    CTX="$(get_ctx_size "$MODEL")"

    if [ -z "$HF_REPO" ]; then
        echo "[skip] unknown model: $MODEL"
        continue
    fi

    echo ""
    echo "##########################################"
    echo "## MODEL: $MODEL ($HF_REPO)"
    echo "##########################################"

    # llama-server バックグラウンド起動
    # --tensor-split 0,1,1,1,1,1,1,1 のように指定すると GPU 0 をスキップできる。
    # ここではデフォルト (全 GPU 均等分散) を使用。
    echo "[serve] starting llama-server..."
    llama-server \
        -hf "$HF_REPO" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --n-gpu-layers "$NGL" \
        --ctx-size "$CTX" \
        --no-warmup \
        > "$LOGDIR/serve_${MODEL}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    SERVE_PID=$!
    echo "[serve] pid=$SERVE_PID"

    if ! wait_server_ready "$PORT" 1800; then
        echo "[error] server not ready, killing pid=$SERVE_PID"
        kill "$SERVE_PID" 2>/dev/null || true
        wait "$SERVE_PID" 2>/dev/null || true
        continue
    fi

    # JA / EN 評価
    for LANG in $LANGS; do
        BENCH="$(get_bench "$LANG")"
        OUTDIR="$(get_outdir "$LANG")"

        if [ ! -f "$BENCH" ]; then
            echo "[skip] $MODEL/$LANG: benchmark not found"
            continue
        fi
        mkdir -p "$OUTDIR"

        # resume: 既に 80 items あればスキップ
        if [ -f "$OUTDIR/${MODEL}_results.json" ]; then
            NLINES=$(python3 -c "import json; print(len(json.load(open('$OUTDIR/${MODEL}_results.json'))))" 2>/dev/null || echo 0)
            if [ "$NLINES" -ge 80 ]; then
                echo "[skip] $MODEL × $LANG: existing $NLINES items"
                continue
            fi
        fi

        echo ""
        echo "------------------------------------------"
        echo "[eval] $MODEL × $LANG → $OUTDIR/${MODEL}_results.json"
        echo "------------------------------------------"

        OPENAI_BASE_URL="http://localhost:${PORT}/v1" \
        OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-dummy}" \
            python3 scripts/phase5_benchmark_validation.py \
                --benchmark "$BENCH" \
                --models "$MODEL" \
                --output-dir "$OUTDIR" \
            || echo "[warn] eval failed for $MODEL/$LANG"
    done

    # stop server
    echo "[serve] stopping pid=$SERVE_PID"
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
    sleep 10  # GPU mem 解放待ち
done

echo ""
echo "=========================================="
echo "Pipeline done."
echo "=========================================="
