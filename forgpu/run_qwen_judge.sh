#!/usr/bin/env bash
# run_qwen_judge.sh — Qwen3.5-122B を 5 つ目の評価者として全 1,120 対話 × 2 ターンを判定する（コンテナ内で実行）。
#   docker run --rm --gpus '"device=1,2,3"' --ipc=host -e HF_HOME=/hf_cache -v $HF_CACHE_DIR:/hf_cache -v $(pwd):/work -w /work \
#       --name cb-qwen-judge cb-vllm:latest bash forgpu/run_qwen_judge.sh
# 1) llama-server を Qwen3.5-122B-A10B GGUF Q4_K_M で起動（port 8000、並列 4 スロット）
# 2) scripts/thesis_rejudge.py を --base-url http://localhost:8000/v1 で JA → EN の順に実行（出力 results/thesis_exp/qwen_judge/）
# 3) llama-server 停止
set -uo pipefail
PORT=8000
HF_REPO="unsloth/Qwen3.5-122B-A10B-GGUF:Q4_K_M"
LOGDIR=logs/thesis_qwen; mkdir -p "$LOGDIR"
pip install --no-cache-dir -q python-dotenv >/dev/null 2>&1 || true
echo "[serve] starting llama-server ($HF_REPO)"
# 思考モードは 1 判定 3,000 トークン超で 20 時間以上かかるため切る（--reasoning-budget 0 が無ければ chat-template-kwargs で無効化）
THINK_OPT="--reasoning off"
echo "[serve] thinking option: $THINK_OPT"
llama-server -hf "$HF_REPO" --host 0.0.0.0 --port "$PORT" --n-gpu-layers -1 --ctx-size 65536 --parallel 8 --no-warmup $THINK_OPT \
    > "$LOGDIR/serve_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
SERVE_PID=$!
elapsed=0
while [ $elapsed -lt 3600 ]; do
    curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "[wait] ready after ${elapsed}s"; break; }
    sleep 15; elapsed=$((elapsed + 15))
done
curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 || { echo "[wait] server not ready"; kill $SERVE_PID; exit 1; }
MODEL_NAME=$(curl -fsS "http://localhost:$PORT/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[serve] model id: $MODEL_NAME"
for LANG_ in JA EN; do
    echo "[judge] $LANG_ start $(date)"
    python3 scripts/thesis_rejudge.py --lang "$LANG_" --judge qwen3.5-122b --tag qwen_judge --base-url "http://localhost:$PORT/v1" \
        --slug "$MODEL_NAME" --workers 8 --max-tokens 2048 2>&1 | tee -a "$LOGDIR/judge_${LANG_}.log" | grep -v "cost so far" | tail -3
    echo "[judge] $LANG_ done $(date)"
done
kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null
echo "[done] $(date)"
