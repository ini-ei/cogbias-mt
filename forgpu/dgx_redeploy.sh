#!/bin/bash
# dgx_redeploy.sh — DGX 側で再展開 + 実行をまとめる。
#
# 前提:
#   1) ~/cb_phase7_open_bundle.tar.gz が転送済 (ローカルから scp 済)
#   2) ~/.hf_token に HF_TOKEN を保存済 (echo "$HF_TOKEN" > ~/.hf_token && chmod 600 ~/.hf_token)
#
# 動作:
#   - 旧 docker image (cb-vllm:latest) 削除
#   - 旧 logs/phase7_open をタイムスタンプ付きで退避
#   - bundle 展開
#   - 環境変数 set
#   - nohup で fire-and-forget 起動
#
# 使い方:
#   ssh muds-dgxa100-nakamura "bash ~/dgx_redeploy.sh"

set -uo pipefail

ROOT="/raid/data/nakamura-lab/Inui"
BUNDLE="$HOME/cb_phase7_open_bundle.tar.gz"
TOKEN_FILE="$HOME/.hf_token"

if [ ! -f "$BUNDLE" ]; then
    echo "ERROR: bundle not found: $BUNDLE"
    echo "  scp it from local first: scp cb_phase7_open_bundle.tar.gz muds-dgxa100-nakamura:~/"
    exit 1
fi
if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: HF token file not found: $TOKEN_FILE"
    echo "  Save it once: read -s HF_TOKEN && echo \"\$HF_TOKEN\" > ~/.hf_token && chmod 600 ~/.hf_token"
    exit 1
fi

cd "$ROOT" || { echo "ERROR: cannot cd $ROOT"; exit 1; }

# 旧 image 削除 (Dockerfile 変更時に必須)
if docker image inspect cb-vllm:latest >/dev/null 2>&1; then
    echo "[redeploy] removing old image cb-vllm:latest"
    docker rmi -f cb-vllm:latest 2>/dev/null
fi

# 既存コンテナ停止
if docker ps -a --format '{{.Names}}' | grep -q "^cb-vllm-phase7$"; then
    echo "[redeploy] stopping existing container cb-vllm-phase7"
    docker stop cb-vllm-phase7 2>/dev/null
    docker rm cb-vllm-phase7 2>/dev/null
fi

# 旧 logs を退避
if [ -d "$ROOT/cognitive_bias_research/logs/phase7_open" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    echo "[redeploy] archiving old logs to logs/phase7_open.$TS"
    mv "$ROOT/cognitive_bias_research/logs/phase7_open" \
       "$ROOT/cognitive_bias_research/logs/phase7_open.$TS"
fi

# bundle 展開 (utils/ scripts/ forgpu/ 等が上書きされる)
echo "[redeploy] extracting bundle..."
tar -xzf "$BUNDLE"

cd cognitive_bias_research

# 環境変数 set
export HF_CACHE_DIR="$ROOT/hf_cache"
mkdir -p "$HF_CACHE_DIR"
export HF_TOKEN=$(cat "$TOKEN_FILE")
mkdir -p logs/phase7_open

# fire-and-forget
echo "[redeploy] launching dgx_fire_and_forget.sh"
nohup bash forgpu/dgx_fire_and_forget.sh > /dev/null 2>&1 &
PID=$!
echo "[redeploy] PID=$PID"

sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[redeploy] WARNING: process $PID died immediately"
fi

echo ""
echo "=========================================="
echo "[redeploy] done. Check logs:"
echo "  ssh muds-dgxa100-nakamura"
echo "  tail -f $ROOT/cognitive_bias_research/logs/phase7_open/main.log"
echo "=========================================="
