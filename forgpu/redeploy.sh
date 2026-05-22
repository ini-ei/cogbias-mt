#!/bin/bash
# redeploy.sh (ローカルで実行)
#
# 1. bundle ビルド
# 2. DGX 転送
# 3. dgx_redeploy.sh をリモート実行
# 4. log tail (任意)
#
# 使い方:
#   bash forgpu/redeploy.sh           # ビルド + 転送 + 起動
#   bash forgpu/redeploy.sh tail      # 上記 + main.log を tail -f

set -euo pipefail

REMOTE="muds-dgxa100-nakamura"
BUNDLE="cb_phase7_open_bundle.tar.gz"

# 1. ビルド
echo "[local] building bundle..."
bash forgpu/make_openmodels_bundle.sh > /tmp/bundle_build.log 2>&1
echo "[local] bundle: $(du -sh $BUNDLE | awk '{print $1}')"

# 2. 転送 + dgx_redeploy.sh も別途送る (bundle 内とは別に home 直下にも置く)
echo "[local] uploading bundle..."
scp -q "$BUNDLE" "$REMOTE:~/"
echo "[local] uploading dgx_redeploy.sh..."
scp -q forgpu/dgx_redeploy.sh "$REMOTE:~/dgx_redeploy.sh"

# 3. リモート実行
echo "[local] executing remote redeploy..."
ssh "$REMOTE" "bash ~/dgx_redeploy.sh"

# 4. オプション: log tail
if [ "${1:-}" = "tail" ]; then
    echo ""
    echo "[local] tailing remote log (Ctrl+C to exit, remote process continues)..."
    sleep 3
    ssh "$REMOTE" "tail -f /raid/data/nakamura-lab/Inui/cognitive_bias_research/logs/phase7_open/main.log"
fi
