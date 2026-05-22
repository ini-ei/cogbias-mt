# forgpu/ — サーバー実行用 Docker / スクリプト

詳細は `docs/GPU_ENVIRONMENT.md` を参照。本ファイルはサーバ用スクリプトのインデックス。

## ファイル一覧

### Phase 6/7 frontier (r750xa / OpenRouter API)

| ファイル | 用途 |
|:---|:---|
| `Dockerfile.api` | API ベースのジョブ実行用 (GPU 不要、python:3.11-slim ベース、~200MB) |
| `run_phase6_en.sh` | EN ベンチマーク全パイプライン (gen → quality → clearce → selfeval → crosseval → merge) |
| `run_retry_en.sh` | HTTP 402 retry ラッパー (`scripts/retry_errored_evaluations.py` を呼ぶ) |
| `run_patch_en.sh` | quality-fail 6 件回復用 patch (n=74 → n=80) |
| `run_patch_en_resume.sh` | Stage A skip 版 (新規 1 件のみ再評価 + マージ) |
| `make_bundle.sh` / `make_patch_bundle.sh` / `make_retry_bundle.sh` | バンドル作成 |
| `pull_results.sh` | r750xa → ローカル rsync |

### Phase 7 open weights (dgxa100 / llama.cpp + GGUF)

| ファイル | 用途 |
|:---|:---|
| `Dockerfile.vllm` | nvidia/cuda:12.4.1-devel + cuda-compat-12-4 + llama.cpp ソースビルド (driver 535 互換) |
| `run_openmodels_pipeline.sh` | llama-server (`-hf <repo>:Q4_K_M`) で 3 open weights × JA/EN を順次評価 |
| `dgx_fire_and_forget.sh` | DGX 上で build → CUDA check → pipeline を一気通貫実行 |
| `dgx_redeploy.sh` | DGX 上で旧 image 削除 + 展開 + 起動 (1 コマンド、 `~/.hf_token` 自動 source) |
| `redeploy.sh` | ローカル: build → scp → ssh + dgx_redeploy.sh を 1 行で実行 |
| `make_openmodels_bundle.sh` | open weights バンドル作成 (両 bench 込み、~430KB) |
| `README_phase7.md` | Phase 7 デプロイ手順 (バンドル内に含まれる) |
| `run_vllm_serve.sh` | (旧、現在は llama-server に置換) |

### 旧 / Phase 1

| ファイル | 用途 | 備考 |
|:---|:---|:---|
| `Dockerfile.plamo` | Phase 1 で PLaMo-2-Translate を vLLM で動かす用 | 旧、現在不使用 |
| `test_plamo.py` | PLaMo 動作確認用 | 旧 |

---

## Phase 6 EN パイプラインをサーバーで動かす

### 1. ローカルで現状をローカルから rsync する

```bash
# サーバーへ
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='results/phase5_v2' \
  --exclude='results/phase5' \
  --exclude='results/phase4' \
  --exclude='results/phase3' \
  --exclude='results/phase2' \
  --exclude='results/phase1' \
  /path/to/cognitive_bias_research/ \
  user@server:/path/to/cognitive_bias_research/
```

`results/phase5_v2_en/` 内に既に部分的な生成結果があれば、それも一緒に転送される（途中再開可能）。

### 2. サーバー側で Docker イメージをビルド

```bash
cd /path/to/cognitive_bias_research/
docker build -f forgpu/Dockerfile.api -t cb-api:latest .
```

### 3. パイプライン実行

```bash
docker run --rm \
  -e OPENROUTER_API_KEY="sk-or-..." \
  -v $(pwd):/work \
  -w /work \
  --name cb-api-phase6 \
  cb-api:latest \
  bash forgpu/run_phase6_en.sh all
```

`all` の代わりに段階別実行も可能:
- `gen` — EN 80 問生成のみ
- `quality` — 品質審査 + 不合格抽出
- `clearce` — Exp C clear CE 書き換え
- `selfeval` — 6 モデル self-eval（並列、〜30-60 分）
- `crosseval` — 6 × 3 = 18 cross-eval ジョブ並列（〜2-4 時間）
- `merge` — 集計のみ（API 呼び出しなし、すぐ終わる）

各ステージは intermediate save / 既存結果スキップに対応しているので、途中で止まっても再起動で継続できる。

### 4. バックグラウンド実行（laptop 閉じていい状態）

```bash
docker run -d \
  -e OPENROUTER_API_KEY="sk-or-..." \
  -v $(pwd):/work \
  -w /work \
  --name cb-api-phase6 \
  --restart unless-stopped \
  cb-api:latest \
  bash forgpu/run_phase6_en.sh all
```

進捗確認:
```bash
docker logs -f cb-api-phase6
ls -la results/phase5_v2_en/
```

止めるとき:
```bash
docker stop cb-api-phase6
docker rm cb-api-phase6
```

### 5. 結果の取得

```bash
# サーバーで完走後、結果をローカルに rsync
rsync -avz user@server:/path/to/cognitive_bias_research/results/phase5_v2_en/ \
  /path/to/cognitive_bias_research/results/phase5_v2_en/
rsync -avz user@server:/path/to/cognitive_bias_research/logs/phase6_en/ \
  /path/to/cognitive_bias_research/logs/phase6_en/
```

---

## API キーの扱い

サーバーに `.env` が置けない環境のため、**実行時に `-e OPENROUTER_API_KEY=...` で環境変数として渡す**。コンテナ終了後はキーは残らない。

シェル履歴に残したくなければ、`docker run --env-file <(echo OPENROUTER_API_KEY=$KEY)` のようにファイル渡しもできる（`<()` は Bash の process substitution、ファイルは作られない）。

---

## 推定所要時間（API 呼び出しのみ、サーバー性能依存しない）

| ステージ | 推定時間 | API コール数 |
|:---|:---:|:---:|
| gen (v1 + v2) | 〜10-15 分 | 〜80 |
| quality | 〜5-10 分 | 〜80 |
| clearce | 〜10-15 分 | 〜80 |
| selfeval (6 モデル並列) | 〜30-60 分 | 6 × 80 × 4 = 1,920 |
| crosseval (18 ジョブ並列) | 〜2-4 時間 | 18 × 80 = 1,440 |
| merge | 即時 | 0 |
| **合計** | **〜3-5 時間** | **〜3,600 コール** |

OpenRouter 推定コスト: $30-60 程度（既存の Phase 6 コストと同水準）。

---

## トラブルシューティング

**Q: コンテナ内から OpenRouter に繋がらない**  
→ `docker run` 時に `--network=host` を追加するか、サーバーのプロキシ設定を確認。`curl https://openrouter.ai/api/v1/models -H "Authorization: Bearer $OPENROUTER_API_KEY"` で疎通確認。

**Q: 途中で OOM**  
→ 現状の Phase 6 は API 呼び出しのみなのでメモリ消費は数百 MB 以下。OOM が出たら `docker run --memory=2g` 等で明示的に上限を指定し、別プロセスとの競合を確認。

**Q: ジョブが死んだ**  
→ 各ステージは intermediate save 対応。`bash forgpu/run_phase6_en.sh <stage>` で該当ステージを再実行すれば既存出力をスキップして続きから走る。
