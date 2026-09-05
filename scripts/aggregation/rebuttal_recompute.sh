#!/bin/bash
# rebuttal_recompute.sh — データ修復後の再計算手順（camera-ready）
# 前提: DGX の results/rebuttal/regen/*_full80.json と *_q8_full80.json を pull 済み、
#       判定側の既定値セルは rejudge_cells.py --todo で再判定済み。
# 各ステップは冪等（同じ入力なら同じ出力）。STEP=n で途中から再開できる。
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${STEP:-1}"; UNTIL="${UNTIL:-5}"   # STEP..UNTIL の範囲だけ実行
TS=$(date +%Y%m%d_%H%M%S)   # ログ名に付けて、分割実行で上書きしない
if [ -f .env ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then set -a; source .env; set +a; fi
unset OPENAI_BASE_URL LOCAL_MODEL; export PYTHONUNBUFFERED=1
R=results/rebuttal/regen; J=results/rebuttal/regen_for_judge; mkdir -p "$J/ja" "$J/en" logs/rebuttal/rejudge
JA=results/phase5_v2/hard_all_80_clear_ce.jsonl; EN=results/phase5_v2_en/hard_all_80_en_clear_ce.jsonl; JA_CR=results/phase5_v2/hard_all_80_claude_rubric.jsonl

# MODELS: この実行で judge 用ディレクトリに入れる（= STEP 2 で再判定する）被験。ブロックが終わった順に分けて回せる。
MODELS="${MODELS:-qwen3.5-27b qwen3.5-122b gemini-3.1-pro}"
if [ "$STEP" -le 1 ] && [ "$UNTIL" -ge 1 ]; then echo "## STEP 1: self-eval ファイルを再生成＋再自己評価版で差し替え、judge 用ディレクトリ作成 (MODELS=$MODELS)"
  rm -f "$J"/ja/*.json "$J"/en/*.json "$J"/ja_open/*.json "$J"/ja_frontier/*.json "$J"/en_open/*.json 2>/dev/null
  for m in $MODELS; do
    [ "$m" = gemini-3.1-pro ] && continue
    [ -f "$R/ja_${m}_full80.json" ] && cp "$R/ja_${m}_full80.json" "results/phase5_v2/full_n80_open/${m}_results.json" && echo "  ja $m self-eval 差し替え"
    [ -f "$R/en_${m}_full80.json" ] && cp "$R/en_${m}_full80.json" "results/phase5_v2_en/full_n80_en_open/${m}_results.json" && echo "  en $m self-eval 差し替え"
    [ -f "$R/ja_${m}.json" ] && cp "$R/ja_${m}.json" "$J/ja/${m}_results.json"
    [ -f "$R/en_${m}.json" ] && cp "$R/en_${m}.json" "$J/en/${m}_results.json"
  done
  # Gemini JA 4 軌跡: self-eval ファイルに差し替え＋judge 用
  case " $MODELS " in *" gemini-3.1-pro "*)
    .venv/bin/python scripts/merge_regen.py --regen "$R/ja_gemini-3.1-pro.json" --target results/phase5_v2/full_n80/gemini-3.1-pro_results.json --judge-dir "$J/ja";; esac
  .venv/bin/python - <<'PY'
import json,glob
for f in glob.glob("results/phase5_v2/full_n80_open/qwen*_results.json")+glob.glob("results/phase5_v2_en/full_n80_en_open/qwen*_results.json")+["results/phase5_v2/full_n80/gemini-3.1-pro_results.json"]:
    d=json.load(open(f)); e=sum(1 for r in d if not (r.get("turn2_response") or "").strip() or not (r.get("turn4_response") or "").strip())
    df=sum(1 for r in d if str(r.get("turn2_bias_confidence"))=="0.5" and str(r.get("turn2_justification_score"))=="0.5")
    print(f"  {f}: n={len(d)} 空応答={e} self既定値(T2)={df}")
PY
fi

if [ "$STEP" -le 2 ] && [ "$UNTIL" -ge 2 ]; then echo "## STEP 2: 再生成軌跡を 4 judge で再判定（JA/EN、既存 judge ファイルに差し替え）"
  # judge dir → (ja frontier dir, ja open dir, en dir, en open dir)
  run() { # evaluator regen_subdir(ja_open|ja_frontier|en_open) target_dir
    local b="$EN"; case "$2" in ja*) b="$JA";; esac
    nohup .venv/bin/python -u scripts/rejudge_cells.py --regen-dir "$J/$2" --judge-file "$3" --evaluator "$1" --benchmark "$b" > "logs/rebuttal/rejudge/regen_${TS}_${1}_${2}_$(basename $3).log" 2>&1 &
  }
  # Qwen（open）と Gemini（frontier）は judge ファイルの置き場所が違うので regen dir を分ける
  mkdir -p "$J/ja_open" "$J/ja_frontier" "$J/en_open"
  cp "$J"/ja/qwen*_results.json "$J/ja_open/" 2>/dev/null; cp "$J"/ja/gemini-3.1-pro_results.json "$J/ja_frontier/" 2>/dev/null; cp "$J"/en/qwen*_results.json "$J/en_open/" 2>/dev/null
  run gpt-5 ja_open results/phase5_v2/cross_eval_gpt5_open;              run gpt-5 ja_frontier results/phase5_v2/cross_eval_gpt5_n80
  run claude-opus-4.7 ja_open results/phase5_v2/cross_eval_claude_opus_n80; run claude-opus-4.7 ja_frontier results/phase5_v2/cross_eval_claude_opus_n80
  run gemini-3.1-pro ja_open results/phase5_v2/cross_eval_gemini_open;   run gemini-3.1-pro ja_frontier results/phase5_v2/cross_eval_gemini_n80
  run llama-4-maverick ja_open results/phase5_v2/cross_eval_llama_n80;   run llama-4-maverick ja_frontier results/phase5_v2/cross_eval_llama_n80
  run gpt-5 en_open results/phase5_v2_en/cross_eval_gpt5_en_open;        run claude-opus-4.7 en_open results/phase5_v2_en/cross_eval_claude_opus_en
  run gemini-3.1-pro en_open results/phase5_v2_en/cross_eval_gemini_en_open; run llama-4-maverick en_open results/phase5_v2_en/cross_eval_llama_en
  # 実験B（Claude rubric × GPT judge）の Qwen 再生成分
  [ "${SKIP_EXPB:-0}" = 1 ] || nohup .venv/bin/python -u scripts/rejudge_cells.py --regen-dir "$J/ja_open" --judge-file results/rebuttal/cross_eval_gpt5_claude_rubric_open --evaluator gpt-5 --benchmark "$JA_CR" > "logs/rebuttal/rejudge/regen_${TS}_expB.log" 2>&1 &
  wait; echo "  STEP 2 done"
fi

if { [ "$STEP" -le 2 ] && [ "$UNTIL" -ge 2 ] && [ -f "$R/ja_qwen3.5-122b_full80.json" ] && [ -f "$J/ja_open/qwen3.5-122b_results.json" ]; } || [ "${STEP2B:-0}" = 1 ]; then echo "## STEP 2b: Opus × Claude ルーブリックをオープン 3 被験にも（補遺: Claude ルーブリック下の主結果2 を 7 被験で）"
  mkdir -p results/rebuttal/cross_eval_claude_rubric_opus_open
  nohup .venv/bin/python -u scripts/phase5_cross_eval.py --results-dir results/phase5_v2/full_n80_open --benchmark "$JA_CR" \
    --evaluator claude-opus-4.7 --models qwen3.5-27b gemma-4-31b qwen3.5-122b --output-dir results/rebuttal/cross_eval_claude_rubric_opus_open > logs/rebuttal/expB_opus_claude_rubric_open.log 2>&1 &
  wait; echo "  STEP 2b done"
fi

if [ "$STEP" -le 3 ] && [ "$UNTIL" -ge 3 ]; then echo "## STEP 3: 実験C（Q8_0）: 再生成版で置き換え → 4 judge 判定"
  [ -f "$R/ja_qwen3.5-27b_q8_full80.json" ] && cp "$R/ja_qwen3.5-27b_q8_full80.json" results/rebuttal/full_n80_open_q8/qwen3.5-27b_results.json
  [ -f "$R/en_qwen3.5-27b_q8_full80.json" ] && cp "$R/en_qwen3.5-27b_q8_full80.json" results/rebuttal/full_n80_en_open_q8/qwen3.5-27b_results.json
  bash scripts/rebuttal_judge_q8.sh
fi

if [ "$STEP" -le 4 ] && [ "$UNTIL" -ge 4 ]; then echo "## STEP 4: 残存する既定値の点検（0 であること）"
  .venv/bin/python - <<'PY'
import json,glob
def dflt(r,t): return r.get(f"{t}_bias_detected") in (False,"False") and str(r.get(f"{t}_bias_confidence"))=="0.5" and str(r.get(f"{t}_justification_score"))=="0.5"
tot=0
for f in glob.glob("results/phase5_v2/cross_eval_*/*_results.json")+glob.glob("results/phase5_v2_en/cross_eval_*/*_results.json")+glob.glob("results/phase5_v2/full_n80*/*_results.json")+glob.glob("results/phase5_v2_en/full_n80*/*_results.json")+glob.glob("results/rebuttal/cross_eval_*/*_results.json"):
    if "metrics" in f or "gpt-5.5" in f or "claude-4-sonnet" in f: continue
    d=json.load(open(f)); k=sum(1 for r in d if dflt(r,"turn2") or dflt(r,"turn4") or r.get("turn2_bias_detected") is None or r.get("turn4_bias_detected") is None)
    e=sum(1 for r in d if not (r.get("turn2_response") or "").strip() or not (r.get("turn4_response") or "").strip())
    if k or e: print(f"  {f}: 既定値/None {k}, 空応答 {e}"); tot+=k+e
print("  合計問題レコード:", tot)
PY
fi

if [ "$STEP" -le 5 ] && [ "$UNTIL" -ge 5 ]; then echo "## STEP 5: 論文の数値を再計算"
  mkdir -p results/rebuttal_checks
  .venv/bin/python scripts/aggregate_multilingual.py > results/rebuttal/recompute_aggregate.txt 2>&1 && echo "  aggregate ok"
  .venv/bin/python scripts/rebuttal_cluster_checks.py > results/rebuttal/recompute_permutation.txt 2>&1 && echo "  permutation ok"
  .venv/bin/python scripts/rebuttal_twoway_cluster.py > results/rebuttal/recompute_twoway.txt 2>&1 && echo "  two-way bootstrap ok"
  .venv/bin/python scripts/rebuttal_loso.py > results/rebuttal/recompute_loso.txt 2>&1 && echo "  loso ok"
  .venv/bin/python scripts/make_fig2_wiiat.py > results/rebuttal/recompute_fig2.txt 2>&1 && echo "  fig2 ok"
  echo "  → 各 recompute_*.txt を確認して本文・図1 SVG・表を更新"
fi
echo "[recompute] done $(date -Iseconds)"
