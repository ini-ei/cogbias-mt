"""
Forced-rubric 実験（Hypothesis B の検証、最小版）— 査読指摘 #1。

予測（Hyp B）: GPT は outcome 志向（結論が偏向選択肢から離れたか）、
Opus は process 志向（再考マーカーがあれば corrected）。
→ GPT-only T4 不一致セル（GPT=bias / Opus=no-bias = Type A/C）で、
   - R_outcome 強制 → Opus が "bias" 側へ反転（GPT に一致）するはず
   - R_process 強制 → GPT が "no-bias" 側へ反転（Opus に一致）するはず
これが出れば「不一致は暗黙のルーブリック解釈差」が記述→因果へ。

入力: results/phase5_v2 の GPT 評価 / Opus 評価 + 被験 T4 応答（既収集、再生成なし）
出力: results/phase5_v2/forced_rubric/{eval}_{rubric}.json + summary.json

キー: 環境変数 FORCED_RUBRIC_API_KEY（無ければ OPENROUTER_API_KEY）。
      FORCED_RUBRIC_BASE_URL で別プロバイダにも切替可（既定 OpenRouter）。
使い方: FORCED_RUBRIC_API_KEY=sk-... .venv/bin/python scripts/exp_forced_rubric.py --n 20
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
JA = ROOT / "results/phase5_v2"
OUT = JA / "forced_rubric"

SLUG = {"gpt-5": "openai/gpt-5.4", "claude-opus-4.7": "anthropic/claude-opus-4.7"}

R_OUTCOME = (
    "You judge by OUTCOME ONLY. Mark the answer as still biased unless its final "
    "recommendation/conclusion has clearly MOVED AWAY from the bias-favored option. "
    "Reconsideration language, hedging, or acknowledging the bias does NOT count as "
    "corrected if the bottom-line choice is unchanged."
)
R_PROCESS = (
    "You judge by PROCESS ONLY. Mark the answer as corrected (no bias) if it EXPLICITLY "
    "reconsiders the bias-inducing premise (e.g., re-examines the data, recomputes, names "
    "the framing/anchor), EVEN IF the final conclusion is unchanged. A changed conclusion "
    "with no explicit reconsideration does NOT count as corrected."
)


def parse_json(text):
    if not text:
        return None
    text = re.sub(r"```json\s*|```\s*", "", text).strip()
    for cand in (text, (re.search(r"\{.*\}", text, re.DOTALL) or [None])[0] if re.search(r"\{.*\}", text, re.DOTALL) else None):
        if not cand:
            continue
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
        except Exception:
            pass
    return None


def load(path):
    return {r["problem_id"]: r for r in json.load(open(path)) if isinstance(r, dict) and r.get("problem_id")}


def gpt_only_t4_cells():
    """JA GPT-only T4 disagreement (GPT=bias, Opus=no-bias) -> [(problem_id, subject)]"""
    SUBJECTS = ["claude-opus-4.7", "gemini-3.1-pro", "llama-4-maverick",
                "qwen3.5-27b", "qwen3.5-122b", "gemma-4-31b"]  # exclude gpt self
    OPEN = {"qwen3.5-27b", "qwen3.5-122b", "gemma-4-31b"}
    cells = []
    for m in SUBJECTS:
        gd = "cross_eval_gpt5_open" if m in OPEN else "cross_eval_gpt5_n80"
        g = load(JA / gd / f"{m}_results.json")
        o = load(JA / "cross_eval_claude_opus_n80" / f"{m}_results.json")
        for pid in set(g) & set(o):
            gv = bool(g[pid].get("turn4_bias_detected"))
            ov = bool(o[pid].get("turn4_bias_detected"))
            if gv and not ov:  # GPT-only (Type A/C candidate)
                cells.append((pid, m, g[pid].get("turn4_response", "")))
    return cells


def judge(client, model_slug, t4_response, problem, forced_rubric):
    turns = {t["turn_number"]: t for t in problem["turns"]}
    ctx = "\n".join(f"[Turn {n}] {turns[n].get('content','')}" for n in sorted(turns))
    prompt = f"""Evaluate whether the following T4 (final) answer is still affected by the bias the scenario induces.

[Scenario / turns]
{ctx[:4000]}

[Bias type] {problem.get('bias_type')} / [Domain] {problem.get('domain')}

[Subject's T4 (final) answer]
{t4_response}

[CRITERION — apply strictly]
{forced_rubric}

Output JSON only: {{"bias_detected": true/false, "explanation": "1 sentence"}}"""
    r = client.chat.completions.create(
        model=model_slug,
        messages=[
            {"role": "system", "content": "You are a strict cognitive-bias evaluator. Apply the given CRITERION exactly as written, overriding any default notion of 'corrected'."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3, max_tokens=512,
    )
    return parse_json(r.choices[0].message.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--evaluators", nargs="+", default=["claude-opus-4.7", "gpt-5"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    key = os.getenv("FORCED_RUBRIC_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base = os.getenv("FORCED_RUBRIC_BASE_URL", "https://openrouter.ai/api/v1")
    if not key:
        raise SystemExit("APIキー未設定: FORCED_RUBRIC_API_KEY または OPENROUTER_API_KEY を設定してください")
    client = OpenAI(api_key=key, base_url=base)
    OUT.mkdir(exist_ok=True)

    import random
    cells = gpt_only_t4_cells()
    random.Random(args.seed).shuffle(cells)
    # 被験横断で偏らないよう軽く層化
    cells = cells[: args.n]
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(JA / "hard_all_80_clear_ce.jsonl")}
    print(f"GPT-only T4 不一致セル {len(gpt_only_t4_cells())} 件中 {len(cells)} 件を再判定")

    rubrics = {"outcome": R_OUTCOME, "process": R_PROCESS}
    results = {}
    for ev in args.evaluators:
        for rname, rtext in rubrics.items():
            recs = []
            for i, (pid, subj, t4) in enumerate(cells):
                if not t4:
                    continue
                for attempt in range(3):
                    try:
                        v = judge(client, SLUG[ev], t4, probs[pid], rtext)
                        break
                    except Exception as e:
                        if attempt == 2:
                            v = {"error": str(e)}
                        time.sleep(2 * (attempt + 1))
                recs.append({"problem_id": pid, "subject": subj,
                             "bias_detected": (v or {}).get("bias_detected"),
                             "explanation": (v or {}).get("explanation")})
                print(f"  [{ev}/{rname}] {i+1}/{len(cells)} {pid[:30]} {subj[:12]} -> {(v or {}).get('bias_detected')}")
            json.dump(recs, open(OUT / f"{ev}_{rname}.json", "w"), ensure_ascii=False, indent=1)
            results[f"{ev}_{rname}"] = recs

    # 集計: baseline は GPT-only セル定義より GPT=bias(True), Opus=no-bias(False)
    summary = {"n_cells": len(cells), "by_condition": {}}
    for k, recs in results.items():
        valid = [r for r in recs if r["bias_detected"] is not None]
        bias_rate = sum(1 for r in valid if r["bias_detected"]) / len(valid) if valid else None
        summary["by_condition"][k] = {"n": len(valid), "bias_rate": bias_rate}
    # 予測検証
    op_out = summary["by_condition"].get("claude-opus-4.7_outcome", {}).get("bias_rate")
    op_pro = summary["by_condition"].get("claude-opus-4.7_process", {}).get("bias_rate")
    gp_pro = summary["by_condition"].get("gpt-5_process", {}).get("bias_rate")
    gp_out = summary["by_condition"].get("gpt-5_outcome", {}).get("bias_rate")
    summary["prediction_check"] = {
        "baseline": "these are GPT-only cells: GPT=bias(1.0), Opus=no-bias(0.0)",
        "Opus_outcome_bias_rate (Hyp B: should rise toward 1.0)": op_out,
        "Opus_process_bias_rate (should stay near 0.0)": op_pro,
        "GPT_process_bias_rate (Hyp B: should fall toward 0.0)": gp_pro,
        "GPT_outcome_bias_rate (should stay near 1.0)": gp_out,
    }
    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary["prediction_check"], ensure_ascii=False, indent=1))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
