#!/usr/bin/env python3
"""
rebuttal_verify_rejudge.py — 再生成軌跡 (results/rebuttal/regen/{ja,en}_{model}.json) が、
すべての judge ファイルで「regenerated trajectory judged」として再判定済みかをファイル側から検証する。
（ログに頼らない。分割実行でログが上書きされても確認できる。）
"""
import json, os, sys
R = "results/rebuttal/regen"
TARGETS = {  # (lang, model) -> judge files
    ("ja", "qwen3.5-27b"): ["results/phase5_v2/cross_eval_gpt5_open/qwen3.5-27b_results.json", "results/phase5_v2/cross_eval_claude_opus_n80/qwen3.5-27b_results.json",
                            "results/phase5_v2/cross_eval_gemini_open/qwen3.5-27b_results.json", "results/phase5_v2/cross_eval_llama_n80/qwen3.5-27b_results.json",
                            "results/rebuttal/cross_eval_gpt5_claude_rubric_open/qwen3.5-27b_results.json"],
    ("ja", "qwen3.5-122b"): ["results/phase5_v2/cross_eval_gpt5_open/qwen3.5-122b_results.json", "results/phase5_v2/cross_eval_claude_opus_n80/qwen3.5-122b_results.json",
                             "results/phase5_v2/cross_eval_gemini_open/qwen3.5-122b_results.json", "results/phase5_v2/cross_eval_llama_n80/qwen3.5-122b_results.json",
                             "results/rebuttal/cross_eval_gpt5_claude_rubric_open/qwen3.5-122b_results.json"],
    ("ja", "gemini-3.1-pro"): ["results/phase5_v2/cross_eval_gpt5_n80/gemini-3.1-pro_results.json", "results/phase5_v2/cross_eval_claude_opus_n80/gemini-3.1-pro_results.json",
                               "results/phase5_v2/cross_eval_gemini_n80/gemini-3.1-pro_results.json", "results/phase5_v2/cross_eval_llama_n80/gemini-3.1-pro_results.json"],
    ("en", "qwen3.5-27b"): ["results/phase5_v2_en/cross_eval_gpt5_en_open/qwen3.5-27b_results.json", "results/phase5_v2_en/cross_eval_claude_opus_en/qwen3.5-27b_results.json",
                            "results/phase5_v2_en/cross_eval_gemini_en_open/qwen3.5-27b_results.json", "results/phase5_v2_en/cross_eval_llama_en/qwen3.5-27b_results.json"],
    ("en", "qwen3.5-122b"): ["results/phase5_v2_en/cross_eval_gpt5_en_open/qwen3.5-122b_results.json", "results/phase5_v2_en/cross_eval_claude_opus_en/qwen3.5-122b_results.json",
                             "results/phase5_v2_en/cross_eval_gemini_en_open/qwen3.5-122b_results.json", "results/phase5_v2_en/cross_eval_llama_en/qwen3.5-122b_results.json"],
}
only = sys.argv[1:]  # 任意: model 名で絞る
ok_all = True
for (lang, m), files in TARGETS.items():
    if only and m not in only: continue
    rf = f"{R}/{lang}_{m}.json"
    if not os.path.exists(rf): print(f"[skip] {rf} なし"); continue
    regen = {r["problem_id"]: r for r in json.load(open(rf))}
    for f in files:
        d = {r["problem_id"]: r for r in json.load(open(f))}
        judged = [p for p in regen if d.get(p, {}).get("rejudge_note") == "regenerated trajectory judged"]
        same_resp = [p for p in judged if d[p].get("turn2_response") == regen[p]["turn2_response"] and d[p].get("turn4_response") == regen[p]["turn4_response"]]
        dflt = [p for p in judged if any(d[p].get(f"{t}_bias_detected") in (False, None) and str(d[p].get(f"{t}_bias_confidence")) == "0.5" and str(d[p].get(f"{t}_justification_score")) == "0.5" for t in ("turn2", "turn4"))]
        st = "OK " if len(judged) == len(regen) and len(same_resp) == len(regen) and not dflt else "NG "
        if st == "NG ": ok_all = False
        print(f"{st}{lang} {m:14s} {f}: judged {len(judged)}/{len(regen)}, response一致 {len(same_resp)}, 既定値 {len(dflt)}")
print("ALL OK" if ok_all else "SOME NG")
