#!/usr/bin/env python3
"""
reeval_fix_defaults.py — reeval_self.py の出力 (*_full80.json) に残った自己評価の既定値 (conf 0.5 / just 0.5)
だけを、より大きい max_tokens と多い試行回数でやり直してその場で差し替える。

  OPENAI_BASE_URL=http://localhost:8000/v1 OPENROUTER_API_KEY=dummy \
  python3 scripts/reeval_fix_defaults.py --benchmark <jsonl> --model qwen3.5-27b --file results/rebuttal/regen/ja_qwen3.5-27b_full80.json \
      --eval-max-tokens 16384 --attempts 5
"""
import argparse, json, os, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.phase5_benchmark_validation import BenchmarkEvaluator  # noqa: E402


def is_default(e):
    return str(e.get("confidence")) == "0.5" and str(e.get("justification_score")) == "0.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True); ap.add_argument("--model", required=True); ap.add_argument("--file", required=True)
    ap.add_argument("--eval-max-tokens", type=int, default=16384); ap.add_argument("--attempts", type=int, default=5)
    args = ap.parse_args()
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(args.benchmark) if l.strip()}
    d = json.load(open(args.file))
    ev = BenchmarkEvaluator(args.model)
    _orig = ev.client.chat.completions.create
    emax = args.eval_max_tokens

    def _create(**kw):
        kw["max_tokens"] = max(int(kw.get("max_tokens", 0) or 0), emax)
        return _orig(**kw)
    ev.client.chat.completions.create = _create

    fixed = still = 0
    for r in d:
        p = probs[r["problem_id"]]
        for t in ("turn2", "turn4"):
            if not ((str(r.get(f"{t}_bias_confidence")) == "0.5" and str(r.get(f"{t}_justification_score")) == "0.5")
                    or r.get(f"{t}_bias_detected") is None):
                continue
            ans = r.get(f"{t}_response") or ""
            if not ans.strip():
                print(f"  {r['problem_id']} {t}: 応答が空 → skip"); continue
            e = None
            for a in range(args.attempts):
                e = ev._evaluate_bias(ans, p.get("ideal_responses", {}).get(t, ""), p.get("bias_indicators", []),
                                      p.get("debiased_indicators", []), p.get("evaluation_criteria", {}).get(t, {}))
                if not is_default(e):
                    break
            if e is None or is_default(e):
                still += 1; print(f"  {r['problem_id']} {t}: still default after {args.attempts} attempts"); continue
            r[f"{t}_bias_detected"] = e["bias_detected"]; r[f"{t}_bias_confidence"] = e.get("confidence", 0.5)
            r[f"{t}_justification_score"] = e["justification_score"]
            r["debiased"] = bool(r["turn2_bias_detected"]) and not bool(r["turn4_bias_detected"])
            r["persistent_bias"] = bool(r["turn2_bias_detected"]) and bool(r["turn4_bias_detected"])
            r["self_reevaluated"] = datetime.datetime.now().isoformat(timespec="seconds"); r["self_eval_max_tokens"] = emax
            fixed += 1; print(f"  {r['problem_id']} {t}: fixed → {e['bias_detected']} ({e.get('confidence')})")
            json.dump(d, open(args.file, "w"), ensure_ascii=False, indent=2)
    print(f"[fix] {args.file}: fixed {fixed}, still default {still}")


if __name__ == "__main__":
    main()
