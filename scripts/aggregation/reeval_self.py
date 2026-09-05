#!/usr/bin/env python3
"""
reeval_self.py — 自己評価だけを全レコードについてやり直す（Qwen3.5 の思考モードで評価 JSON が
max_tokens=2048 に収まらず既定値 0.5/0.5 に落ちていた問題への対処）。

入力: 元の results.json (--orig) と、空応答を再生成したレコード (--regen、任意)。
処理: 応答は regen があればそれを、無ければ元のものを使い、_evaluate_bias を max_tokens 引き上げ＋再試行で再実行。
出力: 80 件そろった新しい results.json (--out)。

  OPENAI_BASE_URL=http://localhost:8000/v1 OPENROUTER_API_KEY=dummy \
  python3 scripts/reeval_self.py --benchmark results/phase5_v2/hard_all_80_clear_ce.jsonl --model qwen3.5-27b \
      --orig results/phase5_v2/full_n80_open/qwen3.5-27b_results.json \
      --regen results/rebuttal/regen/ja_qwen3.5-27b.json --out results/rebuttal/regen/ja_qwen3.5-27b_full80.json
"""
import argparse, json, os, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.phase5_benchmark_validation import BenchmarkEvaluator  # noqa: E402


def is_default(e):
    return str(e.get("confidence")) == "0.5" and str(e.get("justification_score")) == "0.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--regen", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-max-tokens", type=int, default=8192)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--slug", default=None)
    args = ap.parse_args()

    probs = {}
    for line in open(args.benchmark):
        if line.strip():
            p = json.loads(line); probs[p["id"]] = p
    orig = json.load(open(args.orig))
    regen = {r["problem_id"]: r for r in json.load(open(args.regen))} if args.regen and os.path.exists(args.regen) else {}
    ev = BenchmarkEvaluator(args.model)
    if args.slug:
        ev.model = args.slug
    _orig_create = ev.client.chat.completions.create
    emax = args.eval_max_tokens

    def _create(**kw):
        kw["max_tokens"] = max(int(kw.get("max_tokens", 0) or 0), emax)
        return _orig_create(**kw)
    ev.client.chat.completions.create = _create

    done = {r["problem_id"]: r for r in json.load(open(args.out))} if os.path.exists(args.out) else {}
    out = []
    n_def = 0
    for i, r0 in enumerate(orig, 1):
        pid = r0["problem_id"]
        if pid in done:
            out.append(done[pid]); continue
        r = dict(regen.get(pid, r0))
        p = probs[pid]
        if not (r.get("turn2_response") or "").strip() or not (r.get("turn4_response") or "").strip():
            print(f"  [{i}/80] {pid}: 応答が空のまま（regen 未了）→ そのまま保持"); out.append(r); continue

        def se(ans, turn):
            crit = p.get("evaluation_criteria", {}).get(turn, {})
            e = None
            for a in range(args.attempts):
                e = ev._evaluate_bias(ans, p.get("ideal_responses", {}).get(turn, ""),
                                      p.get("bias_indicators", []), p.get("debiased_indicators", []), crit)
                if not is_default(e):
                    return e
            return e
        e2 = se(r["turn2_response"], "turn2"); e4 = se(r["turn4_response"], "turn4")
        if is_default(e2) or is_default(e4):
            n_def += 1
        r.update({
            "turn2_bias_detected": e2["bias_detected"], "turn2_bias_confidence": e2.get("confidence", 0.5),
            "turn2_justification_score": e2["justification_score"],
            "turn4_bias_detected": e4["bias_detected"], "turn4_bias_confidence": e4.get("confidence", 0.5),
            "turn4_justification_score": e4["justification_score"],
            "debiased": bool(e2["bias_detected"]) and not bool(e4["bias_detected"]),
            "persistent_bias": bool(e2["bias_detected"]) and bool(e4["bias_detected"]),
            "self_reevaluated": datetime.datetime.now().isoformat(timespec="seconds"),
            "self_eval_max_tokens": emax,
        })
        out.append(r); done[pid] = r
        json.dump(out + [x for x in orig if x["problem_id"] not in done and x["problem_id"] != pid][:0], open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"  [{i}/80] {pid}: self T2={e2['bias_detected']} ({e2.get('confidence')}) T4={e4['bias_detected']} ({e4.get('confidence')})")
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[reeval] {args.out}: {len(out)} 件, 既定値のまま {n_def} 件")


if __name__ == "__main__":
    main()
