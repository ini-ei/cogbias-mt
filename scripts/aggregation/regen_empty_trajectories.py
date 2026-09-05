#!/usr/bin/env python3
"""
regen_empty_trajectories.py — 空応答だった軌跡だけを同一設定で再生成する（camera-ready 修正）。

背景: Qwen3.5 系（思考モード）で max_tokens=4096 を推論部で使い切り、content が空のまま
      記録・評価されていた（エラーなし）。Gemini JA でも 4 軌跡が空。
方針: 生成設定（system=T1, T2 提示, T4 提示, temperature 0.7）は phase5_benchmark_validation.py と
      同一のまま、max_tokens だけ引き上げ、content が空なら再試行する。自己評価も同じ関数で行う。
出力: 再生成したレコードだけを --out に書く（元ファイルは merge_regen.py で差し替える）。

使い方（ローカル / DGX 共通）:
  OPENAI_BASE_URL=http://localhost:8000/v1 OPENROUTER_API_KEY=dummy \
  python3 scripts/regen_empty_trajectories.py --benchmark results/phase5_v2/hard_all_80_clear_ce.jsonl \
      --model qwen3.5-27b --todo results/rebuttal/regen_todo.json --key ja:qwen3.5-27b \
      --out results/rebuttal/regen/ja_qwen3.5-27b.json
  # API 経由で slug を上書きする場合: --slug google/gemini-3.1-pro-preview
"""
import argparse, json, os, sys, time, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.phase5_benchmark_validation import BenchmarkEvaluator  # noqa: E402


def load_benchmark(path):
    probs = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                probs[p["id"]] = p
    return probs


def nonempty(s):
    return bool((s or "").strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--model", required=True, help="短縮名 (qwen3.5-27b など)")
    ap.add_argument("--todo", required=True, help="regen_todo.json")
    ap.add_argument("--key", required=True, help="todo 内のキー (例 ja:qwen3.5-27b)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--eval-max-tokens", type=int, default=8192)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--slug", default=None, help="model id の上書き (API slug 変更時)")
    args = ap.parse_args()

    probs = load_benchmark(args.benchmark)
    ids = json.load(open(args.todo))[args.key]
    ev = BenchmarkEvaluator(args.model)
    if args.slug:
        ev.model = args.slug
    # 自己評価呼び出しの max_tokens も引き上げる（思考モードで JSON が空になるのを防ぐ）
    _orig_create = ev.client.chat.completions.create
    eval_max = args.eval_max_tokens

    def _create(**kw):
        kw["max_tokens"] = max(int(kw.get("max_tokens", 0) or 0), eval_max)
        return _orig_create(**kw)
    ev.client.chat.completions.create = _create

    done = {}
    if os.path.exists(args.out):
        for r in json.load(open(args.out)):
            done[r["problem_id"]] = r
    out = list(done.values())
    print(f"[regen] {args.key}: {len(ids)} 軌跡, 既存 {len(done)}, model={ev.model}, max_tokens={args.max_tokens}")

    for i, pid in enumerate(ids, 1):
        if pid in done:
            continue
        p = probs[pid]
        t1 = next(t for t in p["turns"] if t["turn_number"] == 1)
        t2 = next(t for t in p["turns"] if t["turn_number"] == 2)
        t3 = next(t for t in p["turns"] if t["turn_number"] == 3)
        t4 = next(t for t in p["turns"] if t["turn_number"] == 4)
        t2_msg = t2["content"] + (f"\n\n{t2['question']}" if t2.get("question") else "")
        t4_msg = t3["content"]
        if t4.get("content"):
            t4_msg += f"\n\n{t4['content']}"
        if t4.get("question"):
            t4_msg += f"\n\n{t4['question']}"

        def gen(messages):
            for a in range(args.attempts):
                try:
                    r = _orig_create(model=ev.model, messages=messages, temperature=0.7, max_tokens=args.max_tokens)
                    c = r.choices[0].message.content
                    if nonempty(c):
                        return c
                    print(f"    [empty] attempt {a+1}: finish={getattr(r.choices[0],'finish_reason',None)}")
                except Exception as e:
                    print(f"    [err] attempt {a+1}: {str(e)[:120]}")
                    time.sleep(5)
            return None

        ans2 = gen([{"role": "system", "content": t1["content"]}, {"role": "user", "content": t2_msg}])
        if ans2 is None:
            print(f"  [{i}/{len(ids)}] {pid}: T2 生成失敗（スキップ）"); continue
        ans4 = gen([{"role": "system", "content": t1["content"]}, {"role": "user", "content": t2_msg},
                    {"role": "assistant", "content": ans2}, {"role": "user", "content": t4_msg}])
        if ans4 is None:
            print(f"  [{i}/{len(ids)}] {pid}: T4 生成失敗（スキップ）"); continue

        def selfeval(ans, turn):
            crit = p.get("evaluation_criteria", {}).get(turn, {})
            for a in range(args.attempts):
                e = ev._evaluate_bias(ans, p.get("ideal_responses", {}).get(turn, ""),
                                      p.get("bias_indicators", []), p.get("debiased_indicators", []), crit)
                # 既定値 (0.5/0.5) は解析失敗の印なので再試行
                if not (str(e.get("confidence")) == "0.5" and str(e.get("justification_score")) == "0.5"):
                    return e
            return e
        e2 = selfeval(ans2, "turn2"); e4 = selfeval(ans4, "turn4")
        rec = {
            "problem_id": pid, "bias_type": p["bias_type"], "domain": p["domain"],
            "difficulty": p.get("difficulty", "unknown"), "harm_type": p.get("harm_type", "unknown"),
            "harm_description": p.get("harm_description", ""), "model": args.model,
            "turn2_response": ans2, "turn4_response": ans4,
            "turn2_bias_detected": e2["bias_detected"], "turn2_bias_confidence": e2.get("confidence", 0.5),
            "turn2_justification_score": e2["justification_score"],
            "turn4_bias_detected": e4["bias_detected"], "turn4_bias_confidence": e4.get("confidence", 0.5),
            "turn4_justification_score": e4["justification_score"],
            "debiased": bool(e2["bias_detected"]) and not bool(e4["bias_detected"]),
            "persistent_bias": bool(e2["bias_detected"]) and bool(e4["bias_detected"]),
            "regenerated": datetime.datetime.now().isoformat(timespec="seconds"),
            "regen_max_tokens": args.max_tokens, "regen_model_id": ev.model,
        }
        out.append(rec); done[pid] = rec
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"  [{i}/{len(ids)}] {pid}: T2 {len(ans2)} chars / T4 {len(ans4)} chars, self T2={e2['bias_detected']} T4={e4['bias_detected']}")
    print(f"[regen] done: {len(out)}/{len(ids)} → {args.out}")


if __name__ == "__main__":
    main()
