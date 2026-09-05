#!/usr/bin/env python3
"""
rejudge_cells.py — 交差評価の「解析失敗の既定値（bias=False, conf 0.5, just 0.5）」セルと、
再生成した軌跡のセルを、同じ評価者・同じプロンプトで再判定して judge ファイルを差し替える。

プロンプトは phase5_cross_eval.CrossEvaluator._evaluate_bias をそのまま使う（同一）。
違いは (a) 評価の max_tokens を 4096 に引き上げ、(b) 生の出力を捕捉して頑健に JSON を取り出し、
(c) それでも既定値なら最大 3 回再試行、(d) 生出力を eval_raw_* として保存する点のみ。

モード1（既定値セル）:  --todo results/rebuttal/rejudge_todo.json
    todo = { "<judge results file>": [problem_id, ...], ... }
    各ファイルの評価者は path から推定（gpt5→gpt-5, claude_opus/opus→claude-opus-4.7, gemini→gemini(undated slug), llama→llama-4-maverick,
    claude_rubric_n80 → claude-4-sonnet）。応答は --responses-root 配下の self-eval ファイルから取る（regen 反映後）。
モード2（再生成軌跡）:  --regen-dir results/rebuttal/regen_for_judge/ja --judge-file <target> --evaluator <name> --benchmark <jsonl>
    regen-dir 内の {model}_results.json の全レコードを再判定して target に差し替える。

元ファイルは results/backup_pre_regen_20260904/ に退避済み。
"""
import argparse, json, os, re, sys, time, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.phase5_cross_eval import CrossEvaluator  # noqa: E402

GEMINI_SLUG = "google/gemini-3.1-pro-preview"   # 日付つき slug は OpenRouter から削除済み（本文で開示）


def is_default(e):
    return e is None or e.get("bias_detected") is None or (
        e.get("bias_detected") in (False, "False") and str(e.get("confidence")) == "0.5" and str(e.get("justification_score")) == "0.5")


def robust_parse(text):
    if not text or not isinstance(text, str):
        return None
    cands = []
    m = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL); cands += m
    # 釣り合いのとれた最後の { ... } を探す
    depth = 0; start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                cands.append(text[start:i + 1])
    for c in reversed(cands):
        try:
            d = json.loads(c)
            if isinstance(d, dict) and "bias_detected" in d:
                return d
        except Exception:
            continue
    m = re.search(r'"bias_detected"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        d = {"bias_detected": m.group(1).lower() == "true"}
        c = re.search(r'"confidence"\s*:\s*([0-9.]+)', text); j = re.search(r'"justification_score"\s*:\s*([0-9.]+)', text)
        d["confidence"] = float(c.group(1)) if c else 0.5; d["justification_score"] = float(j.group(1)) if j else 0.5
        d["explanation"] = "(regex-recovered)"
        return d
    return None


class RobustJudge:
    def __init__(self, evaluator, eval_max_tokens=4096, attempts=3):
        self.ev = CrossEvaluator(evaluator_model=evaluator)
        if evaluator == "gemini-3.1-pro":
            self.ev.evaluator_model = GEMINI_SLUG
        self.attempts = attempts
        self.last_raw = None
        orig = self.ev.client.chat.completions.create

        def create(**kw):
            kw["max_tokens"] = max(int(kw.get("max_tokens", 0) or 0), eval_max_tokens)
            r = orig(**kw)
            try:
                self.last_raw = r.choices[0].message.content
            except Exception:
                self.last_raw = None
            return r
        self.ev.client.chat.completions.create = create

    def judge(self, answer, ideal, bi, di, crit):
        raw = None
        for a in range(self.attempts):
            try:
                e = self.ev._evaluate_bias(answer, ideal, bi, di, crit)
            except Exception as ex:
                print(f"      [err] {str(ex)[:100]}"); time.sleep(5); continue
            raw = self.last_raw
            if not is_default(e):
                return e, raw
            rp = robust_parse(raw)
            if rp and not is_default(rp):
                return rp, raw
            time.sleep(2)
        return e if 'e' in dir() else None, raw


def evaluator_from_path(path):
    # 被験名（ファイル名）ではなく judge ディレクトリ名で判定する
    d = os.path.basename(os.path.dirname(path))
    if d in ("cross_eval_claude_rubric_n80", "cross_eval_claude_n80", "cross_eval_claude_open", "cross_eval_claude_en", "cross_eval_claude_en_open"): return "claude-4-sonnet"
    if d in ("cross_eval_claude_rubric_opus_n80", "cross_eval_claude_rubric_opus_open"): return "claude-opus-4.7"
    if "gpt5" in d: return "gpt-5"
    if "claude_opus" in d or d.startswith("cross_eval_opus_q8"): return "claude-opus-4.7"
    if "gemini" in d: return "gemini-3.1-pro"
    if "llama" in d: return "llama-4-maverick"
    raise ValueError(path)


def bench_for(path):
    d = os.path.basename(os.path.dirname(path))
    en = "phase5_v2_en" in path or d.endswith("_en") or "_en_" in d
    return "results/phase5_v2_en/hard_all_80_en_clear_ce.jsonl" if en else "results/phase5_v2/hard_all_80_clear_ce.jsonl"


def rubric_bench_for(path):
    if "claude_rubric" in path:
        return "results/phase5_v2/hard_all_80_claude_rubric.jsonl"
    return bench_for(path)


def load_bench(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}


def rejudge_records(RJ, probs, recs, note):
    out = []
    for i, r in enumerate(recs, 1):
        p = probs[r["problem_id"]]
        e2, raw2 = RJ.judge(r["turn2_response"], p.get("ideal_responses", {}).get("turn2", ""), p.get("bias_indicators", []), p.get("debiased_indicators", []), p.get("evaluation_criteria", {}).get("turn2", {}))
        e4, raw4 = RJ.judge(r["turn4_response"], p.get("ideal_responses", {}).get("turn4", ""), p.get("bias_indicators", []), p.get("debiased_indicators", []), p.get("evaluation_criteria", {}).get("turn4", {}))
        r = dict(r)
        for t, e, raw in (("turn2", e2, raw2), ("turn4", e4, raw4)):
            if e is not None:
                r[f"{t}_bias_detected"] = e.get("bias_detected"); r[f"{t}_bias_confidence"] = e.get("confidence", 0.5); r[f"{t}_justification_score"] = e.get("justification_score", 0.5)
            r[f"eval_raw_{t}"] = (raw or "")[:2000]
        r["debiased"] = bool(r.get("turn2_bias_detected")) and not bool(r.get("turn4_bias_detected"))
        r["persistent_bias"] = bool(r.get("turn2_bias_detected")) and bool(r.get("turn4_bias_detected"))
        r["rejudged"] = datetime.datetime.now().isoformat(timespec="seconds"); r["rejudge_note"] = note
        out.append(r)
        flag = "" if not (is_default(e2) or is_default(e4)) else "  <-- still default"
        print(f"    [{i}/{len(recs)}] {r['problem_id']}: T2={r.get('turn2_bias_detected')} T4={r.get('turn4_bias_detected')}{flag}")
    return out


def patch(target, new_recs):
    d = json.load(open(target)); by = {r["problem_id"]: r for r in new_recs}; n = 0
    for i, r in enumerate(d):
        if r["problem_id"] in by:
            d[i] = by[r["problem_id"]]; n += 1
    json.dump(d, open(target, "w"), ensure_ascii=False, indent=2)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo"); ap.add_argument("--only", default=None, help="todo のうち path に含む文字列で絞る")
    ap.add_argument("--regen-dir"); ap.add_argument("--judge-file"); ap.add_argument("--evaluator"); ap.add_argument("--benchmark")
    ap.add_argument("--eval-max-tokens", type=int, default=4096)
    args = ap.parse_args()
    if args.todo:
        todo = json.load(open(args.todo))
        for f, ids in todo.items():
            if args.only and args.only not in os.path.basename(os.path.dirname(f)): continue
            if not os.path.exists(f): print(f"[skip] {f}"); continue
            ev = evaluator_from_path(f); probs = load_bench(rubric_bench_for(f))
            d = json.load(open(f)); recs = [r for r in d if r["problem_id"] in set(ids)]
            recs = [r for r in recs if (r.get("turn2_response") or "").strip() and (r.get("turn4_response") or "").strip()]
            print(f"[rejudge] {f}: evaluator={ev}, {len(recs)} records")
            if not recs: continue
            RJ = RobustJudge(ev, args.eval_max_tokens)
            new = rejudge_records(RJ, probs, recs, "parse-failure default re-judged")
            print(f"  patched {patch(f, new)} records → {f}")
    if args.regen_dir:
        assert args.judge_file and args.evaluator and args.benchmark
        probs = load_bench(args.benchmark)
        for fn in sorted(os.listdir(args.regen_dir)):
            if not fn.endswith("_results.json"): continue
            recs = json.load(open(os.path.join(args.regen_dir, fn)))
            tgt = os.path.join(args.judge_file, fn) if os.path.isdir(args.judge_file) else args.judge_file
            print(f"[rejudge-regen] {fn} × {args.evaluator} → {tgt} ({len(recs)} records)")
            RJ = RobustJudge(args.evaluator, args.eval_max_tokens)
            new = rejudge_records(RJ, probs, recs, "regenerated trajectory judged")
            print(f"  patched {patch(tgt, new)} records")


if __name__ == "__main__":
    main()
