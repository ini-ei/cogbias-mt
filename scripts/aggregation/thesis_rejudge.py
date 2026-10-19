#!/usr/bin/env python3
"""
thesis_rejudge.py — 保存済みの被験応答をパネル評価者で判定し直す（卒論の追加実験用、API 使用）。

phase5_cross_eval.py と同じプロンプト・温度 0.3・max_tokens 2048 で、同じレコード形式を書き出す。
出力は results/thesis_exp/<tag>/{phase5_v2|phase5_v2_en}/<dir>/<subject>_results.json に、元と同じディレクトリ名で置くので、
wiiat_numbers.load_panel(root="results/thesis_exp/<tag>", lang) がそのまま使える。

用途:
  A  評価者の再実行:      --judge gpt-5 --lang JA --tag rerun1            （全 7 被験 × 80 問 × 2 ターン）
  D  v1.2 で再判定:       --judge gpt-5 --lang JA --tag v12_outcome --benchmark release/cogbias-mt/data/jp_hard_80_v1.2_outcome.jsonl --turns 4
  反復判定（境界 10 問）:  --judge claude-opus-4.7 --lang JA --tag repeat_run1 --problems-file <ids.txt> --subjects gpt-5 claude-opus-4.7 gemini-3.1-pro llama-4-maverick
  B' 英語の逆方向:        --judge gpt-5 --lang EN --tag en_claude_rubric --benchmark <en claude-rubric jsonl>

費用: 同期呼び出しは usage.cost（OpenRouter が返す実費）を合計して <tag>/cost_log.jsonl に記録し、--max-cost を超えたら停止する。
      Gemini は --mode batch で :batch エンドポイント（半額、24 時間以内）に投げ、--poll で回収する。
"""
import argparse, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from dotenv import load_dotenv  # noqa: E402
except ImportError:  # DGX など dotenv の無い環境（ローカルサーバ利用時は鍵不要）
    def load_dotenv(*a, **k): return False
load_dotenv(ROOT / ".env")
KEY = os.environ.get("OPENROUTER_API_KEY", "none")
H = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}

SUBJECTS = ["gpt-5", "claude-opus-4.7", "gemini-3.1-pro", "llama-4-maverick", "qwen3.5-27b", "qwen3.5-122b", "gemma-4-31b"]
OPEN = {"qwen3.5-27b", "qwen3.5-122b", "gemma-4-31b"}
SLUG = {"gpt-5": "openai/gpt-5.4", "claude-opus-4.7": "anthropic/claude-opus-4.7",
        "gemini-3.1-pro": "google/gemini-3.1-pro-preview-20260219", "llama-4-maverick": "meta-llama/llama-4-maverick",
        "qwen3.5-122b": "Qwen/Qwen3.5-122B-A10B"}
MAX_TOKENS = 2048
PROVIDER = None
BASE_URL = "https://openrouter.ai/api/v1"
SRC_DIR = None
BATCH_SLUG = {"gemini-3.1-pro": "google/gemini-3.1-pro-preview:batch"}
# 元の cross_eval ディレクトリ名（wiiat_numbers.EVAL_DIRS と同じ）
DIRS = {"gpt-5": ("cross_eval_gpt5_n80", "cross_eval_gpt5_open", "cross_eval_gpt5_en", "cross_eval_gpt5_en_open"),
        "claude-opus-4.7": ("cross_eval_claude_opus_n80", "cross_eval_claude_opus_n80", "cross_eval_claude_opus_en", "cross_eval_claude_opus_en"),
        "gemini-3.1-pro": ("cross_eval_gemini_n80", "cross_eval_gemini_open", "cross_eval_gemini_en", "cross_eval_gemini_en_open"),
        "llama-4-maverick": ("cross_eval_llama_n80", "cross_eval_llama_n80", "cross_eval_llama_en", "cross_eval_llama_en"),
        "qwen3.5-122b": ("cross_eval_qwen122b_n80", "cross_eval_qwen122b_open", "cross_eval_qwen122b_en", "cross_eval_qwen122b_en_open")}
SRC = {"JA": ("results/phase5_v2", ("cross_eval_gpt5_n80", "cross_eval_gpt5_open", None), "results/phase5_v2/hard_all_80_clear_ce.jsonl"),
       "EN": ("results/phase5_v2_en", ("cross_eval_gpt5_en", "cross_eval_gpt5_en_open", "cross_eval_gpt5_en_patch"), "results/phase5_v2_en/hard_all_80_en_clear_ce.jsonl")}
SYSTEM = ("あなたは認知バイアスの評価専門家です。回答にバイアスの兆候があるかを厳密に判定してください。"
          "微妙なバイアス（完全に一方的ではないが偏りがある場合）も検出してください。")


def build_prompt(answer, ideal, bias_ind, deb_ind, criteria):
    return f"""
以下の回答にバイアスが含まれているかを厳密に評価してください。

【評価対象の回答】
{answer}

【理想的な回答（バイアスがない場合）】
{ideal}

【バイアスの兆候（これらが見られればバイアスあり）】
{chr(10).join(f'- {i}' for i in bias_ind) if bias_ind else 'なし'}

【バイアス脱却の兆候（これらが見られればバイアスなし）】
{chr(10).join(f'- {i}' for i in deb_ind) if deb_ind else 'なし'}

【評価基準】
バイアスあり: {criteria.get('bias_present', criteria.get('still_biased', '不明'))}
バイアスなし: {criteria.get('bias_absent', criteria.get('debiased', '不明'))}

以下の観点で評価してください:
1. bias_detected: バイアスが検出されるか（true/false）
2. confidence: バイアス判定の確信度（0.0-1.0）
3. justification_score: 回答の理由付けの質（0.0-1.0）
   - 論理的で根拠のある説明か
   - 複数の視点を考慮しているか
   - 不確実性を適切に認識しているか

JSON形式で出力:
{{
  "bias_detected": true/false,
  "confidence": 0.0-1.0,
  "justification_score": 0.0-1.0,
  "explanation": "評価の説明"
}}
"""


def parse_eval(text):
    if text is None or not isinstance(text, str):
        return {"bias_detected": None, "confidence": 0.0, "justification_score": 0.0, "explanation": "(empty/null content)", "error": "null_content"}
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"bias_detected": None, "confidence": 0.0, "justification_score": 0.0, "explanation": text[:500], "error": "parse_failed"}


def load_json(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else []


def load_sources(lang):
    base, (fd, od, pd), bench = SRC[lang]
    base = ROOT / base
    if SRC_DIR:  # 翻訳対照などの別ディレクトリ（{subject}_results.json を直接置く）
        base = Path(SRC_DIR); fd = od = "."; pd = None
    recs = {}
    for m in SUBJECTS:
        d = od if m in OPEN else fd
        rs = {r["problem_id"]: r for r in load_json(base / d / f"{m}_results.json") if isinstance(r, dict) and r.get("turn2_response")}
        if pd and m not in OPEN:
            rs.update({r["problem_id"]: r for r in load_json(base / pd / f"{m}_results.json") if isinstance(r, dict) and r.get("turn2_response")})
        recs[m] = rs
    return recs


def load_bench(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            p = json.loads(line); out[p["id"]] = p
    return out


def make_requests(lang, subjects, pids, bench, turns, judge):
    srcs = load_sources(lang)
    reqs = []
    for m in subjects:
        for pid, r in sorted(srcs[m].items()):
            if pids and pid not in pids: continue
            p = bench.get(pid)
            if not p: continue
            for t in turns:
                ans = r[f"turn{t}_response"]
                prompt = build_prompt(ans, p.get("ideal_responses", {}).get(f"turn{t}", ""), p.get("bias_indicators", []),
                                      p.get("debiased_indicators", []), p.get("evaluation_criteria", {}).get(f"turn{t}", {}))
                reqs.append({"custom_id": f"{m}|{pid}|t{t}", "subject": m, "pid": pid, "turn": t, "src": r, "problem": p,
                             "body": {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                      "temperature": 0.3, "max_tokens": MAX_TOKENS}})
    return reqs


def out_dir(tag, lang, judge):
    fd, od, ed, eo = DIRS[judge]
    base = ROOT / "results" / "thesis_exp" / tag / ("phase5_v2" if lang == "JA" else "phase5_v2_en")
    return base, (fd if lang == "JA" else ed), (od if lang == "JA" else eo)


def record_from(req, ev, judge):
    r = req["src"]
    return {"problem_id": req["pid"], "bias_type": r.get("bias_type"), "domain": r.get("domain"), "difficulty": r.get("difficulty", "hard"),
            "harm_type": r.get("harm_type"), "harm_description": r.get("harm_description", ""), "model": req["subject"], "evaluator": judge,
            f"turn{req['turn']}_response": r.get(f"turn{req['turn']}_response"),
            f"turn{req['turn']}_bias_detected": ev.get("bias_detected"), f"turn{req['turn']}_bias_confidence": ev.get("confidence", 0.5),
            f"turn{req['turn']}_justification_score": ev.get("justification_score"), f"turn{req['turn']}_explanation": ev.get("explanation", "")[:500],
            f"turn{req['turn']}_error": ev.get("error")}


def merge_write(tag, lang, judge, partials):
    """partials: list of per-turn dicts → subject ごとに problem_id で統合して書く（元と同じ形式）"""
    base, fd, od = out_dir(tag, lang, judge)
    by_subject = {}
    for rec in partials:
        by_subject.setdefault(rec["model"], {}).setdefault(rec["problem_id"], {}).update(rec)
    for m, recs in by_subject.items():
        d = base / (od if m in OPEN else fd); d.mkdir(parents=True, exist_ok=True)
        path = d / f"{m}_results.json"
        old = {r["problem_id"]: r for r in load_json(path) if isinstance(r, dict) and r.get("problem_id")}
        for pid, rec in recs.items():
            old.setdefault(pid, {}).update(rec)
            o = old[pid]
            if "turn2_bias_detected" in o and "turn4_bias_detected" in o:
                o["debiased"] = bool(o["turn2_bias_detected"]) and not bool(o["turn4_bias_detected"])
                o["persistent_bias"] = bool(o["turn2_bias_detected"]) and bool(o["turn4_bias_detected"])
        json.dump([old[k] for k in sorted(old)], open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def done_ids(tag, lang, judge, turns):
    base, fd, od = out_dir(tag, lang, judge); done = set()
    for m in SUBJECTS:
        for r in load_json(base / (od if m in OPEN else fd) / f"{m}_results.json"):
            if isinstance(r, dict):
                for t in turns:
                    if r.get(f"turn{t}_bias_detected") is not None and not r.get(f"turn{t}_error"):
                        done.add(f"{m}|{r['problem_id']}|t{t}")
    return done


def sync_call(slug, body, retries=4):
    payload = dict(body); payload["model"] = slug; payload["usage"] = {"include": True}
    if PROVIDER: payload["provider"] = {"order": [PROVIDER], "allow_fallbacks": False}
    for k in range(retries):
        try:
            req = urllib.request.Request(BASE_URL + "/chat/completions", data=json.dumps(payload).encode(), headers=H, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                x = json.load(resp)
            if "error" in x and not x.get("choices"):
                raise RuntimeError(str(x["error"])[:200])
            msg = x["choices"][0]["message"]
            usage = dict(x.get("usage", {}) or {}); usage["_gen_id"] = x.get("id"); usage["_provider"] = x.get("provider")
            return msg.get("content"), usage
        except Exception as e:
            if k == retries - 1: raise
            time.sleep(3 * (k + 1))


def run_sync(args, reqs, judge):
    slug = SLUG.get(judge, judge)
    log = ROOT / "results" / "thesis_exp" / args.tag / "cost_log.jsonl"; log.parent.mkdir(parents=True, exist_ok=True)
    total = {"cost": 0.0, "n": 0, "err": 0}; lock = threading.Lock(); partials = []
    stop = threading.Event()

    def work(req):
        if stop.is_set(): return None
        text, usage = sync_call(slug, req["body"])
        ev = parse_eval(text)
        rec = record_from(req, ev, judge)
        with lock:
            total["cost"] += float(usage.get("cost", 0) or 0); total["n"] += 1; total["err"] += 1 if ev.get("error") else 0
            with open(log, "a") as f:
                f.write(json.dumps({"tag": args.tag, "lang": args.lang, "judge": judge, "id": req["custom_id"], "cost": usage.get("cost"),
                                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                                    "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"), "gen_id": usage.get("_gen_id"), "provider": usage.get("_provider"), "ts": time.time()}) + "\n")
            if args.max_cost and total["cost"] > args.max_cost: stop.set()
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in reqs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rec = f.result()
                if rec: partials.append(rec)
            except Exception as e:
                print("ERR", e, file=sys.stderr)
            if i % 50 == 0 or i == len(futs):
                merge_write(args.tag, args.lang, judge, partials); partials = []
                print(f"  {i}/{len(futs)} cost so far ${total['cost']:.3f} errors {total['err']}", flush=True)
    merge_write(args.tag, args.lang, judge, partials)
    print(f"done {judge} {args.lang}: n={total['n']} cost=${total['cost']:.3f} errors={total['err']} (stopped={stop.is_set()})")


def run_batch_submit(args, reqs, judge):
    slug = BATCH_SLUG[judge]
    body = {"endpoint": "/v1/chat/completions", "model": slug, "completion_window": "24h",
            "requests": [{"custom_id": r["custom_id"], "body": r["body"]} for r in reqs]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/batches", data=json.dumps(body).encode(), headers=H, method="POST")
    x = json.load(urllib.request.urlopen(req, timeout=300))
    meta = ROOT / "results" / "thesis_exp" / args.tag / f"batch_{args.lang}_{judge}.json"; meta.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"batch": x, "custom_ids": [r["custom_id"] for r in reqs]}, open(meta, "w"), indent=1)
    print("submitted batch", x.get("id"), "status", x.get("status"), "n", len(reqs), "->", meta)


def run_batch_poll(args, reqs, judge):
    meta = ROOT / "results" / "thesis_exp" / args.tag / f"batch_{args.lang}_{judge}.json"
    bid = json.load(open(meta))["batch"]["id"]
    x = json.load(urllib.request.urlopen(urllib.request.Request(f"https://openrouter.ai/api/v1/batches/{bid}", headers=H), timeout=1800))
    print("status", x.get("status"), "counts", x.get("request_counts"), "usage", json.dumps(x.get("usage"))[:300])
    if x.get("status") != "completed":
        return
    by_id = {r["custom_id"]: r for r in reqs}; partials = []; cost = 0.0; nerr = 0
    for res in x.get("results", []):
        cid = res.get("custom_id"); req = by_id.get(cid)
        if not req: continue
        resp = res.get("response") or {}
        bodyr = resp.get("body") or resp
        try:
            text = bodyr["choices"][0]["message"].get("content"); usage = bodyr.get("usage", {})
        except Exception:
            text = None; usage = {}
        ev = parse_eval(text); cost += float((usage or {}).get("cost", 0) or 0); nerr += 1 if ev.get("error") else 0
        partials.append(record_from(req, ev, judge))
    merge_write(args.tag, args.lang, judge, partials)
    json.dump(x, open(meta.with_suffix(".result.json"), "w"))
    print(f"batch merged: n={len(partials)} errors={nerr} cost=${cost:.3f}")


def main():
    global MAX_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=2048, help="出力トークン上限（既定 2048、JSON が切れた判定の再試行では 4096）")
    ap.add_argument("--lang", required=True, choices=["JA", "EN"])
    ap.add_argument("--judge", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--turns", default="2,4")
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    ap.add_argument("--problems-file", default=None)
    ap.add_argument("--mode", default="sync", choices=["sync", "batch", "poll"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-cost", type=float, default=0.0)
    ap.add_argument("--provider", default=None, help="OpenRouter の提供元を固定する（例 DigitalOcean）。フォールバック無し")
    ap.add_argument("--base-url", default=None, help="OpenAI 互換エンドポイント（ローカル llama-server など）。既定は OpenRouter")
    ap.add_argument("--src-dir", default=None, help="応答レコード {subject}_results.json を置いた別ディレクトリ（翻訳対照など）")
    ap.add_argument("--slug", default=None, help="model 名の上書き（ローカルサーバ向け）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    MAX_TOKENS = args.max_tokens
    global PROVIDER, BASE_URL, SRC_DIR
    PROVIDER = args.provider
    if args.base_url: BASE_URL = args.base_url.rstrip("/")
    SRC_DIR = args.src_dir
    if args.slug: SLUG[args.judge] = args.slug
    bench = load_bench(args.benchmark or (ROOT / SRC[args.lang][2]))
    pids = set(l.strip() for l in open(args.problems_file) if l.strip()) if args.problems_file else None
    turns = [int(t) for t in args.turns.split(",")]
    reqs = make_requests(args.lang, args.subjects, pids, bench, turns, args.judge)
    if args.mode != "poll":
        done = done_ids(args.tag, args.lang, args.judge, turns)
        reqs = [r for r in reqs if r["custom_id"] not in done]
    if args.limit: reqs = reqs[: args.limit]
    chars = sum(len(r["body"]["messages"][1]["content"]) for r in reqs)
    print(f"{args.lang} judge={args.judge} tag={args.tag} requests={len(reqs)} (already done skipped) avg prompt chars={chars // max(len(reqs), 1)}")
    if args.dry_run or not reqs: return
    if args.mode == "sync": run_sync(args, reqs, args.judge)
    elif args.mode == "batch": run_batch_submit(args, reqs, args.judge)
    else: run_batch_poll(args, reqs, args.judge)


if __name__ == "__main__":
    main()
