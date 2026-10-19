#!/usr/bin/env python3
"""
thesis_e5_translate.py — 翻訳対照セット: 日本語 20 問（バイアス 4 種 × ドメイン 5 の各 1 問）を英訳し、同じ ID で JSONL に書く。
  .venv/bin/python scripts/thesis_e5_translate.py --out results/thesis_exp/e5_trans/jp20_to_en.jsonl
翻訳は Opus 4.7（OpenRouter）。構造・ID・数値・固有名詞は保ち、文章フィールドだけを英語にする。
"""
import argparse, json, os, re, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for line in open(ROOT / ".env"):
    if line.startswith("OPENROUTER_API_KEY="): os.environ.setdefault("OPENROUTER_API_KEY", line.split("=", 1)[1].strip().strip('"'))
H = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"}
MODEL = "anthropic/claude-opus-4.7"
PROMPT = """You are translating a Japanese cognitive-bias benchmark item into natural English for use with English-speaking language models.
Translate EVERY Japanese string value in the JSON below into fluent, natural English. Keep the JSON structure, all keys, the "id", numbers, amounts, dates, names and list lengths exactly as they are. Do not add or drop information, do not soften or strengthen the persuasive framing, and do not add commentary. Keep currency as in the source (e.g. 300万円 → 3 million yen). Return ONLY the translated JSON object, no code fences.

{json}"""


def select20(bench):
    seen = {}
    for p in bench:
        key = (p["bias_type"], p["domain"])
        if key not in seen: seen[key] = p
    return [seen[k] for k in sorted(seen)]


def translate(p):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.replace("{json}", json.dumps(p, ensure_ascii=False, indent=1))}],
            "temperature": 0.2, "max_tokens": 12000, "usage": {"include": True}}
    for k in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(), headers=H, method="POST")
            x = json.load(urllib.request.urlopen(req, timeout=300))
            text = x["choices"][0]["message"]["content"]; m = re.search(r"\{.*\}", text, re.S)
            out = json.loads(m.group()); out["id"] = p["id"]
            return out, float(x.get("usage", {}).get("cost", 0) or 0)
        except Exception as e:
            print("retry", k, str(e)[:100]); time.sleep(5)
    raise RuntimeError("translation failed " + p["id"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default=str(ROOT / "results/phase5_v2/hard_all_80_clear_ce.jsonl"))
    ap.add_argument("--out", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    bench = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
    sel = select20(bench); print("selected", len(sel), [p["id"] for p in sel])
    if a.dry_run: return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")} if Path(a.out).exists() else set()
    total = 0.0
    with open(a.out, "a", encoding="utf-8") as f:
        for p in sel:
            if p["id"] in done: continue
            out, cost = translate(p); total += cost
            ja_keys = set(json.dumps(p, ensure_ascii=False)) ; 
            f.write(json.dumps(out, ensure_ascii=False) + "\n"); f.flush()
            print(p["id"], f"${cost:.3f}", "turns", len(out.get("turns", [])), "ja chars left", len(re.findall(r"[぀-ヿ一-鿿]", json.dumps(out, ensure_ascii=False))))
    print(f"total ${total:.2f}")


if __name__ == "__main__":
    main()
