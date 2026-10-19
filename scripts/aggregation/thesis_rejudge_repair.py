#!/usr/bin/env python3
"""
thesis_rejudge_repair.py — thesis_rejudge.py の出力で JSON として読めなかった判定のうち、
先頭に bias_detected / confidence / justification_score が書かれているもの（JSON が途中で切れた場合）を
正規表現で読み直して記録を直す。読めないものはエラーのまま残す（同じコマンドで再判定される）。
  python scripts/thesis_rejudge_repair.py results/thesis_exp/rerun1 results/thesis_exp/repeat_run1 ...
"""
import json, re, sys, glob

def fix(text):
    m = re.search(r'"bias_detected"\s*:\s*(true|false)', text)
    if not m: return None
    out = {"bias_detected": m.group(1) == "true"}
    for k in ("confidence", "justification_score"):
        mm = re.search(rf'"{k}"\s*:\s*([0-9.]+)', text)
        out[k] = float(mm.group(1)) if mm else 0.5
    me = re.search(r'"explanation"\s*:\s*"(.*)', text, re.DOTALL)
    out["explanation"] = (me.group(1) if me else text)[:500]
    return out

for root in sys.argv[1:]:
    nfix = nleft = 0
    for f in glob.glob(f"{root}/phase5_v2*/cross_eval_*/*_results.json"):
        recs = json.load(open(f, encoding="utf-8")); changed = False
        for r in recs:
            for t in ("turn2", "turn4"):
                if r.get(f"{t}_error") != "parse_failed": continue
                ev = fix(r.get(f"{t}_explanation") or "")
                if ev is None: nleft += 1; continue
                r[f"{t}_bias_detected"] = ev["bias_detected"]; r[f"{t}_bias_confidence"] = ev["confidence"]
                r[f"{t}_justification_score"] = ev["justification_score"]; r[f"{t}_explanation"] = ev["explanation"]
                r[f"{t}_error"] = None; r[f"{t}_repaired"] = "regex_from_truncated_json"; changed = True; nfix += 1
            if "turn2_bias_detected" in r and "turn4_bias_detected" in r and r["turn2_bias_detected"] is not None and r["turn4_bias_detected"] is not None:
                r["debiased"] = bool(r["turn2_bias_detected"]) and not bool(r["turn4_bias_detected"])
                r["persistent_bias"] = bool(r["turn2_bias_detected"]) and bool(r["turn4_bias_detected"])
        if changed: json.dump(recs, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{root}: repaired {nfix}, still unreadable {nleft}")
