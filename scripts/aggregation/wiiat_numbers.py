#!/usr/bin/env python3
"""
wiiat_numbers.py — WI-IAT 論文 (camera-ready) 本文・表・図に載る数値を、判定レコードから一括再計算する。
  python scripts/wiiat_numbers.py                      # 現行データ (results/)
  python scripts/wiiat_numbers.py --root results/backup_pre_regen_20260904/results   # 提出版データで定義を検証
定義は本文どおり:
  global T2 strictness = 各 judge の、自分以外 6 被験 (480 軌跡, JA) に対する初回 (T2) Bias 率の平均
  pair-specific marginal = ペアが共有する 560 レコード上での T4 bias-call 率
  Table II = T4 判定が食い違うレコード数、優勢側 share、優勢/劣勢の数
"""
import argparse, json, math, os
from itertools import combinations
from collections import defaultdict
import numpy as np

SUBJECTS = ["gpt-5", "claude-opus-4.7", "gemini-3.1-pro", "llama-4-maverick", "qwen3.5-27b", "qwen3.5-122b", "gemma-4-31b"]
OPEN = {"qwen3.5-27b", "qwen3.5-122b", "gemma-4-31b"}
FRONTIER = [m for m in SUBJECTS if m not in OPEN]
SELF_OF = {"GPT-5.4": "gpt-5", "Opus4.7": "claude-opus-4.7", "Gemini3.1": "gemini-3.1-pro", "LLaMA4": "llama-4-maverick"}
SHORT = {"GPT-5.4": "G", "Opus4.7": "O", "Gemini3.1": "Ge", "LLaMA4": "L"}
EVAL_DIRS = {
    "GPT-5.4": ("cross_eval_gpt5_n80", "cross_eval_gpt5_open", "cross_eval_gpt5_en", "cross_eval_gpt5_en_open", "cross_eval_gpt5_en_patch"),
    "Opus4.7": ("cross_eval_claude_opus_n80", "cross_eval_claude_opus_n80", "cross_eval_claude_opus_en", "cross_eval_claude_opus_en", None),
    "Gemini3.1": ("cross_eval_gemini_n80", "cross_eval_gemini_open", "cross_eval_gemini_en", "cross_eval_gemini_en_open", "cross_eval_gemini_en_patch"),
    "LLaMA4": ("cross_eval_llama_n80", "cross_eval_llama_n80", "cross_eval_llama_en", "cross_eval_llama_en", None),
}
EVALS = list(EVAL_DIRS); PAIRS = [(a, b) for i, a in enumerate(EVALS) for b in EVALS[i + 1:]]


def load(path):
    if not os.path.exists(path): return {}
    out = {}
    for r in json.load(open(path)):
        if isinstance(r, dict) and r.get("problem_id") and (r.get("turn2_bias_confidence") is not None or r.get("turn4_bias_confidence") is not None):
            out[r["problem_id"]] = (bool(r.get("turn2_bias_detected")), bool(r.get("turn4_bias_detected")))
    return out


def load_dir_set(root, lang, dirs):
    """dirs = (frontier_dir, open_dir, patch_dir|None) → subject -> pid -> (t2,t4)"""
    base = f"{root}/phase5_v2" if lang == "JA" else f"{root}/phase5_v2_en"
    f_d, o_d, p_d = dirs
    out = {}
    for m in SUBJECTS:
        recs = load(f"{base}/{o_d if m in OPEN else f_d}/{m}_results.json")
        if p_d and m not in OPEN: recs.update(load(f"{base}/{p_d}/{m}_results.json"))
        out[m] = recs
    return out


def load_panel(root, lang):
    J = {}
    for ev, (jf, jo, ef, eo, ep) in EVAL_DIRS.items():
        J[ev] = load_dir_set(root, lang, (jf, jo, None) if lang == "JA" else (ef, eo, ep))
    return J


def load_self(root, lang):
    return load_dir_set(root, lang, ("full_n80", "full_n80_open", None) if lang == "JA" else ("full_n80_en", "full_n80_en_open", "full_n80_en_patch"))


def br(recs, t, ids=None):
    v = [x[t] for p, x in recs.items() if ids is None or p in ids]
    return 100 * np.mean(v) if v else float("nan"), len(v)


def strictness(J, ev, t=0, ids=None):
    vals = []
    for m in SUBJECTS:
        if m == SELF_OF[ev]: continue
        vals += [x[t] for p, x in J[ev][m].items() if ids is None or p in ids]
    return 100 * np.mean(vals), len(vals)


def pair_cells(J, a, b, t, subjects=SUBJECTS, ids=None):
    """→ dict(n, a_only, b_only, pa, pb, both, neither)"""
    ao = bo = both = nei = 0
    for m in subjects:
        A, B = J[a].get(m, {}), J[b].get(m, {})
        for p in set(A) & set(B):
            if ids is not None and p not in ids: continue
            va, vb = A[p][t], B[p][t]
            ao += va and not vb; bo += vb and not va; both += va and vb; nei += (not va) and (not vb)
    n = ao + bo + both + nei
    pa = (ao + both) / n if n else float("nan"); pb = (bo + both) / n if n else float("nan")
    return dict(n=n, a_only=ao, b_only=bo, both=both, neither=nei, pa=pa, pb=pb)


def baseline(pa, pb):
    x = pa * (1 - pb); y = (1 - pa) * pb
    return x / (x + y) if x + y else float("nan")


def kappa(c):
    n = c["n"]; po = (c["both"] + c["neither"]) / n
    pe = c["pa"] * c["pb"] + (1 - c["pa"]) * (1 - c["pb"])
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cp_halfwidth(k, n):
    """Clopper–Pearson 95% CI half-width (pp) for k/n"""
    from scipy.stats import beta
    if n == 0: return float("nan")
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return 100 * (hi - lo) / 2


def fmt_pair(c, a, b):
    dom, opp = (a, b) if c["a_only"] >= c["b_only"] else (b, a)
    d, o = max(c["a_only"], c["b_only"]), min(c["a_only"], c["b_only"])
    n = d + o
    return dom, d, o, (100 * d / n if n else float("nan"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results"); ap.add_argument("--json", default=None)
    args = ap.parse_args(); R = args.root
    out = {}
    J = {L: load_panel(R, L) for L in ("JA", "EN")}; S = {L: load_self(R, L) for L in ("JA", "EN")}
    for L in ("JA", "EN"):
        print(f"[{L}] " + " ".join(f"{ev}:{sum(len(v) for v in J[L][ev].values())}" for ev in EVALS) + "  self:" + str(sum(len(v) for v in S[L].values())))

    print("\n=== Fig.3a: global T2 strictness (JA, 6 non-self subjects) ===")
    st = {}
    for L in ("JA", "EN"):
        st[L] = {ev: strictness(J[L], ev) for ev in EVALS}
        print(f"  {L}: " + "  ".join(f"{ev} {v[0]:.1f}% (n={v[1]})" for ev, v in st[L].items()))
    order = sorted(EVALS, key=lambda e: -st["JA"][e][0])
    print(f"  JA spread = {st['JA'][order[0]][0] - st['JA'][order[-1]][0]:.1f}pp  order: {' > '.join(order)}")
    out["strictness"] = {L: {ev: round(v[0], 2) for ev, v in st[L].items()} for L in st}

    print("\n=== Fig.3b: Opus 4.7 subject (JA) T2 BR by judge ===")
    opus = {ev: br(J["JA"][ev]["claude-opus-4.7"], 0)[0] for ev in EVALS}; opus["self"] = br(S["JA"]["claude-opus-4.7"], 0)[0]
    print("  " + "  ".join(f"{k} {v:.1f}%" for k, v in opus.items()) + f"   LLaMA→GPT ratio {opus['GPT-5.4'] / opus['LLaMA4']:.1f}x, diff {opus['GPT-5.4'] - opus['LLaMA4']:.1f}pp")
    out["opus_subject"] = {k: round(v, 2) for k, v in opus.items()}

    print("\n=== GPT-5.4 subject (JA): panel swing ===")
    g = {ev: br(J["JA"][ev]["gpt-5"], 0)[0] for ev in EVALS if ev != "GPT-5.4"}; g["self"] = br(S["JA"]["gpt-5"], 0)[0]
    cross = [v for k, v in g.items() if k != "self"]
    panels = {"+".join(c): np.mean([g[x] for x in c]) for c in combinations(g, 3)}
    print("  " + "  ".join(f"{k} {v:.1f}%" for k, v in g.items()))
    print(f"  cross judges span {min(cross):.1f}–{max(cross):.1f}%; 3-judge subsets {min(panels.values()):.1f}–{max(panels.values()):.1f}% swing {max(panels.values()) - min(panels.values()):.1f}pp")
    out["gpt_subject"] = {k: round(v, 2) for k, v in g.items()}; out["gpt_panels"] = {k: round(v, 2) for k, v in panels.items()}

    print("\n=== Table II: T4 disagreement direction (560 shared per pair) + baseline/residual/kappa ===")
    out["table2"] = {}; resid = []; kap = []; t2_share = []
    for a, b in PAIRS:
        row = []
        for L in ("JA", "EN"):
            c = pair_cells(J[L], a, b, 1); dom, d, o, sh = fmt_pair(c, a, b)
            pa, pb = c["pa"], c["pb"]; base = 100 * (baseline(pa, pb) if dom == a else baseline(pb, pa)); res = sh - base
            hi_marg = a if pa > pb else b; k = kappa(c)
            resid.append(res); kap.append(k)
            c2 = pair_cells(J[L], a, b, 0); dom2, d2, o2, sh2 = fmt_pair(c2, a, b); t2_share.append(sh2)
            row.append(f"{L}: {d + o} ({sh:.1f}%, {d}/{o}, {SHORT[dom]}) n={c['n']} pA={100 * pa:.1f} pB={100 * pb:.1f} base={base:.1f} resid={res:+.1f} marg-side={'ok' if hi_marg == dom else 'MISMATCH'} κ={k:.2f} | T2: {d2 + o2} ({sh2:.1f}%, {SHORT[dom2]})")
            out["table2"][f"{a}-{b}_{L}"] = dict(n_dis=d + o, share=round(sh, 2), dom=dom, d=d, o=o, pa=round(100 * pa, 2), pb=round(100 * pb, 2), base=round(base, 2), resid=round(res, 2), kappa=round(k, 3), t2_n=d2 + o2, t2_share=round(sh2, 2), t2_dom=dom2)
        print(f"  {a}-{b}:\n    " + "\n    ".join(row))
    print(f"  residual range {min(resid):+.1f}..{max(resid):+.1f}pp; κ range {min(kap):.2f}..{max(kap):.2f}; T2 share ≥78%: {sum(s >= 78 for s in t2_share)}/12")
    keep = sum(out["table2"][f"{a}-{b}_JA"]["dom"] == out["table2"][f"{a}-{b}_EN"]["dom"] for a, b in PAIRS)
    print(f"  pairs with same dominant judge in JA and EN: {keep}/6; flips: " + ", ".join(f"{a}-{b}" for a, b in PAIRS if out['table2'][f'{a}-{b}_JA']['dom'] != out['table2'][f'{a}-{b}_EN']['dom']))

    print("\n=== Strata (GPT-5.4–Opus4.7, T4): 20 bias×domain cells ===")
    out["strata"] = {}
    for L in ("JA", "EN"):
        cells = defaultdict(lambda: [0, 0])
        A, B = J[L]["GPT-5.4"], J[L]["Opus4.7"]
        for m in SUBJECTS:
            for p in set(A[m]) & set(B[m]):
                parts = p.split("-"); key = (parts[0], parts[1])
                va, vb = A[m][p][1], B[m][p][1]
                if va and not vb: cells[key][0] += 1
                elif vb and not va: cells[key][1] += 1
        gpt_side = sum(1 for v in cells.values() if v[0] > v[1]); ties = sum(1 for v in cells.values() if v[0] == v[1]); opus_side = sum(1 for v in cells.values() if v[1] > v[0])
        ns = [v[0] + v[1] for v in cells.values()]; hw = [cp_halfwidth(max(v), v[0] + v[1]) for v in cells.values()]
        print(f"  {L}: strata={len(cells)} GPT-side {gpt_side}, ties {ties}, Opus-side {opus_side}; disagreements per stratum {min(ns)}–{max(ns)}; CP half-width ±{min(hw):.0f}–{max(hw):.0f}pp")
        out["strata"][L] = dict(gpt=gpt_side, tie=ties, opus=opus_side, n_range=[min(ns), max(ns)], hw_range=[round(min(hw), 1), round(max(hw), 1)])

    print("\n=== Generator version (JA): gpt-5-authored 40 vs gpt-5.4-authored 40 ===")
    p40 = f"{R}/phase5_v2/hard_all_40_clear_ce.jsonl"
    if os.path.exists(p40):
        orig = {json.loads(l)["id"] for l in open(p40) if l.strip()}
        allids = {p for ev in EVALS for m in SUBJECTS for p in J["JA"][ev][m]}; extra = allids - orig
        out["genver"] = {}
        for ev in EVALS:
            so, se = strictness(J["JA"], ev, 0, orig)[0], strictness(J["JA"], ev, 0, extra)[0]
            print(f"  {ev}: gpt-5-authored {so:.1f}%  gpt-5.4-authored {se:.1f}%  Δ {se - so:+.1f}pp"); out["genver"][ev] = [round(so, 2), round(se, 2)]
        for ids, nm in ((orig, "gpt-5-authored"), (extra, "gpt-5.4-authored")):
            c = pair_cells(J["JA"], "GPT-5.4", "Opus4.7", 1, ids=ids); dom, d, o, sh = fmt_pair(c, "GPT-5.4", "Opus4.7")
            print(f"  GPT–Opus share on {nm}: {sh:.1f}% ({d}/{o})"); out["genver"][f"share_{nm}"] = round(sh, 2)

    print("\n=== Exp B: Claude rubric × GPT-5.4 judge (JA) ===")
    cr = {}
    for m in SUBJECTS:
        d = "cross_eval_gpt5_claude_rubric_open" if m in OPEN else "cross_eval_gpt5_claude_rubric_n80"
        cr[m] = load(f"{R}/rebuttal/{d}/{m}_results.json")
    if any(cr.values()):
        JB = {"GPT-5.4": cr}
        sb, nb = strictness(JB, "GPT-5.4"); print(f"  global T2 strictness under Claude rubric: {sb:.1f}% (n={nb}) vs GPT rubric {st['JA']['GPT-5.4'][0]:.1f}%")
        out["expB"] = {"strictness": round(sb, 2), "per_subject": {}}
        d2s, d4s = [], []
        for m in SUBJECTS:
            g2, _ = br(J["JA"]["GPT-5.4"][m], 0); c2, _ = br(cr[m], 0); g4, _ = br(J["JA"]["GPT-5.4"][m], 1); c4, n4 = br(cr[m], 1)
            d2s.append(c2 - g2); d4s.append(c4 - g4)
            print(f"    {m:18s} T2 {g2:.1f}→{c2:.1f} ({c2 - g2:+.1f})   T4 {g4:.1f}→{c4:.1f} ({c4 - g4:+.1f})  n={n4}")
            out["expB"]["per_subject"][m] = dict(t2=[round(g2, 2), round(c2, 2)], t4=[round(g4, 2), round(c4, 2)])
        print(f"  T2 Δ range {min(d2s):+.1f}..{max(d2s):+.1f}; T4 Δ range {min(d4s):+.1f}..{max(d4s):+.1f}; subjects with T4 fall ≥10pp: {sum(d < -10 for d in d4s)}/7")
        # Opus under Claude rubric
        co = {}
        for m in SUBJECTS:
            f = f"{R}/phase5_v2/cross_eval_claude_rubric_opus_n80/{m}_results.json" if m not in OPEN else f"{R}/rebuttal/cross_eval_claude_rubric_opus_open/{m}_results.json"
            co[m] = load(f)
        JB["Opus4.7"] = co
        for subs, nm in ((FRONTIER, "4 frontier"), (SUBJECTS, "7 subjects")):
            c = pair_cells(JB, "GPT-5.4", "Opus4.7", 1, subjects=subs)
            if c["n"] == 0: continue
            dom, d, o, sh = fmt_pair(c, "GPT-5.4", "Opus4.7"); base = 100 * (baseline(c["pa"], c["pb"]) if dom == "GPT-5.4" else baseline(c["pb"], c["pa"]))
            print(f"  GPT–Opus both under Claude rubric ({nm}, n={c['n']}): {d + o} dis, {sh:.1f}% {SHORT[dom]} ({d}/{o}), pG={100 * c['pa']:.1f} pO={100 * c['pb']:.1f} base={base:.1f} resid={sh - base:+.1f}")
            out["expB"][f"gpt_opus_{nm}"] = dict(n=c["n"], n_dis=d + o, share=round(sh, 2), base=round(base, 2), d=d, o=o)

    print("\n=== Rubric author (judge fixed): Claude Sonnet judge, GPT rubric vs Claude rubric (JA, 4 frontier) ===")
    a1 = {m: load(f"{R}/phase5_v2/cross_eval_claude_n80/{m}_results.json") for m in FRONTIER}
    a2 = {m: load(f"{R}/phase5_v2/cross_eval_claude_rubric_n80/{m}_results.json") for m in FRONTIER}
    if any(a1.values()) and any(a2.values()):
        rows = {m: (br(a1[m], 0)[0], br(a2[m], 0)[0]) for m in FRONTIER}
        for m, (x, y) in rows.items(): print(f"    {m:18s} {x:.1f} → {y:.1f} ({y - x:+.1f})")
        r1 = sorted(FRONTIER, key=lambda m: rows[m][0]); r2 = sorted(FRONTIER, key=lambda m: rows[m][1])
        print(f"  max |Δ| = {max(abs(y - x) for x, y in rows.values()):.1f}pp; rank same: {r1 == r2}")
        out["rubric_fixed"] = {m: [round(x, 2), round(y, 2)] for m, (x, y) in rows.items()}

    print("\n=== Self-eval T2 BR (dedicated self line) ===")
    out["self"] = {}
    for L in ("JA", "EN"):
        vals = {m: br(S[L][m], 0)[0] for m in SUBJECTS}; out["self"][L] = {m: round(v, 2) for m, v in vals.items()}
        print(f"  {L}: " + "  ".join(f"{m} {v:.1f}" for m, v in vals.items()))

    # significance tallies from the rebuttal_checks outputs (if present at this root's sibling)
    ck = f"{R}/rebuttal_checks/cluster_checks.json"
    if os.path.exists(ck):
        cc = json.load(open(ck)); ps = sorted(c["perm_p"] for c in cc["combos"]); m = len(ps)
        bonf = sum(p < 0.05 / m for p in ps); bh = max([i + 1 for i, p in enumerate(ps) if p <= (i + 1) / m * 0.05] or [0])
        ps2 = sorted(min(1, 2 * p) for p in ps); bonf2 = sum(p < 0.05 / m for p in ps2); bh2 = max([i + 1 for i, p in enumerate(ps2) if p <= (i + 1) / m * 0.05] or [0])
        print(f"\n=== permutation (from cluster_checks.json): one-sided BH {bh}/12, Bonferroni {bonf}/12; two-sided (2p) BH {bh2}/12, Bonferroni {bonf2}/12")
        out["perm"] = dict(bh=bh, bonf=bonf, bh2=bh2, bonf2=bonf2)
    tw = f"{R}/rebuttal_checks/twoway_cluster.json"
    if os.path.exists(tw):
        t = json.load(open(tw)); rows = t.get("twoway") or t.get("A") or []
        try:
            n_pos = sum(1 for r in rows if r["ci"][0] > 0); print(f"=== two-way bootstrap CI > 0: {n_pos}/{len(rows)}; mixed: {t.get('mixed')}")
        except Exception: print("=== two-way json keys:", list(t))
    lo = f"{R}/rebuttal_checks/loso.json"
    if os.path.exists(lo):
        l = json.load(open(lo)); print("=== loso json keys:", list(l)[:5] if isinstance(l, dict) else type(l))
    if args.json: json.dump(out, open(args.json, "w"), indent=1, ensure_ascii=False); print("saved", args.json)


if __name__ == "__main__" and not os.environ.get("SUPP"):
    main()


# ---------------------------------------------------------------- supplement tables
def supplement(R="results"):
    from scipy.stats import beta as _b
    J = {L: load_panel(R, L) for L in ("JA", "EN")}; S = {L: load_self(R, L) for L in ("JA", "EN")}
    NAME = {"GPT-5.4": "GPT-5.4", "Opus4.7": "Opus 4.7", "Gemini3.1": "Gemini 3.1", "LLaMA4": "LLaMA 4"}
    print("\n=== S14: JA GPT–Opus T4 disagreements by bias × domain (n / GPT-only share) ===")
    A, B = J["JA"]["GPT-5.4"], J["JA"]["Opus4.7"]; cells = defaultdict(lambda: [0, 0])
    for m in SUBJECTS:
        for p in set(A[m]) & set(B[m]):
            bias, dom = p.split("-")[0], p.split("-")[1]; va, vb = A[m][p][1], B[m][p][1]
            if va and not vb: cells[(bias, dom)][0] += 1
            elif vb and not va: cells[(bias, dom)][1] += 1
    biases = ["confirmation_bias", "anchoring", "representativeness", "framing"]; doms = ["medical", "legal", "education", "labor", "daily_life"]
    def cell(g, o): n = g + o; return f"{n} / {100 * g / n:.0f}%" if n else "0 / –"
    for b in biases:
        row = [cell(*cells[(b, d)]) for d in doms]; g = sum(cells[(b, d)][0] for d in doms); o = sum(cells[(b, d)][1] for d in doms)
        print(f"| {b} | " + " | ".join(row) + f" | {g + o} / {100 * g / (g + o):.1f}% |")
    tot = [(sum(cells[(b, d)][0] for b in biases), sum(cells[(b, d)][1] for b in biases)) for d in doms]
    G = sum(t[0] for t in tot); O = sum(t[1] for t in tot)
    print("| **Total** | " + " | ".join(f"{g + o} / {100 * g / (g + o):.1f}%" for g, o in tot) + f" | {G + O} / {100 * G / (G + O):.1f}% |")

    print("\n=== S17: four frontier subjects × judge (JA, T2 BR; † same-model) ===")
    for m in FRONTIER:
        print(f"| {m} | " + " | ".join(f"{br(J['JA'][ev][m], 0)[0]:.1f}{'†' if SELF_OF[ev] == m else ''}" for ev in EVALS) + " |")

    print("\n=== S20: self-eval T2 BR with problem-level bootstrap 95% CI ===")
    rng = np.random.default_rng(42)
    for m in SUBJECTS:
        cols = []
        for L in ("JA", "EN"):
            v = np.array([x[0] for x in S[L][m].values()], float); n = len(v)
            bs = v[rng.integers(0, n, (10000, n))].mean(1) * 100
            cols.append(f"{100 * v.mean():.1f} [{np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}]")
        print(f"| {m} | " + " | ".join(cols) + " |")
    ja = [br(S['JA'][m], 0)[0] for m in SUBJECTS]; en = [br(S['EN'][m], 0)[0] for m in SUBJECTS]
    from scipy.stats import spearmanr
    print(f"  self JA–EN Spearman ρ = {spearmanr(ja, en)[0]:.3f} (p={spearmanr(ja, en)[1]:.3f})")
    print("\n=== S9: cross-evaluator BR (4-judge mean, self-pair excluded), T2 ===")
    cja, cen = [], []
    for m in SUBJECTS:
        r = []
        for L in ("JA", "EN"):
            vals = [br(J[L][ev][m], 0)[0] for ev in EVALS if SELF_OF[ev] != m]; r.append(np.mean(vals))
        cja.append(r[0]); cen.append(r[1]); print(f"| {m} | self {br(S['JA'][m], 0)[0]:.1f} / {br(S['EN'][m], 0)[0]:.1f} | cross {r[0]:.1f} / {r[1]:.1f} |")
    print(f"  cross JA–EN Spearman ρ = {spearmanr(cja, cen)[0]:.3f} (p={spearmanr(cja, cen)[1]:.3f})")
    print("  per-judge JA–EN ρ (6 non-self subjects): " + ", ".join(f"{ev} {spearmanr([br(J['JA'][ev][m], 0)[0] for m in SUBJECTS if SELF_OF[ev] != m], [br(J['EN'][ev][m], 0)[0] for m in SUBJECTS if SELF_OF[ev] != m])[0]:.2f}" for ev in EVALS))

    print("\n=== S5: self-pairs excluded (5 cross-only subject rows per pair), T4 ===")
    for a, b in PAIRS:
        for L in ("JA", "EN"):
            subs = [m for m in SUBJECTS if m not in (SELF_OF[a], SELF_OF[b])]
            c = pair_cells(J[L], a, b, 1, subjects=subs); dom, d, o, sh = fmt_pair(c, a, b)
            c0 = pair_cells(J[L], a, b, 1); dom0, d0, o0, sh0 = fmt_pair(c0, a, b)
            print(f"| {NAME[a]} – {NAME[b]} | {L} | {NAME[dom]} | {sh:.1f}% | {d + o} | {sh0:.1f}% |" + ("" if dom == dom0 else "  <-- side differs"))

    print("\n=== S13: per-axis strictness (JA/EN × Frontier/Open), T2 ===")
    for ev in EVALS:
        cols = []
        for L in ("JA", "EN"):
            for grp in (FRONTIER, sorted(OPEN)):
                vals = []
                for m in grp:
                    if m == SELF_OF[ev]: continue
                    vals += [x[0] for x in J[L][ev][m].values()]
                cols.append(f"{100 * np.mean(vals):.1f}")
        print(f"| {NAME[ev]} | " + " | ".join(cols) + " |")


if __name__ == "__main__" and os.environ.get("SUPP"):
    supplement(os.environ.get("ROOT", "results"))
