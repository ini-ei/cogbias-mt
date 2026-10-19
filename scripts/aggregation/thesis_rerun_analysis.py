#!/usr/bin/env python3
"""
thesis_rerun_analysis.py — 卒論の追加実験（評価者の再実行・反復判定）を元の判定と比べる。
  python scripts/thesis_rerun_analysis.py --tag rerun1 --repeat repeat_run1 repeat_run2 --json results/thesis_exp/analysis.json

(1) 反復判定: 境界 10 問 × 4 被験 × 2 ターンを同じ評価者で 2 回判定し直し、元の判定との一致率、
    3 回の多数決と元判定の一致率、反転したセル数を出す。
(2) 再実行: 全 560 対話 × 2 ターンを判定し直し、
    - セル単位の一致率と κ（元 vs 再実行、評価者・言語・ターンごと）
    - 初回ターンの判定率（自分以外 6 被験）と評価者間の開き
    - ペアごとの反証後ターンの不一致件数・優勢側の割合・式 (strict) の予測値・残差
    を元の値と並べる。定義は wiiat_numbers.py と同じ。
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wiiat_numbers as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "results", "thesis_exp")
FRONTIER = W.FRONTIER


def load_turns(path):
    """problem_id -> {0: bool|None, 1: bool|None}（判定が無い・エラーのターンは None）"""
    if not os.path.exists(path): return {}
    out = {}
    for r in json.load(open(path)):
        if not isinstance(r, dict) or not r.get("problem_id"): continue
        v = {}
        for t, key in ((0, "turn2"), (1, "turn4")):
            if r.get(f"{key}_bias_confidence") is None or r.get(f"{key}_error"): v[t] = None
            elif r.get(f"{key}_bias_detected") is None: v[t] = None
            else: v[t] = bool(r[f"{key}_bias_detected"])
        out[r["problem_id"]] = v
    return out


def panel_turns(root, lang):
    """evaluator -> subject -> pid -> {0: v, 1: v}"""
    base = f"{root}/phase5_v2" if lang == "JA" else f"{root}/phase5_v2_en"
    J = {}
    for ev, (jf, jo, ef, eo, ep) in W.EVAL_DIRS.items():
        f_d, o_d, p_d = (jf, jo, None) if lang == "JA" else (ef, eo, ep)
        J[ev] = {}
        for m in W.SUBJECTS:
            recs = load_turns(f"{base}/{o_d if m in W.OPEN else f_d}/{m}_results.json")
            if p_d and m not in W.OPEN:
                for p, v in load_turns(f"{base}/{p_d}/{m}_results.json").items():
                    recs.setdefault(p, {0: None, 1: None})
                    for t in (0, 1):
                        if v[t] is not None: recs[p][t] = v[t]
            J[ev][m] = recs
    return J


def to_bool_panel(J):
    """両ターンが揃ったレコードだけを wiiat_numbers 形式 (pid -> (t2, t4)) にする"""
    return {ev: {m: {p: (v[0], v[1]) for p, v in recs.items() if v[0] is not None and v[1] is not None}
                 for m, recs in J[ev].items()} for ev in J}


def agree_stats(A, B, t, subjects, ids=None):
    """A, B: subject -> pid -> {t: v}. 両方に判定があるセルの一致率と κ"""
    a = b = both = nei = 0
    for m in subjects:
        for p in set(A.get(m, {})) & set(B.get(m, {})):
            if ids is not None and p not in ids: continue
            va, vb = A[m][p].get(t), B[m][p].get(t)
            if va is None or vb is None: continue
            a += va and not vb; b += vb and not va; both += va and vb; nei += (not va) and (not vb)
    n = a + b + both + nei
    if n == 0: return dict(n=0)
    pa, pb = (a + both) / n, (b + both) / n
    po = (both + nei) / n; pe = pa * pb + (1 - pa) * (1 - pb)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return dict(n=n, agree=round(100 * po, 2), kappa=round(k, 3), a_only=a, b_only=b, both=both, neither=nei,
                pa=round(100 * pa, 2), pb=round(100 * pb, 2))


def table2(Jb, lang):
    out = {}
    for a, b in W.PAIRS:
        if a not in Jb or b not in Jb: continue
        c = W.pair_cells(Jb, a, b, 1)
        if c["n"] == 0: continue
        dom, d, o, sh = W.fmt_pair(c, a, b); pa, pb = c["pa"], c["pb"]
        base = 100 * (W.baseline(pa, pb) if dom == a else W.baseline(pb, pa))
        out[f"{a}-{b}"] = dict(n=c["n"], n_dis=d + o, share=round(sh, 2), dom=dom, d=d, o=o,
                               pa=round(100 * pa, 2), pb=round(100 * pb, 2), base=round(base, 2), resid=round(sh - base, 2), kappa=round(W.kappa(c), 3))
    return out


def analyze_repeat(tags, lang, J0, ids):
    """反復判定: tag ごとの判定を元 (J0) と比べる"""
    res = {}
    runs = {tag: panel_turns(os.path.join(EXP, tag), lang) for tag in tags}
    for ev in W.EVALS:
        if not any(sum(len(v) for v in runs[t][ev].values()) for t in tags): continue
        r = {}
        for t in (0, 1):
            tn = "T2" if t == 0 else "T4"
            r[tn] = {}
            for tag in tags:
                r[tn][f"orig_vs_{tag}"] = agree_stats(J0[ev], runs[tag][ev], t, FRONTIER, ids)
            if len(tags) >= 2:
                r[tn][f"{tags[0]}_vs_{tags[1]}"] = agree_stats(runs[tags[0]][ev], runs[tags[1]][ev], t, FRONTIER, ids)
                # 3 回（元 + 2 回）の多数決と元判定の一致、全回一致のセル数
                n = same = flip_any = 0
                for m in FRONTIER:
                    for p in ids:
                        vs = [J0[ev][m].get(p, {}).get(t)] + [runs[tag][ev].get(m, {}).get(p, {}).get(t) for tag in tags]
                        if any(v is None for v in vs): continue
                        n += 1; maj = sum(vs) >= 2; same += maj == vs[0]; flip_any += len(set(vs)) > 1
                r[tn]["majority"] = dict(n=n, majority_eq_orig=round(100 * same / n, 2) if n else None, cells_with_any_flip=flip_any,
                                         unanimous=round(100 * (n - flip_any) / n, 2) if n else None)
        res[ev] = r
    return res


def analyze_rerun(tag, lang, J0):
    J1 = panel_turns(os.path.join(EXP, tag), lang)
    res = {"n_records": {ev: sum(len(v) for v in J1[ev].values()) for ev in W.EVALS}}
    res["cell_agreement"] = {}
    for ev in W.EVALS:
        if res["n_records"][ev] == 0: continue
        res["cell_agreement"][ev] = {"T2": agree_stats(J0[ev], J1[ev], 0, W.SUBJECTS), "T4": agree_stats(J0[ev], J1[ev], 1, W.SUBJECTS)}
    J0b, J1b = to_bool_panel(J0), to_bool_panel(J1)
    # 判定率（初回ターン、自分以外 6 被験）: 元と再実行を同じレコード集合で比べる
    res["strictness"] = {}
    for ev in W.EVALS:
        if res["n_records"][ev] == 0: continue
        ids_common = {m: set(J0b[ev][m]) & set(J1b[ev][m]) for m in W.SUBJECTS}
        v0 = [J0b[ev][m][p][0] for m in W.SUBJECTS if m != W.SELF_OF[ev] for p in ids_common[m]]
        v1 = [J1b[ev][m][p][0] for m in W.SUBJECTS if m != W.SELF_OF[ev] for p in ids_common[m]]
        v0b = [J0b[ev][m][p][1] for m in W.SUBJECTS if m != W.SELF_OF[ev] for p in ids_common[m]]
        v1b = [J1b[ev][m][p][1] for m in W.SUBJECTS if m != W.SELF_OF[ev] for p in ids_common[m]]
        res["strictness"][ev] = dict(n=len(v0), T2_orig=round(100 * np.mean(v0), 2) if v0 else None, T2_rerun=round(100 * np.mean(v1), 2) if v1 else None,
                                     T4_orig=round(100 * np.mean(v0b), 2) if v0b else None, T4_rerun=round(100 * np.mean(v1b), 2) if v1b else None)
    # 被験モデルごとの判定率（両ターン）
    res["bias_rate"] = {}
    for ev in W.EVALS:
        if res["n_records"][ev] == 0: continue
        res["bias_rate"][ev] = {}
        for m in W.SUBJECTS:
            ids = set(J0b[ev][m]) & set(J1b[ev][m])
            if not ids: continue
            res["bias_rate"][ev][m] = dict(n=len(ids),
                                           T2_orig=round(W.br(J0b[ev][m], 0, ids)[0], 1), T2_rerun=round(W.br(J1b[ev][m], 0, ids)[0], 1),
                                           T4_orig=round(W.br(J0b[ev][m], 1, ids)[0], 1), T4_rerun=round(W.br(J1b[ev][m], 1, ids)[0], 1))
    # ペアの不一致（反証後）: 再実行同士、および再実行が揃ったペアのみ
    done = [ev for ev in W.EVALS if res["n_records"][ev] > 0]
    J1full = {ev: J1b[ev] for ev in done}
    res["table2_rerun"] = table2(J1full, lang) if len(done) >= 2 else {}
    # 元データを同じレコード集合に制限した表 2
    J0sub = {ev: {m: {p: v for p, v in J0b[ev][m].items() if p in J1b[ev][m]} for m in W.SUBJECTS} for ev in done}
    res["table2_orig_same_ids"] = table2(J0sub, lang) if len(done) >= 2 else {}
    # 元 (全評価者) の表 2
    res["table2_orig"] = table2(J0b, lang)
    # 混合パネル: 再実行のある評価者は再実行、無い評価者は元
    Jmix = {ev: (J1b[ev] if ev in done else J0b[ev]) for ev in W.EVALS}
    res["table2_mixed"] = table2(Jmix, lang)
    return res


def fmt_agree(d):
    if d.get("n", 0) == 0: return "n=0"
    return f"n={d['n']} agree={d['agree']:.1f}% κ={d['kappa']:.2f} (orig {d['pa']:.1f}% → new {d['pb']:.1f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rerun1"); ap.add_argument("--repeat", nargs="*", default=["repeat_run1", "repeat_run2"])
    ap.add_argument("--langs", nargs="*", default=["JA", "EN"]); ap.add_argument("--json", default=None)
    ap.add_argument("--ids-file", default=os.path.join(EXP, "boundary10_ids.txt"))
    args = ap.parse_args()
    ids = {l.strip() for l in open(args.ids_file) if l.strip()} if os.path.exists(args.ids_file) else set()
    out = {}
    for L in args.langs:
        J0 = panel_turns(os.path.join(ROOT, "results"), L)
        out[L] = {}
        if args.repeat and L == "JA":
            print(f"\n=== [{L}] 反復判定（境界 {len(ids)} 問 × 4 被験） ===")
            rep = analyze_repeat(args.repeat, L, J0, ids); out[L]["repeat"] = rep
            for ev, r in rep.items():
                for tn in ("T2", "T4"):
                    print(f"  {ev} {tn}: " + " | ".join(f"{k}: {fmt_agree(v)}" for k, v in r[tn].items() if k != "majority"))
                    if "majority" in r[tn]: print(f"      majority: {r[tn]['majority']}")
        print(f"\n=== [{L}] 再実行 {args.tag} ===")
        rr = analyze_rerun(args.tag, L, J0); out[L]["rerun"] = rr
        print("  records: " + " ".join(f"{ev}:{n}" for ev, n in rr["n_records"].items()))
        for ev, d in rr["cell_agreement"].items():
            print(f"  {ev}: T2 {fmt_agree(d['T2'])} | T4 {fmt_agree(d['T4'])}")
        for ev, d in rr["strictness"].items():
            print(f"  strictness {ev} (n={d['n']}): T2 {d['T2_orig']} → {d['T2_rerun']} | T4 {d['T4_orig']} → {d['T4_rerun']}")
        for key in ("table2_orig_same_ids", "table2_rerun", "table2_mixed"):
            if rr[key]:
                print(f"  {key}:")
                for pr, d in rr[key].items():
                    print(f"    {pr}: dis={d['n_dis']} share={d['share']:.1f}% dom={d['dom']} pA={d['pa']:.1f} pB={d['pb']:.1f} base={d['base']:.1f} resid={d['resid']:+.1f} κ={d['kappa']:.2f}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1, ensure_ascii=False); print("wrote", args.json)


if __name__ == "__main__":
    main()
