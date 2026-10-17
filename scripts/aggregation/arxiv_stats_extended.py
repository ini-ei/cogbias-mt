#!/usr/bin/env python3
"""
arxiv_stats_extended.py — arXiv 拡張版のための追加統計 (全て既存レコードから; API 呼び出しなし)。

対応する指摘 (2026-09 の統計レビュー):
  A  主結果 1 の不確実性: 厳格度・spread・Opus 被験 (5.0 vs 43.8)・パネル振れ (15.4pp) の問題クラスタ bootstrap CI
  B  κ の有病率依存: PABAK / Gwet AC1 / prevalence index を併記
  C  残差 CI の方法依存性: percentile / basic / normal (two-way), problem-only, subject-only, subject-LOSO range,
     + Mantel–Haenszel 条件付き関連 (problem 層別 80 層 / subject 層別 7 層)
  D  12 検定の依存性: Westfall–Young max-T 同時置換 (問題クラスタ符号反転を言語内で共有)
  E  Table III の「効果なし」の検出力: 各チェックの効果量に paired bootstrap CI を付ける
  F  再実行レジーム混在: 再生成軌跡 (97) と再判定セルを除いた集計で方向・残差符号が不変か
  G  温度チェックの評価者・被験セットの明示 (Sonnet はパネル外; 被験に Sonnet を含む)

出力: results/rebuttal_checks/arxiv_extended.json + stdout (Markdown 風の表)
実行: .venv/bin/python scripts/arxiv_stats_extended.py
"""
import json, os, sys
from collections import defaultdict
from itertools import combinations
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("SUPP", "")  # wiiat_numbers の main() を走らせない
from wiiat_numbers import SUBJECTS, OPEN, FRONTIER, SELF_OF, EVAL_DIRS, EVALS, PAIRS, baseline, kappa  # noqa: E402

R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "rebuttal_checks", "arxiv_extended.json")
N_BOOT = 10_000
N_PERM = 100_000
SEED = 42
NAME = {"GPT-5.4": "GPT-5.4", "Opus4.7": "Opus 4.7", "Gemini3.1": "Gemini 3.1", "LLaMA4": "LLaMA 4"}


# ------------------------------------------------------------------ loading (with re-run flags)
def load_flag(path):
    """pid -> (t2, t4, september_flag)"""
    if not os.path.exists(path):
        return {}
    out = {}
    for r in json.load(open(path)):
        if not (isinstance(r, dict) and r.get("problem_id")):
            continue
        if r.get("turn2_bias_confidence") is None and r.get("turn4_bias_confidence") is None:
            continue
        sept = bool(r.get("rejudged") or r.get("rejudge_note") or r.get("regenerated") or r.get("self_reevaluated"))
        out[r["problem_id"]] = (bool(r.get("turn2_bias_detected")), bool(r.get("turn4_bias_detected")), sept)
    return out


def load_dir_set(lang, dirs):
    base = f"{R}/phase5_v2" if lang == "JA" else f"{R}/phase5_v2_en"
    f_d, o_d, p_d = dirs
    out = {}
    for m in SUBJECTS:
        recs = load_flag(f"{base}/{o_d if m in OPEN else f_d}/{m}_results.json")
        if p_d and m not in OPEN:
            recs.update(load_flag(f"{base}/{p_d}/{m}_results.json"))
        out[m] = recs
    return out


def load_panel(lang):
    J = {}
    for ev, (jf, jo, ef, eo, ep) in EVAL_DIRS.items():
        J[ev] = load_dir_set(lang, (jf, jo, None) if lang == "JA" else (ef, eo, ep))
    return J


def load_self(lang):
    return load_dir_set(lang, ("full_n80", "full_n80_open", None) if lang == "JA" else ("full_n80_en", "full_n80_en_open", "full_n80_en_patch"))


def regen_sets():
    """lang -> set of (subject, pid) regenerated trajectories (from regen_todo.json)"""
    d = json.load(open(f"{R}/rebuttal/regen_todo.json"))
    out = {"JA": set(), "EN": set()}
    for k, pids in d.items():
        lang, m = k.split(":")
        out[lang.upper()].update((m, p) for p in pids)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def pct(x, q=(2.5, 97.5)):
    return [float(np.nanpercentile(x, q[0])), float(np.nanpercentile(x, q[1]))]


# ------------------------------------------------------------------ A. Main Result 1 CIs
def main_result_1(J, S, rng):
    out = {}
    pids = sorted({p for ev in EVALS for m in SUBJECTS for p in J["JA"][ev][m]})
    P = len(pids); pidx = {p: i for i, p in enumerate(pids)}
    # per-judge matrix P x (6 subjects) of T2 verdicts
    mats = {}
    for ev in EVALS:
        cols = []
        for m in SUBJECTS:
            if m == SELF_OF[ev]:
                continue
            col = np.full(P, np.nan)
            for p, v in J["JA"][ev][m].items():
                col[pidx[p]] = v[0]
            cols.append(col)
        mats[ev] = np.column_stack(cols)
    idx = rng.integers(0, P, (N_BOOT, P))
    boot = {ev: np.array([np.nanmean(mats[ev][i]) for i in idx]) * 100 for ev in EVALS}
    point = {ev: float(np.nanmean(mats[ev]) * 100) for ev in EVALS}
    spread_b = np.max(np.column_stack([boot[ev] for ev in EVALS]), 1) - np.min(np.column_stack([boot[ev] for ev in EVALS]), 1)
    order = sorted(EVALS, key=lambda e: -point[e])
    fixed_spread = boot[order[0]] - boot[order[-1]]
    out["strictness"] = {ev: dict(point=round(point[ev], 1), ci=[round(x, 1) for x in pct(boot[ev])]) for ev in EVALS}
    out["spread"] = dict(point=round(point[order[0]] - point[order[-1]], 1), ci_fixed_pair=[round(x, 1) for x in pct(fixed_spread)],
                         ci_max_minus_min=[round(x, 1) for x in pct(spread_b)], strictest=order[0], most_lenient=order[-1])
    # adjacent differences (Opus vs Gemini etc.)
    out["strictness_diffs"] = {}
    for a, b in combinations(EVALS, 2):
        d = boot[a] - boot[b]
        out["strictness_diffs"][f"{a}-{b}"] = dict(point=round(point[a] - point[b], 1), ci=[round(x, 1) for x in pct(d)])
    print("\n### A1. Global T2 strictness (JA), problem-cluster bootstrap 95% CI")
    for ev in order:
        print(f"| {NAME[ev]} | {point[ev]:.1f} | [{out['strictness'][ev]['ci'][0]:.1f}, {out['strictness'][ev]['ci'][1]:.1f}] |")
    print(f"spread {out['spread']['point']} pp; CI (fixed strictest-lenient pair) {out['spread']['ci_fixed_pair']}; CI (max-min, selection-inclusive) {out['spread']['ci_max_minus_min']}")
    for k, v in out["strictness_diffs"].items():
        print(f"  diff {k}: {v['point']:+.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]")

    # Opus subject: LLaMA vs GPT judge
    o = J["JA"]; opus = "claude-opus-4.7"
    common = sorted(set(o["LLaMA4"][opus]) & set(o["GPT-5.4"][opus]))
    l = np.array([o["LLaMA4"][opus][p][0] for p in common], float); g = np.array([o["GPT-5.4"][opus][p][0] for p in common], float)
    n = len(common); ii = rng.integers(0, n, (N_BOOT, n))
    lb, gb = l[ii].mean(1) * 100, g[ii].mean(1) * 100
    ratio = np.where(lb > 0, gb / np.maximum(lb, 1e-9), np.nan)
    out["opus_subject"] = dict(n=n, llama=dict(k=int(l.sum()), point=round(100 * l.mean(), 1), wilson=[round(x, 1) for x in wilson(l.sum(), n)]),
                               gpt=dict(k=int(g.sum()), point=round(100 * g.mean(), 1), wilson=[round(x, 1) for x in wilson(g.sum(), n)]),
                               diff_pp=dict(point=round(100 * (g.mean() - l.mean()), 1), ci=[round(x, 1) for x in pct(gb - lb)]),
                               ratio=dict(point=round(g.mean() / l.mean(), 1), ci=[round(x, 1) for x in pct(ratio)], n_resamples_llama_zero=int((lb == 0).sum())),
                               mcnemar=dict(g_only=int(((g == 1) & (l == 0)).sum()), l_only=int(((g == 0) & (l == 1)).sum())))
    print("\n### A2. Opus 4.7 subject (JA T2): LLaMA vs GPT judge")
    print(json.dumps(out["opus_subject"]))

    # GPT subject panel swing
    gs = "gpt-5"
    lines = {"self": S["JA"][gs]}
    for ev in EVALS:
        if ev != "GPT-5.4":
            lines[ev] = o[ev][gs]
    common = sorted(set.intersection(*[set(v) for v in lines.values()]))
    arr = {k: np.array([v[p][0] for p in common], float) for k, v in lines.items()}
    n = len(common); ii = rng.integers(0, n, (N_BOOT, n))
    bl = {k: v[ii].mean(1) * 100 for k, v in arr.items()}
    pt = {k: 100 * v.mean() for k, v in arr.items()}
    panels = list(combinations(lines, 3))
    ppt = {"+".join(c): np.mean([pt[x] for x in c]) for c in panels}
    pb = {"+".join(c): np.mean([bl[x] for x in c], 0) for c in panels}
    lo_k, hi_k = min(ppt, key=ppt.get), max(ppt, key=ppt.get)
    fixed = pb[hi_k] - pb[lo_k]
    sel = np.max(np.column_stack(list(pb.values())), 1) - np.min(np.column_stack(list(pb.values())), 1)
    cross = [k for k in lines if k != "self"]
    cross_range_fixed = bl[max(cross, key=pt.get)] - bl[min(cross, key=pt.get)]
    out["gpt_subject"] = dict(n=n, per_line={k: round(v, 1) for k, v in pt.items()},
                              cross_range=dict(point=round(max(pt[c] for c in cross) - min(pt[c] for c in cross), 1), ci=[round(x, 1) for x in pct(cross_range_fixed)]),
                              panels={k: round(v, 1) for k, v in ppt.items()},
                              swing=dict(point=round(ppt[hi_k] - ppt[lo_k], 1), max_panel=hi_k, min_panel=lo_k,
                                         ci_fixed_panels=[round(x, 1) for x in pct(fixed)], ci_max_minus_min=[round(x, 1) for x in pct(sel)]))
    print("\n### A3. GPT-5.4 subject (JA T2): panel swing")
    print(json.dumps(out["gpt_subject"]))
    return out


# ------------------------------------------------------------------ B. agreement statistics
def cells_arrays(J, lang, a, b, t=1, exclude=None, drop_sept=False):
    """→ va, vb (bool), pidx, sidx, pids, subs for the common cells of pair (a,b)"""
    va, vb, pp, ss = [], [], [], []
    for m in SUBJECTS:
        A, B = J[lang][a].get(m, {}), J[lang][b].get(m, {})
        for p in sorted(set(A) & set(B)):
            if exclude and (m, p) in exclude:
                continue
            if drop_sept and (A[p][2] or B[p][2]):
                continue
            va.append(A[p][t]); vb.append(B[p][t]); pp.append(p); ss.append(m)
    pids = sorted(set(pp)); subs = sorted(set(ss))
    return (np.array(va, bool), np.array(vb, bool), np.array([pids.index(p) for p in pp]), np.array([subs.index(s) for s in ss]), pids, subs)


def share_stats(va, vb, w=None):
    """weighted: returns (dominant share, baseline, residual, pa, pb, side_a)"""
    if w is None:
        w = np.ones(len(va))
    n = w.sum()
    a_only = (w * (va & ~vb)).sum(); b_only = (w * (vb & ~va)).sum()
    pa = (w * va).sum() / n; pb = (w * vb).sum() / n
    if a_only + b_only == 0:
        return (np.nan,) * 5 + (True,)
    side_a = a_only >= b_only
    obs = max(a_only, b_only) / (a_only + b_only)
    base = baseline(pa, pb) if side_a else baseline(pb, pa)
    return obs, base, obs - base, pa, pb, side_a


def agreement_stats(va, vb):
    n = len(va); both = (va & vb).mean(); nei = (~va & ~vb).mean(); po = both + nei
    pa, pb = va.mean(), vb.mean()
    pe = pa * pb + (1 - pa) * (1 - pb); k = (po - pe) / (1 - pe)
    pabak = 2 * po - 1
    pi = (pa + pb) / 2; pe_g = 2 * pi * (1 - pi); ac1 = (po - pe_g) / (1 - pe_g)
    return dict(n=n, po=round(100 * po, 1), kappa=round(k, 3), pabak=round(pabak, 3), ac1=round(ac1, 3), prevalence_index=round(abs(both - nei), 3), pa=round(100 * pa, 1), pb=round(100 * pb, 1))


# ------------------------------------------------------------------ C. residual CIs (method sensitivity) + MH
def residual_methods(J, rng):
    from statsmodels.stats.contingency_tables import StratifiedTable
    out = []
    print("\n### C. Residual: CI method sensitivity and conditional (MH) association")
    print("| pair | lang | resid | 2-way pct | 2-way basic | 2-way normal | problem-only | subject-only | LOSO min..max | MH OR|problem [CI] p | MH OR|subject [CI] p |")
    for lang in ("JA", "EN"):
        for a, b in PAIRS:
            va, vb, pidx, sidx, pids, subs = cells_arrays(J, lang, a, b)
            P, Sn = len(pids), len(subs)
            obs, base, res, pa, pb, side_a = share_stats(va, vb)
            wp = np.array([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(N_BOOT)])
            ws = np.array([np.bincount(rng.integers(0, Sn, Sn), minlength=Sn) for _ in range(N_BOOT)])
            def dist(use_p, use_s):
                r = np.empty(N_BOOT)
                for t in range(N_BOOT):
                    w = (wp[t][pidx] if use_p else 1) * (ws[t][sidx] if use_s else 1)
                    r[t] = share_stats(va, vb, w)[2]
                return r * 100
            d2 = dist(True, True); dp = dist(True, False); ds = dist(False, True)
            res_pp = res * 100
            two_pct = pct(d2); two_basic = [2 * res_pp - two_pct[1], 2 * res_pp - two_pct[0]]
            sd = float(np.nanstd(d2)); two_norm = [res_pp - 1.96 * sd, res_pp + 1.96 * sd]
            loso = []
            for s in range(Sn):
                m = sidx != s
                loso.append(share_stats(va[m], vb[m])[2] * 100)
            # Mantel–Haenszel: association between A and B verdicts, stratified by problem / by subject
            def mh(strata_idx, K):
                tabs = []
                for k in range(K):
                    m = strata_idx == k
                    n11 = int((va[m] & vb[m]).sum()); n10 = int((va[m] & ~vb[m]).sum()); n01 = int((~va[m] & vb[m]).sum()); n00 = int((~va[m] & ~vb[m]).sum())
                    tabs.append(np.array([[n11, n10], [n01, n00]]))
                st = StratifiedTable(tabs)
                orr = float(st.oddsratio_pooled); lo, hi = st.oddsratio_pooled_confint(); p = float(st.test_null_odds().pvalue)
                return dict(or_=round(orr, 2), ci=[round(float(lo), 2), round(float(hi), 2)], p=p)
            mhp = mh(pidx, P); mhs = mh(sidx, Sn)
            row = dict(pair=f"{a}-{b}", lang=lang, residual_pp=round(res_pp, 2), twoway_percentile=[round(x, 2) for x in two_pct], twoway_basic=[round(x, 2) for x in two_basic],
                       twoway_normal=[round(x, 2) for x in two_norm], twoway_sd=round(sd, 2), problem_only=[round(x, 2) for x in pct(dp)], subject_only=[round(x, 2) for x in pct(ds)],
                       loso_range=[round(min(loso), 2), round(max(loso), 2)], mh_by_problem=mhp, mh_by_subject=mhs, tail_share_le0=float((d2 <= 0).mean()))
            out.append(row)
            f = lambda c: f"[{c[0]:+.1f}, {c[1]:+.1f}]"
            print(f"| {NAME[a]}–{NAME[b]} | {lang} | {res_pp:+.1f} | {f(two_pct)} | {f(two_basic)} | {f(two_norm)} | {f(row['problem_only'])} | {f(row['subject_only'])} | {min(loso):+.1f}..{max(loso):+.1f} | {mhp['or_']} [{mhp['ci'][0]}, {mhp['ci'][1]}] p={mhp['p']:.2g} | {mhs['or_']} [{mhs['ci'][0]}, {mhs['ci'][1]}] p={mhs['p']:.2g} |")
    for key in ("twoway_percentile", "twoway_basic", "twoway_normal", "problem_only", "subject_only"):
        print(f"  {key}: lower bound > 0 in {sum(r[key][0] > 0 for r in out)}/12")
    print(f"  LOSO residual min > 0 in {sum(r['loso_range'][0] > 0 for r in out)}/12; MH|problem p<0.05: {sum(r['mh_by_problem']['p'] < 0.05 for r in out)}/12 (Bonferroni {sum(r['mh_by_problem']['p'] < 0.05 / 12 for r in out)}/12); MH|subject p<0.05: {sum(r['mh_by_subject']['p'] < 0.05 for r in out)}/12 (Bonferroni {sum(r['mh_by_subject']['p'] < 0.05 / 12 for r in out)}/12)")
    return out


# ------------------------------------------------------------------ D. Westfall–Young joint permutation for the 12 raw-asymmetry tests
def joint_permutation(J, rng):
    print("\n### D. Raw asymmetry: single-step max-T permutation (flips shared within language) — FWER-adjusted p")
    combos = []
    for lang in ("JA", "EN"):
        va_list = []
        pids = sorted({p for ev in EVALS for m in SUBJECTS for p in J[lang][ev][m]}); P = len(pids); pidx = {p: i for i, p in enumerate(pids)}
        flips = rng.random((N_PERM, P)) < 0.5
        for a, b in PAIRS:
            av = np.zeros(P); bv = np.zeros(P)
            for m in SUBJECTS:
                A, B = J[lang][a].get(m, {}), J[lang][b].get(m, {})
                for p in set(A) & set(B):
                    x, y = A[p][1], B[p][1]
                    if x and not y: av[pidx[p]] += 1
                    elif y and not x: bv[pidx[p]] += 1
            side_a = av.sum() >= bv.sum(); dom, oth = (av, bv) if side_a else (bv, av)
            tot = dom.sum() + oth.sum(); obs = dom.sum() / tot
            null = (np.where(flips, oth, dom).sum(1)) / tot
            sd = null.std(); z_null = (null - 0.5) / sd; z_obs = (obs - 0.5) / sd
            p_raw = max(float((null >= obs - 1e-12).mean()), 1 / N_PERM)
            combos.append(dict(pair=f"{a}-{b}", lang=lang, n=int(tot), share=round(100 * obs, 1), z_obs=float(z_obs), z_null=np.abs(z_null), p_raw_onesided=p_raw))
    maxT = np.max(np.column_stack([c["z_null"] for c in combos]), 1)  # JA and EN flips are independent draws → joint null
    for c in combos:
        c["p_adj_maxT_twosided"] = float((maxT >= abs(c["z_obs"])).mean()); c["p_adj_maxT_twosided"] = max(c["p_adj_maxT_twosided"], 1 / N_PERM); del c["z_null"]
        print(f"| {c['pair']} | {c['lang']} | n={c['n']} | share {c['share']} | z={c['z_obs']:.2f} | raw one-sided p={c['p_raw_onesided']:.2g} | max-T adj. two-sided p={c['p_adj_maxT_twosided']:.3g} |")
    print(f"  max-T adjusted p < 0.05: {sum(c['p_adj_maxT_twosided'] < 0.05 for c in combos)}/12")
    return combos


# ------------------------------------------------------------------ E. Table III effect sizes with CIs
def paired_delta(x, y, rng):
    """x,y arrays aligned by problem (0/1) → Δ=mean(y)-mean(x) pp with problem-bootstrap CI"""
    n = len(x); ii = rng.integers(0, n, (N_BOOT, n)); d = (y[ii].mean(1) - x[ii].mean(1)) * 100
    return dict(n=n, x=round(100 * x.mean(), 1), y=round(100 * y.mean(), 1), delta=round(100 * (y.mean() - x.mean()), 1), ci=[round(v, 1) for v in pct(d)])


def table3_effects(J, S, rng):
    out = {}
    base = f"{R}/phase5_v2"
    # E1 rubric author, judge fixed (Claude Sonnet 4.6 judge, GPT rubric vs Claude rubric)
    print("\n### E1. Rubric author, judge fixed (Sonnet 4.6 judge): GPT rubric → Claude rubric, T2 BR, paired by problem")
    subs = ["gpt-5", "claude-4-sonnet", "gemini-3.1-pro", "llama-4-maverick", "claude-opus-4.7"]
    e1 = {}
    for m in subs:
        A = load_flag(f"{base}/cross_eval_claude_n80/{m}_results.json"); B = load_flag(f"{base}/cross_eval_claude_rubric_n80/{m}_results.json")
        c = sorted(set(A) & set(B))
        if not c:
            e1[m] = None; print(f"| {m} | (no Claude-rubric records) |"); continue
        x = np.array([A[p][0] for p in c], float); y = np.array([B[p][0] for p in c], float)
        e1[m] = paired_delta(x, y, rng); e1[m]["t4"] = paired_delta(np.array([A[p][1] for p in c], float), np.array([B[p][1] for p in c], float), rng)
        print(f"| {m} | n={e1[m]['n']} | {e1[m]['x']} → {e1[m]['y']} | Δ {e1[m]['delta']:+.1f} [{e1[m]['ci'][0]:+.1f}, {e1[m]['ci'][1]:+.1f}] | T4 Δ {e1[m]['t4']['delta']:+.1f} [{e1[m]['t4']['ci'][0]:+.1f}, {e1[m]['t4']['ci'][1]:+.1f}] |")
    valid = [m for m in subs if e1[m]]
    r1 = sorted(valid, key=lambda m: e1[m]["x"]); r2 = sorted(valid, key=lambda m: e1[m]["y"])
    e1["_subjects_with_both"] = valid; e1["_rank_same"] = r1 == r2; e1["_max_abs_delta"] = max(abs(e1[m]["delta"]) for m in valid)
    print(f"  subjects with both rubrics: {valid}; rank same: {r1 == r2}; max|Δ| = {e1['_max_abs_delta']}")
    out["rubric_fixed"] = e1

    # E2 reciprocal: GPT-5.4 judge, GPT rubric vs Claude rubric (7 subjects)
    print("\n### E2. Rubric author, reciprocal (GPT-5.4 judge): GPT rubric → Claude rubric")
    e2 = {"per_subject": {}}
    allx, ally, allx4, ally4, pid_all = [], [], [], [], []
    for m in SUBJECTS:
        d = "cross_eval_gpt5_claude_rubric_open" if m in OPEN else "cross_eval_gpt5_claude_rubric_n80"
        B = load_flag(f"{R}/rebuttal/{d}/{m}_results.json"); A = J["JA"]["GPT-5.4"][m]
        c = sorted(set(A) & set(B))
        x = np.array([A[p][0] for p in c], float); y = np.array([B[p][0] for p in c], float)
        x4 = np.array([A[p][1] for p in c], float); y4 = np.array([B[p][1] for p in c], float)
        e2["per_subject"][m] = dict(t2=paired_delta(x, y, rng), t4=paired_delta(x4, y4, rng))
        if m != "gpt-5":
            allx.append(x); ally.append(y); pid_all += c
        print(f"| {m} | T2 {e2['per_subject'][m]['t2']['x']} → {e2['per_subject'][m]['t2']['y']} (Δ {e2['per_subject'][m]['t2']['delta']:+.1f} [{e2['per_subject'][m]['t2']['ci'][0]:+.1f}, {e2['per_subject'][m]['t2']['ci'][1]:+.1f}]) | T4 {e2['per_subject'][m]['t4']['x']} → {e2['per_subject'][m]['t4']['y']} (Δ {e2['per_subject'][m]['t4']['delta']:+.1f} [{e2['per_subject'][m]['t4']['ci'][0]:+.1f}, {e2['per_subject'][m]['t4']['ci'][1]:+.1f}]) |")
    # global strictness Δ with problem-cluster bootstrap (6 non-self subjects)
    pids = sorted(set(pid_all)); P = len(pids); pidx = np.array([pids.index(p) for p in pid_all])
    X = np.concatenate(allx); Y = np.concatenate(ally)
    d = np.empty(N_BOOT)
    for t in range(N_BOOT):
        w = np.bincount(rng.integers(0, P, P), minlength=P)[pidx]
        d[t] = ((w * Y).sum() - (w * X).sum()) / w.sum() * 100
    e2["global_strictness"] = dict(gpt_rubric=round(100 * X.mean(), 1), claude_rubric=round(100 * Y.mean(), 1), delta=round(100 * (Y.mean() - X.mean()), 1), ci=[round(v, 1) for v in pct(d)])
    print(f"  global T2 strictness: {e2['global_strictness']}")
    out["reciprocal"] = e2

    # E3 generator version: strictness on gpt-5-authored 40 vs gpt-5.4-authored 40 (independent problem sets)
    print("\n### E3. Generator version (JA): strictness on gpt-5-authored 40 vs own 40, per judge, two-sample problem bootstrap")
    orig = {json.loads(l)["id"] for l in open(f"{base}/hard_all_40_clear_ce.jsonl") if l.strip()}
    e3 = {}
    for ev in EVALS:
        rows_o, rows_e = defaultdict(list), defaultdict(list)
        for m in SUBJECTS:
            if m == SELF_OF[ev]: continue
            for p, v in J["JA"][ev][m].items():
                (rows_o if p in orig else rows_e)[p].append(v[0])
        mo = np.array([np.mean(v) for v in rows_o.values()]); me = np.array([np.mean(v) for v in rows_e.values()])
        bo = mo[rng.integers(0, len(mo), (N_BOOT, len(mo)))].mean(1); be = me[rng.integers(0, len(me), (N_BOOT, len(me)))].mean(1)
        e3[ev] = dict(gpt5_authored=round(100 * mo.mean(), 1), own=round(100 * me.mean(), 1), delta_own_minus_gpt5=round(100 * (me.mean() - mo.mean()), 1), ci=[round(v, 1) for v in pct((be - bo) * 100)])
        print(f"| {NAME[ev]} | {e3[ev]['gpt5_authored']} | {e3[ev]['own']} | Δ {e3[ev]['delta_own_minus_gpt5']:+.1f} [{e3[ev]['ci'][0]:+.1f}, {e3[ev]['ci'][1]:+.1f}] |")
    out["generator_version"] = e3

    # E4 quantization: Q4 vs Q8 per judge line, paired by problem (responses differ → generation noise included)
    print("\n### E4. Quantization (Qwen3.5-27B): Q4_K_M → Q8_0 per evaluator line, paired by problem")
    q4dirs = {"JA": {"GPT-5.4": "phase5_v2/cross_eval_gpt5_open", "Opus4.7": "phase5_v2/cross_eval_claude_opus_n80", "Gemini3.1": "phase5_v2/cross_eval_gemini_open", "LLaMA4": "phase5_v2/cross_eval_llama_n80", "self": "phase5_v2/full_n80_open"},
              "EN": {"GPT-5.4": "phase5_v2_en/cross_eval_gpt5_en_open", "Opus4.7": "phase5_v2_en/cross_eval_claude_opus_en", "Gemini3.1": "phase5_v2_en/cross_eval_gemini_en_open", "LLaMA4": "phase5_v2_en/cross_eval_llama_en", "self": "phase5_v2_en/full_n80_en_open"}}
    q8dirs = {"GPT-5.4": "cross_eval_gpt5_q8", "Opus4.7": "cross_eval_opus_q8", "Gemini3.1": "cross_eval_gemini_q8", "LLaMA4": "cross_eval_llama_q8"}
    e4 = {}; pooled2, pooled4 = [], []
    for lang in ("JA", "EN"):
        for ev in list(EVALS) + ["self"]:
            A = load_flag(f"{R}/{q4dirs[lang][ev]}/qwen3.5-27b_results.json")
            Bp = f"{R}/rebuttal/{q8dirs[ev]}_{lang.lower()}/qwen3.5-27b_results.json" if ev != "self" else f"{R}/rebuttal/full_n80{'_en' if lang == 'EN' else ''}_open_q8/qwen3.5-27b_results.json"
            B = load_flag(Bp); c = sorted(set(A) & set(B))
            x2 = np.array([A[p][0] for p in c], float); y2 = np.array([B[p][0] for p in c], float); x4 = np.array([A[p][1] for p in c], float); y4 = np.array([B[p][1] for p in c], float)
            e4[f"{lang}_{ev}"] = dict(t2=paired_delta(x2, y2, rng), t4=paired_delta(x4, y4, rng))
            if ev != "self":
                pooled2.append(y2 - x2); pooled4.append(y4 - x4)
            r = e4[f"{lang}_{ev}"]
            print(f"| {lang} | {ev} | T2 {r['t2']['x']} → {r['t2']['y']} (Δ {r['t2']['delta']:+.1f} [{r['t2']['ci'][0]:+.1f}, {r['t2']['ci'][1]:+.1f}]) | T4 {r['t4']['x']} → {r['t4']['y']} (Δ {r['t4']['delta']:+.1f} [{r['t4']['ci'][0]:+.1f}, {r['t4']['ci'][1]:+.1f}]) |")
    # pooled mean Δ over the 8 judge cells, resampling problems (same 80 problems shared across judges within a language → cluster by (lang, problem))
    def pooled(dl):
        M = np.column_stack(dl)  # n_problems(80) × 8? no: JA and EN problems differ → stack as 80 × 8 with JA/EN separate columns; cluster by row index within lang
        ii = rng.integers(0, M.shape[0], (N_BOOT, M.shape[0]))
        return dict(mean_delta=round(100 * M.mean(), 1), ci=[round(v, 1) for v in pct(M[ii].mean((1, 2)) * 100)], max_abs=round(100 * np.abs(M.mean(0)).max(), 1))
    e4["pooled_judges_t2"] = pooled(pooled2); e4["pooled_judges_t4"] = pooled(pooled4)
    print(f"  pooled over 8 judge cells: T2 {e4['pooled_judges_t2']}  T4 {e4['pooled_judges_t4']}")
    out["quantization"] = e4

    # E5 temperature check: agreement with Wilson CI; subjects and evaluators disclosed
    print("\n### E5. Evaluation-temperature check (80 boundary verdicts; judges GPT-5.4 and Claude Sonnet 4.6 — Sonnet is not a panel member; subjects gpt-5, claude-4-sonnet, gemini-3.1-pro, llama-4-maverick)")
    e5 = {}
    for ev in ("gpt-5", "claude-4-sonnet"):
        v = {}
        for t in ("0.0", "0.3", "0.7"):
            d = json.load(open(f"{base}/exp_evaluator_temp/{ev}_temp{t}.json"))
            v[t] = {(r["subject"], r["problem_id"], r["turn"]): bool(r["bias_detected"]) for r in d if r.get("error") is None}
        keys = sorted(set.intersection(*[set(x) for x in v.values()]))
        e5[ev] = dict(n=len(keys), subjects=sorted({k[0] for k in keys}), br={t: round(100 * np.mean([v[t][k] for k in keys]), 1) for t in v})
        for t1, t2 in (("0.0", "0.3"), ("0.0", "0.7"), ("0.3", "0.7")):
            agree = sum(v[t1][k] == v[t2][k] for k in keys)
            e5[ev][f"agree_{t1}_{t2}"] = dict(k=agree, pct=round(100 * agree / len(keys), 1), wilson=[round(x, 1) for x in wilson(agree, len(keys))])
        print(f"| {ev} | n={e5[ev]['n']} | BR {e5[ev]['br']} | " + " | ".join(f"{k}: {x['pct']} [{x['wilson'][0]}, {x['wilson'][1]}]" for k, x in e5[ev].items() if k.startswith('agree')) + " |")
    out["temperature"] = e5
    return out


# ------------------------------------------------------------------ F. regime sensitivity (exclude re-run records)
def regime_sensitivity(J, S):
    print("\n### F. Regime sensitivity: exclude regenerated trajectories (97) / additionally every re-judged cell")
    regen = regen_sets()
    out = {}
    for label, excl, drop in (("all", None, False), ("no_regenerated", regen, False), ("no_regenerated_no_rejudged", regen, True)):
        res = {"strictness": {}, "table2": {}}
        for lang in ("JA", "EN"):
            for ev in EVALS:
                vals = []
                for m in SUBJECTS:
                    if m == SELF_OF[ev]: continue
                    for p, v in J[lang][ev][m].items():
                        if excl and (m, p) in excl[lang]: continue
                        if drop and v[2]: continue
                        vals.append(v[0])
                res["strictness"][f"{lang}_{ev}"] = dict(point=round(100 * np.mean(vals), 1), n=len(vals))
            for a, b in PAIRS:
                va, vb, pidx, sidx, pids, subs = cells_arrays(J, lang, a, b, 1, exclude=excl[lang] if excl else None, drop_sept=drop)
                obs, base, r, pa, pb, side_a = share_stats(va, vb)
                n_dis = int((va & ~vb).sum() + (vb & ~va).sum())
                res["table2"][f"{a}-{b}_{lang}"] = dict(n=int(len(va)), n_dis=n_dis, dom=a if side_a else b, share=round(100 * obs, 1), base=round(100 * base, 1), resid=round(100 * r, 1), pa=round(100 * pa, 1), pb=round(100 * pb, 1))
        opus = {ev: J["JA"][ev]["claude-opus-4.7"] for ev in ("LLaMA4", "GPT-5.4")}
        res["opus_subject"] = {ev: round(100 * np.mean([v[0] for p, v in d.items() if not (drop and v[2])]), 1) for ev, d in opus.items()}
        out[label] = res
        print(f"\n[{label}] strictness JA: " + ", ".join(f"{ev} {res['strictness'][f'JA_{ev}']['point']} (n={res['strictness'][f'JA_{ev}']['n']})" for ev in EVALS) + f"; Opus subject LLaMA/GPT {res['opus_subject']}")
        for k, v in res["table2"].items():
            print(f"  {k}: n={v['n']} dis={v['n_dis']} {v['dom']} {v['share']}% base {v['base']} resid {v['resid']:+.1f}")
    same_dom = all(out["all"]["table2"][k]["dom"] == out["no_regenerated_no_rejudged"]["table2"][k]["dom"] for k in out["all"]["table2"])
    pos = sum(v["resid"] > 0 for v in out["no_regenerated_no_rejudged"]["table2"].values())
    print(f"\n  dominant judge identical in all 12 after both exclusions: {same_dom}; residual > 0: {pos}/12; max |Δ share| = {max(abs(out['all']['table2'][k]['share'] - out['no_regenerated_no_rejudged']['table2'][k]['share']) for k in out['all']['table2']):.1f}pp")
    # regenerated-only subset for information
    reg = {"table2": {}}
    for lang in ("JA", "EN"):
        for a, b in PAIRS:
            va, vb, *_ = cells_arrays(J, lang, a, b, 1)
            # keep only regenerated
            keep = []
            for m in SUBJECTS:
                A, B = J[lang][a].get(m, {}), J[lang][b].get(m, {})
                for p in sorted(set(A) & set(B)):
                    keep.append((m, p) in regen[lang])
            keep = np.array(keep)
            if keep.sum() == 0: continue
            obs, base, r, pa, pb, side_a = share_stats(va[keep], vb[keep])
            reg["table2"][f"{a}-{b}_{lang}"] = dict(n=int(keep.sum()), n_dis=int((va[keep] & ~vb[keep]).sum() + (vb[keep] & ~va[keep]).sum()), dom=a if side_a else b, share=round(100 * obs, 1) if obs == obs else None, resid=round(100 * r, 1) if r == r else None)
    out["regenerated_only"] = reg
    print("  regenerated-only cells: " + "; ".join(f"{k}: n={v['n']} dis={v['n_dis']} {v['dom']} {v['share']}% resid {v['resid']}" for k, v in reg["table2"].items()))
    return out


def main():
    rng = np.random.default_rng(SEED)
    J = {L: load_panel(L) for L in ("JA", "EN")}; S = {L: load_self(L) for L in ("JA", "EN")}
    out = {"seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM}
    out["main_result_1"] = main_result_1(J, S, rng)
    print("\n### B. Agreement statistics per combination (T4): κ, PABAK, Gwet AC1, prevalence index")
    out["agreement"] = {}
    for lang in ("JA", "EN"):
        for a, b in PAIRS:
            va, vb, *_ = cells_arrays(J, lang, a, b)
            st = agreement_stats(va, vb); out["agreement"][f"{a}-{b}_{lang}"] = st
            print(f"| {NAME[a]}–{NAME[b]} | {lang} | po {st['po']} | κ {st['kappa']:.2f} | PABAK {st['pabak']:.2f} | AC1 {st['ac1']:.2f} | PI {st['prevalence_index']:.2f} | pA {st['pa']} pB {st['pb']} |")
    out["residual_methods"] = residual_methods(J, rng)
    out["joint_permutation"] = joint_permutation(J, rng)
    out["table3_effects"] = table3_effects(J, S, rng)
    out["regime"] = regime_sensitivity(J, S)
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
