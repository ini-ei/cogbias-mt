#!/usr/bin/env python3
"""
thesis_v12_analysis.py — ルーブリック v1.2（結果基準）で日本語の反証後ターンを判定し直した結果を v1.1 と比べる。
  .venv/bin/python scripts/thesis_v12_analysis.py --tag v12_outcome --json results/thesis_exp/v12_analysis.json

v1.2 の判定がある評価者だけを使い、同じ評価者・同じセルで v1.1（元の判定）と並べる。
  - 評価者ごとの反証後の判定率（自分以外 6 被験）と評価者の開き
  - 被験モデルごとの評価者間の幅（最大−最小の 7 被験平均）
  - ペアごとの不一致件数・優勢側の割合・式 (strict) の予測・残差・κ
  - 補正との組合せ: 問題を 40/40 に 200 回分割し、学習集合で閾値と平均差を決め、テスト集合で幅・開き・順位一致を測る
    （無補正／平均差し引き／閾値化＋平均差し引き＝提案手法）。定義は thesis_method_eval.py と同じ。
"""
import argparse, json, os, sys, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wiiat_numbers as W  # noqa: E402

SUBJ = W.SUBJECTS


def load_conf(root, lang="JA", t=1):
    base = f"{root}/phase5_v2" if lang == "JA" else f"{root}/phase5_v2_en"; C = {}
    k = f"turn{2 if t == 0 else 4}"
    for ev, (jf, jo, ef, eo, ep) in W.EVAL_DIRS.items():
        C[ev] = {}
        for m in SUBJ:
            d = (jo if m in W.OPEN else jf) if lang == "JA" else (eo if m in W.OPEN else ef)
            paths = [f"{base}/{d}/{m}_results.json"] + ([f"{base}/{ep}/{m}_results.json"] if (lang == "EN" and ep and m not in W.OPEN) else [])
            for pth in paths:
                if not os.path.exists(pth): continue
                for r in json.load(open(pth)):
                    if isinstance(r, dict) and r.get("problem_id") and r.get(f"{k}_bias_confidence") is not None and not r.get(f"{k}_error") and r.get(f"{k}_bias_detected") is not None:
                        C[ev][(m, r["problem_id"])] = (bool(r[f"{k}_bias_detected"]), float(r[f"{k}_bias_confidence"]))
    return C


def matrix(C, evals, cells):
    X = np.full((len(cells), len(evals)), np.nan)
    for j, ev in enumerate(evals):
        for i, cell in enumerate(cells):
            if cell[0] == W.SELF_OF[ev]: continue  # 自己セルは除く（thesis_method_eval と同じ）
            if cell in C[ev]: X[i, j] = 1.0 if C[ev][cell][0] else 0.0
    return X


def score_matrix(X, sidx, k):
    M = np.full((len(SUBJ), k), np.nan)
    for s in range(len(SUBJ)):
        m = sidx == s
        for j in range(k):
            v = X[m, j]; v = v[~np.isnan(v)]
            if len(v): M[s, j] = v.mean()
    return M


def metrics(M):
    from scipy.stats import spearmanr
    rng_ = np.nanmax(M, 1) - np.nanmin(M, 1); strict = np.nanmean(M, 0); rs = []
    for a, b in itertools.combinations(range(M.shape[1]), 2):
        ok = ~np.isnan(M[:, a]) & ~np.isnan(M[:, b])
        if ok.sum() >= 4:
            r = spearmanr(M[ok, a], M[ok, b])[0]
            if r == r: rs.append(r)
    return dict(per_subject_range=100 * float(np.nanmean(rng_)), judge_spread=100 * float(np.nanmax(strict) - np.nanmin(strict)),
                rank_agreement=float(np.mean(rs)) if rs else float("nan"))


def apply_threshold(X, cells, C, evals, taus):
    Xt = X.copy()
    for j, ev in enumerate(evals):
        for i, cell in enumerate(cells):
            if np.isnan(X[i, j]): continue
            v, c = C[ev][cell]; Xt[i, j] = 1.0 if (v and c >= taus[j]) else 0.0
    return Xt


def choose_taus(X, cells, C, evals):
    rates = [np.nanmean(X[:, j]) for j in range(len(evals))]; target = float(np.median(rates)); taus = []
    for j, ev in enumerate(evals):
        best, bt = None, 0.5
        for tau in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            r = np.nanmean([1.0 if (C[ev][cells[i]][0] and C[ev][cells[i]][1] >= tau) else 0.0 for i in range(len(cells)) if not np.isnan(X[i, j])])
            if best is None or abs(r - target) < best: best, bt = abs(r - target), tau
        taus.append(bt)
    return taus


def split_eval(C, evals, cells, n_split=200, seed=42):
    pids = sorted({c[1] for c in cells}); rng = np.random.default_rng(seed)
    sidx = np.array([SUBJ.index(c[0]) for c in cells]); X = matrix(C, evals, cells); k = len(evals)
    out = {m: [] for m in ("raw", "mean_shift", "threshold_mean_shift")}
    for _ in range(n_split):
        perm = rng.permutation(len(pids)); tr = {pids[i] for i in perm[:len(pids) // 2]}
        itr = np.array([c[1] in tr for c in cells]); ite = ~itr
        # raw
        out["raw"].append(metrics(score_matrix(X[ite], sidx[ite], k)))
        # mean shift: 学習集合の評価者平均（自分以外）の差を引く
        Mtr = score_matrix(X[itr], sidx[itr], k); shift = np.nanmean(Mtr, 0) - np.nanmean(np.nanmean(Mtr, 0))
        Mte = score_matrix(X[ite], sidx[ite], k); out["mean_shift"].append(metrics(np.clip(Mte - shift, 0, 1)))
        # threshold then mean shift
        cells_tr = [c for c, f in zip(cells, itr) if f]; taus = choose_taus(X[itr], cells_tr, C, evals)
        Xt = apply_threshold(X, cells, C, evals, taus)
        Mtr2 = score_matrix(Xt[itr], sidx[itr], k); shift2 = np.nanmean(Mtr2, 0) - np.nanmean(np.nanmean(Mtr2, 0))
        Mte2 = score_matrix(Xt[ite], sidx[ite], k); out["threshold_mean_shift"].append(metrics(np.clip(Mte2 - shift2, 0, 1)))
    summ = {}
    for m, rows in out.items():
        summ[m] = {key: dict(mean=round(float(np.mean([r[key] for r in rows])), 2), lo=round(float(np.percentile([r[key] for r in rows], 2.5)), 2),
                             hi=round(float(np.percentile([r[key] for r in rows], 97.5)), 2)) for key in rows[0]}
    return summ


def pair_stats(C, evals, cells):
    out = {}
    for a, b in itertools.combinations(evals, 2):
        ao = bo = both = nei = 0
        for cell in cells:
            if cell not in C[a] or cell not in C[b]: continue
            va, vb = C[a][cell][0], C[b][cell][0]
            ao += va and not vb; bo += vb and not va; both += va and vb; nei += (not va) and (not vb)
        n = ao + bo + both + nei
        if not n: continue
        pa, pb = (ao + both) / n, (bo + both) / n
        dom, d, o = (a, ao, bo) if ao >= bo else (b, bo, ao)
        sh = 100 * d / (d + o) if d + o else float("nan")
        base = 100 * (W.baseline(pa, pb) if dom == a else W.baseline(pb, pa))
        c = dict(n=n, a_only=ao, b_only=bo, both=both, neither=nei, pa=pa, pb=pb)
        out[f"{a}-{b}"] = dict(n=n, n_dis=d + o, share=round(sh, 1), dom=dom, pa=round(100 * pa, 1), pb=round(100 * pb, 1), base=round(base, 1), resid=round(sh - base, 1), kappa=round(W.kappa(c), 3))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v12_outcome"); ap.add_argument("--json", default=None); ap.add_argument("--lang", default="JA")
    args = ap.parse_args()
    C0 = load_conf(os.path.join(ROOT, "results"), args.lang); C1 = load_conf(os.path.join(ROOT, "results", "thesis_exp", args.tag), args.lang)
    evals = [ev for ev in W.EVALS if len(C1[ev]) > 0]
    cells = sorted(set.intersection(*[set(C0[ev]) & set(C1[ev]) for ev in evals]))
    print(f"[{args.lang} 反証後] v1.2 の判定がある評価者: {evals}; 共通セル {len(cells)}")
    res = {"evals": evals, "n_cells": len(cells)}
    for name, C in (("v1.1", C0), ("v1.2", C1)):
        X = matrix(C, evals, cells); sidx = np.array([SUBJ.index(c[0]) for c in cells]); M = score_matrix(X, sidx, len(evals))
        strict = {ev: round(100 * float(np.nanmean(X[:, j])), 1) for j, ev in enumerate(evals)}
        per_subject = {s: {ev: (round(100 * float(M[i, j]), 1) if M[i, j] == M[i, j] else None) for j, ev in enumerate(evals)} for i, s in enumerate(SUBJ)}
        met = metrics(M); pairs = pair_stats(C, evals, cells)
        res[name] = dict(strictness=strict, judge_spread=round(met["judge_spread"], 1), per_subject_range=round(met["per_subject_range"], 1), rank_agreement=round(met["rank_agreement"], 3), per_subject=per_subject, pairs=pairs,
                         split=split_eval(C, evals, cells))
        print(f"  {name}: 判定率 {strict} 開き {met['judge_spread']:.1f} 幅 {met['per_subject_range']:.1f} 順位一致 {met['rank_agreement']:.2f}")
        for pr, d in pairs.items(): print(f"    {pr}: dis={d['n_dis']} share={d['share']} dom={d['dom']} base={d['base']} resid={d['resid']:+.1f} κ={d['kappa']}")
        for m, d in res[name]["split"].items(): print(f"    {m}: 幅 {d['per_subject_range']['mean']} [{d['per_subject_range']['lo']}, {d['per_subject_range']['hi']}] 開き {d['judge_spread']['mean']} 順位一致 {d['rank_agreement']['mean']}")
    # v1.1 → v1.2 のセル一致（評価者ごと）
    res["cell_change"] = {}
    for ev in evals:
        a = np.array([C0[ev][c][0] for c in cells if c[0] != W.SELF_OF[ev]]); b = np.array([C1[ev][c][0] for c in cells if c[0] != W.SELF_OF[ev]])
        res["cell_change"][ev] = dict(n=int(len(a)), agree=round(100 * float((a == b).mean()), 1), v11_rate=round(100 * float(a.mean()), 1), v12_rate=round(100 * float(b.mean()), 1),
                                      to_no=int((a & ~b).sum()), to_yes=int((~a & b).sum()))
        print(f"  {ev}: v1.1→v1.2 一致 {res['cell_change'][ev]['agree']}%  あり→なし {res['cell_change'][ev]['to_no']}  なし→あり {res['cell_change'][ev]['to_yes']}")
    if args.json: json.dump(res, open(args.json, "w"), indent=1, ensure_ascii=False); print("wrote", args.json)


if __name__ == "__main__":
    main()
