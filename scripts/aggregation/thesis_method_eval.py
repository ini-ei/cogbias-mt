#!/usr/bin/env python3
"""
thesis_method_eval.py — 補正法を「提案手法」として未知データで評価する（API 不要）。
  .venv/bin/python scripts/thesis_method_eval.py → results/thesis/method_eval.json + stdout

問題を学習集合（40 問）とテスト集合（40 問）に無作為分割し、学習集合で各補正法のパラメータを推定、
テスト集合で「評価者を替えても同じ値が出るか」（整合性）を測る。200 回の分割で平均と 95% 区間。

比較する方法（各評価者 j・被験 s について補正後スコア ŝ_js を出す）:
  M0 raw           : 観測バイアス率そのまま
  M1 latent-RG     : 学習集合で潜在クラス（4 者）を推定し、TPR̃/FPR̃ で Rogan–Gladen 補正（切り詰め）
  M2 conf-match    : 学習集合で評価者ごとに確信度閾値 τ_j を選び、判定率がパネル中央値に揃うようにする（モデル非依存）
  M3 M2 → M1       : 確信度で閾値化した判定に潜在クラス補正を重ねる
  M4 mean-shift    : 学習集合での評価者の平均厳格度の差を引く（最も単純な較正）
  M5 M2 → M4       : 確信度閾値化の後に平均差を引く（提案手法の候補）
  M6 proxy-majority: Zhang et al. (2026, arXiv:2609.12002) の disagreement-based 推定。学習集合で、他の評価者の多数決を
                     正解の代わりにして各評価者の FPR・FNR を数え、テスト集合の観測率を Rogan–Gladen 式で補正する。
                     他の評価者が 2 つしかないセルは両者が一致したときだけ使い、同点は除く（Zhang らは 5 評価者以上を想定）。
整合性の指標（テスト集合）:
  per-subject range: 被験ごとの評価者間の最大−最小（pp）の 7 被験平均
  judge spread     : 評価者の平均スコア（自分以外の被験）の最大−最小（pp）
  panel swing      : 3 評価者パネル（4 通り）の平均スコアの被験ごとの最大−最小の平均（pp）
  rank agreement   : 評価者ペア間の被験順位 Spearman の平均
"""
import json, os, sys, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ.setdefault("SUPP", "")
from wiiat_numbers import load_panel, SUBJECTS, EVALS, SELF_OF, OPEN, EVAL_DIRS  # noqa: E402
from thesis_latent_class import build, em, rogan_gladen  # noqa: E402

R = os.path.join(ROOT, "results"); OUT = os.path.join(R, "thesis", "method_eval.json")
N_SPLIT, SEED = 200, 42
COND = [("JA", 0, "JA_T2"), ("JA", 1, "JA_T4"), ("EN", 0, "EN_T2"), ("EN", 1, "EN_T4")]
NAME = {"GPT-5.4": "GPT-5.4", "Opus4.7": "Opus 4.7", "Gemini3.1": "Gemini 3.1", "LLaMA4": "LLaMA 4"}


def load_conf(L, t):
    base = f"{R}/phase5_v2" if L == "JA" else f"{R}/phase5_v2_en"; C = {}
    for ev, (jf, jo, ef, eo, ep) in EVAL_DIRS.items():
        C[ev] = {}
        for m in SUBJECTS:
            d = (jo if m in OPEN else jf) if L == "JA" else (eo if m in OPEN else ef)
            paths = [f"{base}/{d}/{m}_results.json"] + ([f"{base}/{ep}/{m}_results.json"] if (L == "EN" and ep and m not in OPEN) else [])
            for pth in paths:
                if not os.path.exists(pth): continue
                for r in json.load(open(pth)):
                    k = f"turn{2 if t == 0 else 4}"
                    if isinstance(r, dict) and r.get("problem_id") and r.get(f"{k}_bias_confidence") is not None:
                        C[ev][(m, r["problem_id"])] = (bool(r.get(f"{k}_bias_detected")), float(r[f"{k}_bias_confidence"]))
    return C


def score_matrix(X, sidx):
    """subjects × judges observed rate (nan for self cells)"""
    M = np.full((len(SUBJECTS), 4), np.nan)
    for s in range(len(SUBJECTS)):
        m = sidx == s
        for j in range(4):
            v = X[m, j]; v = v[~np.isnan(v)]
            if len(v): M[s, j] = v.mean()
    return M


def metrics(M):
    from scipy.stats import spearmanr
    rng_ = np.nanmax(M, 1) - np.nanmin(M, 1)
    strict = np.nanmean(M, 0)
    panels = []
    for combo in itertools.combinations(range(4), 3):
        panels.append(np.nanmean(M[:, list(combo)], 1))
    P = np.column_stack(panels); swing = (np.nanmax(P, 1) - np.nanmin(P, 1))
    rs = []
    for a, b in itertools.combinations(range(4), 2):
        ok = ~np.isnan(M[:, a]) & ~np.isnan(M[:, b])
        if ok.sum() >= 4:
            r = spearmanr(M[ok, a], M[ok, b])[0]
            if r == r: rs.append(r)
    return dict(per_subject_range=100 * float(np.nanmean(rng_)), judge_spread=100 * float(np.nanmax(strict) - np.nanmin(strict)),
                panel_swing=100 * float(np.nanmean(swing)), rank_agreement=float(np.mean(rs)) if rs else float("nan"))


def apply_threshold(X, cells, C, taus):
    Xt = X.copy()
    for j, ev in enumerate(EVALS):
        for i, cell in enumerate(cells):
            if np.isnan(X[i, j]): continue
            v, c = C[ev][cell]; Xt[i, j] = 1.0 if (v and c >= taus[j]) else 0.0
    return Xt


def choose_taus(X, cells, C):
    """学習集合: 各評価者の閾値を、判定率が 4 評価者の生の判定率の中央値に最も近づく値（候補 0.5..0.95）に選ぶ"""
    rates = [np.nanmean(X[:, j]) for j in range(4)]; target = float(np.median(rates)); taus = []
    for j, ev in enumerate(EVALS):
        best, bt = None, 0.5
        for tau in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            r = np.nanmean([1.0 if (C[ev][cells[i]][0] and C[ev][cells[i]][1] >= tau) else 0.0 for i in range(len(cells)) if not np.isnan(X[i, j])])
            if best is None or abs(r - target) < best: best, bt = abs(r - target), tau
        taus.append(bt)
    return taus


def zhang_rates(X):
    """Zhang et al. の disagreement-based 推定: 他の評価者の多数決を proxy 真値として FPR_j, FNR_j を数える"""
    tpr = np.zeros(4); fpr = np.zeros(4)
    for j in range(4):
        others = [k for k in range(4) if k != j]
        o_j = ~np.isnan(X[:, j]); Xo = X[:, others]; cnt = (~np.isnan(Xo)).sum(1); mv = np.nansum(Xo, 1) / np.maximum(cnt, 1)
        use = o_j & (cnt > 0) & (mv != 0.5)
        proxy = mv[use] > 0.5; xj = X[use, j] > 0.5
        n1 = proxy.sum(); n0 = (~proxy).sum()
        fnr = float((~xj & proxy).sum() / n1) if n1 else 0.0
        fpr[j] = float((xj & ~proxy).sum() / n0) if n0 else 0.0
        tpr[j] = 1.0 - fnr
    return tpr, fpr


def main():
    rng = np.random.default_rng(SEED); J = {L: load_panel(R, L) for L in ("JA", "EN")}; out = {}
    for L, t, key in COND:
        X, sidx, pids = build(J, L, t); cells = [(SUBJECTS[s], p) for s, p in zip(sidx, pids)]; C = load_conf(L, t)
        upids = sorted(set(pids)); pid_arr = np.array(pids)
        res = {m: [] for m in ("M0_raw", "M1_latent", "M2_conf", "M3_conf_latent", "M4_meanshift", "M5_conf_meanshift", "M6_zhang")}; taus_hist = []; zr_hist = []
        for _ in range(N_SPLIT):
            perm = rng.permutation(len(upids)); tr = set(np.array(upids)[perm[:40]]); te = set(np.array(upids)[perm[40:]])
            itr = np.array([p in tr for p in pid_arr]); ite = ~itr
            Xtr, str_ = X[itr], sidx[itr]; Xte, ste = X[ite], sidx[ite]; cells_tr = [c for c, k in zip(cells, itr) if k]; cells_te = [c for c, k in zip(cells, ite) if k]
            # M0
            Mte = score_matrix(Xte, ste); res["M0_raw"].append(metrics(Mte))
            # M1
            f = em(Xtr, str_); M1 = np.column_stack([rogan_gladen(Mte[:, j], f["tpr"][j], f["fpr"][j]) for j in range(4)]); res["M1_latent"].append(metrics(M1))
            # M2
            taus = choose_taus(Xtr, cells_tr, C); taus_hist.append(taus); Xte_t = apply_threshold(Xte, cells_te, C, taus); M2 = score_matrix(Xte_t, ste); res["M2_conf"].append(metrics(M2))
            # M3
            Xtr_t = apply_threshold(Xtr, cells_tr, C, taus); f3 = em(Xtr_t, str_); M3 = np.column_stack([rogan_gladen(M2[:, j], f3["tpr"][j], f3["fpr"][j]) for j in range(4)]); res["M3_conf_latent"].append(metrics(M3))
            # M4: subtract judge mean offset (training) from test scores
            Mtr = score_matrix(Xtr, str_); off = np.nanmean(Mtr, 0) - np.nanmean(Mtr); M4 = np.clip(Mte - off[None, :], 0, 1); res["M4_meanshift"].append(metrics(M4))
            # M5: confidence threshold (train) then mean-shift of the thresholded training scores
            Mtr_t = score_matrix(Xtr_t, str_); off5 = np.nanmean(Mtr_t, 0) - np.nanmean(Mtr_t); M5 = np.clip(M2 - off5[None, :], 0, 1); res["M5_conf_meanshift"].append(metrics(M5))
            # M6: Zhang-style proxy-majority FPR/FNR (training) + Rogan–Gladen on test rates
            ztpr, zfpr = zhang_rates(Xtr); zr_hist.append(np.concatenate([ztpr, zfpr])); M6 = np.column_stack([rogan_gladen(Mte[:, j], ztpr[j], zfpr[j]) for j in range(4)]); res["M6_zhang"].append(metrics(M6))
        summary = {}
        for m, lst in res.items():
            summary[m] = {k: dict(mean=round(float(np.nanmean([d[k] for d in lst])), 1 if k != "rank_agreement" else 3), ci=[round(float(np.nanpercentile([d[k] for d in lst], q)), 1 if k != "rank_agreement" else 3) for q in (2.5, 97.5)]) for k in lst[0]}
        th = np.array(taus_hist); summary["tau_median"] = {ev: float(np.median(th[:, j])) for j, ev in enumerate(EVALS)}
        zr = np.array(zr_hist); summary["zhang_rates_median"] = {ev: dict(tpr=round(100 * float(np.median(zr[:, j])), 1), fpr=round(100 * float(np.median(zr[:, 4 + j])), 1)) for j, ev in enumerate(EVALS)}
        out[key] = summary
        print(f"\n### {key} (train 40 / test 40 problems, {N_SPLIT} splits)")
        print("| method | per-subject range | judge spread | panel swing | rank agreement |")
        for m in res:
            s = summary[m]; print(f"| {m} | {s['per_subject_range']['mean']} [{s['per_subject_range']['ci'][0]}, {s['per_subject_range']['ci'][1]}] | {s['judge_spread']['mean']} [{s['judge_spread']['ci'][0]}, {s['judge_spread']['ci'][1]}] | {s['panel_swing']['mean']} [{s['panel_swing']['ci'][0]}, {s['panel_swing']['ci'][1]}] | {s['rank_agreement']['mean']} |")
        print("  median thresholds (M2): " + ", ".join(f"{NAME[ev]} {v}" for ev, v in summary["tau_median"].items()))
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False, default=float); print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
