#!/usr/bin/env python3
"""
thesis_latent_validation.py — 潜在クラス推定の妥当性検証と拡張（卒論 5–7 章の補強、API 不要）。
  .venv/bin/python scripts/thesis_latent_validation.py  → results/thesis/latent_validation.json + stdout

1  leave-one-judge-out (LOJO): 3 評価者で推定した事後確率が残り 1 者の判定をどれだけ予測するか（AUC）、
   LOJO 事後から推定した TPR/FPR と 4 者推定の一致
2  条件付き独立の診断: 事後重み付き級内相関（ペア別）、応答パターンの適合度 G²（パラメトリック bootstrap p）
3  モデル比較: TPR/FPR 共通 vs 被験群別（frontier/open）vs バイアス種別、2 クラス vs 3 クラス（対数尤度・BIC）
4  確信度による補正: 判定者ごとの ROC（LOJO 基準）、閾値を上げた場合の TPR/FPR と評価者間の開き
5  日英差の分解: 観測差 = 被験成分（π の差）＋ 評価者成分（TPR/FPR の言語差）
6  ルーブリック差し替え: Claude 作ルーブリック下の GPT / Opus 判定の TPR/FPR（LOJO 基準で推定）
7  T2→T4 遷移（事後確率ベースの修正率）
8  評価者は何人必要か: 2 者・3 者の部分集合で推定した π̂ と 4 者推定の差
"""
import json, os, sys, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ.setdefault("SUPP", "")
from wiiat_numbers import load_panel, SUBJECTS, EVALS, SELF_OF, OPEN  # noqa: E402
from thesis_latent_class import build, em, observed_rates, rogan_gladen, spread_stats  # noqa: E402

R = os.path.join(ROOT, "results"); OUT = os.path.join(R, "thesis", "latent_validation.json")
SEED = 42; NAME = {"GPT-5.4": "GPT-5.4", "Opus4.7": "Opus 4.7", "Gemini3.1": "Gemini 3.1", "LLaMA4": "LLaMA 4"}
COND = [("JA", 0, "JA_T2"), ("JA", 1, "JA_T4"), ("EN", 0, "EN_T2"), ("EN", 1, "EN_T4")]


# ------------------------------------------------------------- general K-class LCA with group-specific priors
def em_lca(X, gidx, K=2, n_groups=None, theta_groups=None, n_iter=500, tol=1e-7, seed=0, init=None):
    """X: cells × J (nan = missing); gidx: prior group per cell; theta_groups: optional array giving, per cell,
    the index of the item-parameter set (allows judge params to vary by subject group / bias type)."""
    n, J = X.shape; obs = ~np.isnan(X); Xf = np.nan_to_num(X)
    G = n_groups or (gidx.max() + 1); T = 1 if theta_groups is None else (theta_groups.max() + 1)
    tg = np.zeros(n, int) if theta_groups is None else theta_groups
    rng = np.random.default_rng(seed)
    if init is None:
        # init from majority-vote quantiles
        mv = np.where(obs.sum(1) > 0, Xf.sum(1) / np.maximum(obs.sum(1), 1), 0.5)
        q = np.zeros((n, K))
        if K == 2:
            q[:, 1] = mv; q[:, 0] = 1 - mv
        else:
            cuts = np.quantile(mv, np.linspace(0, 1, K + 1)[1:-1])
            lab = np.searchsorted(cuts, mv, side="right")
            q[np.arange(n), lab] = 0.8; q += 0.2 / K
        q = q + rng.random((n, K)) * 0.05; q /= q.sum(1, keepdims=True)
    else:
        q = init
    pi = np.full((G, K), 1 / K); theta = np.full((T, K, J), 0.5); ll_old = -np.inf
    for it in range(n_iter):
        # M
        for g in range(G):
            m = gidx == g
            pi[g] = np.clip(q[m].mean(0), 1e-4, 1) if m.any() else 1 / K; pi[g] /= pi[g].sum()
        for t in range(T):
            mt = tg == t
            for k in range(K):
                w = q[mt, k]
                for j in range(J):
                    o = obs[mt, j]
                    theta[t, k, j] = np.clip((w[o] * Xf[mt][o, j]).sum() / max(w[o].sum(), 1e-9), 1e-4, 1 - 1e-4)
        # E
        logp = np.log(pi[gidx])  # n × K
        for j in range(J):
            o = obs[:, j]; x = Xf[:, j]
            th = theta[tg][:, :, j]  # n × K
            logp = logp + (o[:, None] * (x[:, None] * np.log(th) + (1 - x[:, None]) * np.log(1 - th)))
        mx = logp.max(1, keepdims=True); pj = np.exp(logp - mx); ll = float((np.log(pj.sum(1)) + mx[:, 0]).sum())
        q_new = pj / pj.sum(1, keepdims=True)
        if abs(ll - ll_old) < tol:
            q = q_new; break
        q = q_new; ll_old = ll
    n_params = G * (K - 1) + T * K * J
    return dict(pi=pi, theta=theta, q=q, loglik=ll, n_params=n_params, bic=-2 * ll + n_params * np.log(n))


def order_classes(fit):
    """K=2: class 1 = 'biased' (higher mean theta). K=3: sort by mean theta ascending."""
    th = fit["theta"]; order = np.argsort(th.mean((0, 2)))
    fit["theta"] = th[:, order, :]; fit["pi"] = fit["pi"][:, order]; fit["q"] = fit["q"][:, order]
    return fit


def auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); s = np.asarray(score, float)
    if y.all() or (~y).all(): return float("nan")
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    rng = np.random.default_rng(SEED)
    J = {L: load_panel(R, L) for L in ("JA", "EN")}
    out = {}
    data = {}
    for L, t, key in COND:
        X, sidx, pids = build(J, L, t); data[key] = (X, sidx, pids)

    # ---------------------------------------------------------------- 1. LOJO
    print("\n### 1. Leave-one-judge-out: AUC of 3-judge posterior for the held-out judge; TPR/FPR from LOJO posterior vs 4-judge fit")
    out["lojo"] = {}
    for L, t, key in COND:
        X, sidx, pids = data[key]; full = em(X, sidx); res = {}
        for j, ev in enumerate(EVALS):
            keep = [k for k in range(4) if k != j]
            f3 = em(X[:, keep], sidx)
            q = f3["q"]; o = ~np.isnan(X[:, j]); y = X[o, j] > 0.5
            a = auc(q[o], y)
            # majority-vote baseline of the other judges
            mv = np.nanmean(X[:, keep], 1); a_mv = auc(mv[o], y)
            tpr = float((q[o] * X[o, j]).sum() / q[o].sum()); fpr = float(((1 - q[o]) * X[o, j]).sum() / (1 - q[o]).sum())
            # bootstrap CI of AUC over problems
            up = sorted(set(pids)); pm = {p: i for i, p in enumerate(up)}; cl = np.array([pm[p] for p in pids]); mem = [np.where(cl == i)[0] for i in range(len(up))]
            aucs = []
            for _ in range(300):
                idx = np.concatenate([mem[c] for c in rng.integers(0, len(up), len(up))])
                f = em(X[idx][:, keep], sidx[idx], n_iter=100); oo = ~np.isnan(X[idx, j])
                aucs.append(auc(f["q"][oo], X[idx][oo, j] > 0.5))
            res[ev] = dict(auc=round(a, 3), auc_ci=[round(float(np.nanpercentile(aucs, 2.5)), 3), round(float(np.nanpercentile(aucs, 97.5)), 3)], auc_majority=round(a_mv, 3),
                           tpr_lojo=round(100 * tpr, 1), fpr_lojo=round(100 * fpr, 1), tpr_full=round(100 * full["tpr"][j], 1), fpr_full=round(100 * full["fpr"][j], 1))
            print(f"| {key} | {NAME[ev]} | AUC {a:.3f} [{res[ev]['auc_ci'][0]}, {res[ev]['auc_ci'][1]}] (majority {a_mv:.3f}) | TPR LOJO {100*tpr:.1f} / full {100*full['tpr'][j]:.1f} | FPR LOJO {100*fpr:.1f} / full {100*full['fpr'][j]:.1f} |")
        out["lojo"][key] = res

    # ---------------------------------------------------------------- 2. conditional dependence diagnostics
    print("\n### 2. Conditional independence: posterior-weighted within-class correlations (pairs) and pattern goodness-of-fit")
    out["dependence"] = {}
    for L, t, key in COND:
        X, sidx, pids = data[key]; f = em(X, sidx); q = f["q"]; res = {"within_class_corr": {}}
        for a, b in itertools.combinations(range(4), 2):
            o = ~np.isnan(X[:, a]) & ~np.isnan(X[:, b])
            xa, xb = X[o, a], X[o, b]
            def wcorr(w):
                w = w / w.sum(); ma, mb = (w * xa).sum(), (w * xb).sum()
                cov = (w * (xa - ma) * (xb - mb)).sum(); va = (w * (xa - ma) ** 2).sum(); vb = (w * (xb - mb) ** 2).sum()
                return float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else float("nan")
            res["within_class_corr"][f"{EVALS[a]}-{EVALS[b]}"] = dict(biased=round(wcorr(q[o]), 3), not_biased=round(wcorr(1 - q[o]), 3), n=int(o.sum()))
        # pattern GOF: group by (subject, observed set)
        def gof(Xm, sm, fit):
            G2 = 0.0; df = 0
            for s in range(len(SUBJECTS)):
                m = sm == s
                if not m.any(): continue
                obsj = [j for j in range(4) if not np.isnan(Xm[m][0, j])]
                pats = list(itertools.product([0, 1], repeat=len(obsj)))
                # expected per pattern
                th1 = fit["tpr"][obsj]; th0 = fit["fpr"][obsj]; pi_s = fit["pi"][s]
                E = []; O = []
                sub = Xm[m][:, obsj]
                for pat in pats:
                    pat = np.array(pat)
                    p1 = np.prod(th1 ** pat * (1 - th1) ** (1 - pat)); p0 = np.prod(th0 ** pat * (1 - th0) ** (1 - pat))
                    E.append(m.sum() * (pi_s * p1 + (1 - pi_s) * p0)); O.append(int((sub == pat).all(1).sum()))
                E = np.array(E); O = np.array(O)
                G2 += 2 * float(np.sum(np.where(O > 0, O * np.log(np.maximum(O, 1e-12) / np.maximum(E, 1e-12)), 0)))
                df += len(pats) - 1
            return G2, df
        g2, df = gof(X, sidx, f)
        # parametric bootstrap
        sims = []
        for _ in range(200):
            Xs = X.copy()
            z = rng.random(len(X)) < f["pi"][sidx]
            for j in range(4):
                o = ~np.isnan(X[:, j]); p = np.where(z, f["tpr"][j], f["fpr"][j])
                Xs[o, j] = (rng.random(o.sum()) < p[o]).astype(float)
            fs = em(Xs, sidx, n_iter=150); sims.append(gof(Xs, sidx, fs)[0])
        p = float(np.mean(np.array(sims) >= g2))
        res["gof"] = dict(G2=round(g2, 1), df_patterns=df, n_params=int(2 * 4 + len(SUBJECTS)), boot_p=p, boot_G2_median=round(float(np.median(sims)), 1))
        out["dependence"][key] = res
        print(f"[{key}] G² = {g2:.1f} (pattern df {df}, params {res['gof']['n_params']}), parametric-bootstrap p = {p:.3f} (sim median {np.median(sims):.1f})")
        print("   within-class corr (biased / not): " + "; ".join(f"{k}: {v['biased']:+.2f}/{v['not_biased']:+.2f}" for k, v in res["within_class_corr"].items()))

    # ---------------------------------------------------------------- 3. model comparison
    print("\n### 3. Model comparison (log-likelihood, BIC): shared TPR/FPR vs by subject group vs by bias type; 2 vs 3 classes")
    out["model_comparison"] = {}
    for L, t, key in COND:
        X, sidx, pids = data[key]; res = {}
        grp = np.array([1 if SUBJECTS[s] in OPEN else 0 for s in sidx])
        biases = ["confirmation_bias", "anchoring", "representativeness", "framing"]; bt = np.array([biases.index(p.split("-")[0]) for p in pids])
        fits = {"2class_shared": em_lca(X, sidx, 2, len(SUBJECTS)), "2class_by_group": em_lca(X, sidx, 2, len(SUBJECTS), theta_groups=grp), "2class_by_bias": em_lca(X, sidx, 2, len(SUBJECTS), theta_groups=bt)}
        best3 = None
        for sd in range(5):
            f3 = em_lca(X, sidx, 3, len(SUBJECTS), seed=sd)
            if best3 is None or f3["loglik"] > best3["loglik"]: best3 = f3
        fits["3class_shared"] = order_classes(best3)
        for nm, f in fits.items():
            res[nm] = dict(loglik=round(f["loglik"], 1), n_params=int(f["n_params"]), bic=round(float(f["bic"]), 1))
        th = fits["3class_shared"]["theta"][0]  # K × J
        res["3class_theta"] = {f"class{k}": {ev: round(100 * float(th[k, j]), 1) for j, ev in enumerate(EVALS)} for k in range(3)}
        res["3class_pi_mean"] = [round(100 * float(v), 1) for v in fits["3class_shared"]["pi"].mean(0)]
        out["model_comparison"][key] = res
        print(f"[{key}] " + "  ".join(f"{nm}: ll {v['loglik']} BIC {v['bic']} (p={v['n_params']})" for nm, v in res.items() if isinstance(v, dict) and 'bic' in v))
        print("   3-class judge P(biased-call | class): " + " | ".join(f"class{k}: " + ", ".join(f"{NAME[ev]} {res['3class_theta'][f'class{k}'][ev]}" for ev in EVALS) for k in range(3)) + f"; mean class shares {res['3class_pi_mean']}")

    # ---------------------------------------------------------------- 4. confidence-based ROC and thresholds
    print("\n### 4. Confidence scores: ROC vs LOJO reference and thresholded TPR/FPR")
    out["confidence"] = {}
    def load_conf(L, t):
        from wiiat_numbers import EVAL_DIRS
        base = f"{R}/phase5_v2" if L == "JA" else f"{R}/phase5_v2_en"; C = {}
        for ev, (jf, jo, ef, eo, ep) in EVAL_DIRS.items():
            C[ev] = {}
            for m in SUBJECTS:
                d = (jo if m in OPEN else jf) if L == "JA" else (eo if m in OPEN else ef)
                paths = [f"{base}/{d}/{m}_results.json"] + ([f"{base}/{ep}/{m}_results.json"] if (L == "EN" and ep and m not in OPEN) else [])
                for pth in paths:
                    if not os.path.exists(pth): continue
                    for r in json.load(open(pth)):
                        if isinstance(r, dict) and r.get("problem_id") and r.get(f"turn{2 if t == 0 else 4}_bias_confidence") is not None:
                            v = bool(r.get(f"turn{2 if t == 0 else 4}_bias_detected")); c = float(r[f"turn{2 if t == 0 else 4}_bias_confidence"])
                            C[ev][(m, r["problem_id"])] = (v, c)
        return C
    for L, t, key in COND:
        X, sidx, pids = data[key]; C = load_conf(L, t); res = {}
        cells = [(SUBJECTS[s], p) for s, p in zip(sidx, pids)]
        for j, ev in enumerate(EVALS):
            keep = [k for k in range(4) if k != j]; f3 = em(X[:, keep], sidx); ref = f3["q"] >= 0.5
            o = ~np.isnan(X[:, j]); idx = np.where(o)[0]
            score = np.array([(C[ev][cells[i]][1] if C[ev][cells[i]][0] else 1 - C[ev][cells[i]][1]) for i in idx])
            a = auc(score, ref[idx]); a_bin = auc(X[idx, j], ref[idx])
            thr = {}
            for th in (0.5, 0.7, 0.8, 0.9, 0.95):
                pred = score >= th
                tpr = float(pred[ref[idx]].mean()) if ref[idx].any() else float("nan"); fpr = float(pred[~ref[idx]].mean()) if (~ref[idx]).any() else float("nan")
                thr[str(th)] = dict(tpr=round(100 * tpr, 1), fpr=round(100 * fpr, 1), call_rate=round(100 * float(pred.mean()), 1))
            res[ev] = dict(auc_conf=round(a, 3), auc_binary=round(a_bin, 3), thresholds=thr, conf_dist_biased={str(c): int(n) for c, n in zip(*np.unique(np.round([C[ev][cells[i]][1] for i in idx if C[ev][cells[i]][0]], 1), return_counts=True))})
            print(f"| {key} | {NAME[ev]} | AUC conf {a:.3f} vs binary {a_bin:.3f} | " + " ".join(f"thr{th}: TPR {v['tpr']} FPR {v['fpr']} call {v['call_rate']}" for th, v in thr.items()) + " |")
        # spread after thresholding GPT at 0.9 (others unchanged): recompute observed strictness and 4-judge fit + correction
        j = 0; Xg = X.copy(); o = ~np.isnan(X[:, j])
        for i in np.where(o)[0]:
            v, c = C["GPT-5.4"][cells[i]]; Xg[i, j] = 1.0 if (v and c >= 0.9) else 0.0
        BR0 = observed_rates(X, sidx); BR1 = observed_rates(Xg, sidx); s0, _ = spread_stats(BR0); s1, _ = spread_stats(BR1)
        f1 = em(Xg, sidx); C1 = np.column_stack([rogan_gladen(BR1[:, k], f1["tpr"][k], f1["fpr"][k]) for k in range(4)]); sc, rc = spread_stats(C1)
        res["gpt_threshold_0.9"] = dict(strictness_before={ev: round(float(s0[k]), 1) for k, ev in enumerate(EVALS)}, strictness_after={ev: round(float(s1[k]), 1) for k, ev in enumerate(EVALS)},
                                       spread_before=round(float(np.nanmax(s0) - np.nanmin(s0)), 1), spread_after=round(float(np.nanmax(s1) - np.nanmin(s1)), 1),
                                       gpt_tpr_fpr_after=[round(100 * f1["tpr"][0], 1), round(100 * f1["fpr"][0], 1)], spread_after_corrected=round(float(np.nanmax(sc) - np.nanmin(sc)), 1), mean_range_after_corrected=round(float(np.nanmean(rc)), 1))
        print(f"   GPT thresholded at conf≥0.9: strictness {res['gpt_threshold_0.9']['strictness_before']['GPT-5.4']} → {res['gpt_threshold_0.9']['strictness_after']['GPT-5.4']}; spread {res['gpt_threshold_0.9']['spread_before']} → {res['gpt_threshold_0.9']['spread_after']}; GPT TPR/FPR refit {res['gpt_threshold_0.9']['gpt_tpr_fpr_after']}; corrected spread {res['gpt_threshold_0.9']['spread_after_corrected']}, mean range {res['gpt_threshold_0.9']['mean_range_after_corrected']}")
        out["confidence"][key] = res

    # ---------------------------------------------------------------- 5. JA−EN decomposition
    print("\n### 5. JA−EN gap decomposition per judge: subject component (Δπ at JA judge params) + judge component")
    out["gap_decomposition"] = {}
    for t, tn in ((0, "T2"), (1, "T4")):
        XJ, sJ, _ = data[f"JA_{tn}"]; XE, sE, _ = data[f"EN_{tn}"]; fJ = em(XJ, sJ); fE = em(XE, sE)
        BRJ = observed_rates(XJ, sJ); BRE = observed_rates(XE, sE); res = {"per_judge": {}, "per_subject": {}}
        for j, ev in enumerate(EVALS):
            tot, subj, judg = [], [], []
            for s, m in enumerate(SUBJECTS):
                if np.isnan(BRJ[s, j]) or np.isnan(BRE[s, j]): continue
                total = BRJ[s, j] - BRE[s, j]
                sub = (fJ["tpr"][j] - fJ["fpr"][j]) * (fJ["pi"][s] - fE["pi"][s])  # subject component at JA judge params
                sub2 = (fE["tpr"][j] - fE["fpr"][j]) * (fJ["pi"][s] - fE["pi"][s])  # at EN judge params
                subc = 0.5 * (sub + sub2); jud = total - subc
                tot.append(total); subj.append(subc); judg.append(jud)
                res["per_subject"].setdefault(m, {})[ev] = dict(total=round(100 * total, 1), subject=round(100 * subc, 1), judge=round(100 * jud, 1))
            res["per_judge"][ev] = dict(total=round(100 * float(np.mean(tot)), 1), subject=round(100 * float(np.mean(subj)), 1), judge=round(100 * float(np.mean(judg)), 1),
                                        judge_params=dict(tpr_ja=round(100 * fJ["tpr"][j], 1), fpr_ja=round(100 * fJ["fpr"][j], 1), tpr_en=round(100 * fE["tpr"][j], 1), fpr_en=round(100 * fE["fpr"][j], 1)))
            print(f"| {tn} | {NAME[ev]} | mean gap {res['per_judge'][ev]['total']:+.1f} = subject {res['per_judge'][ev]['subject']:+.1f} + judge {res['per_judge'][ev]['judge']:+.1f} |")
        res["latent_pi_gap"] = {m: round(100 * float(fJ["pi"][s] - fE["pi"][s]), 1) for s, m in enumerate(SUBJECTS)}
        out["gap_decomposition"][tn] = res

    # ---------------------------------------------------------------- 6. rubric swap
    print("\n### 6. Claude-authored rubric: TPR/FPR of GPT-5.4 and Opus 4.7 judges under the Claude rubric, LOJO reference (JA)")
    out["rubric_swap"] = {}
    def load_line(paths):
        d = {}
        for pth in paths:
            if not os.path.exists(pth): continue
            for r in json.load(open(pth)):
                if isinstance(r, dict) and r.get("problem_id") and (r.get("turn2_bias_confidence") is not None or r.get("turn4_bias_confidence") is not None):
                    d[(r["model"] if r.get("model") else None, r["problem_id"])] = (bool(r.get("turn2_bias_detected")), bool(r.get("turn4_bias_detected")))
        return d
    lines = {"GPT-5.4": {m: load_line([f"{R}/rebuttal/cross_eval_gpt5_claude_rubric_{'open' if m in OPEN else 'n80'}/{m}_results.json"]) for m in SUBJECTS},
             "Opus4.7": {m: load_line([f"{R}/rebuttal/cross_eval_claude_rubric_opus_open/{m}_results.json" if m in OPEN else f"{R}/phase5_v2/cross_eval_claude_rubric_opus_n80/{m}_results.json"]) for m in SUBJECTS}}
    for t, tn in ((0, "T2"), (1, "T4")):
        X, sidx, pids = data[f"JA_{tn}"]; cells = [(SUBJECTS[s], p) for s, p in zip(sidx, pids)]; res = {}
        for ev in ("GPT-5.4", "Opus4.7"):
            j = EVALS.index(ev); keep = [k for k in range(4) if k != j]; f3 = em(X[:, keep], sidx); q = f3["q"]
            # GPT-rubric line (original) and Claude-rubric line for the same judge, same reference
            def tf(getter):
                num1 = den1 = num0 = den0 = 0.0; n = 0
                for i, (m, p) in enumerate(cells):
                    if m == SELF_OF[ev]: continue
                    v = getter(m, p)
                    if v is None: continue
                    num1 += q[i] * v; den1 += q[i]; num0 += (1 - q[i]) * v; den0 += 1 - q[i]; n += 1
                return round(100 * num1 / den1, 1), round(100 * num0 / den0, 1), n
            orig = tf(lambda m, p: (X[cells.index((m, p)), j] if not np.isnan(X[cells.index((m, p)), j]) else None))
            cl = tf(lambda m, p: (float(lines[ev][m][(None, p)][t]) if (None, p) in lines[ev][m] else (float(next((v for (mm, pp), v in lines[ev][m].items() if pp == p), (None, None))[t]) if any(pp == p for (mm, pp) in lines[ev][m]) else None)))
            res[ev] = dict(gpt_rubric=dict(tpr=orig[0], fpr=orig[1], n=orig[2]), claude_rubric=dict(tpr=cl[0], fpr=cl[1], n=cl[2]))
            print(f"| JA {tn} | {NAME[ev]} | GPT rubric TPR {orig[0]} FPR {orig[1]} (n={orig[2]}) | Claude rubric TPR {cl[0]} FPR {cl[1]} (n={cl[2]}) |")
        out["rubric_swap"][tn] = res

    # ---------------------------------------------------------------- 7. T2→T4 transitions on posteriors
    print("\n### 7. T2→T4 transitions from posteriors: P(T4 biased | T2 biased) (persistence) and P(T4 | T2 not)")
    out["transitions"] = {}
    for L in ("JA", "EN"):
        X2, s2, p2 = data[f"{L}_T2"]; X4, s4, p4 = data[f"{L}_T4"]; q2 = em(X2, s2)["q"]; q4 = em(X4, s4)["q"]
        assert p2 == p4 and (s2 == s4).all()
        res = {}
        for s, m in enumerate(SUBJECTS):
            mm = s2 == s
            persist = float((q2[mm] * q4[mm]).sum() / q2[mm].sum()); emerge = float(((1 - q2[mm]) * q4[mm]).sum() / (1 - q2[mm]).sum())
            res[m] = dict(pi_t2=round(100 * float(q2[mm].mean()), 1), pi_t4=round(100 * float(q4[mm].mean()), 1), persist=round(100 * persist, 1), correction=round(100 * (1 - persist), 1), emerge=round(100 * emerge, 1))
        out["transitions"][L] = res
        print(f"[{L}] " + "; ".join(f"{m}: π2 {v['pi_t2']} π4 {v['pi_t4']} persist {v['persist']} emerge {v['emerge']}" for m, v in res.items()))

    # ---------------------------------------------------------------- 8. how many judges
    print("\n### 8. Judge subsets: mean |π̂_s(subset) − π̂_s(4 judges)| and naive mean-of-observed error")
    out["subsets"] = {}
    for L, t, key in COND:
        X, sidx, pids = data[key]; full = em(X, sidx)["pi"]; res = {}
        for size in (2, 3):
            for combo in itertools.combinations(range(4), size):
                f = em(X[:, list(combo)], sidx); d = np.abs(f["pi"] - full) * 100
                naive = np.abs(np.nanmean(observed_rates(X[:, list(combo)], sidx), 1) - full) * 100
                res["+".join(EVALS[c] for c in combo)] = dict(mean_abs_err=round(float(d.mean()), 1), max_abs_err=round(float(d.max()), 1), naive_mean_abs_err=round(float(np.nanmean(naive)), 1))
        out["subsets"][key] = res
        print(f"[{key}] " + "; ".join(f"{k}: {v['mean_abs_err']} (max {v['max_abs_err']}, naive {v['naive_mean_abs_err']})" for k, v in res.items()))

    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False, default=float); print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
