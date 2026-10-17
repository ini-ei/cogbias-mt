#!/usr/bin/env python3
"""
thesis_latent_dependence.py — 評価者ペア 1 組に局所依存（クラス内の 2×2 同時分布）を許した 2 クラス潜在クラスモデルを当てはめ、
条件付き独立モデルと BIC で比べる（卒論 5.4 節・付録 C の補足。Balasubramanian et al. 2026 / Kohli 2026 への対応）。
  .venv/bin/python scripts/thesis_latent_dependence.py → results/thesis/latent_dependence.json + stdout

モデル: 各セルに潜在クラス Z∈{0,1}、P(Z=1)=π_s（被験ごと）。ペア (a,b) 以外の評価者は Z を与えて条件付き独立（Bernoulli θ_kj）。
       ペア (a,b) はクラス k ごとに自由な 2×2 同時分布 P(x_a, x_b | Z=k)（3 パラメータ）を持つ。独立モデルより 2 パラメータ多い。
欠測（評価者が自分の応答を判定したセル）: ペアの片方が欠測なら同時分布の周辺を用い、M ステップでは期待計数で補う。
出力: 6 ペアそれぞれの対数尤度・BIC と、クラス内のオッズ比。独立モデル（thesis_latent_validation.py の 2class_shared）との差。
"""
import json, os, sys, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ.setdefault("SUPP", "")
from wiiat_numbers import load_panel, SUBJECTS, EVALS  # noqa: E402
from thesis_latent_class import build, em  # noqa: E402

R = os.path.join(ROOT, "results"); OUT = os.path.join(R, "thesis", "latent_dependence.json")
COND = [("JA", 0, "JA_T2"), ("JA", 1, "JA_T4"), ("EN", 0, "EN_T2"), ("EN", 1, "EN_T4")]
NAME = {"GPT-5.4": "GPT-5.4", "Opus4.7": "Opus 4.7", "Gemini3.1": "Gemini 3.1", "LLaMA4": "LLaMA 4"}


def em_dep(X, sidx, a, b, n_iter=500, tol=1e-7):
    n, J = X.shape; obs = ~np.isnan(X); Xf = np.nan_to_num(X); S = len(SUBJECTS)
    base = em(X, sidx)
    q = base["q"].copy()
    pi = np.clip(base["pi"].copy(), 1e-4, 1 - 1e-4)
    theta = np.vstack([base["fpr"], base["tpr"]])  # 2 × J  (class0 = not biased, class1 = biased)
    # pair joint tables T[k, xa, xb], init from independence
    T = np.zeros((2, 2, 2))
    for k in range(2):
        for xa in (0, 1):
            for xb in (0, 1):
                T[k, xa, xb] = (theta[k, a] if xa else 1 - theta[k, a]) * (theta[k, b] if xb else 1 - theta[k, b])
    others = [j for j in range(J) if j not in (a, b)]
    ll_old = -np.inf
    oa, ob = obs[:, a], obs[:, b]; xa_, xb_ = Xf[:, a].astype(int), Xf[:, b].astype(int)
    for it in range(n_iter):
        # ---- E step: class posteriors
        logp = np.column_stack([np.log(1 - pi[sidx]), np.log(pi[sidx])])  # n × 2
        for j in others:
            o = obs[:, j]; x = Xf[:, j]
            for k in range(2):
                th = theta[k, j]; logp[:, k] += o * (x * np.log(th) + (1 - x) * np.log(1 - th))
        pair_lik = np.ones((n, 2))
        for k in range(2):
            both = oa & ob
            pair_lik[both, k] = T[k, xa_[both], xb_[both]]
            only_a = oa & ~ob; pair_lik[only_a, k] = T[k, xa_[only_a], :].sum(1)
            only_b = ~oa & ob; pair_lik[only_b, k] = T[k, :, xb_[only_b]].sum(1)
        logp += np.log(np.maximum(pair_lik, 1e-300))
        mx = logp.max(1, keepdims=True); pj = np.exp(logp - mx); ll = float((np.log(pj.sum(1)) + mx[:, 0]).sum())
        q = pj[:, 1] / pj.sum(1)
        if abs(ll - ll_old) < tol: break
        ll_old = ll
        # ---- M step
        w = np.column_stack([1 - q, q])  # n × 2
        for s in range(S):
            m = sidx == s
            pi[s] = np.clip(q[m].mean(), 1e-4, 1 - 1e-4) if m.any() else 0.5
        for j in others:
            o = obs[:, j]
            for k in range(2):
                theta[k, j] = np.clip((w[o, k] * Xf[o, j]).sum() / max(w[o, k].sum(), 1e-9), 1e-4, 1 - 1e-4)
        for k in range(2):
            C = np.full((2, 2), 1e-3)
            both = oa & ob
            np.add.at(C, (xa_[both], xb_[both]), w[both, k])
            # partial rows: distribute by current conditional
            only_a = oa & ~ob
            for i in np.where(only_a)[0]:
                row = T[k, xa_[i], :]; C[xa_[i], :] += w[i, k] * row / row.sum()
            only_b = ~oa & ob
            for i in np.where(only_b)[0]:
                col = T[k, :, xb_[i]]; C[:, xb_[i]] += w[i, k] * col / col.sum()
            T[k] = C / C.sum()
    n_params = S + 2 * len(others) + 2 * 3
    bic = -2 * ll + n_params * np.log(n)
    # ensure class 1 is the "biased" class (higher marginal call rate)
    return dict(loglik=ll, n_params=n_params, bic=float(bic), T=T, theta=theta, pi=pi)


def main():
    J = {L: load_panel(R, L) for L in ("JA", "EN")}
    base_bic = json.load(open(os.path.join(R, "thesis", "latent_validation.json")))["model_comparison"]
    out = {}
    for L, t, key in COND:
        X, sidx, _ = build(J, L, t); n = len(X)
        b = em(X, sidx); ll0 = b["loglik"]; p0 = len(SUBJECTS) + 8; bic0 = -2 * ll0 + p0 * np.log(n)
        res = {"independence": dict(loglik=round(ll0, 1), n_params=p0, bic=round(float(bic0), 1), bic_validation_script=base_bic[key]["2class_shared"]["bic"]), "pairs": {}}
        best = None
        for a, bb in itertools.combinations(range(4), 2):
            f = em_dep(X, sidx, a, bb)
            ors = []
            for k in range(2):
                T = f["T"][k]; ors.append(float((T[1, 1] * T[0, 0]) / max(T[1, 0] * T[0, 1], 1e-9)))
            name = f"{EVALS[a]}-{EVALS[bb]}"
            res["pairs"][name] = dict(loglik=round(f["loglik"], 1), n_params=f["n_params"], bic=round(f["bic"], 1), delta_bic=round(f["bic"] - float(bic0), 1),
                                      odds_ratio_class0=round(ors[0], 2), odds_ratio_class1=round(ors[1], 2),
                                      tpr_others={EVALS[j]: round(100 * float(f["theta"][1, j]), 1) for j in range(4) if j not in (a, bb)},
                                      fpr_others={EVALS[j]: round(100 * float(f["theta"][0, j]), 1) for j in range(4) if j not in (a, bb)})
            if best is None or f["bic"] < best[1]: best = (name, f["bic"])
        res["best_pair"] = dict(pair=best[0], bic=round(best[1], 1), delta_bic=round(best[1] - float(bic0), 1))
        out[key] = res
        print(f"\n### {key}: independence ll {ll0:.1f} BIC {bic0:.1f} (p={p0})")
        for name, v in res["pairs"].items():
            print(f"   {name:24s} ll {v['loglik']:8.1f} BIC {v['bic']:7.1f} ΔBIC {v['delta_bic']:+7.1f}  OR(class0) {v['odds_ratio_class0']:6.2f} OR(class1) {v['odds_ratio_class1']:6.2f}")
        print(f"   best: {res['best_pair']}")
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False, default=float); print("saved ->", OUT)


if __name__ == "__main__":
    main()
