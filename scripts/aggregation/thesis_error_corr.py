#!/usr/bin/env python3
"""
thesis_error_corr.py — 評価者ペアの誤り相関（φ）と Kish の有効評価者数 n_eff（卒論 5.4 節、仮定 (i) の補足）。
  .venv/bin/python scripts/thesis_error_corr.py  → results/thesis/error_corr.json + stdout

定義:
  潜在クラス z = 1[q >= 0.5]（4 評価者の Dawid–Skene 推定の事後確率を 2 値化）。
  評価者 j の誤り e_j = 1[x_j != z]。ペア (j,k) の誤り相関 φ_jk = 両者が観測されたセル上の e_j, e_k のピアソン相関（= φ 係数）。
  判定そのものの相関 r_jk も併記。
  Kish の有効評価者数 n_eff = k / (1 + (k-1) * ρ̄)、ρ̄ は 6 ペアの φ の平均（負は 0 に切り詰め）。k = 4。
  参考: フロンティア被験モデルのセルは 3 判定しか持たないので、ペアごとの n は 400〜480。
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ.setdefault("SUPP", "")
from wiiat_numbers import load_panel, EVALS  # noqa: E402
from thesis_latent_class import build, em  # noqa: E402

OUT = os.path.join(ROOT, "results", "thesis", "error_corr.json")


def phi(a, b):
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    R = os.path.join(ROOT, "results")
    J = {L: load_panel(R, L) for L in ("JA", "EN")}
    out = {}
    for lang in ("JA", "EN"):
        for t, tn in ((0, "T2"), (1, "T4")):
            X, sidx, _ = build(J, lang, t)
            fit = em(X, sidx)
            z = (fit["q"] >= 0.5).astype(float)
            obs = ~np.isnan(X)
            pairs, phis = {}, []
            for a in range(4):
                for b in range(a + 1, 4):
                    m = obs[:, a] & obs[:, b]
                    ea = (X[m, a] != z[m]).astype(float); eb = (X[m, b] != z[m]).astype(float)
                    p = phi(ea, eb); r = phi(X[m, a], X[m, b])
                    pairs[f"{EVALS[a]}-{EVALS[b]}"] = dict(phi_error=round(p, 3), r_judgment=round(r, 3), n=int(m.sum()),
                                                            err_rate_a=round(float(ea.mean()), 3), err_rate_b=round(float(eb.mean()), 3))
                    phis.append(p)
            phis = np.array(phis); rho = float(np.nanmean(np.clip(phis, 0, None)))
            n_eff = 4 / (1 + 3 * rho)
            out[f"{lang}_{tn}"] = dict(pairs=pairs, mean_phi_error=round(float(np.nanmean(phis)), 3),
                                     mean_phi_clipped=round(rho, 3), n_eff=round(n_eff, 2))
            print(f"{lang} {tn}: mean phi(error) = {np.nanmean(phis):+.3f}  n_eff = {n_eff:.2f}")
            for k, v in pairs.items():
                print(f"   {k:24s} phi_err {v['phi_error']:+.3f}  r_judg {v['r_judgment']:+.3f}  n={v['n']}")
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
