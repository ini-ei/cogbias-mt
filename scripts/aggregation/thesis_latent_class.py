#!/usr/bin/env python3
"""
thesis_latent_class.py — 人間基準なしで評価モデルの感度・特異度を推定する（Dawid–Skene 型 2 クラス潜在クラスモデル）。
  .venv/bin/python scripts/thesis_latent_class.py     → results/thesis/latent_class.json + stdout

設計:
  セル = (被験, 問題)。観測 = 4 パネル判定者の 0/1 判定（T2, T4 別）。同一モデルセル（判定者が自分の応答を判定）は欠測扱い。
  モデル: 各セルに潜在真値 Z∈{0,1}。P(Z=1)=π_s（被験ごと）。判定者 j は Z=1 のとき確率 TPR_j で 1、Z=0 のとき FPR_j で 1。
          判定者は Z を与えたとき条件付き独立（Dawid–Skene / Hui–Walter の標準仮定。限界として本文に明記）。
  推定: EM（初期値 = 多数決）。識別性のため TPR_j > FPR_j を課す。問題クラスタ bootstrap で CI。
  補正: 判定者別観測率 BR_js を Rogan–Gladen 式 (BR − FPR)/(TPR − FPR) で補正し、判定者間の開き・順位一致が縮むかを見る。
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ.setdefault("SUPP", "")
from wiiat_numbers import load_panel, load_self, SUBJECTS, EVALS, SELF_OF  # noqa: E402

R = os.path.join(ROOT, "results"); OUT = os.path.join(R, "thesis", "latent_class.json")
N_BOOT, SEED = 1000, 42
NAME = {"GPT-5.4": "GPT-5.4", "Opus4.7": "Opus 4.7", "Gemini3.1": "Gemini 3.1", "LLaMA4": "LLaMA 4"}


def build(J, lang, t):
    """→ X (cells × 4) with nan for missing, subj_idx, pid list"""
    rows, sidx, pids = [], [], []
    for si, m in enumerate(SUBJECTS):
        common = sorted(set.union(*[set(J[lang][ev][m]) for ev in EVALS]))
        for p in common:
            r = []
            for ev in EVALS:
                if SELF_OF[ev] == m or p not in J[lang][ev][m]:
                    r.append(np.nan)
                else:
                    r.append(float(J[lang][ev][m][p][t]))
            rows.append(r); sidx.append(si); pids.append(p)
    return np.array(rows), np.array(sidx), pids


def em(X, sidx, n_iter=300, tol=1e-7):
    n, k = X.shape; obs = ~np.isnan(X); Xf = np.nan_to_num(X)
    # init: majority vote
    q = np.where(obs.sum(1) > 0, Xf.sum(1) / np.maximum(obs.sum(1), 1), 0.5)
    pi_s = np.zeros(len(SUBJECTS)); tpr = np.full(k, 0.8); fpr = np.full(k, 0.2)
    for _ in range(n_iter):
        # M-step
        for s in range(len(SUBJECTS)):
            m = sidx == s
            pi_s[s] = np.clip(q[m].mean(), 1e-4, 1 - 1e-4) if m.any() else 0.5
        for j in range(k):
            o = obs[:, j]
            tpr[j] = np.clip((q[o] * Xf[o, j]).sum() / max((q[o]).sum(), 1e-9), 1e-4, 1 - 1e-4)
            fpr[j] = np.clip(((1 - q[o]) * Xf[o, j]).sum() / max((1 - q[o]).sum(), 1e-9), 1e-4, 1 - 1e-4)
            if tpr[j] < fpr[j]:  # identifiability
                tpr[j], fpr[j] = fpr[j], tpr[j]
        # E-step
        l1 = np.log(pi_s[sidx]); l0 = np.log(1 - pi_s[sidx])
        for j in range(k):
            o = obs[:, j]; x = Xf[:, j]
            l1 = l1 + o * (x * np.log(tpr[j]) + (1 - x) * np.log(1 - tpr[j]))
            l0 = l0 + o * (x * np.log(fpr[j]) + (1 - x) * np.log(1 - fpr[j]))
        q_new = 1 / (1 + np.exp(l0 - l1))
        if np.abs(q_new - q).max() < tol:
            q = q_new; break
        q = q_new
    ll = float(np.logaddexp(l1, l0).sum())
    return dict(pi=pi_s.copy(), tpr=tpr.copy(), fpr=fpr.copy(), q=q, loglik=ll)


def observed_rates(X, sidx):
    """BR_js: judge j observed rate on subject s (nan if same-model)"""
    out = np.full((len(SUBJECTS), X.shape[1]), np.nan)
    for s in range(len(SUBJECTS)):
        m = sidx == s
        for j in range(X.shape[1]):
            v = X[m, j]; v = v[~np.isnan(v)]
            if len(v): out[s, j] = v.mean()
    return out


def rogan_gladen(br, tpr, fpr):
    return np.clip((br - fpr) / np.maximum(tpr - fpr, 1e-6), 0, 1)


def kendall_w(M):
    """判定者ペアごとに、両者が採点した被験（5–6 体）上の Spearman 順位相関を取り、6 ペアの平均を返す（判定者間の被験順位の一致度）。"""
    from scipy.stats import spearmanr
    rs = []
    for a in range(M.shape[1]):
        for b in range(a + 1, M.shape[1]):
            ok = ~np.isnan(M[:, a]) & ~np.isnan(M[:, b])
            if ok.sum() >= 4:
                r = spearmanr(M[ok, a], M[ok, b])[0]
                if r == r: rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def spread_stats(M):
    """judge-level spread on the shared subject set: for each subject, max−min over judges; and global strictness per judge (mean over non-self subjects)"""
    strict = np.nanmean(M, 0) * 100
    per_subject_range = (np.nanmax(M, 1) - np.nanmin(M, 1)) * 100
    return strict, per_subject_range


def run(J, lang, t, rng):
    X, sidx, pids = build(J, lang, t)
    fit = em(X, sidx)
    BR = observed_rates(X, sidx)
    corrected = np.column_stack([rogan_gladen(BR[:, j], fit["tpr"][j], fit["fpr"][j]) for j in range(4)])
    strict_b, rng_b = spread_stats(BR); strict_a, rng_a = spread_stats(corrected)
    res = dict(n_cells=int(len(X)), loglik=fit["loglik"],
               judges={ev: dict(tpr=round(100 * fit["tpr"][j], 1), fpr=round(100 * fit["fpr"][j], 1), strictness_obs=round(float(strict_b[j]), 1), strictness_corrected=round(float(strict_a[j]), 1)) for j, ev in enumerate(EVALS)},
               pi=dict(zip(SUBJECTS, [round(100 * v, 1) for v in fit["pi"]])),
               br_obs={m: {ev: (None if np.isnan(BR[s, j]) else round(100 * BR[s, j], 1)) for j, ev in enumerate(EVALS)} for s, m in enumerate(SUBJECTS)},
               br_corrected={m: {ev: (None if np.isnan(corrected[s, j]) else round(100 * corrected[s, j], 1)) for j, ev in enumerate(EVALS)} for s, m in enumerate(SUBJECTS)},
               spread_obs=round(float(np.nanmax(strict_b) - np.nanmin(strict_b)), 1), spread_corrected=round(float(np.nanmax(strict_a) - np.nanmin(strict_a)), 1),
               per_subject_range_obs={m: round(float(v), 1) for m, v in zip(SUBJECTS, rng_b)}, per_subject_range_corrected={m: round(float(v), 1) for m, v in zip(SUBJECTS, rng_a)},
               mean_range_obs=round(float(np.nanmean(rng_b)), 1), mean_range_corrected=round(float(np.nanmean(rng_a)), 1),
               rank_corr_obs=round(kendall_w(BR), 3), rank_corr_corrected=round(kendall_w(corrected), 3))
    # bootstrap over problems (clusters): resample problem ids, keep all cells of drawn problems
    upids = sorted(set(pids)); pmap = {p: i for i, p in enumerate(upids)}; cl = np.array([pmap[p] for p in pids])
    members = [np.where(cl == i)[0] for i in range(len(upids))]
    tprs, fprs, pis, spreads, ranges = [], [], [], [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[c] for c in rng.integers(0, len(upids), len(upids))])
        f = em(X[idx], sidx[idx], n_iter=150)
        tprs.append(f["tpr"]); fprs.append(f["fpr"]); pis.append(f["pi"])
        B = observed_rates(X[idx], sidx[idx]); C = np.column_stack([rogan_gladen(B[:, j], f["tpr"][j], f["fpr"][j]) for j in range(4)])
        sa, ra = spread_stats(C); spreads.append(np.nanmax(sa) - np.nanmin(sa)); ranges.append(np.nanmean(ra))
    tprs, fprs, pis = np.array(tprs), np.array(fprs), np.array(pis)
    for j, ev in enumerate(EVALS):
        res["judges"][ev]["tpr_ci"] = [round(100 * float(np.percentile(tprs[:, j], q)), 1) for q in (2.5, 97.5)]
        res["judges"][ev]["fpr_ci"] = [round(100 * float(np.percentile(fprs[:, j], q)), 1) for q in (2.5, 97.5)]
    res["pi_ci"] = {m: [round(100 * float(np.percentile(pis[:, s], q)), 1) for q in (2.5, 97.5)] for s, m in enumerate(SUBJECTS)}
    res["spread_corrected_ci"] = [round(float(np.percentile(spreads, q)), 1) for q in (2.5, 97.5)]
    res["mean_range_corrected_ci"] = [round(float(np.percentile(ranges, q)), 1) for q in (2.5, 97.5)]
    return res


def main():
    J = {L: load_panel(R, L) for L in ("JA", "EN")}
    rng = np.random.default_rng(SEED); out = {"n_boot": N_BOOT, "seed": SEED}
    for L in ("JA", "EN"):
        for t, tn in ((0, "T2"), (1, "T4")):
            r = run(J, L, t, rng); out[f"{L}_{tn}"] = r
            print(f"\n### {L} {tn}  (cells={r['n_cells']}, loglik={r['loglik']:.1f})")
            print("| 判定者 | TPR [CI] | FPR [CI] | 観測厳格度 | 補正後 |")
            for ev in EVALS:
                d = r["judges"][ev]; print(f"| {NAME[ev]} | {d['tpr']} [{d['tpr_ci'][0]}, {d['tpr_ci'][1]}] | {d['fpr']} [{d['fpr_ci'][0]}, {d['fpr_ci'][1]}] | {d['strictness_obs']} | {d['strictness_corrected']} |")
            print("π̂ per subject: " + ", ".join(f"{m} {v} [{r['pi_ci'][m][0]}, {r['pi_ci'][m][1]}]" for m, v in r["pi"].items()))
            print(f"judge spread (max−min strictness): obs {r['spread_obs']} → corrected {r['spread_corrected']} [{r['spread_corrected_ci'][0]}, {r['spread_corrected_ci'][1]}]")
            print(f"mean per-subject range over judges: obs {r['mean_range_obs']} → corrected {r['mean_range_corrected']} [{r['mean_range_corrected_ci'][0]}, {r['mean_range_corrected_ci'][1]}]")
            print(f"mean pairwise Spearman of subject rankings: obs {r['rank_corr_obs']} → corrected {r['rank_corr_corrected']}")
            print("per-subject obs → corrected by judge:")
            for m in SUBJECTS:
                print("  " + m + ": " + "  ".join(f"{ev} {r['br_obs'][m][ev]}→{r['br_corrected'][m][ev]}" for ev in EVALS if r['br_obs'][m][ev] is not None))
    # JA−EN gap on latent π
    out["gap_T4_latent"] = {m: round(out["JA_T4"]["pi"][m] - out["EN_T4"]["pi"][m], 1) for m in SUBJECTS}
    out["gap_T2_latent"] = {m: round(out["JA_T2"]["pi"][m] - out["EN_T2"]["pi"][m], 1) for m in SUBJECTS}
    print("\nJA−EN gap on latent π: T4 " + str(out["gap_T4_latent"]) + "\n                        T2 " + str(out["gap_T2_latent"]))
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False, default=float); print("saved ->", OUT)


if __name__ == "__main__":
    main()
