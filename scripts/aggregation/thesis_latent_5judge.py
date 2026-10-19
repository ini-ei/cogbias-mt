#!/usr/bin/env python3
"""
thesis_latent_5judge.py — Qwen3.5-122B を 5 つ目の評価者に加えた潜在クラス推定。
  .venv/bin/python scripts/thesis_latent_5judge.py --tag qwen_judge --json results/thesis_exp/latent_5judge.json
4 評価者（本文）と 5 評価者で、各評価者の感度・偽陽性率、被験モデルごとの π̃、評価者の開き（補正前後）を比べる。
さらに GPT-5.4 を除いた 4 評価者（Opus・Gemini・LLaMA・Qwen）でも推定し、潜在クラスが GPT-5.4 を含むかどうかにどれだけ左右されるかを見る。
Qwen の自己セル（被験 qwen3.5-122b）は用いない。
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wiiat_numbers as W  # noqa: E402
from thesis_latent_class import em, rogan_gladen  # noqa: E402
SUBJ = W.SUBJECTS
QDIRS = ("cross_eval_qwen122b_n80", "cross_eval_qwen122b_open", "cross_eval_qwen122b_en", "cross_eval_qwen122b_en_open", None)


def load_all(tag):
    J = {}
    for L in ("JA", "EN"):
        J[L] = W.load_panel(os.path.join(ROOT, "results"), L)
        jf, jo, ef, eo, ep = QDIRS
        J[L]["Qwen122B"] = W.load_dir_set(os.path.join(ROOT, "results", "thesis_exp", tag), L, (jf, jo, None) if L == "JA" else (ef, eo, ep))
    return J


def build(J, L, t, evals, self_of):
    rows, sidx = [], []
    for si, m in enumerate(SUBJ):
        common = sorted(set.intersection(*[set(J[L][ev][m]) for ev in evals if self_of[ev] != m]))
        for p in common:
            rows.append([np.nan if (self_of[ev] == m or p not in J[L][ev][m]) else float(J[L][ev][m][p][t]) for ev in evals]); sidx.append(si)
    return np.array(rows), np.array(sidx)


def obs_rates(X, sidx):
    M = np.full((len(SUBJ), X.shape[1]), np.nan)
    for s in range(len(SUBJ)):
        m = sidx == s
        for j in range(X.shape[1]):
            v = X[m, j]; v = v[~np.isnan(v)]
            if len(v): M[s, j] = v.mean()
    return M


def fit_set(J, L, t, evals, self_of, nboot, rng):
    X, sidx = build(J, L, t, evals, self_of); f = em(X, sidx)
    BR = obs_rates(X, sidx); C = np.column_stack([rogan_gladen(BR[:, j], f["tpr"][j], f["fpr"][j]) for j in range(len(evals))])
    strict_b = np.nanmean(BR, 0) * 100; strict_a = np.nanmean(C, 0) * 100
    res = dict(n_cells=int(len(X)), judges={ev: dict(tpr=round(100 * f["tpr"][j], 1), fpr=round(100 * f["fpr"][j], 1), strictness_obs=round(float(strict_b[j]), 1), strictness_corrected=round(float(strict_a[j]), 1)) for j, ev in enumerate(evals)},
               pi={m: round(100 * float(v), 1) for m, v in zip(SUBJ, f["pi"])}, spread_obs=round(float(np.nanmax(strict_b) - np.nanmin(strict_b)), 1), spread_corrected=round(float(np.nanmax(strict_a) - np.nanmin(strict_a)), 1),
               mean_range_obs=round(100 * float(np.nanmean(np.nanmax(BR, 1) - np.nanmin(BR, 1))), 1), mean_range_corrected=round(100 * float(np.nanmean(np.nanmax(C, 1) - np.nanmin(C, 1))), 1))
    if nboot:
        tprs, fprs, pis = [], [], []
        # 問題単位のブートストラップ（セルの問題 ID は build で捨てたので被験×問題の行を被験内で再標本）
        for _ in range(nboot):
            idx = np.concatenate([rng.choice(np.where(sidx == s)[0], (sidx == s).sum(), replace=True) for s in range(len(SUBJ)) if (sidx == s).any()])
            g = em(X[idx], sidx[idx], n_iter=150); tprs.append(g["tpr"]); fprs.append(g["fpr"]); pis.append(g["pi"])
        tprs, fprs, pis = np.array(tprs), np.array(fprs), np.array(pis)
        for j, ev in enumerate(evals):
            res["judges"][ev]["tpr_ci"] = [round(100 * float(np.percentile(tprs[:, j], q)), 1) for q in (2.5, 97.5)]
            res["judges"][ev]["fpr_ci"] = [round(100 * float(np.percentile(fprs[:, j], q)), 1) for q in (2.5, 97.5)]
        res["pi_ci"] = {m: [round(100 * float(np.percentile(pis[:, s], q)), 1) for q in (2.5, 97.5)] for s, m in enumerate(SUBJ)}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="qwen_judge"); ap.add_argument("--json", default=None); ap.add_argument("--nboot", type=int, default=300)
    a = ap.parse_args(); rng = np.random.default_rng(42)
    J = load_all(a.tag); self_of = dict(W.SELF_OF); self_of["Qwen122B"] = "qwen3.5-122b"
    E4 = list(W.EVALS); E5 = E4 + ["Qwen122B"]; E4noG = [e for e in E5 if e != "GPT-5.4"]
    out = {}
    for L in ("JA", "EN"):
        nq = sum(len(v) for v in J[L]["Qwen122B"].values()); print(f"[{L}] Qwen judge records: {nq}")
        if nq == 0: continue
        out[L] = {}
        for t, tn in ((0, "T2"), (1, "T4")):
            # Qwen の判定率と他評価者との一致
            qs = W.strictness({"Qwen122B": J[L]["Qwen122B"]} | {k: J[L][k] for k in E4}, "Qwen122B", t) if False else None
            vals = [x[t] for m in SUBJ if m != "qwen3.5-122b" for p, x in J[L]["Qwen122B"][m].items()]
            agree = {}
            for ev in E4:
                c = W.pair_cells({**J[L]}, ev, "Qwen122B", t, subjects=[m for m in SUBJ if m not in (self_of[ev], "qwen3.5-122b")])
                agree[ev] = dict(n=c["n"], agree=round(100 * (c["both"] + c["neither"]) / c["n"], 1) if c["n"] else None, kappa=round(W.kappa(c), 3) if c["n"] else None)
            r = dict(qwen_strictness=round(100 * float(np.mean(vals)), 1), n=len(vals), agreement_with=agree,
                     judges4=fit_set(J, L, t, E4, self_of, a.nboot, rng), judges5=fit_set(J, L, t, E5, self_of, a.nboot, rng), judges4_noGPT=fit_set(J, L, t, E4noG, self_of, a.nboot, rng))
            out[L][tn] = r
            print(f"  {tn}: Qwen 判定率 {r['qwen_strictness']} 一致 {agree}")
            for k in ("judges4", "judges5", "judges4_noGPT"):
                d = r[k]; print(f"    {k}: " + " ".join(f"{ev} TPR {v['tpr']} FPR {v['fpr']}" for ev, v in d["judges"].items()) + f" | π̃ {d['pi']} | 開き {d['spread_obs']}→{d['spread_corrected']} 幅 {d['mean_range_obs']}→{d['mean_range_corrected']}")
    if a.json: json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False); print("wrote", a.json)


if __name__ == "__main__":
    main()
