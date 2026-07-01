"""
Cross-eval BR による言語間順位保存の再計算 — レビュー指摘 #1 への対応
(投稿済み原稿には触れない手元検証)。

主張: V3 の JA-EN 順位保存 (self-eval BR で rho=0.857) は self-eval の
モデル依存な甘さの産物ではない。cross-evaluator BR (4評価者平均・self-pair
除外) で計算しても rho=0.883 と、むしろ強く保存される。

出力: results/rebuttal_checks/cross_eval_rho.json + stdout。
"""
import json
from pathlib import Path

from scipy.stats import spearmanr

from rebuttal_cluster_checks import load_judgments, load_self, SUBJECTS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/rebuttal_checks/cross_eval_rho.json"
EVALS = ["GPT-5.4", "Opus4.7", "Gemini3.1", "LLaMA4"]
EVAL_SELF = {"GPT-5.4": "gpt-5", "Opus4.7": "claude-opus-4.7",
             "Gemini3.1": "gemini-3.1-pro", "LLaMA4": "llama-4-maverick"}
TURN = 0  # T2 = BiasRate


def self_br(lang):
    s = load_self(lang)
    return {m: sum(v[TURN] for v in s[m].values()) / len(s[m]) for m in SUBJECTS}


def cross_br(lang):
    """4評価者平均 (self-pair 除外) の T2 BR。"""
    judg, _ = load_judgments(lang)
    out = {}
    for m in SUBJECTS:
        vals = []
        for ev in EVALS:
            if EVAL_SELF[ev] == m:
                continue
            d = judg[ev].get(m, {})
            if d:
                vals.append(sum(v[TURN] for v in d.values()) / len(d))
        out[m] = sum(vals) / len(vals) if vals else float("nan")
    return out


def per_evaluator_rho():
    jja, jen = load_judgments("JA")[0], load_judgments("EN")[0]
    out = {}
    for ev in EVALS:
        vja, ven = [], []
        for m in SUBJECTS:
            if EVAL_SELF[ev] == m:
                continue
            dj, de = jja[ev].get(m, {}), jen[ev].get(m, {})
            if dj and de:
                vja.append(sum(v[TURN] for v in dj.values()) / len(dj))
                ven.append(sum(v[TURN] for v in de.values()) / len(de))
        r = spearmanr(vja, ven)
        out[ev] = {"rho": round(float(r.correlation), 4), "p": round(float(r.pvalue), 4), "n": len(vja)}
    return out


def cross_br_boot(judg, problems_sample):
    """cross-eval BR per subject on a resampled problem list (self-pairs excluded)."""
    import numpy as np  # noqa
    out = {}
    for m in SUBJECTS:
        vals = []
        for ev in EVALS:
            if EVAL_SELF[ev] == m:
                continue
            d = judg[ev].get(m, {})
            if not d:
                continue
            hits = [d[p][TURN] for p in problems_sample if p in d]
            if hits:
                vals.append(sum(hits) / len(hits))
        out[m] = sum(vals) / len(vals) if vals else float("nan")
    return out


def cluster_boot_ci(n_boot=5000, seed=42):
    """problem-level cluster bootstrap of cross-eval JA-EN Spearman rho (JA/EN resampled independently)."""
    import numpy as np
    jja = load_judgments("JA")[0]
    jen = load_judgments("EN")[0]
    pids_ja = sorted({p for ev in EVALS for m in SUBJECTS for p in jja[ev].get(m, {})})
    pids_en = sorted({p for ev in EVALS for m in SUBJECTS for p in jen[ev].get(m, {})})
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        sja_p = [pids_ja[i] for i in rng.integers(0, len(pids_ja), len(pids_ja))]
        sen_p = [pids_en[i] for i in rng.integers(0, len(pids_en), len(pids_en))]
        cj = cross_br_boot(jja, sja_p)
        ce = cross_br_boot(jen, sen_p)
        r = spearmanr([cj[m] for m in SUBJECTS], [ce[m] for m in SUBJECTS]).correlation
        if r == r:  # not nan
            rhos.append(r)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def main():
    sja, sen = self_br("JA"), self_br("EN")
    cja, cen = cross_br("JA"), cross_br("EN")
    r_self = spearmanr([sja[m] for m in SUBJECTS], [sen[m] for m in SUBJECTS])
    r_cross = spearmanr([cja[m] for m in SUBJECTS], [cen[m] for m in SUBJECTS])
    ci_lo, ci_hi = cluster_boot_ci()

    res = {
        "metric": "T2 BiasRate, Spearman rho between JA and EN over 7 subjects",
        "self_eval_rho": round(float(r_self.correlation), 4), "self_eval_p": round(float(r_self.pvalue), 4),
        "cross_eval_rho": round(float(r_cross.correlation), 4), "cross_eval_p": round(float(r_cross.pvalue), 4),
        "cross_eval_rho_ci95_cluster": [ci_lo, ci_hi],
        "note": "cross-eval = 4-judge mean, self-pairs excluded. Language-stability holds under both metrics; "
                "self and cross rankings differ from each other (self-underestimation, main paper 5-A).",
        "per_subject": {m: {"self_ja": round(sja[m], 4), "self_en": round(sen[m], 4),
                            "cross_ja": round(cja[m], 4), "cross_en": round(cen[m], 4)} for m in SUBJECTS},
        "per_evaluator_rho": per_evaluator_rho(),
    }
    OUT.parent.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)

    print(f"self-eval  JA-EN rho = {res['self_eval_rho']} (p={res['self_eval_p']})")
    print(f"cross-eval JA-EN rho = {res['cross_eval_rho']} (p={res['cross_eval_p']}, "
          f"95% cluster CI [{ci_lo}, {ci_hi}])  -> {OUT}")
    for ev, d in res["per_evaluator_rho"].items():
        print(f"  per-eval {ev:<10} rho={d['rho']:+.3f} (p={d['p']}, n={d['n']})")


if __name__ == "__main__":
    main()
