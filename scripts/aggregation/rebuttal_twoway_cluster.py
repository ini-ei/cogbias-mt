"""
レビュー指摘 #1 への対応: 「beyond strictness」の残差を problem と subject の
両効果を制御しても示す。

(A) problem × subject 二方向クラスタ Bootstrap で、観測 dominant share が
    独立性 marginal 予測を超える「超過(excess)」の CI を出す（全 12 組）。
(B) mixed-effects logistic regression: 不一致セルで「厳格側に偏ったか」を、
    problem と subject をランダム効果として制御した上で、切片が 0.5 を超えるか。

入力は rebuttal_cluster_checks.py と同じ生判定。
出力: results/rebuttal_checks/twoway_cluster.json + stdout
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JA = ROOT / "results/phase5_v2"
EN = ROOT / "results/phase5_v2_en"
OUT = ROOT / "results/rebuttal_checks"

SUBJECTS = ["gpt-5", "claude-opus-4.7", "gemini-3.1-pro", "llama-4-maverick",
            "qwen3.5-27b", "qwen3.5-122b", "gemma-4-31b"]
OPEN = {"qwen3.5-27b", "qwen3.5-122b", "gemma-4-31b"}
EVAL_DIRS = {
    "GPT-5.4": ("cross_eval_gpt5_n80", "cross_eval_gpt5_open",
                "cross_eval_gpt5_en", "cross_eval_gpt5_en_open", "cross_eval_gpt5_en_patch"),
    "Opus4.7": ("cross_eval_claude_opus_n80", "cross_eval_claude_opus_n80",
                "cross_eval_claude_opus_en", "cross_eval_claude_opus_en", None),
    "Gemini3.1": ("cross_eval_gemini_n80", "cross_eval_gemini_open",
                  "cross_eval_gemini_en", "cross_eval_gemini_en_open", "cross_eval_gemini_en_patch"),
    "LLaMA4": ("cross_eval_llama_n80", "cross_eval_llama_n80",
               "cross_eval_llama_en", "cross_eval_llama_en", None),
}
EVALUATORS = list(EVAL_DIRS)
PAIRS = [(a, b) for i, a in enumerate(EVALUATORS) for b in EVALUATORS[i + 1:]]
N_BOOT = 10_000
SEED = 42


def load_records(path):
    if not path.exists():
        return {}
    out = {}
    for r in json.load(open(path)):
        if not (isinstance(r, dict) and r.get("problem_id")):
            continue
        if r.get("turn2_bias_confidence") is not None or r.get("turn4_bias_confidence") is not None:
            out[r["problem_id"]] = r
    return out


def load(lang):
    base = JA if lang == "JA" else EN
    judg = {}
    for ev, (dj_f, dj_o, de_f, de_o, de_p) in EVAL_DIRS.items():
        judg[ev] = {}
        for m in SUBJECTS:
            d = (dj_o if m in OPEN else dj_f) if lang == "JA" else (de_o if m in OPEN else de_f)
            recs = load_records(base / d / f"{m}_results.json")
            if lang == "EN" and de_p and m not in OPEN:
                recs.update(load_records(base / de_p / f"{m}_results.json"))
            judg[ev][m] = {pid: bool(r.get("turn4_bias_detected")) for pid, r in recs.items()}
    return judg


def cells(judg, ea, eb):
    """list of (problem_id, subject, va, vb) for common cells"""
    out = []
    for m in SUBJECTS:
        ra, rb = judg[ea].get(m, {}), judg[eb].get(m, {})
        for pid in set(ra) & set(rb):
            out.append((pid, m, ra[pid], rb[pid]))
    return out


def excess(rows):
    """observed dominant share - independence-marginal prediction (on disagreement cells)."""
    va = np.array([r[2] for r in rows]); vb = np.array([r[3] for r in rows])
    a_only = int((va & ~vb).sum()); b_only = int((vb & ~va).sum())
    if a_only + b_only == 0:
        return np.nan
    pa, pb = va.mean(), vb.mean()
    side_a = a_only >= b_only
    obs = max(a_only, b_only) / (a_only + b_only)
    pa_only, pb_only = pa * (1 - pb), (1 - pa) * pb
    denom = pa_only + pb_only
    pred = (pa_only if side_a else pb_only) / denom if denom > 0 else 0.5
    return obs - pred


def twoway_bootstrap(rows, rng):
    probs = sorted({r[0] for r in rows})
    subs = sorted({r[1] for r in rows})
    by_ps = {}
    for r in rows:
        by_ps.setdefault((r[0], r[1]), []).append(r)
    pidx = {p: i for i, p in enumerate(probs)}
    sidx = {s: i for i, s in enumerate(subs)}
    out = np.empty(N_BOOT)
    for t in range(N_BOOT):
        bp = [probs[i] for i in rng.integers(0, len(probs), len(probs))]
        bs = [subs[i] for i in rng.integers(0, len(subs), len(subs))]
        samp = []
        for p in bp:
            for s in bs:
                samp.extend(by_ps.get((p, s), []))
        out[t] = excess(samp) if samp else np.nan
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    OUT.mkdir(exist_ok=True)
    res = {"n_boot": N_BOOT, "seed": SEED, "combos": []}
    print("=== (A) problem × subject two-way cluster bootstrap: excess beyond marginal ===")
    print(f"{'pair':<20}{'lang':<5}{'excess(pp)':>11}{'two-way 95% CI (pp)':>22}")
    print("-" * 58)
    for lang in ("JA", "EN"):
        judg = load(lang)
        for ea, eb in PAIRS:
            rows = cells(judg, ea, eb)
            ex = excess(rows)
            rng = np.random.default_rng(SEED)
            lo, hi = twoway_bootstrap(rows, rng)
            flag = "" if lo > 0 else "  (CI incl. 0)"
            print(f"{ea+'-'+eb:<20}{lang:<5}{ex*100:>+10.1f}{f'[{lo*100:+.1f}, {hi*100:+.1f}]':>22}{flag}")
            res["combos"].append(dict(pair=f"{ea}-{eb}", lang=lang,
                                      excess_pp=round(ex*100, 2),
                                      twoway_ci_pp=[round(lo*100, 2), round(hi*100, 2)]))
    n_pos = sum(1 for c in res["combos"] if c["twoway_ci_pp"][0] > 0)
    print(f"\n  two-way CI が 0 を除外する組: {n_pos} / 12")

    # (B) mixed-effects logistic on GPT-Opus disagreement cells (headline pair)
    print("\n=== (B) mixed-effects logistic: skew toward stricter judge, problem+subject random effects ===")
    import pandas as pd
    import statsmodels.formula.api as smf
    res["mixed"] = {}
    for lang in ("JA", "EN"):
        judg = load(lang)
        rows = cells(judg, "GPT-5.4", "Opus4.7")
        dis = [(p, s, 1 if (va and not vb) else 0)
               for (p, s, va, vb) in rows if va != vb]  # 1 = GPT-side (stricter)
        df = pd.DataFrame(dis, columns=["problem", "subject", "gpt_side"])
        # group by problem, subject as crossed random effects -> approximate with problem groups + subject variance comp
        try:
            md = smf.mixedlm("gpt_side ~ 1", df, groups=df["problem"],
                             re_formula="1", vc_formula={"subject": "0 + C(subject)"})
            mf = md.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            int = mf.params["Intercept"]; se = mf.bse["Intercept"]
            lo, hi = int - 1.96*se, int + 1.96*se
            print(f"  GPT-Opus {lang}: n_disagree={len(df)}, mean GPT-side={df.gpt_side.mean()*100:.1f}%, "
                  f"intercept(prob)={int:.2f} 95%CI[{lo:.2f},{hi:.2f}] -> P={1/(1+np.exp(-int)):.3f} "
                  f"CI[{1/(1+np.exp(-lo)):.3f},{1/(1+np.exp(-hi)):.3f}]")
            res["mixed"][f"GPT-Opus_{lang}"] = dict(
                n=len(df), mean_gpt_side=round(df.gpt_side.mean(), 4),
                intercept=round(int, 3), p_gpt_side=round(1/(1+np.exp(-int)), 4),
                p_ci=[round(1/(1+np.exp(-lo)), 4), round(1/(1+np.exp(-hi)), 4)])
        except Exception as e:
            print(f"  GPT-Opus {lang}: mixed model failed ({e}); reporting raw mean {df.gpt_side.mean()*100:.1f}%")
            res["mixed"][f"GPT-Opus_{lang}"] = dict(n=len(df), mean_gpt_side=round(df.gpt_side.mean(), 4), error=str(e))

    json.dump(res, open(OUT / "twoway_cluster.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved -> {OUT/'twoway_cluster.json'}")


if __name__ == "__main__":
    main()
