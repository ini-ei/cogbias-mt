"""
Rebuttal 用手元検証 (論文は投稿済み — 原稿には触れない):

B-2: 方向性非対称 (11/12) の問題クラスタ頑健性
     - 問題単位クラスタブートストラップ CI (dominant-side share)
     - 問題単位クラスタ置換検定 (符号反転) p 値
B-3: 周辺予測 (独立性仮定) に対する観測超過 (pp) のクラスタブートストラップ CI
C-2: JA-EN self-eval BR の Spearman ρ のクラスタブートストラップ CI
B-6: 原 40 問 vs 追加 40 問のバッチ別保存性 (JA: self BR ランキング + 方向非対称)
C-6: §5.2 評価者セットスイングの self 抜き再計算

入力 (JA): results/phase5_v2/{full_n80,full_n80_open,cross_eval_*_n80,cross_eval_*_open}
入力 (EN): results/phase5_v2_en/{full_n80_en,full_n80_en_open,cross_eval_*_en,*_en_open,*_patch}
出力: results/rebuttal_checks/cluster_checks.json + stdout サマリ
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JA = ROOT / "results/phase5_v2"
EN = ROOT / "results/phase5_v2_en"
OUT_DIR = ROOT / "results/rebuttal_checks"

SUBJECTS = [
    "gpt-5", "claude-opus-4.7", "gemini-3.1-pro", "llama-4-maverick",
    "qwen3.5-27b", "qwen3.5-122b", "gemma-4-31b",
]
OPEN_SUBJECTS = {"qwen3.5-27b", "qwen3.5-122b", "gemma-4-31b"}

# evaluator slug -> (JA frontier dir, JA open dir, EN frontier dir, EN open dir, EN patch dir)
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
N_PERM = 100_000
SEED = 42


def load_records(path: Path) -> dict:
    """problem_id -> record。
    注意: 'error' キーは初回 402 失敗の残骸で、リトライ後も残存している
    (cross_eval_claude_opus_n80 等)。実際に評価済みかは confidence で判定する。"""
    if not path.exists():
        return {}
    out = {}
    for r in json.load(open(path)):
        if not (isinstance(r, dict) and r.get("problem_id")):
            continue
        evaluated = (r.get("turn2_bias_confidence") is not None
                     or r.get("turn4_bias_confidence") is not None)
        if evaluated:
            out[r["problem_id"]] = r
    return out


def load_judgments(lang: str) -> tuple:
    """returns (judg, problems)
    judg[evaluator][subject][problem_id] = (t2: bool, t4: bool)"""
    base = JA if lang == "JA" else EN
    judg = {}
    problems = set()
    for ev, (dj_f, dj_o, de_f, de_o, de_p) in EVAL_DIRS.items():
        judg[ev] = {}
        for m in SUBJECTS:
            d = (dj_o if m in OPEN_SUBJECTS else dj_f) if lang == "JA" \
                else (de_o if m in OPEN_SUBJECTS else de_f)
            recs = load_records(base / d / f"{m}_results.json")
            if lang == "EN" and de_p and m not in OPEN_SUBJECTS:
                recs.update(load_records(base / de_p / f"{m}_results.json"))
            judg[ev][m] = {
                pid: (bool(r.get("turn2_bias_detected")), bool(r.get("turn4_bias_detected")))
                for pid, r in recs.items()
            }
            problems.update(judg[ev][m])
    return judg, sorted(problems)


def load_self(lang: str) -> dict:
    """self-eval: subject -> problem_id -> (t2, t4)"""
    base, d_f, d_o, d_p = (JA, "full_n80", "full_n80_open", None) if lang == "JA" \
        else (EN, "full_n80_en", "full_n80_en_open", "full_n80_en_patch")
    out = {}
    for m in SUBJECTS:
        recs = load_records(base / (d_o if m in OPEN_SUBJECTS else d_f) / f"{m}_results.json")
        if d_p and m not in OPEN_SUBJECTS:
            recs.update(load_records(base / d_p / f"{m}_results.json"))
        out[m] = {pid: (bool(r.get("turn2_bias_detected")), bool(r.get("turn4_bias_detected")))
                  for pid, r in recs.items()}
    return out


def pair_cluster_counts(judg, problems, ev_a, ev_b, turn_idx) -> tuple:
    """per-problem (a_only, b_only) counts summed over 7 subjects -> (a[80], b[80])"""
    a = np.zeros(len(problems), dtype=np.int64)
    b = np.zeros(len(problems), dtype=np.int64)
    pidx = {p: i for i, p in enumerate(problems)}
    for m in SUBJECTS:
        ja_, jb_ = judg[ev_a].get(m, {}), judg[ev_b].get(m, {})
        for pid in set(ja_) & set(jb_):
            va, vb = ja_[pid][turn_idx], jb_[pid][turn_idx]
            if va and not vb:
                a[pidx[pid]] += 1
            elif vb and not va:
                b[pidx[pid]] += 1
    return a, b


def dominant_share(a_tot, b_tot):
    n = a_tot + b_tot
    return (max(a_tot, b_tot) / n if n else np.nan), ("A" if a_tot >= b_tot else "B")


def cluster_bootstrap_share(a, b, rng) -> tuple:
    """dominant side fixed to observed; CI of its share under problem resampling"""
    obs_side_a = a.sum() >= b.sum()
    dom, other = (a, b) if obs_side_a else (b, a)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    ds = dom[idx].sum(axis=1).astype(float)
    os_ = other[idx].sum(axis=1).astype(float)
    tot = ds + os_
    shares = np.where(tot > 0, ds / np.maximum(tot, 1), np.nan)
    return (float(np.nanpercentile(shares, 2.5)), float(np.nanpercentile(shares, 97.5)))


def cluster_permutation_p(a, b, rng) -> float:
    """H0: 方向は問題クラスタ内で交換可能。クラスタ単位で (a,b) を確率 1/2 で反転。
    one-sided: 観測 dominant 側 share 以上になる確率"""
    obs_side_a = a.sum() >= b.sum()
    dom, other = (a, b) if obs_side_a else (b, a)
    obs = dom.sum() / max(dom.sum() + other.sum(), 1)
    flips = rng.random((N_PERM, len(a))) < 0.5
    dom_tot = np.where(flips, other, dom).sum(axis=1).astype(float)
    tot = float(dom.sum() + other.sum())
    shares = dom_tot / tot
    p = float((shares >= obs - 1e-12).mean())
    return max(p, 1.0 / N_PERM)  # 0 hits -> p < 1/N_PERM


def marginal_excess(judg, problems, ev_a, ev_b, turn_idx):
    """observed dominant share - independence prediction (8-19pp claim), on common cells"""
    va, vb = [], []
    pid_of = []
    for m in SUBJECTS:
        ja_, jb_ = judg[ev_a].get(m, {}), judg[ev_b].get(m, {})
        for pid in sorted(set(ja_) & set(jb_)):
            va.append(ja_[pid][turn_idx])
            vb.append(jb_[pid][turn_idx])
            pid_of.append(pid)
    va, vb = np.array(va, bool), np.array(vb, bool)
    pids = sorted(set(pid_of))
    pmap = {p: i for i, p in enumerate(pids)}
    cl = np.array([pmap[p] for p in pid_of])

    def stats(mask_idx):
        v_a, v_b = va[mask_idx], vb[mask_idx]
        pa, pb = v_a.mean(), v_b.mean()
        a_only = (v_a & ~v_b).sum()
        b_only = (v_b & ~v_a).sum()
        if a_only + b_only == 0:
            return np.nan, np.nan
        obs = max(a_only, b_only) / (a_only + b_only)
        pa_only, pb_only = pa * (1 - pb), (1 - pa) * pb
        side_a = a_only >= b_only
        pred = (pa_only if side_a else pb_only) / max(pa_only + pb_only, 1e-12)
        return obs, obs - pred

    all_idx = np.arange(len(va))
    obs_share, obs_excess = stats(all_idx)

    rng = np.random.default_rng(SEED + 1)
    cl_members = [np.where(cl == i)[0] for i in range(len(pids))]
    excesses = np.empty(N_BOOT)
    for t in range(N_BOOT):
        chosen = rng.integers(0, len(pids), size=len(pids))
        idx = np.concatenate([cl_members[c] for c in chosen])
        excesses[t] = stats(idx)[1]
    lo, hi = np.nanpercentile(excesses, [2.5, 97.5])
    return obs_share, obs_excess, float(lo), float(hi)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    out = {"seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM}

    judg = {}
    probs = {}
    selfj = {}
    for lang in ("JA", "EN"):
        judg[lang], probs[lang] = load_judgments(lang)
        selfj[lang] = load_self(lang)
        print(f"[{lang}] problems={len(probs[lang])}  "
              + "  ".join(f"{ev}:{sum(len(v) for v in judg[lang][ev].values())}"
                          for ev in EVALUATORS))

    # ---------------- B-2 + B-3: 12 combos ----------------
    print("\n=== B-2 / B-3: T4 directional asymmetry — cluster-robust re-test ===")
    hdr = (f"{'pair':<22}{'lang':<5}{'n_dis':>6}{'share':>8}{'side':>5}"
           f"{'clusterCI':>18}{'perm p':>12}{'excess':>8}{'excessCI':>20}")
    print(hdr); print("-" * len(hdr))
    out["combos"] = []
    for ev_a, ev_b in PAIRS:
        for lang in ("JA", "EN"):
            a, b = pair_cluster_counts(judg[lang], probs[lang], ev_a, ev_b, 1)
            share, side = dominant_share(int(a.sum()), int(b.sum()))
            rng = np.random.default_rng(SEED)
            lo, hi = cluster_bootstrap_share(a, b, rng)
            p = cluster_permutation_p(a, b, np.random.default_rng(SEED))
            _, excess, elo, ehi = marginal_excess(judg[lang], probs[lang], ev_a, ev_b, 1)
            n_dis = int(a.sum() + b.sum())
            side_name = ev_a if side == "A" else ev_b
            p_str = f"<{1/N_PERM:.0e}" if p <= 1.0 / N_PERM else f"{p:.4f}"
            print(f"{ev_a+'-'+ev_b:<22}{lang:<5}{n_dis:>6}{share*100:>7.1f}%"
                  f"{side_name[:4]:>5}  [{lo*100:5.1f},{hi*100:6.1f}]%"
                  f"{p_str:>12}{excess*100:>+7.1f}p  [{elo*100:+5.1f},{ehi*100:+6.1f}]pp")
            out["combos"].append(dict(
                pair=f"{ev_a}-{ev_b}", lang=lang, n_disagree=n_dis,
                dominant_share=round(share, 4), dominant_side=side_name,
                cluster_ci=[round(lo, 4), round(hi, 4)],
                perm_p=p, excess_pp=round(excess * 100, 2),
                excess_ci_pp=[round(elo * 100, 2), round(ehi * 100, 2)]))

    sig = sum(1 for c in out["combos"] if c["cluster_ci"][0] > 0.5)
    print(f"\n  cluster CI が 50% を除外する組: {sig} / 12")
    sig_p = sum(1 for c in out["combos"] if c["perm_p"] < 0.05)
    print(f"  cluster permutation p < 0.05  : {sig_p} / 12")

    # ---------------- C-2: Spearman rho CI ----------------
    print("\n=== C-2: Spearman ρ (JA vs EN self-eval T2 BR, 7 models) ===")
    mats = {}
    for lang in ("JA", "EN"):
        pids = sorted(set.union(*[set(selfj[lang][m]) for m in SUBJECTS]))
        mat = np.full((len(SUBJECTS), len(pids)), np.nan)
        for i, m in enumerate(SUBJECTS):
            for j, pid in enumerate(pids):
                if pid in selfj[lang][m]:
                    mat[i, j] = selfj[lang][m][pid][0]
        mats[lang] = mat
    br_ja, br_en = np.nanmean(mats["JA"], 1), np.nanmean(mats["EN"], 1)
    rho_obs = spearman(br_ja, br_en)
    rng = np.random.default_rng(SEED + 2)
    rhos = np.empty(N_BOOT)
    nja, nen = mats["JA"].shape[1], mats["EN"].shape[1]
    for t in range(N_BOOT):
        bja = np.nanmean(mats["JA"][:, rng.integers(0, nja, nja)], 1)
        ben = np.nanmean(mats["EN"][:, rng.integers(0, nen, nen)], 1)
        rhos[t] = spearman(bja, ben)
    rlo, rhi = np.percentile(rhos, [2.5, 97.5])
    for m, bj, be in zip(SUBJECTS, br_ja, br_en):
        print(f"  {m:<18} JA {bj*100:5.1f}%  EN {be*100:5.1f}%")
    print(f"  ρ = {rho_obs:.3f}   cluster bootstrap 95% CI [{rlo:.3f}, {rhi:.3f}]")
    out["spearman"] = dict(rho=round(rho_obs, 4), ci=[round(float(rlo), 4), round(float(rhi), 4)],
                           br_ja={m: round(float(v), 4) for m, v in zip(SUBJECTS, br_ja)},
                           br_en={m: round(float(v), 4) for m, v in zip(SUBJECTS, br_en)})

    # ---------------- B-6: batch split (JA) ----------------
    print("\n=== B-6: 原 40 vs 追加 40 (JA) のバッチ別保存性 ===")
    orig_ids = {json.loads(l)["id"]
                for l in open(JA / "hard_all_40_clear_ce.jsonl") if l.strip()}
    all_ids = {json.loads(l)["id"]
               for l in open(JA / "hard_all_80_clear_ce.jsonl") if l.strip()}
    extra_ids = all_ids - orig_ids
    print(f"  original={len(orig_ids)}  additional={len(extra_ids)}")

    def batch_br(batch):
        return np.array([
            np.mean([selfj["JA"][m][p][0] for p in batch if p in selfj["JA"][m]])
            for m in SUBJECTS])

    br_o, br_e = batch_br(orig_ids), batch_br(extra_ids)
    rho_batch = spearman(br_o, br_e)
    print(f"  {'model':<18}{'orig40 BR':>10}{'extra40 BR':>11}{'Δ':>7}")
    for m, o, e in zip(SUBJECTS, br_o, br_e):
        print(f"  {m:<18}{o*100:>9.1f}%{e*100:>10.1f}%{(e-o)*100:>+6.1f}p")
    print(f"  batch間 self BR ランキング Spearman ρ = {rho_batch:.3f}")

    out["batch"] = dict(rho=round(rho_batch, 4),
                        br_orig={m: round(float(v), 4) for m, v in zip(SUBJECTS, br_o)},
                        br_extra={m: round(float(v), 4) for m, v in zip(SUBJECTS, br_e)},
                        pairs=[])
    print(f"\n  {'pair':<22}{'orig: n / share':>18}{'extra: n / share':>19}{'side一致':>9}")
    pidx = {p: i for i, p in enumerate(probs["JA"])}
    o_mask = np.array([p in orig_ids for p in probs["JA"]])
    for ev_a, ev_b in PAIRS:
        a, b = pair_cluster_counts(judg["JA"], probs["JA"], ev_a, ev_b, 1)
        so, _ = dominant_share(int(a[o_mask].sum()), int(b[o_mask].sum()))
        se, _ = dominant_share(int(a[~o_mask].sum()), int(b[~o_mask].sum()))
        side_o = "A" if a[o_mask].sum() >= b[o_mask].sum() else "B"
        side_e = "A" if a[~o_mask].sum() >= b[~o_mask].sum() else "B"
        no = int(a[o_mask].sum() + b[o_mask].sum())
        ne = int(a[~o_mask].sum() + b[~o_mask].sum())
        agree = "✓" if side_o == side_e else "✗ 反転"
        print(f"  {ev_a+'-'+ev_b:<22}{no:>7} /{so*100:5.1f}%{ne:>8} /{se*100:5.1f}%{agree:>9}")
        out["batch"]["pairs"].append(dict(pair=f"{ev_a}-{ev_b}",
                                          orig=[no, round(so, 4)], extra=[ne, round(se, 4)],
                                          same_side=side_o == side_e))

    # ---------------- C-6: evaluator-set swing without self ----------------
    print("\n=== C-6: §5.2 スイングの self 抜き再計算 (JA) ===")
    out["swing"] = {}
    for subj in ("gpt-5", "claude-opus-4.7"):
        brs = {"self": float(np.mean([v[0] for v in selfj["JA"][subj].values()]))}
        for ev in EVALUATORS:
            recs = judg["JA"][ev].get(subj, {})
            if (ev == "GPT-5.4" and subj == "gpt-5") or (ev == "Opus4.7" and subj == "claude-opus-4.7"):
                continue  # self pair: cross 配信版は除外し self 列を使用
            if recs:
                brs[ev] = float(np.mean([v[0] for v in recs.values()]))
        print(f"  subject={subj}: " + "  ".join(f"{k} {v*100:.1f}%" for k, v in brs.items()))
        names = list(brs)
        from itertools import combinations
        panels = {}
        for combo in combinations(names, 3):
            panels["+".join(combo)] = float(np.mean([brs[c] for c in combo]))
        with_self = {k: v for k, v in panels.items() if "self" in k}
        no_self = {k: v for k, v in panels.items() if "self" not in k}
        def rng_str(d):
            if not d:
                return "n/a"
            lo_k = min(d, key=d.get); hi_k = max(d, key=d.get)
            return (f"{d[lo_k]*100:.1f}% ({lo_k}) — {d[hi_k]*100:.1f}% ({hi_k})"
                    f"  swing {100*(d[hi_k]-d[lo_k]):.1f}pp")
        print(f"    3-judge panels incl self: {rng_str(with_self)}")
        print(f"    3-judge panels excl self: {rng_str(no_self)}")
        cross_only = {k: v for k, v in brs.items() if k != "self"}
        print(f"    single cross judge swing: {rng_str({k: v for k, v in cross_only.items()})}")
        out["swing"][subj] = dict(per_judge={k: round(v, 4) for k, v in brs.items()},
                                  panels={k: round(v, 4) for k, v in panels.items()})

    with open(OUT_DIR / "cluster_checks.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {OUT_DIR / 'cluster_checks.json'}")


if __name__ == "__main__":
    main()
