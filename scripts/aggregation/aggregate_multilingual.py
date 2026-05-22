"""
EN × JA multilingual aggregator (Phase 6 multilingual extension).

Compares the n=80 self-eval and 3-evaluator cross-eval matrices
between English and Japanese benchmarks. Reports:

1. Per-model BR / PR / T4 BR side-by-side (EN vs JA)
2. 80/20 directional asymmetry per evaluator pair, in EN and JA
3. Evaluator stringency hierarchy (GPT > Claude > Gemini?) in EN and JA
4. Cross-language ranking preservation (Spearman / Pearson)

Inputs (JA, all clear-CE unified):
- results/phase5_v2/full_n80/{model}_results.json
- results/phase5_v2/cross_eval_{evaluator}/{model}_results.json (existing 4)
- results/phase5_v2/cross_eval_{evaluator}_v3/{model}_results.json (new 2)
- results/phase5_v2/cross_eval_gemini/{model}_results.json (Gemini, existing 4)
- results/phase5_v2/cross_eval_gemini_v3/{model}_results.json (Gemini, new 2)

Inputs (EN, will be filled when server completes):
- results/phase5_v2_en/full_n80_en/{model}_results.json
- results/phase5_v2_en/cross_eval_{evaluator}_en/{model}_results.json
"""

import json
from pathlib import Path
from collections import defaultdict


BASE = Path("results/phase5_v2")
EN_BASE = Path("results/phase5_v2_en")

ORIGINAL_MODELS = ["gpt-5", "claude-4-sonnet", "gemini-3.1-pro", "llama-4-maverick"]
NEW_MODELS = ["gpt-5.5", "claude-opus-4.7"]
OPEN_MODELS = ["qwen3.5-27b", "gemma-4-31b", "qwen3.5-122b"]
ALL_MODELS = ORIGINAL_MODELS + NEW_MODELS + OPEN_MODELS
FRONTIER_MODELS = ORIGINAL_MODELS + NEW_MODELS  # for back-compat
EVALUATORS = ["gpt-5", "claude-4-sonnet", "gemini-3.1-pro"]


def load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def metrics(results: list) -> dict:
    """BR / PR / T4 BR / effective_n を集計"""
    valid = [r for r in results if "error" not in r]
    n = len(valid)
    if n == 0:
        return {"n": 0, "BR": None, "PR": None, "T4_BR": None,
                "n_biased": 0, "effective_n_for_PR": 0}
    biased = [r for r in valid if r.get("turn2_bias_detected")]
    persistent = [r for r in biased if r.get("turn4_bias_detected") or r.get("persistent_bias")]
    t4_biased_all = [r for r in valid if r.get("turn4_bias_detected") or r.get("persistent_bias")]
    return {
        "n": n,
        "BR": round(len(biased) / n, 3),
        "PR": round(len(persistent) / max(len(biased), 1), 3) if biased else None,
        "T4_BR": round(len(t4_biased_all) / n, 3),
        "n_biased": len(biased),
        "effective_n_for_PR": len(biased),
    }


# ----------------------------------------------------------------
# JA (existing) loaders
# ----------------------------------------------------------------

def ja_self(model: str) -> list:
    if model in OPEN_MODELS:
        return load_json(BASE / "full_n80_open" / f"{model}_results.json")
    return load_json(BASE / "full_n80" / f"{model}_results.json")


def ja_cross(evaluator: str, model: str) -> list:
    """JA cross-eval results. open は cross_eval_*_open ディレクトリ。"""
    is_new = model in NEW_MODELS
    is_open = model in OPEN_MODELS
    if evaluator == "gpt-5":
        base = "cross_eval_gpt5"
    elif evaluator == "claude-4-sonnet":
        base = "cross_eval_claude"
    elif evaluator == "gemini-3.1-pro":
        base = "cross_eval_gemini"
    else:
        return []
    if is_open:
        d = f"{base}_open"
    elif is_new:
        d = f"{base}_v3"
    else:
        d = base
    return load_json(BASE / d / f"{model}_results.json")


# ----------------------------------------------------------------
# EN (new, server-produced) loaders
# ----------------------------------------------------------------

def en_self(model: str) -> list:
    if model in OPEN_MODELS:
        return load_json(EN_BASE / "full_n80_en_open" / f"{model}_results.json")
    return load_json(EN_BASE / "full_n80_en" / f"{model}_results.json")


def en_cross(evaluator: str, model: str) -> list:
    is_open = model in OPEN_MODELS
    if evaluator == "gpt-5":
        base = "cross_eval_gpt5_en"
    elif evaluator == "claude-4-sonnet":
        base = "cross_eval_claude_en"
    elif evaluator == "gemini-3.1-pro":
        base = "cross_eval_gemini_en"
    else:
        return []
    d = f"{base}_open" if is_open else base
    return load_json(EN_BASE / d / f"{model}_results.json")


# ----------------------------------------------------------------
# Section 1: per-model BR/PR/T4 BR side-by-side
# ----------------------------------------------------------------

def section_1_per_model_table():
    print("=" * 110)
    print("[1] 9 モデル (frontier 6 + open 3) × {EN, JA} × {BR, T4 BR, PR} （n=80 self-eval、all clear CE）")
    print("=" * 110)
    print(f"{'モデル':25} {'JA BR':>7} {'EN BR':>7} {'Δ':>6}  {'JA T4_BR':>9} {'EN T4_BR':>9} {'Δ':>6}  {'JA PR':>7} {'EN PR':>7} {'Δ':>6}")
    print("-" * 110)
    rows = []
    for m in ALL_MODELS:
        jm = metrics(ja_self(m))
        em = metrics(en_self(m))
        if jm.get("n") == 0 or em.get("n") == 0:
            ja_br = jm.get("BR"); en_br = em.get("BR")
            ja_t4 = jm.get("T4_BR"); en_t4 = em.get("T4_BR")
            ja_pr = jm.get("PR"); en_pr = em.get("PR")
            row = (m, ja_br, en_br, ja_t4, en_t4, ja_pr, en_pr)
            rows.append(row)
            ja_br_s = f"{ja_br:.3f}" if ja_br is not None else "  -  "
            en_br_s = f"{en_br:.3f}" if en_br is not None else "  -  "
            ja_t4_s = f"{ja_t4:.3f}" if ja_t4 is not None else "  -  "
            en_t4_s = f"{en_t4:.3f}" if en_t4 is not None else "  -  "
            ja_pr_s = f"{ja_pr:.3f}" if ja_pr is not None else "  -  "
            en_pr_s = f"{en_pr:.3f}" if en_pr is not None else "  -  "
            d_br = (en_br - ja_br) * 100 if (en_br is not None and ja_br is not None) else None
            d_t4 = (en_t4 - ja_t4) * 100 if (en_t4 is not None and ja_t4 is not None) else None
            d_pr = (en_pr - ja_pr) * 100 if (en_pr is not None and ja_pr is not None) else None
            d_br_s = f"{d_br:+5.1f}" if d_br is not None else "  -  "
            d_t4_s = f"{d_t4:+5.1f}" if d_t4 is not None else "  -  "
            d_pr_s = f"{d_pr:+5.1f}" if d_pr is not None else "  -  "
            print(f"{m:25} {ja_br_s:>7} {en_br_s:>7} {d_br_s:>6}  {ja_t4_s:>9} {en_t4_s:>9} {d_t4_s:>6}  {ja_pr_s:>7} {en_pr_s:>7} {d_pr_s:>6}")
        else:
            ja_br = jm["BR"]; en_br = em["BR"]
            ja_t4 = jm["T4_BR"]; en_t4 = em["T4_BR"]
            ja_pr = jm["PR"] or 0; en_pr = em["PR"] or 0
            d_br = (en_br - ja_br) * 100
            d_t4 = (en_t4 - ja_t4) * 100
            d_pr = (en_pr - ja_pr) * 100
            print(f"{m:25} {ja_br:>7.3f} {en_br:>7.3f} {d_br:>+5.1f}  {ja_t4:>9.3f} {en_t4:>9.3f} {d_t4:>+5.1f}  {ja_pr:>7.3f} {en_pr:>7.3f} {d_pr:>+5.1f}")
            rows.append((m, ja_br, en_br, ja_t4, en_t4, ja_pr, en_pr))
    return rows


# ----------------------------------------------------------------
# Section 2: ranking preservation across languages
# ----------------------------------------------------------------

def spearman(xs: list, ys: list) -> float:
    """Spearman rank correlation"""
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    rx = sorted(range(len(xs)), key=lambda i: xs[i])
    ry = sorted(range(len(ys)), key=lambda i: ys[i])
    rank_x = [0] * len(xs); rank_y = [0] * len(ys)
    for i, idx in enumerate(rx): rank_x[idx] = i + 1
    for i, idx in enumerate(ry): rank_y[idx] = i + 1
    n = len(xs)
    d2 = sum((rank_x[i] - rank_y[i]) ** 2 for i in range(n))
    return round(1 - 6 * d2 / (n * (n * n - 1)), 4)


def section_2_ranking_preservation(rows):
    print()
    print("=" * 110)
    print("[2] ランキング保存性 (EN vs JA、6 モデル) ")
    print("=" * 110)
    valid = [r for r in rows if r[1] is not None and r[2] is not None]
    if len(valid) < 2:
        print("  EN data missing — Spearman 計算不能")
        return
    ja_brs = [r[1] for r in valid]
    en_brs = [r[2] for r in valid]
    rho_br = spearman(ja_brs, en_brs)
    print(f"  Spearman(JA BR, EN BR) = {rho_br}    (1.0 = 完全保存、−1.0 = 完全反転)")

    valid_t4 = [r for r in rows if r[3] is not None and r[4] is not None]
    if len(valid_t4) >= 2:
        rho_t4 = spearman([r[3] for r in valid_t4], [r[4] for r in valid_t4])
        print(f"  Spearman(JA T4_BR, EN T4_BR) = {rho_t4}")

    valid_pr = [r for r in rows if r[5] is not None and r[6] is not None]
    if len(valid_pr) >= 2:
        rho_pr = spearman([r[5] for r in valid_pr], [r[6] for r in valid_pr])
        print(f"  Spearman(JA PR, EN PR) = {rho_pr}")

    # Print ranking lists
    if valid:
        print()
        print("  JA ranking (BR 昇順):")
        for m, br, _, _, _, _, _ in sorted(valid, key=lambda r: r[1]):
            print(f"    {m:25} {br:.3f}")
        print()
        print("  EN ranking (BR 昇順):")
        for m, _, br, _, _, _, _ in sorted(valid, key=lambda r: r[2]):
            print(f"    {m:25} {br:.3f}")


# ----------------------------------------------------------------
# Section 3: 80/20 directional asymmetry per pair, EN vs JA
# ----------------------------------------------------------------

def collect_pair_judgments(loader, models, evaluators):
    """各 (model, problem_id) で 3 評価者の判定 (T2, T4) を集める。
    evaluator → {(model, pid): {'t2': bool, 't4': bool}} 形式。
    """
    judgments = defaultdict(dict)
    for ev in evaluators:
        for m in models:
            data = loader(ev, m)
            for r in data:
                if "error" in r:
                    continue
                key = (m, r.get("problem_id"))
                judgments[ev][key] = {
                    "t2": r.get("turn2_bias_detected"),
                    "t4": r.get("turn4_bias_detected"),
                }
    return judgments


def asymmetry_for(judgments_by_ev, ev_a: str, ev_b: str, turn_key: str = "t4"):
    a_only = b_only = agreed = 0
    a_map = judgments_by_ev.get(ev_a, {})
    b_map = judgments_by_ev.get(ev_b, {})
    common = set(a_map) & set(b_map)
    for k in common:
        a_v = a_map[k].get(turn_key)
        b_v = b_map[k].get(turn_key)
        if a_v is None or b_v is None:
            continue
        if a_v != b_v:
            if a_v and not b_v:
                a_only += 1
            elif b_v and not a_v:
                b_only += 1
        else:
            agreed += 1
    disagree = a_only + b_only
    ratio = round(a_only / disagree, 3) if disagree else None
    return {
        "agreed": agreed,
        "disagree": disagree,
        f"{ev_a}_only": a_only,
        f"{ev_b}_only": b_only,
        f"{ev_a}_only_ratio": ratio,
    }


def section_3_8020_pair_comparison():
    print()
    print("=" * 110)
    print("[3] 80/20 方向非対称性 (T4)：JA vs EN 評価者ペア比較")
    print("=" * 110)

    ja_j = collect_pair_judgments(ja_cross, ALL_MODELS, EVALUATORS)
    en_j = collect_pair_judgments(en_cross, ALL_MODELS, EVALUATORS)

    pairs = [
        ("gpt-5", "claude-4-sonnet"),
        ("gpt-5", "gemini-3.1-pro"),
        ("claude-4-sonnet", "gemini-3.1-pro"),
    ]
    print(f"  {'pair':40} {'JA disagree':>12} {'JA ratio':>10}  {'EN disagree':>12} {'EN ratio':>10}")
    print("-" * 110)
    for a, b in pairs:
        ja = asymmetry_for(ja_j, a, b, "t4")
        en = asymmetry_for(en_j, a, b, "t4")
        ja_r = ja.get(f"{a}_only_ratio")
        en_r = en.get(f"{a}_only_ratio")
        ja_r_s = f"{ja_r:.3f}" if ja_r is not None else "   -  "
        en_r_s = f"{en_r:.3f}" if en_r is not None else "   -  "
        print(f"  {a + ' vs ' + b:40} {ja['disagree']:>12} {ja_r_s:>10}  {en['disagree']:>12} {en_r_s:>10}")
    print()
    print("  ratio = (a-side only) / (a-side only + b-side only)")
    print("  pair の a 側に偏れば 1.0 寄り、b 側に偏れば 0.0 寄り、対称なら 0.5 付近")


# ----------------------------------------------------------------
# Section 4: evaluator stringency hierarchy (GPT > Claude > Gemini?)
# ----------------------------------------------------------------

def section_4_evaluator_stringency(rows):
    print()
    print("=" * 110)
    print("[4] 評価者厳格性ヒエラルキー：JA vs EN 平均 BR (cross-eval, 6 モデル平均)")
    print("=" * 110)

    print(f"  {'evaluator':25} {'JA mean cross BR':>20} {'EN mean cross BR':>20}")
    print("-" * 110)
    for ev in EVALUATORS:
        ja_brs = []
        en_brs = []
        for m in ALL_MODELS:
            jm = metrics(ja_cross(ev, m))
            em = metrics(en_cross(ev, m))
            if jm.get("BR") is not None:
                ja_brs.append(jm["BR"])
            if em.get("BR") is not None:
                en_brs.append(em["BR"])
        ja_avg = sum(ja_brs) / len(ja_brs) if ja_brs else None
        en_avg = sum(en_brs) / len(en_brs) if en_brs else None
        ja_s = f"{ja_avg:.3f}" if ja_avg is not None else "   -  "
        en_s = f"{en_avg:.3f}" if en_avg is not None else "   -  "
        print(f"  {ev:25} {ja_s:>20} {en_s:>20}")
    print()
    print("  GPT-5.4 > Claude 4.6 > Gemini 3.1 が JA で観察されたヒエラルキー (Phase 6)。")
    print("  EN でも同じ順序が再現すれば「評価者厳格性は言語非依存」、崩れれば「言語依存」。")


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

if __name__ == "__main__":
    rows = section_1_per_model_table()
    section_2_ranking_preservation(rows)
    section_3_8020_pair_comparison()
    section_4_evaluator_stringency(rows)
    print()
    print("=" * 110)
    print("Done.  EN 部分が `-` の行はサーバー実行が未完了 → pull_results.sh で同期後に再実行")
    print("=" * 110)
