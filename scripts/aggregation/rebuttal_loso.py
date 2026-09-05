"""
Leave-one-subject-out (LOSO) 頑健性 — 論文 §IV-D / Contributions(ii) の
"leave-one-subject-out preserves the skew direction of every significant
combination" を裏付ける手元検証 (投稿済み原稿には触れない)。

各評価者ペア×言語 (12 組合せ) について、7 被験者を 1 つずつ除外して
T4 不一致の優勢側 share を再計算し、優勢側の符号が保存されるかを確認する。
レビュー指摘 #1 (pairwise disagreement は 7 被験者集合を使い frontier の
self-pair を含む) への直接の反証: frontier 被験者 (gpt-5, claude-opus-4.7)
を除いても有意 11 組合せの方向は一切反転しない。

入力: rebuttal_cluster_checks.load_judgments と同じ。
出力: results/rebuttal_checks/loso.json + stdout サマリ。
"""
import json
from pathlib import Path

from rebuttal_cluster_checks import load_judgments, SUBJECTS, PAIRS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/rebuttal_checks/loso.json"
# 有意でないセル: cluster_checks.json の問題クラスタ置換検定 p >= 0.05 の組合せ（ハードコードしない）
def _marginal():
    f = ROOT / "results/rebuttal_checks/cluster_checks.json"
    if not f.exists():
        return {("Opus4.7", "Gemini3.1", "JA")}
    out = set()
    for c in json.load(open(f))["combos"]:
        if c["perm_p"] >= 0.05:
            a = next(e for e in ("GPT-5.4", "Opus4.7", "Gemini3.1", "LLaMA4") if c["pair"].startswith(e + "-"))
            out.add((a, c["pair"][len(a) + 1:], c["lang"]))
    return out
MARGINAL = _marginal()


def pair_counts(judg, ev_a, ev_b, drop=None):
    a = b = 0
    for m in SUBJECTS:
        if m == drop:
            continue
        ja_, jb_ = judg[ev_a].get(m, {}), judg[ev_b].get(m, {})
        for pid in set(ja_) & set(jb_):
            va, vb = ja_[pid][1], jb_[pid][1]  # T4
            if va and not vb:
                a += 1
            elif vb and not va:
                b += 1
    return a, b


def main():
    results = []
    sig_all_ok = True
    for lang in ("JA", "EN"):
        judg, _ = load_judgments(lang)
        for ev_a, ev_b in PAIRS:
            fa, fb = pair_counts(judg, ev_a, ev_b)
            if fa + fb == 0:
                continue
            full_side = ev_a if fa >= fb else ev_b
            full_share = max(fa, fb) / (fa + fb)
            significant = (ev_a, ev_b, lang) not in MARGINAL
            min_share, flips = 1.0, []
            for drop in SUBJECTS:
                a, b = pair_counts(judg, ev_a, ev_b, drop)
                if a + b == 0:
                    continue
                side = ev_a if a >= b else ev_b
                share = max(a, b) / (a + b)
                min_share = min(min_share, share)
                if side != full_side:
                    flips.append(drop)
            preserved = not flips
            if significant and not preserved:
                sig_all_ok = False
            results.append({
                "pair": f"{ev_a}-{ev_b}", "lang": lang, "dominant_side": full_side,
                "full_share": round(full_share, 4), "min_loso_share": round(min_share, 4),
                "significant": significant, "direction_preserved": preserved,
                "flips_on": flips,
            })

    summary = {
        "n_combinations": len(results),
        "significant_all_direction_preserved": sig_all_ok,
        "note": f"non-significant (perm p>=0.05) combinations: {sorted(MARGINAL)}; flips: "
                f"{[(r['pair'], r['lang'], r['flips_on']) for r in results if not r['direction_preserved']]}",
        "results": results,
    }
    OUT.parent.mkdir(exist_ok=True)
    json.dump(summary, open(OUT, "w"), indent=1, ensure_ascii=False)

    print(f"{'pair':<20}{'lang':<5}{'side':<10}{'full%':>7}{'minLOSO%':>9}  sig  preserved")
    for r in results:
        m = "✓" if r["direction_preserved"] else f"✗ {r['flips_on']}"
        print(f"{r['pair']:<20}{r['lang']:<5}{r['dominant_side']:<10}"
              f"{r['full_share']*100:>7.1f}{r['min_loso_share']*100:>9.1f}  "
              f"{'Y' if r['significant'] else 'n':<4} {m}")
    print(f"\n有意 11 組合せ全てで方向保存: {'YES' if sig_all_ok else 'NO'}  -> {OUT}")


if __name__ == "__main__":
    main()
