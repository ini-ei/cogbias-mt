#!/usr/bin/env python3
"""
thesis_gap_numbers.py — 卒論 4 章「日英バイアス差」の数値を camera-ready レコードから再計算する。
  .venv/bin/python scripts/thesis_gap_numbers.py            # 表を stdout に、JSON を results/thesis/ja_en_gap.json に
定義:
  self  = 専用自己評価ライン (full_n80*/)
  cross = 自分以外のパネル判定者 (3 or 4) の平均 (cross_eval_*/)、同一モデルセルは除外
  Δ     = JA − EN (pp)。CI は日英の問題を独立に再標本する問題ブートストラップ (10,000, seed 42)
  被験別・判定者別・バイアス種別・ドメイン別の T2/T4 を出す。
"""
import json, os, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ.setdefault("SUPP", "")
from wiiat_numbers import load_panel, load_self, SUBJECTS, EVALS, SELF_OF, br  # noqa: E402

R = os.path.join(ROOT, "results"); OUT_DIR = os.path.join(R, "thesis"); os.makedirs(OUT_DIR, exist_ok=True)
N_BOOT, SEED = 10_000, 42


def main():
    J = {L: load_panel(R, L) for L in ("JA", "EN")}; S = {L: load_self(R, L) for L in ("JA", "EN")}
    rng = np.random.default_rng(SEED); out = {"subjects": {}, "by_judge": {}, "by_bias": {}, "by_domain": {}}

    def cross_mat(L, m, t):
        evs = [ev for ev in EVALS if SELF_OF[ev] != m]
        pids = sorted(set.intersection(*[set(J[L][ev][m]) for ev in evs]))
        return np.array([[J[L][ev][m][p][t] for ev in evs] for p in pids], float)

    def self_vec(L, m, t):
        return np.array([v[t] for v in S[L][m].values()], float)

    def gap(A, B):
        """A: JA array (problems × k), B: EN array → Δpp, CI"""
        A2 = A.reshape(len(A), -1); B2 = B.reshape(len(B), -1); n, k = len(A2), len(B2)
        d = 100 * (A2[rng.integers(0, n, (N_BOOT, n))].mean((1, 2)) - B2[rng.integers(0, k, (N_BOOT, k))].mean((1, 2)))
        return round(100 * (A2.mean() - B2.mean()), 1), [round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)]

    print("| 被験 | self T2 JA/EN (Δ [CI]) | self T4 JA/EN (Δ [CI]) | cross T2 JA/EN (Δ [CI]) | cross T4 JA/EN (Δ [CI]) |")
    print("|:--|:--|:--|:--|:--|")
    for m in SUBJECTS:
        row = {}; cells = []
        for line, fn in (("self", self_vec), ("cross", cross_mat)):
            for t, tn in ((0, "t2"), (1, "t4")):
                A, B = fn("JA", m, t), fn("EN", m, t); dlt, ci = gap(A, B)
                row[f"{line}_{tn}"] = dict(ja=round(100 * A.mean(), 1), en=round(100 * B.mean(), 1), delta=dlt, ci=ci)
                cells.append(f"{100 * A.mean():.1f}/{100 * B.mean():.1f} ({dlt:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}])")
        out["subjects"][m] = row; print(f"| {m} | " + " | ".join(cells) + " |")

    print("\n判定者別 cross T4 Δ(JA−EN) — 行=被験, 列=判定者:")
    for m in SUBJECTS:
        out["by_judge"][m] = {}
        parts = []
        for ev in EVALS:
            if SELF_OF[ev] == m: continue
            a, b = br(J["JA"][ev][m], 1)[0], br(J["EN"][ev][m], 1)[0]; out["by_judge"][m][ev] = round(a - b, 1); parts.append(f"{ev} {a - b:+.1f}")
        print(f"  {m:18s} " + "  ".join(parts))

    for idx, key in ((0, "by_bias"), (1, "by_domain")):
        acc = {L: defaultdict(lambda: defaultdict(list)) for L in ("JA", "EN")}
        for L in ("JA", "EN"):
            for ev in EVALS:
                for m in SUBJECTS:
                    if SELF_OF[ev] == m: continue
                    for p, v in J[L][ev][m].items():
                        acc[L][p.split("-")[idx]][p].append(v[1])
        print(f"\ncross T4 {key}:")
        for k in acc["JA"]:
            A = np.array([np.mean(v) for v in acc["JA"][k].values()]); B = np.array([np.mean(v) for v in acc["EN"][k].values()])
            dlt, ci = gap(A[:, None], B[:, None]); out[key][k] = dict(ja=round(100 * A.mean(), 1), en=round(100 * B.mean(), 1), delta=dlt, ci=ci)
            print(f"  {k:20s} JA {100 * A.mean():.1f}  EN {100 * B.mean():.1f}  Δ {dlt:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]")

    p = os.path.join(OUT_DIR, "ja_en_gap.json"); json.dump(out, open(p, "w"), indent=1, ensure_ascii=False); print("\nsaved ->", p)


if __name__ == "__main__":
    main()
