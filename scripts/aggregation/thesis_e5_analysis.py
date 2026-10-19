#!/usr/bin/env python3
"""
thesis_e5_analysis.py — 翻訳対照セット（日本語 20 問を英訳し、フロンティア 4 被験に解かせ、4 評価者で判定）の集計。
  .venv/bin/python scripts/thesis_e5_analysis.py --tag e5_trans --json results/thesis_exp/e5_analysis.json
比べるもの（評価者 × ターン、4 フロンティア被験の交差評価 = 自分以外 3 被験 × 20 問 = 60 判定）:
  JA20   : 日本語の元の判定（同じ 20 問）
  ENtr20 : 英訳版の判定（同じ 20 問）
  EN20   : 英語の元のセットのうち同じバイアス種 × ドメインの -001 問（別項目）
  EN80   : 英語の元のセット全 80 問
  言語の効果 = JA20 − ENtr20（同じ項目）、項目の効果 = ENtr20 − EN20（同じ言語）。区間は問題の対応ありブートストラップ 1 万回。
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wiiat_numbers as W  # noqa: E402
from thesis_rerun_analysis import panel_turns  # noqa: E402
FR = W.FRONTIER


def rates(J, ev, t, subjects, ids):
    """subject -> pid -> value（評価者 ev、自分以外）"""
    out = {}
    for m in subjects:
        if m == W.SELF_OF[ev]: continue
        for p, v in J[ev].get(m, {}).items():
            if p in ids and v.get(t) is not None: out[(m, p)] = float(v[t])
    return out


def mean_ci(vals_by_pid, pids, nboot=10000, seed=0):
    rng = np.random.default_rng(seed); pids = [p for p in pids if p in vals_by_pid]
    per = np.array([np.mean(vals_by_pid[p]) for p in pids])
    if len(per) == 0: return None
    boots = [np.mean(per[rng.integers(0, len(per), len(per))]) for _ in range(nboot)]
    return dict(mean=round(100 * float(per.mean()), 1), lo=round(100 * float(np.percentile(boots, 2.5)), 1), hi=round(100 * float(np.percentile(boots, 97.5)), 1), n_pid=len(per))


def paired_diff(a, b, pids, nboot=10000, seed=0):
    rng = np.random.default_rng(seed); pids = [p for p in pids if p in a and p in b]
    d = np.array([np.mean(a[p]) - np.mean(b[p]) for p in pids])
    if len(d) == 0: return None
    boots = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(nboot)]
    return dict(mean=round(100 * float(d.mean()), 1), lo=round(100 * float(np.percentile(boots, 2.5)), 1), hi=round(100 * float(np.percentile(boots, 97.5)), 1), n_pid=len(d))


def by_pid(cells):
    out = {}
    for (m, p), v in cells.items(): out.setdefault(p, []).append(v)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="e5_trans"); ap.add_argument("--ids", default=os.path.join(ROOT, "results/thesis_exp/e5_trans/jp20_to_en.jsonl"))
    ap.add_argument("--json", default=None); a = ap.parse_args()
    ids20 = [json.loads(l)["id"] for l in open(a.ids, encoding="utf-8") if l.strip()]
    cells20 = {tuple(p.split("-")[:2]) for p in ids20}
    JA = panel_turns(os.path.join(ROOT, "results"), "JA"); EN = panel_turns(os.path.join(ROOT, "results"), "EN")
    TR = panel_turns(os.path.join(ROOT, "results", "thesis_exp", a.tag), "EN")
    en_ids = sorted({p for ev in W.EVALS for m in W.SUBJECTS for p in EN[ev][m]})
    en20 = [p for p in en_ids if tuple(p.split("-")[:2]) in cells20 and p.endswith("-001")]
    print(f"JA20 {len(ids20)} ids, EN80 {len(en_ids)} ids, EN20 {len(en20)} ids; translated records:", {ev: sum(len(TR[ev][m]) for m in FR) for ev in W.EVALS})
    out = {"ids20": ids20, "en20": en20, "judges": {}}
    for ev in W.EVALS:
        out["judges"][ev] = {}
        for t, tn in ((0, "T2"), (1, "T4")):
            ja = by_pid(rates(JA, ev, t, FR, set(ids20))); tr = by_pid(rates(TR, ev, t, FR, set(ids20)))
            en20r = by_pid(rates(EN, ev, t, FR, set(en20))); en80 = by_pid(rates(EN, ev, t, FR, set(en_ids)))
            r = dict(JA20=mean_ci(ja, ids20), ENtr20=mean_ci(tr, ids20), EN20=mean_ci(en20r, en20), EN80=mean_ci(en80, en_ids),
                     lang_effect_JA_minus_ENtr=paired_diff(ja, tr, ids20))
            # 項目の効果: 同じセルの -001 どうしを対応づける
            key = lambda p: tuple(p.split("-")[:2])
            trk = {key(p): v for p, v in tr.items()}; enk = {key(p): v for p, v in en20r.items()}
            r["item_effect_ENtr_minus_EN20"] = paired_diff(trk, enk, sorted(set(trk) & set(enk)))
            jak = {key(p): v for p, v in ja.items()}
            r["orig_gap_JA_minus_EN20"] = paired_diff(jak, enk, sorted(set(jak) & set(enk)))
            out["judges"][ev][tn] = r
            f = lambda d: "—" if d is None else f"{d['mean']} [{d['lo']}, {d['hi']}]"
            print(f"  {ev} {tn}: JA20 {f(r['JA20'])} | ENtr20 {f(r['ENtr20'])} | EN20 {f(r['EN20'])} | EN80 {f(r['EN80'])} || 言語 {f(r['lang_effect_JA_minus_ENtr'])} 項目 {f(r['item_effect_ENtr_minus_EN20'])} 元の日英差 {f(r['orig_gap_JA_minus_EN20'])}")
    # 被験モデル別（評価者平均）
    out["subjects"] = {}
    for m in FR:
        out["subjects"][m] = {}
        for t, tn in ((0, "T2"), (1, "T4")):
            def avg(J, ids):
                vals = [float(J[ev][m][p][t]) for ev in W.EVALS if W.SELF_OF[ev] != m for p in ids if p in J[ev].get(m, {}) and J[ev][m][p].get(t) is not None]
                return round(100 * float(np.mean(vals)), 1) if vals else None
            out["subjects"][m][tn] = dict(JA20=avg(JA, ids20), ENtr20=avg(TR, ids20), EN20=avg(EN, en20), EN80=avg(EN, en_ids))
        print(f"  subject {m}: {out['subjects'][m]}")
    # ペアの優勢側（反証後、英訳版）
    TRb = {ev: {m: {p: (v[0], v[1]) for p, v in TR[ev][m].items() if v[0] is not None and v[1] is not None} for m in W.SUBJECTS} for ev in W.EVALS}
    out["pairs_ENtr_T4"] = {}
    for a_, b_ in W.PAIRS:
        c = W.pair_cells(TRb, a_, b_, 1)
        if c["n"] == 0: continue
        dom, d, o, sh = W.fmt_pair(c, a_, b_); out["pairs_ENtr_T4"][f"{a_}-{b_}"] = dict(n=c["n"], n_dis=d + o, share=round(sh, 1), dom=dom)
        print(f"  pair {a_}-{b_}: n={c['n']} dis={d + o} share={sh:.1f} dom={dom}")
    if a.json: json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False); print("wrote", a.json)


if __name__ == "__main__":
    main()
