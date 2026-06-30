"""
forced-rubric 実験の集計: Type A/C 反転率 + κ 回復。
baseline: 対象は GPT-only T4 不一致セル → GPT=bias(1), Opus=no-bias(0)。
forced 後の各評価者の判定から、判定者間 κ がどれだけ回復したかを出す。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/phase5_v2/forced_rubric"


def load(name):
    p = OUT / f"{name}.json"
    if not p.exists():
        return {}
    return {(r["problem_id"], r["subject"]): r["bias_detected"]
            for r in json.load(open(p)) if r.get("bias_detected") is not None}


def kappa(a, b):
    keys = set(a) & set(b)
    n = len(keys)
    if n == 0:
        return None, 0
    n11 = sum(1 for k in keys if a[k] and b[k])
    n00 = sum(1 for k in keys if not a[k] and not b[k])
    po = (n11 + n00) / n
    pa = sum(1 for k in keys if a[k]) / n
    pb = sum(1 for k in keys if b[k]) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return ((po - pe) / (1 - pe) if pe != 1 else 1.0), n


def main():
    op_out, op_pro = load("claude-opus-4.7_outcome"), load("claude-opus-4.7_process")
    gp_out, gp_pro = load("gpt-5_outcome"), load("gpt-5_process")

    def rate(d):
        return (sum(1 for v in d.values() if v) / len(d), len(d)) if d else (None, 0)

    print("=== Forced-rubric results (GPT-only T4 disagreement cells; baseline GPT=1, Opus=0) ===")
    for name, d in [("Opus + outcome-forced", op_out), ("Opus + process-forced", op_pro),
                    ("GPT  + outcome-forced", gp_out), ("GPT  + process-forced", gp_pro)]:
        r, n = rate(d)
        print(f"  {name}: bias_rate = {r*100:.1f}% (n={n})" if r is not None else f"  {name}: (no data)")

    print("\n=== Hypothesis B prediction check ===")
    ro, _ = rate(op_out); rp, _ = rate(op_pro)
    go, _ = rate(gp_out); gp, _ = rate(gp_pro)
    if ro is not None:
        print(f"  Opus under OUTCOME → {ro*100:.0f}% bias (Hyp B: should RISE toward 100%; baseline Opus=0%)  {'✓' if ro>0.5 else '—'}")
    if rp is not None:
        print(f"  Opus under PROCESS → {rp*100:.0f}% bias (should stay LOW near 0%)  {'✓' if rp<0.5 else '—'}")
    if gp is not None:
        print(f"  GPT  under PROCESS → {gp*100:.0f}% bias (Hyp B: should FALL toward 0%; baseline GPT=100%)  {'✓' if gp<0.5 else '—'}")
    if go is not None:
        print(f"  GPT  under OUTCOME → {go*100:.0f}% bias (should stay HIGH near 100%)  {'✓' if go>0.5 else '—'}")

    print("\n=== Inter-judge kappa recovery (GPT vs Opus on these cells) ===")
    print("  baseline (paper): kappa approx 0 (these are disagreement cells, GPT=1/Opus=0)")
    ko, no = kappa(gp_out, op_out)
    kp, np_ = kappa(gp_pro, op_pro)
    if ko is not None:
        print(f"  under OUTCOME-forced (both judges): kappa = {ko:+.3f} (n={no})")
    if kp is not None:
        print(f"  under PROCESS-forced (both judges): kappa = {kp:+.3f} (n={np_})")
    # agreement rate too
    for lab, a, b in [("outcome", gp_out, op_out), ("process", gp_pro, op_pro)]:
        keys = set(a) & set(b)
        if keys:
            agree = sum(1 for k in keys if a[k] == b[k]) / len(keys)
            print(f"  agreement under {lab}-forced: {agree*100:.0f}% (n={len(keys)})")


if __name__ == "__main__":
    main()
