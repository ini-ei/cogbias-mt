"""
Exp E: 再現性検証（提案E）

目的
----
temperature=0.7 固定での単一実行結果が、確率性によるブレではなく
モデル本来の傾向を捉えているかを検証する（教員フィードバック 2026-04-17）。

設計
----
- 境界10問（評価者間不一致スコア上位、選定根拠は PROGRESS_LOG 2026-04-18）
- 各モデル × 境界問題 × 3回実行（temp=0.7）
- T2/T4 の bias_detected フラグが試行間でどれだけ揺らぐかを測定

出力
----
- results/phase5_v2/exp_e_reproducibility/<model>_runs.json : 全試行の生回答
- results/phase5_v2/exp_e_reproducibility/summary.json : 試行間分散・一致率
"""

import json
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.phase5_benchmark_validation import BenchmarkEvaluator


# 境界10問（disagreement_cases.json で評価者間不一致スコア上位）
BOUNDARY_PROBLEM_IDS = [
    "framing-labor-hard-001",
    "anchoring-legal-hard-001",
    "confirmation_bias-labor-hard-002",
    "framing-legal-hard-002",
    "framing-daily_life-hard-002",
    "representativeness-medical-hard-001",
    "confirmation_bias-legal-hard-001",
    "confirmation_bias-education-hard-001",
    "framing-legal-hard-001",
    "confirmation_bias-medical-hard-001",
]

N_RUNS = 3
TEMPERATURE = 0.7


def load_boundary_problems(benchmark_path: str) -> list:
    problems = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p["id"] in BOUNDARY_PROBLEM_IDS:
                    problems.append(p)
    missing = set(BOUNDARY_PROBLEM_IDS) - {p["id"] for p in problems}
    if missing:
        raise ValueError(f"境界問題が見つからない: {missing}")
    problems.sort(key=lambda p: BOUNDARY_PROBLEM_IDS.index(p["id"]))
    return problems


def run_model(model: str, problems: list, n_runs: int, output_dir: str) -> list:
    """各問題を n_runs 回実行し、生結果を保存"""
    evaluator = BenchmarkEvaluator(model=model)
    # NOTE: BenchmarkEvaluator は内部で temperature=0.7 を使用している（デフォルト）
    all_runs = []
    for run_idx in range(n_runs):
        print(f"\n--- {model}: run {run_idx + 1}/{n_runs} ---")
        for p in problems:
            print(f"  [{run_idx + 1}] {p['id']}")
            try:
                result = evaluator._route_evaluate(p)
                result["run_idx"] = run_idx
                all_runs.append(result)
            except Exception as e:
                print(f"    エラー: {e}")
                all_runs.append(
                    {"problem_id": p["id"], "run_idx": run_idx, "model": model, "error": str(e)}
                )
    out_path = os.path.join(output_dir, f"{model}_runs.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {out_path}")
    return all_runs


def compute_consistency(all_runs_by_model: dict) -> dict:
    """試行間の一致率を計算

    各 (model, problem_id) について、T2/T4 の bias_detected が
    n_runs 回で何回一致するかを集計。
    """
    summary = {}
    for model, runs in all_runs_by_model.items():
        by_pid = defaultdict(list)
        for r in runs:
            if "error" in r:
                continue
            by_pid[r["problem_id"]].append(r)

        per_problem = {}
        t2_stable = 0
        t4_stable = 0
        total = 0
        for pid, rs in by_pid.items():
            t2_vals = [bool(r.get("turn2_bias_detected")) for r in rs]
            t4_vals = [bool(r.get("turn4_bias_detected")) for r in rs]
            t2_same = len(set(t2_vals)) == 1
            t4_same = len(set(t4_vals)) == 1
            per_problem[pid] = {
                "n_runs_completed": len(rs),
                "t2_values": t2_vals,
                "t4_values": t4_vals,
                "t2_stable": t2_same,
                "t4_stable": t4_same,
            }
            total += 1
            if t2_same:
                t2_stable += 1
            if t4_same:
                t4_stable += 1
        summary[model] = {
            "n_problems": total,
            "t2_stable_count": t2_stable,
            "t2_stable_rate": round(t2_stable / total, 3) if total else None,
            "t4_stable_count": t4_stable,
            "t4_stable_rate": round(t4_stable / total, 3) if total else None,
            "per_problem": per_problem,
        }
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Exp E: 再現性検証")
    parser.add_argument(
        "--benchmark",
        default="results/phase5/hard_all_40.jsonl",
        help="ベンチマーク JSONL",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5", "claude-4-sonnet", "gemini-3.1-pro", "llama-4-maverick"],
    )
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--output-dir", default="results/phase5_v2/exp_e_reproducibility")
    args = parser.parse_args()

    print("=" * 80)
    print("Exp E: 再現性検証（temp=0.7 × 3回、境界10問）")
    print("=" * 80)
    print(f"モデル: {args.models}")
    print(f"試行数: {args.n_runs}")
    print(f"境界問題: {len(BOUNDARY_PROBLEM_IDS)}問")

    problems = load_boundary_problems(args.benchmark)
    print(f"ロード完了: {len(problems)}問")

    all_runs_by_model = {}
    for m in args.models:
        all_runs_by_model[m] = run_model(m, problems, args.n_runs, args.output_dir)

    summary = compute_consistency(all_runs_by_model)

    out_summary = os.path.join(args.output_dir, "summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "n_runs": args.n_runs,
                    "temperature": TEMPERATURE,
                    "boundary_problem_ids": BOUNDARY_PROBLEM_IDS,
                },
                "summary_by_model": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nサマリ保存: {out_summary}")

    print("\n" + "=" * 80)
    print("【一致率サマリ】")
    print("=" * 80)
    print(f"{'model':<25} {'T2 stable':<15} {'T4 stable':<15}")
    for m, s in summary.items():
        t2 = f"{s['t2_stable_count']}/{s['n_problems']} ({s['t2_stable_rate']})"
        t4 = f"{s['t4_stable_count']}/{s['n_problems']} ({s['t4_stable_rate']})"
        print(f"{m:<25} {t2:<15} {t4:<15}")


if __name__ == "__main__":
    main()
