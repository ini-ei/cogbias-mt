"""
Exp I: Ablation（本体 vs 周辺の寄与度分離）

目的
----
教員フィードバック（2026-04-17）の指摘
「寄与度が大きいのは LLM本体なのか、周辺（system prompt / temperature 等）なのか」
に対して、同一モデル内で周辺設定を変えた時の BiasRate 揺らぎを測定し、
モデル間差と比較する。

設計
----
- 境界10問（exp_e_reproducibility と同じ集合）
- 条件:
    (A) with_system_prompt × temperature = 0.7  ← ベースライン（Phase 5 v2 と同等）
    (B) without_system_prompt × temperature = 0.7 ← system prompt の寄与
    (C) with_system_prompt × temperature = 0.3 ← temperature の寄与（低）
    (D) with_system_prompt × temperature = 1.0 ← temperature の寄与（高）
- 各モデル × 4条件 × 境界10問 × 1回

判定
----
- モデル間 BiasRate 分散 ≫ 条件間 BiasRate 分散 なら「モデル本体が支配的」
- モデル間 BiasRate 分散 ≈ 条件間 BiasRate 分散 なら「周辺設定が交絡」
"""

import json
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.phase5_benchmark_validation import BenchmarkEvaluator
from scripts.exp_e_reproducibility import BOUNDARY_PROBLEM_IDS, load_boundary_problems


class AblationEvaluator(BenchmarkEvaluator):
    """周辺条件を差し替え可能にした評価器"""

    def __init__(self, model: str, with_system_prompt: bool, temperature: float):
        super().__init__(model=model)
        self.with_system_prompt = with_system_prompt
        self.response_temperature = temperature

    def _run_multiturn(self, problem: dict) -> tuple:
        """T2 と T4 の応答を条件下で生成"""
        turn1 = next(t for t in problem["turns"] if t["turn_number"] == 1)
        turn2 = next(t for t in problem["turns"] if t["turn_number"] == 2)
        turn3 = next(t for t in problem["turns"] if t["turn_number"] == 3)
        turn4 = next(t for t in problem["turns"] if t["turn_number"] == 4)

        t2_user_msg = turn2["content"]
        if turn2.get("question"):
            t2_user_msg += f"\n\n{turn2['question']}"

        messages_t2 = []
        if self.with_system_prompt:
            messages_t2.append({"role": "system", "content": turn1["content"]})
        else:
            # system prompt を落とす代わりに T1 の内容を user 側に移して情報量は保存
            messages_t2.append({"role": "user", "content": turn1["content"]})
            messages_t2.append({"role": "assistant", "content": "承知しました。"})
        messages_t2.append({"role": "user", "content": t2_user_msg})

        resp_t2 = self.client.chat.completions.create(
            model=self.model,
            messages=messages_t2,
            temperature=self.response_temperature,
        )
        answer_t2 = resp_t2.choices[0].message.content

        t4_user_msg = turn3["content"]
        if turn4.get("content"):
            t4_user_msg += f"\n\n{turn4['content']}"
        if turn4.get("question"):
            t4_user_msg += f"\n\n{turn4['question']}"

        messages_t4 = list(messages_t2) + [
            {"role": "assistant", "content": answer_t2},
            {"role": "user", "content": t4_user_msg},
        ]
        resp_t4 = self.client.chat.completions.create(
            model=self.model,
            messages=messages_t4,
            temperature=self.response_temperature,
        )
        answer_t4 = resp_t4.choices[0].message.content
        return answer_t2, answer_t4

    def evaluate_with_condition(self, problem: dict) -> dict:
        answer_t2, answer_t4 = self._run_multiturn(problem)

        # 評価はベースと同じロジック（temperature=0.3 で評価する _evaluate_bias を流用）
        bias_eval_t2 = self._evaluate_bias(
            answer_t2,
            problem.get("ideal_responses", {}).get("turn2", ""),
            problem.get("bias_indicators", []),
            problem.get("debiased_indicators", []),
            problem.get("evaluation_criteria", {}).get("turn2", {}),
        )
        bias_eval_t4 = self._evaluate_bias(
            answer_t4,
            problem.get("ideal_responses", {}).get("turn4", ""),
            problem.get("bias_indicators", []),
            problem.get("debiased_indicators", []),
            problem.get("evaluation_criteria", {}).get("turn4", {}),
        )
        return {
            "problem_id": problem["id"],
            "bias_type": problem["bias_type"],
            "domain": problem["domain"],
            "model": self.model_short,
            "with_system_prompt": self.with_system_prompt,
            "temperature": self.response_temperature,
            "turn2_response": answer_t2,
            "turn4_response": answer_t4,
            "turn2_bias_detected": bias_eval_t2["bias_detected"],
            "turn4_bias_detected": bias_eval_t4["bias_detected"],
        }


CONDITIONS = [
    {"label": "A_baseline", "with_system_prompt": True, "temperature": 0.7},
    {"label": "B_no_sysprompt", "with_system_prompt": False, "temperature": 0.7},
    {"label": "C_temp_low", "with_system_prompt": True, "temperature": 0.3},
    {"label": "D_temp_high", "with_system_prompt": True, "temperature": 1.0},
]


def run_condition(model: str, condition: dict, problems: list, output_dir: str) -> list:
    label = condition["label"]
    print(f"\n--- {model} / {label} "
          f"(sysprompt={condition['with_system_prompt']}, temp={condition['temperature']}) ---")
    evaluator = AblationEvaluator(
        model=model,
        with_system_prompt=condition["with_system_prompt"],
        temperature=condition["temperature"],
    )
    results = []
    for p in problems:
        print(f"  {p['id']}")
        try:
            r = evaluator.evaluate_with_condition(p)
            r["condition"] = label
            results.append(r)
        except Exception as e:
            print(f"    エラー: {e}")
            results.append(
                {
                    "problem_id": p["id"],
                    "model": model,
                    "condition": label,
                    "error": str(e),
                }
            )
    out_path = os.path.join(output_dir, f"{model}_{label}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  保存: {out_path}")
    return results


def aggregate_variance(all_results: list) -> dict:
    """条件間分散とモデル間分散を比較

    BiasRate = T2 で bias_detected が true の割合
    """
    # (model, condition) ごとに BiasRate を計算
    br = defaultdict(lambda: {"n": 0, "biased": 0})
    for r in all_results:
        if "error" in r:
            continue
        key = (r["model"], r["condition"])
        br[key]["n"] += 1
        br[key]["biased"] += int(bool(r.get("turn2_bias_detected")))

    bias_rate = {
        k: (v["biased"] / v["n"] if v["n"] else None) for k, v in br.items()
    }

    # モデルごとに条件間分散
    per_model_variance = {}
    models = sorted({k[0] for k in bias_rate})
    conds = sorted({k[1] for k in bias_rate})
    for m in models:
        rates = [bias_rate[(m, c)] for c in conds if (m, c) in bias_rate]
        if len(rates) >= 2:
            mean = sum(rates) / len(rates)
            var = sum((r - mean) ** 2 for r in rates) / len(rates)
            per_model_variance[m] = {
                "mean_bias_rate": round(mean, 3),
                "variance": round(var, 4),
                "range": round(max(rates) - min(rates), 3),
                "by_condition": {c: round(bias_rate[(m, c)], 3) for c in conds if (m, c) in bias_rate},
            }

    # ベースライン（A）でのモデル間分散
    baseline_rates = [bias_rate[(m, "A_baseline")] for m in models if (m, "A_baseline") in bias_rate]
    if len(baseline_rates) >= 2:
        mean = sum(baseline_rates) / len(baseline_rates)
        model_variance = sum((r - mean) ** 2 for r in baseline_rates) / len(baseline_rates)
        cross_model_variance = {
            "baseline_mean_bias_rate": round(mean, 3),
            "variance": round(model_variance, 4),
            "range": round(max(baseline_rates) - min(baseline_rates), 3),
            "by_model": {m: round(bias_rate[(m, "A_baseline")], 3) for m in models if (m, "A_baseline") in bias_rate},
        }
    else:
        cross_model_variance = None

    return {
        "bias_rate_table": {f"{m}|{c}": round(v, 3) for (m, c), v in bias_rate.items()},
        "per_model_condition_variance": per_model_variance,
        "cross_model_variance_at_baseline": cross_model_variance,
        "interpretation": (
            "モデル間分散 (cross_model) > 条件間分散 (per_model) の平均値 なら、"
            "モデル本体の寄与が周辺設定より支配的と解釈できる。"
        ),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Exp I: Ablation 実験")
    parser.add_argument(
        "--benchmark",
        default="results/phase5/hard_all_40.jsonl",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5", "claude-4-sonnet", "gemini-3.1-pro", "llama-4-maverick"],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[c["label"] for c in CONDITIONS],
        help="実行する条件ラベル",
    )
    parser.add_argument("--output-dir", default="results/phase5_v2/exp_i_ablation")
    args = parser.parse_args()

    print("=" * 80)
    print("Exp I: Ablation（本体 vs 周辺の寄与度分離）")
    print("=" * 80)

    problems = load_boundary_problems(args.benchmark)
    print(f"境界問題: {len(problems)}問")

    selected_conditions = [c for c in CONDITIONS if c["label"] in args.conditions]
    print(f"条件数: {len(selected_conditions)}")
    for c in selected_conditions:
        print(f"  {c['label']}: sysprompt={c['with_system_prompt']}, temp={c['temperature']}")

    all_results = []
    for m in args.models:
        for cond in selected_conditions:
            rs = run_condition(m, cond, problems, args.output_dir)
            all_results.extend(rs)

    agg = aggregate_variance(all_results)
    out_summary = os.path.join(args.output_dir, "ablation_summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"\nサマリ保存: {out_summary}")

    print("\n" + "=" * 80)
    print("【寄与度サマリ】")
    print("=" * 80)
    if agg["cross_model_variance_at_baseline"]:
        cmv = agg["cross_model_variance_at_baseline"]
        print(f"モデル間分散 (baseline A):  variance={cmv['variance']}, range={cmv['range']}")
    for m, v in agg["per_model_condition_variance"].items():
        print(f"  {m}: 条件間 variance={v['variance']}, range={v['range']}")


if __name__ == "__main__":
    main()
