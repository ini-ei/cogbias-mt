"""
失敗 (`error` フィールドあり) の評価エントリだけを再走するスクリプト。

self-eval / cross-eval どちらの結果ファイルにも対応。
2026-05-08 EN run の HTTP 402 (クレジット不足) 大量発生への対応。

仕組み:
  入力ファイル {model}_results.json を読み込み、
  - 'error' を持つエントリ → benchmark から該当 problem を引き、再評価
  - error 無しのエントリ → そのまま
  結果は同じファイルに上書き保存（intermediate save あり）。

使用:
  # self-eval の retry
  python3 scripts/retry_errored_evaluations.py self-eval \\
    --benchmark results/phase5_v2_en/hard_all_80_en_clear_ce.jsonl \\
    --results results/phase5_v2_en/full_n80_en/claude-opus-4.7_results.json \\
    --model claude-opus-4.7

  # cross-eval の retry
  python3 scripts/retry_errored_evaluations.py cross-eval \\
    --benchmark results/phase5_v2_en/hard_all_80_en_clear_ce.jsonl \\
    --results results/phase5_v2_en/cross_eval_gpt5_en/claude-opus-4.7_results.json \\
    --evaluator gpt-5 \\
    --subject claude-opus-4.7

  # 一括 retry（全モデル × 全評価者を順番に）
  python3 scripts/retry_errored_evaluations.py all \\
    --en-dir results/phase5_v2_en
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def index_benchmark(benchmark_path: str) -> dict:
    """benchmark jsonl を {problem_id: problem} に索引化"""
    idx = {}
    with open(benchmark_path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                idx[p["id"]] = p
    return idx


def is_broken_self_eval(r: dict) -> bool:
    """self-eval entry が「壊れている」かを判定。

    壊れているケース:
      1. 'error' フィールドがある (HTTP 402 等で明示的に失敗)
      2. 'error' は無いが turn2_response / turn4_response が空文字列 / None
         (API が空応答を返したが例外にならなかったケース)
    """
    if "error" in r:
        return True
    t2 = r.get("turn2_response")
    t4 = r.get("turn4_response")
    if t2 in (None, "") or t4 in (None, ""):
        return True
    return False


def retry_self_eval(benchmark: str, results_path: str, model: str,
                    max_attempts: int = 5):
    """self-eval の壊れた entry を retry。

    各 entry につき最大 max_attempts 回まで再試行。
    gemini 等の確率的空応答を吸収する。
    """
    from scripts.phase5_benchmark_validation import BenchmarkEvaluator

    bench_idx = index_benchmark(benchmark)
    with open(results_path) as f:
        results = json.load(f)

    error_entries = [(i, r) for i, r in enumerate(results) if is_broken_self_eval(r)]
    n_with_error = sum(1 for _, r in error_entries if "error" in r)
    n_empty_turn = len(error_entries) - n_with_error
    print(f"[self-eval-{model}] {len(error_entries)} broken entries to retry "
          f"({n_with_error} with 'error' field, {n_empty_turn} with empty turn response), "
          f"max_attempts={max_attempts}")

    if not error_entries:
        print("  no broken entries to retry, exiting.")
        return

    evaluator = BenchmarkEvaluator(model=model)
    fixed = 0
    for i, r in error_entries:
        pid = r.get("problem_id")
        if pid not in bench_idx:
            print(f"  [skip] {pid} not in benchmark")
            continue

        new_r = None
        success = False
        last_err = ""
        for attempt in range(1, max_attempts + 1):
            try:
                new_r = evaluator._route_evaluate(bench_idx[pid])
                if not is_broken_self_eval(new_r):
                    results[i] = new_r
                    fixed += 1
                    print(f"  [fixed attempt={attempt}] {pid}")
                    success = True
                    break
                else:
                    # まだ壊れてる: 何が空か特定
                    issues = []
                    if "error" in new_r:
                        issues.append(f"error={new_r.get('error', '')[:50]}")
                    if not new_r.get("turn2_response"):
                        issues.append("turn2_empty")
                    if not new_r.get("turn4_response"):
                        issues.append("turn4_empty")
                    last_err = ", ".join(issues)
                    print(f"  [retry attempt={attempt}] {pid}: {last_err}")
            except Exception as e:
                last_err = str(e)
                print(f"  [exception attempt={attempt}] {pid}: {e}")
        if not success and new_r is not None:
            results[i] = new_r  # 最後の試行結果を保存（broken のまま）
            print(f"  [still broken after {max_attempts} attempts] {pid}: {last_err}")

        # intermediate save every 5 fixed
        if fixed % 5 == 0 and fixed > 0:
            with open(results_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    remaining = sum(1 for r in results if is_broken_self_eval(r))
    print(f"[self-eval-{model}] done. fixed: {fixed}, remaining broken: {remaining}")


def is_broken_cross_eval(r: dict) -> bool:
    """cross-eval entry が壊れているかを判定。

    壊れているケース:
      1. 'error' フィールドがある
      2. 'error' は無いが turn2_bias_detected / turn4_bias_detected が None
         (評価判定が出ていない)
    """
    if "error" in r:
        return True
    if r.get("turn2_bias_detected") is None or r.get("turn4_bias_detected") is None:
        return True
    return False


def retry_cross_eval(benchmark: str, results_path: str, evaluator_model: str,
                     subject_results_path: str = None,
                     max_attempts: int = 5):
    """cross-eval の壊れた entry を retry。

    cross-eval は subject の self-eval 応答 (turn{2,4}_response) を読んで評価のみ実行する。
    壊れている entry には turn{2,4}_response が無い可能性が高いので、
    subject の self-eval 結果ファイル (subject_results_path) から取得する。
    subject 側も「error は無いが turn response が空」のケースがあるので、
    is_broken_self_eval で valid な subject だけを索引化する。
    """
    from scripts.phase5_cross_eval import CrossEvaluator

    bench_idx = index_benchmark(benchmark)
    with open(results_path) as f:
        results = json.load(f)

    # subject の self-eval 結果を {pid: result} で索引化（turn responses 取得用）
    # error 無し + turn{2,4}_response が両方ある entry のみ採用
    subject_idx = {}
    if subject_results_path and os.path.exists(subject_results_path):
        with open(subject_results_path) as f:
            for r in json.load(f):
                if not is_broken_self_eval(r):
                    subject_idx[r.get("problem_id")] = r

    error_entries = [(i, r) for i, r in enumerate(results) if is_broken_cross_eval(r)]
    n_with_error = sum(1 for _, r in error_entries if "error" in r)
    n_empty_judgment = len(error_entries) - n_with_error
    print(f"[cross-eval] {len(error_entries)} broken entries to retry "
          f"({n_with_error} with 'error' field, {n_empty_judgment} with missing judgment) "
          f"(evaluator={evaluator_model}, results={results_path})")

    if not error_entries:
        print("  no broken entries to retry, exiting.")
        return

    cross = CrossEvaluator(evaluator_model=evaluator_model)
    fixed = 0
    skipped = 0
    for i, r in error_entries:
        pid = r.get("problem_id")
        if pid not in bench_idx:
            print(f"  [skip] {pid} not in benchmark")
            skipped += 1
            continue
        problem = bench_idx[pid]

        # turn{2,4}_response を取得: r 自体 → subject_idx の順で探す
        t2 = r.get("turn2_response") or subject_idx.get(pid, {}).get("turn2_response")
        t4 = r.get("turn4_response") or subject_idx.get(pid, {}).get("turn4_response")
        if not t2 or not t4:
            print(f"  [skip] {pid}: no subject response available "
                  f"(subject_results: {subject_results_path})")
            skipped += 1
            continue

        # phase5_cross_eval.CrossEvaluator は re_evaluate_bias_result(result, problem) を持つ
        synth_result = {
            "problem_id": pid,
            "bias_type": problem.get("bias_type", "?"),
            "domain": problem.get("domain", "?"),
            "difficulty": problem.get("difficulty", "hard"),
            "harm_type": problem.get("harm_type", "?"),
            "harm_description": problem.get("harm_description", ""),
            "model": r.get("model", "?"),
            "turn2_response": t2,
            "turn4_response": t4,
            "turn2_bias_detected": None,
            "turn4_bias_detected": None,
        }

        new_r = None
        success = False
        last_err = ""
        for attempt in range(1, max_attempts + 1):
            try:
                new_r = cross.re_evaluate_bias_result(synth_result, problem)
                if not is_broken_cross_eval(new_r):
                    results[i] = new_r
                    fixed += 1
                    print(f"  [fixed attempt={attempt}] {pid}")
                    success = True
                    break
                else:
                    issues = []
                    if "error" in new_r:
                        issues.append(f"error={new_r.get('error', '')[:50]}")
                    if new_r.get("turn2_bias_detected") is None:
                        issues.append("t2_judgment=None")
                    if new_r.get("turn4_bias_detected") is None:
                        issues.append("t4_judgment=None")
                    last_err = ", ".join(issues)
                    print(f"  [retry attempt={attempt}] {pid}: {last_err}")
            except Exception as e:
                last_err = str(e)
                print(f"  [exception attempt={attempt}] {pid}: {e}")
        if not success and new_r is not None:
            results[i] = new_r
            print(f"  [still broken after {max_attempts} attempts] {pid}: {last_err}")

        if fixed % 5 == 0 and fixed > 0:
            with open(results_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    remaining = sum(1 for r in results if is_broken_cross_eval(r))
    print(f"[cross-eval] done. fixed: {fixed}, skipped: {skipped}, remaining broken: {remaining}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_self = sub.add_parser("self-eval")
    p_self.add_argument("--benchmark", required=True)
    p_self.add_argument("--results", required=True)
    p_self.add_argument("--model", required=True)

    p_cross = sub.add_parser("cross-eval")
    p_cross.add_argument("--benchmark", required=True)
    p_cross.add_argument("--results", required=True)
    p_cross.add_argument("--evaluator", required=True)
    p_cross.add_argument("--subject", required=True)

    p_all = sub.add_parser("all")
    p_all.add_argument("--en-dir", default="results/phase5_v2_en")
    p_all.add_argument("--benchmark", default="results/phase5_v2_en/hard_all_80_en_clear_ce.jsonl")

    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    if args.cmd == "self-eval":
        retry_self_eval(args.benchmark, args.results, args.model)

    elif args.cmd == "cross-eval":
        retry_cross_eval(args.benchmark, args.results, args.evaluator, args.subject)

    elif args.cmd == "all":
        en_dir = Path(args.en_dir)
        bench = args.benchmark

        # self-eval 全モデル
        models = ["gpt-5", "gpt-5.5", "claude-4-sonnet", "claude-opus-4.7",
                  "gemini-3.1-pro", "llama-4-maverick"]
        print("\n========== self-eval retries ==========")
        for m in models:
            results_path = en_dir / "full_n80_en" / f"{m}_results.json"
            if results_path.exists():
                print(f"\n--- {m} ---")
                retry_self_eval(bench, str(results_path), m)
            else:
                print(f"  [skip] {results_path} not found")

        # cross-eval 全評価者 × 全モデル
        ev_dirs = {
            "gpt-5": "cross_eval_gpt5_en",
            "claude-4-sonnet": "cross_eval_claude_en",
            "gemini-3.1-pro": "cross_eval_gemini_en",
        }
        print("\n========== cross-eval retries ==========")
        for ev, ev_dir in ev_dirs.items():
            for m in models:
                results_path = en_dir / ev_dir / f"{m}_results.json"
                # subject self-eval (turn2/4 response 取得元) — 同じモデル名のファイル
                subject_path = en_dir / "full_n80_en" / f"{m}_results.json"
                if results_path.exists():
                    print(f"\n--- {ev} → {m} (subject: {subject_path}) ---")
                    retry_cross_eval(bench, str(results_path), ev, str(subject_path))
                else:
                    print(f"  [skip] {results_path} not found")


if __name__ == "__main__":
    main()
