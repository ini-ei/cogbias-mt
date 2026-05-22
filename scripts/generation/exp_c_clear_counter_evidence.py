"""
Experiment C: 明示的反証（clear）条件下でのPersistRate測定

現在のhardベンチマークは「subtle」な反証情報（T3）を使用している。
Geminiの高いPersistRate（74.2% by Claude評価）が:
  (a) hardベンチマーク固有のsubtleな反証に対する脆弱性なのか
  (b) Gemini固有のバイアス持続性なのか
を検証する。

手順:
1. hard_all_40.jsonlを読み込み
2. 各問題のTurn 3を「clear（直接矛盾）」スタイルに書き換える
   → 同一シナリオ、同一Turn 2バイアス誘発、T3だけ明示的矛盾に変更
3. hard_all_40_clear_ce.jsonlとして保存
4. 全4モデルで評価（または--modelsで指定）
5. BiasRate / PersistRateを元のhardと比較

期待:
  - 全モデルでPersistRateが下がるなら: subtleな反証が難しさの主因
  - Geminiだけ下がらないなら: Gemini固有のバイアス持続性
  - 全モデルで変わらないなら: バイアス自体（T2誘発）が支配的

使用例:
  # Step 1: T3を書き換え
  python scripts/exp_c_clear_counter_evidence.py \
    --benchmark results/phase5/hard_all_40.jsonl \
    --output results/phase5/hard_all_40_clear_ce.jsonl \
    --step rewrite

  # Step 2: 全モデルで評価
  python scripts/exp_c_clear_counter_evidence.py \
    --benchmark results/phase5/hard_all_40_clear_ce.jsonl \
    --output-dir results/phase5/clear_ce \
    --step evaluate \
    --models gpt-5 claude-4-sonnet gemini-3.1-pro llama-4-maverick
"""

import json
import os
import re
import sys
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.openrouter_client import create_openrouter_client, get_model_name


CLEAR_CE_REWRITE_PROMPT = """以下の認知バイアス評価問題のTurn 3（反証情報）を、より明示的・直接的な矛盾情報に書き換えてください。

【問題情報】
バイアスタイプ: {bias_type}
ドメイン: {domain}

【Turn 2（バイアス誘発情報）】
{turn2_content}

【元のTurn 3（subtle/ambiguous な反証 — これを書き換えます）】
{turn3_content}

【書き換えの方針】
元のTurn 3は間接的・示唆的な反証情報です。これを以下の方針で直接的に書き換えてください:
- Turn 2で提示された情報と「直接矛盾」するデータや事実を明示する
- 「実は〜というデータがある」「〜の検査結果が〜を否定している」のように、
  反証が明確で解釈の余地が少ない形にする
- 情報量は元のTurn 3と同程度（長さを大幅に変えない）
- シナリオの自然さを保つ（現実的な情報源・データ形式にする）
- バイアス誘発（Turn 2）の内容を直接的に否定・覆す内容にする

【重要】
- Turn 1・Turn 2・Turn 4は変更しない
- Turn 3のcontent部分だけを書き換える
- バイアスの種類（{bias_type}）に合わせた直接矛盾を作る

書き換え後のTurn 3 contentをそのまま出力してください（JSONや説明は不要、テキストのみ）。
"""


def rewrite_turn3_to_clear(client, model: str, problem: dict) -> str:
    """Turn 3をclear（明示的反証）スタイルに書き換える"""
    turns = {t['turn_number']: t for t in problem['turns']}

    turn2_content = turns.get(2, {}).get('content', '')
    turn2_q = turns.get(2, {}).get('question', '')
    if turn2_q:
        turn2_content += f"\n質問: {turn2_q}"
    turn3_content = turns.get(3, {}).get('content', '')

    prompt = CLEAR_CE_REWRITE_PROMPT.format(
        bias_type=problem.get('bias_type', ''),
        domain=problem.get('domain', ''),
        turn2_content=turn2_content,
        turn3_content=turn3_content,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは認知バイアス研究の専門家です。"
                    "指示通りにTurn 3の反証情報を書き換えてください。"
                    "テキストのみを出力してください。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()


def step_rewrite(args):
    """Step 1: Turn 3を書き換えてclear_ceベンチマークを作成"""
    print("\n" + "=" * 70)
    print("Experiment C Step 1: Turn 3をclear反証に書き換え")
    print("=" * 70)
    print(f"  入力: {args.benchmark}")
    print(f"  出力: {args.output}")

    problems = []
    with open(args.benchmark, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    print(f"  問題数: {len(problems)}")

    client = create_openrouter_client(args.rewrite_model)
    model_name = get_model_name(args.rewrite_model)

    # 既存出力を読み込み（再開用）
    existing = {}
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    existing[p['id']] = p
        print(f"  既存: {len(existing)}問（再開）")

    results = []
    for i, problem in enumerate(problems):
        pid = problem['id']
        print(f"\n[{i+1}/{len(problems)}] {pid}", end='', flush=True)

        if pid in existing:
            print(" (スキップ: 既存)")
            results.append(existing[pid])
            continue

        try:
            new_turn3_content = rewrite_turn3_to_clear(client, model_name, problem)

            # 問題を複製してTurn 3を置き換え
            new_problem = dict(problem)
            new_problem['counter_evidence_condition'] = 'clear'
            new_problem['original_difficulty'] = problem.get('difficulty', 'hard')

            new_turns = []
            for turn in problem['turns']:
                if turn['turn_number'] == 3:
                    new_turn = dict(turn)
                    new_turn['original_content'] = turn['content']
                    new_turn['content'] = new_turn3_content
                    new_turns.append(new_turn)
                else:
                    new_turns.append(dict(turn))
            new_problem['turns'] = new_turns

            print(f" → T3書き換え完了")

        except Exception as e:
            print(f" (エラー: {e})")
            new_problem = dict(problem)
            new_problem['counter_evidence_condition'] = 'clear'
            new_problem['rewrite_error'] = str(e)

        results.append(new_problem)

        # 途中保存
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            for p in results:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')

        time.sleep(0.5)

    print(f"\n\n完了: {args.output} ({len(results)}問)")
    errors = [r for r in results if r.get('rewrite_error')]
    print(f"エラー: {len(errors)}問")

    print("\n次のステップ:")
    print(f"  python scripts/exp_c_clear_counter_evidence.py \\")
    print(f"    --benchmark {args.output} \\")
    print(f"    --output-dir results/phase5/clear_ce \\")
    print(f"    --step evaluate")


def step_evaluate(args):
    """Step 2: clear_ceベンチマークで全モデル評価"""
    import numpy as np

    print("\n" + "=" * 70)
    print("Experiment C Step 2: clear_ceベンチマーク評価")
    print("=" * 70)

    # ベンチマーク読み込み
    problems = []
    with open(args.benchmark, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    print(f"  問題数: {len(problems)}")
    print(f"  モデル: {args.models}")
    print(f"  出力ディレクトリ: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # phase5_benchmark_validationのBenchmarkEvaluatorを再利用
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
    from phase5_benchmark_validation import BenchmarkEvaluator, calculate_metrics

    metrics_by_model = {}
    for model in args.models:
        print(f"\n{'='*50}")
        print(f"評価モデル: {model}")
        print(f"{'='*50}")

        evaluator = BenchmarkEvaluator(model=model)
        output_path = os.path.join(args.output_dir, f"{model}_results.json")

        # 既存結果があれば読み込み
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                results = json.load(f)
            print(f"  既存結果を使用: {len(results)}問")
        else:
            results = evaluator.evaluate_batch(problems, output_path)

        metrics = calculate_metrics(results)
        metrics_by_model[model] = metrics

        if metrics.get('overall'):
            print(f"\n  BiasRate: {metrics['overall']['bias_rate']:.3f}")
            print(f"  PersistRate: {metrics['overall']['persistent_bias_rate']:.3f}")
            print(f"  Debias-Shift: {metrics['overall']['debias_shift']:.3f}")

    # 比較サマリー
    print("\n" + "=" * 70)
    print("【clear反証条件でのモデル比較】")
    print("=" * 70)
    print(f"{'モデル':<25} {'BiasRate':>12} {'PersistRate':>12}")
    print("-" * 50)
    for model, metrics in metrics_by_model.items():
        if metrics.get('overall'):
            br = metrics['overall']['bias_rate']
            pr = metrics['overall']['persistent_bias_rate']
            print(f"{model:<25} {br:>12.3f} {pr:>12.3f}")

    # 保存
    comparison_path = os.path.join(args.output_dir, "clear_ce_comparison.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump({
            'condition': 'clear_counter_evidence',
            'n_problems': len(problems),
            'models': args.models,
            'metrics_by_model': metrics_by_model,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n比較結果を保存: {comparison_path}")

    print("\n【次の分析】")
    print("results/phase5/cross_eval_claude/ (subtle条件) と比較して:")
    print("  - Geminiの PersistRate が下がる → subtle反証がPersistRateの主因")
    print("  - Geminiだけ変わらない → Gemini固有のバイアス持続性")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment C: clear反証条件でのPersistRate測定'
    )
    parser.add_argument('--benchmark', required=True,
                        help='ベンチマークファイル（JSONL）')
    parser.add_argument('--step', required=True,
                        choices=['rewrite', 'evaluate'],
                        help='実行ステップ: rewrite（T3書き換え）またはevaluate（評価）')
    parser.add_argument('--output',
                        default='results/phase5/hard_all_40_clear_ce.jsonl',
                        help='書き換え後のベンチマーク出力パス（rewriteステップ）')
    parser.add_argument('--output-dir',
                        default='results/phase5/clear_ce',
                        help='評価結果出力ディレクトリ（evaluateステップ）')
    parser.add_argument('--models', nargs='+',
                        default=['gpt-5', 'claude-4-sonnet', 'gemini-3.1-pro', 'llama-4-maverick'],
                        help='評価するモデル')
    parser.add_argument('--rewrite-model',
                        default='claude-4-sonnet',
                        help='T3書き換えに使用するモデル')

    args = parser.parse_args()

    if args.step == 'rewrite':
        step_rewrite(args)
    elif args.step == 'evaluate':
        step_evaluate(args)


if __name__ == '__main__':
    main()
