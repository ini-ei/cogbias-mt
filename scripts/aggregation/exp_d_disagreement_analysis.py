"""
Experiment D: LLM評価者間の不一致分析

GPT-5評価者（cross_eval_gpt5_40/）とClaude評価者（cross_eval_claude/）の
判定を問題単位で照合し、「いつ・どこで・なぜ評価者が食い違うか」を定量化する。

目的:
  軸1: LLM judgeの信頼性の根拠として、不一致の構造を明示する
  軸2: 「バイアス評価の難しさ自体」を研究対象化する
        - T2（バイアス誘導後）より T4（反証後）の不一致が大きい
          = 反証後の判断変化の評価は、バイアスの存在そのものの評価より難しい

出力:
  - results/phase5/disagreement_analysis/disagreement_cases.json
  - results/phase5/disagreement_analysis/disagreement_summary.json
  - コンソール: 不一致率テーブル（ターン別・バイアスタイプ別・ドメイン別・モデル別）

使用例:
  python scripts/exp_d_disagreement_analysis.py
  python scripts/exp_d_disagreement_analysis.py --output-dir results/phase5/disagreement_analysis
"""

import json
import os
import argparse
from collections import defaultdict


MODELS = ['gpt-5', 'claude-4-sonnet', 'gemini-3.1-pro', 'llama-4-maverick']


def load_cross_eval(results_dir: str) -> dict:
    """cross_evalディレクトリから全モデルの結果を読み込み、{problem_id: entry}に変換"""
    all_results = {}  # {model: {problem_id: entry}}
    for model in MODELS:
        path = os.path.join(results_dir, f'{model}_results.json')
        if not os.path.exists(path):
            print(f"  警告: {path} が見つかりません")
            continue
        with open(path) as f:
            entries = json.load(f)
        all_results[model] = {e['problem_id']: e for e in entries}
    return all_results


def classify_agreement(gpt5_val: bool, claude_val: bool) -> str:
    """2評価者の判定を分類"""
    if gpt5_val == claude_val:
        return 'agree_bias' if gpt5_val else 'agree_nobias'
    elif gpt5_val and not claude_val:
        return 'disagree_gpt5bias'   # GPT-5がバイアスあり、Claudeがなし
    else:
        return 'disagree_claudebias'  # Claudeがバイアスあり、GPT-5がなし


def is_disagree(label: str) -> bool:
    return label.startswith('disagree')


def main():
    parser = argparse.ArgumentParser(description='Exp D: LLM評価者間の不一致分析')
    parser.add_argument('--gpt5-dir', default='results/phase5/cross_eval_gpt5_40',
                        help='GPT-5クロス評価ディレクトリ')
    parser.add_argument('--claude-dir', default='results/phase5/cross_eval_claude',
                        help='Claudeクロス評価ディレクトリ')
    parser.add_argument('--output-dir', default='results/phase5/disagreement_analysis',
                        help='出力ディレクトリ')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("Experiment D: LLM評価者間の不一致分析")
    print("=" * 70)

    # データ読み込み
    print(f"\nGPT-5評価: {args.gpt5_dir}")
    gpt5_results = load_cross_eval(args.gpt5_dir)
    print(f"Claude評価: {args.claude_dir}")
    claude_results = load_cross_eval(args.claude_dir)

    # 問題単位で照合
    disagreement_cases = []
    all_cases = []

    for model in MODELS:
        if model not in gpt5_results or model not in claude_results:
            continue
        gpt5_by_pid = gpt5_results[model]
        claude_by_pid = claude_results[model]

        for pid in gpt5_by_pid:
            if pid not in claude_by_pid:
                continue
            g = gpt5_by_pid[pid]
            c = claude_by_pid[pid]

            t2_label = classify_agreement(g['turn2_bias_detected'], c['turn2_bias_detected'])
            t4_label = classify_agreement(g['turn4_bias_detected'], c['turn4_bias_detected'])

            case = {
                'problem_id': pid,
                'evaluated_model': model,
                'bias_type': g['bias_type'],
                'domain': g['domain'],
                'harm_type': g.get('harm_type', ''),
                'gpt5_t2': g['turn2_bias_detected'],
                'claude_t2': c['turn2_bias_detected'],
                'gpt5_t4': g['turn4_bias_detected'],
                'claude_t4': c['turn4_bias_detected'],
                't2_agreement': t2_label,
                't4_agreement': t4_label,
                't2_disagree': is_disagree(t2_label),
                't4_disagree': is_disagree(t4_label),
                # 不一致の方向性
                't2_disagree_direction': t2_label if is_disagree(t2_label) else None,
                't4_disagree_direction': t4_label if is_disagree(t4_label) else None,
                # T4の文脈: T2でバイアスありだったか
                't2_both_agree_bias': t2_label == 'agree_bias',
                # 応答の長さ（文字数）
                'turn2_response_len': len(g.get('turn2_response', '')),
                'turn4_response_len': len(g.get('turn4_response', '')),
            }
            all_cases.append(case)
            if is_disagree(t2_label) or is_disagree(t4_label):
                disagreement_cases.append(case)

    n_total = len(all_cases)
    n_t2_disagree = sum(1 for c in all_cases if c['t2_disagree'])
    n_t4_disagree = sum(1 for c in all_cases if c['t4_disagree'])

    print(f"\n総ケース数: {n_total}（{len(MODELS)}モデル × 40問）")
    print(f"T2不一致: {n_t2_disagree}/{n_total} = {100*n_t2_disagree/n_total:.1f}%")
    print(f"T4不一致: {n_t4_disagree}/{n_total} = {100*n_t4_disagree/n_total:.1f}%")

    # === 1. ターン別・モデル別 不一致率 ===
    print("\n" + "=" * 70)
    print("【1】評価対象モデル別 不一致率")
    print("=" * 70)
    print(f"{'モデル':<25} {'T2不一致':>10} {'T4不一致':>10} {'両方不一致':>12}")
    print("-" * 60)
    model_summary = {}
    for model in MODELS:
        cases = [c for c in all_cases if c['evaluated_model'] == model]
        if not cases:
            continue
        n = len(cases)
        t2d = sum(1 for c in cases if c['t2_disagree'])
        t4d = sum(1 for c in cases if c['t4_disagree'])
        both = sum(1 for c in cases if c['t2_disagree'] and c['t4_disagree'])
        print(f"{model:<25} {t2d}/{n}={100*t2d/n:.0f}%{'':<2} {t4d}/{n}={100*t4d/n:.0f}%{'':<2} {both}/{n}={100*both/n:.0f}%")
        model_summary[model] = {'n': n, 't2_disagree': t2d, 't4_disagree': t4d, 'both_disagree': both}

    # === 2. バイアスタイプ別 不一致率 ===
    print("\n" + "=" * 70)
    print("【2】バイアスタイプ別 不一致率")
    print("=" * 70)
    print(f"{'バイアスタイプ':<30} {'T2不一致':>10} {'T4不一致':>10}")
    print("-" * 55)
    bias_types = sorted(set(c['bias_type'] for c in all_cases))
    bias_summary = {}
    for bt in bias_types:
        cases = [c for c in all_cases if c['bias_type'] == bt]
        n = len(cases)
        t2d = sum(1 for c in cases if c['t2_disagree'])
        t4d = sum(1 for c in cases if c['t4_disagree'])
        print(f"{bt:<30} {t2d}/{n}={100*t2d/n:.0f}%{'':<3} {t4d}/{n}={100*t4d/n:.0f}%")
        bias_summary[bt] = {'n': n, 't2_disagree': t2d, 't4_disagree': t4d}

    # === 3. ドメイン別 不一致率 ===
    print("\n" + "=" * 70)
    print("【3】ドメイン別 不一致率")
    print("=" * 70)
    print(f"{'ドメイン':<20} {'T2不一致':>10} {'T4不一致':>10}")
    print("-" * 45)
    domains = sorted(set(c['domain'] for c in all_cases))
    domain_summary = {}
    for dom in domains:
        cases = [c for c in all_cases if c['domain'] == dom]
        n = len(cases)
        t2d = sum(1 for c in cases if c['t2_disagree'])
        t4d = sum(1 for c in cases if c['t4_disagree'])
        print(f"{dom:<20} {t2d}/{n}={100*t2d/n:.0f}%{'':<3} {t4d}/{n}={100*t4d/n:.0f}%")
        domain_summary[dom] = {'n': n, 't2_disagree': t2d, 't4_disagree': t4d}

    # === 4. 不一致の方向性 ===
    print("\n" + "=" * 70)
    print("【4】不一致の方向性（どちらが「バイアスあり」と言いやすいか）")
    print("=" * 70)
    t2_gpt5_strict = sum(1 for c in all_cases if c['t2_disagree_direction'] == 'disagree_gpt5bias')
    t2_claude_strict = sum(1 for c in all_cases if c['t2_disagree_direction'] == 'disagree_claudebias')
    t4_gpt5_strict = sum(1 for c in all_cases if c['t4_disagree_direction'] == 'disagree_gpt5bias')
    t4_claude_strict = sum(1 for c in all_cases if c['t4_disagree_direction'] == 'disagree_claudebias')
    print(f"T2: GPT-5のみ「バイアスあり」= {t2_gpt5_strict}件, Claudeのみ「バイアスあり」= {t2_claude_strict}件")
    print(f"T4: GPT-5のみ「バイアスあり」= {t4_gpt5_strict}件, Claudeのみ「バイアスあり」= {t4_claude_strict}件")

    # === 5. T4不一致のT2コンテキスト ===
    print("\n" + "=" * 70)
    print("【5】T4不一致ケースのT2状況（反証後の判断がどんな文脈で食い違うか）")
    print("=" * 70)
    t4_disagree_cases = [c for c in all_cases if c['t4_disagree']]
    if t4_disagree_cases:
        t2_agree_bias = sum(1 for c in t4_disagree_cases if c['t2_agreement'] == 'agree_bias')
        t2_agree_nobias = sum(1 for c in t4_disagree_cases if c['t2_agreement'] == 'agree_nobias')
        t2_also_disagree = sum(1 for c in t4_disagree_cases if c['t2_disagree'])
        n = len(t4_disagree_cases)
        print(f"T4不一致 {n}件のうち:")
        print(f"  T2は両者一致「バイアスあり」: {t2_agree_bias}件 ({100*t2_agree_bias/n:.0f}%)")
        print(f"  T2は両者一致「バイアスなし」: {t2_agree_nobias}件 ({100*t2_agree_nobias/n:.0f}%)")
        print(f"  T2も不一致だった: {t2_also_disagree}件 ({100*t2_also_disagree/n:.0f}%)")

    # === 6. 応答長と不一致の関係 ===
    print("\n" + "=" * 70)
    print("【6】応答長と不一致の関係")
    print("=" * 70)
    t2_agree_len = [c['turn2_response_len'] for c in all_cases if not c['t2_disagree']]
    t2_disagree_len = [c['turn2_response_len'] for c in all_cases if c['t2_disagree']]
    t4_agree_len = [c['turn4_response_len'] for c in all_cases if not c['t4_disagree']]
    t4_disagree_len = [c['turn4_response_len'] for c in all_cases if c['t4_disagree']]
    if t2_agree_len and t2_disagree_len:
        print(f"T2 一致ケースの平均応答長: {sum(t2_agree_len)/len(t2_agree_len):.0f}文字")
        print(f"T2 不一致ケースの平均応答長: {sum(t2_disagree_len)/len(t2_disagree_len):.0f}文字")
    if t4_agree_len and t4_disagree_len:
        print(f"T4 一致ケースの平均応答長: {sum(t4_agree_len)/len(t4_agree_len):.0f}文字")
        print(f"T4 不一致ケースの平均応答長: {sum(t4_disagree_len)/len(t4_disagree_len):.0f}文字")

    # === 保存 ===
    summary = {
        'n_total': n_total,
        'n_t2_disagree': n_t2_disagree,
        'n_t4_disagree': n_t4_disagree,
        't2_disagree_rate': n_t2_disagree / n_total if n_total else 0,
        't4_disagree_rate': n_t4_disagree / n_total if n_total else 0,
        'by_model': model_summary,
        'by_bias_type': bias_summary,
        'by_domain': domain_summary,
        'direction': {
            't2_gpt5_strict': t2_gpt5_strict,
            't2_claude_strict': t2_claude_strict,
            't4_gpt5_strict': t4_gpt5_strict,
            't4_claude_strict': t4_claude_strict,
        }
    }

    summary_path = os.path.join(args.output_dir, 'disagreement_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nサマリー保存: {summary_path}")

    cases_path = os.path.join(args.output_dir, 'disagreement_cases.json')
    with open(cases_path, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print(f"全ケース保存: {cases_path}")

    # 不一致ケースのみ別途保存
    disagree_only_path = os.path.join(args.output_dir, 'disagree_only.json')
    with open(disagree_only_path, 'w', encoding='utf-8') as f:
        json.dump(disagreement_cases, f, ensure_ascii=False, indent=2)
    print(f"不一致ケースのみ保存: {disagree_only_path} ({len(disagreement_cases)}件)")

    print("\n【次のステップ】")
    print("  - disagree_only.json の応答テキストを読んで、不一致の質的分析")
    print("  - §11.1.3のκ値と合わせてRESEARCH_NOTE.mdに結果を追加")


if __name__ == '__main__':
    main()
