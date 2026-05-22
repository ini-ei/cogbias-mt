"""
Phase 5: Bootstrap信頼区間の計算

BiasRate / PersistRate / SycophancyRate に対する
95% Bootstrap CIを計算する。

少サンプル（n=19-20問）での統計的信頼性を正直に報告するため、
χ²検定（検出力不足）ではなくBootstrap CIを採用。

使用例:
  python scripts/phase5_bootstrap_ci.py \
    --results-dir results/phase5/full \
    --syco-results-dir results/phase5/sycophancy \
    --output results/phase5/bootstrap_ci.json
"""

import json
import os
import sys
import argparse
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

MODELS = ['gpt-5', 'claude-4-sonnet', 'gemini-3.1-pro', 'llama-4-maverick']
N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
ALPHA = 1 - CI_LEVEL


def bootstrap_ci(values: list, stat_fn, n_boot: int = N_BOOTSTRAP) -> dict:
    """
    Bootstrap法で統計量の信頼区間を計算する。

    Args:
        values: 0/1のリスト（各問題の結果）
        stat_fn: 集計関数（例: np.mean）
        n_boot: Bootstrap反復数

    Returns:
        {'mean': float, 'ci_lower': float, 'ci_upper': float, 'n': int}
    """
    values = np.array(values, dtype=float)
    n = len(values)
    if n == 0:
        return {'mean': None, 'ci_lower': None, 'ci_upper': None, 'n': 0}

    observed = stat_fn(values)

    # Bootstrap sampling
    rng = np.random.default_rng(42)  # 再現性のため固定シード
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats.append(stat_fn(sample))

    boot_stats = np.array(boot_stats)
    lower = np.percentile(boot_stats, 100 * ALPHA / 2)
    upper = np.percentile(boot_stats, 100 * (1 - ALPHA / 2))

    return {
        'mean': round(float(observed), 4),
        'ci_lower': round(float(lower), 4),
        'ci_upper': round(float(upper), 4),
        'n': n,
    }


def calc_bias_ci(results: list) -> dict:
    """bias_susceptibility結果からBiasRate/PersistRate/T4 BR(無条件)のCIを計算

    2026-05-08 拡張: T4 BR (unconditional) と effective_n_for_PR を追加。
    PR は T2 biased サンプル条件付きのため低 BR モデルで分母が小さい問題への対応。
    """
    valid = [r for r in results if 'error' not in r]
    biased = [r for r in valid if r.get('turn2_bias_detected')]

    bias_flags = [1 if r.get('turn2_bias_detected') else 0 for r in valid]
    persist_flags = [1 if r.get('persistent_bias') else 0 for r in biased]
    debias_flags = [1 if r.get('debiased') else 0 for r in biased]
    # T4 BR (unconditional): 全 valid サンプルを分母に取る
    t4_flags = [
        1 if (r.get('turn4_bias_detected') or r.get('persistent_bias')) else 0
        for r in valid
    ]

    return {
        'bias_rate': bootstrap_ci(bias_flags, np.mean),
        'persistent_bias_rate': bootstrap_ci(persist_flags, np.mean) if persist_flags else None,
        'debias_shift': bootstrap_ci(debias_flags, np.mean) if debias_flags else None,
        't4_bias_rate': bootstrap_ci(t4_flags, np.mean),  # 無条件 T4 BR
        'effective_n_for_PR': len(biased),  # PR の分母 (= n_biased@T2)
    }


def calc_syco_ci(results: list) -> dict:
    """sycophancy_test結果からSycophancyRateのCIを計算"""
    valid = [r for r in results if 'error' not in r]
    unbiased = [r for r in valid if r.get('initial_unbiased')]

    init_flags = [1 if r.get('initial_unbiased') else 0 for r in valid]
    syco_flags = [1 if r.get('sycophancy_detected') else 0 for r in unbiased]
    resist_flags = [1 if r.get('resistance_maintained') else 0 for r in unbiased]

    return {
        'initial_unbiased_rate': bootstrap_ci(init_flags, np.mean),
        'sycophancy_rate': bootstrap_ci(syco_flags, np.mean) if syco_flags else None,
        'resistance_rate': bootstrap_ci(resist_flags, np.mean) if resist_flags else None,
    }


def calc_bias_by_type_ci(results: list) -> dict:
    """バイアスタイプ別 BiasRate CI"""
    valid = [r for r in results if 'error' not in r]
    bias_types = set(r['bias_type'] for r in valid)

    out = {}
    for bt in sorted(bias_types):
        sub = [r for r in valid if r['bias_type'] == bt]
        flags = [1 if r.get('turn2_bias_detected') else 0 for r in sub]
        out[bt] = bootstrap_ci(flags, np.mean)
    return out


def calc_syco_by_type_ci(results: list) -> dict:
    """バイアスタイプ別 SycophancyRate CI"""
    valid = [r for r in results if 'error' not in r]
    bias_types = set(r['bias_type'] for r in valid)

    out = {}
    for bt in sorted(bias_types):
        sub_u = [r for r in valid if r['bias_type'] == bt and r.get('initial_unbiased')]
        flags = [1 if r.get('sycophancy_detected') else 0 for r in sub_u]
        out[bt] = bootstrap_ci(flags, np.mean) if flags else None
    return out


def format_ci(ci_dict: dict) -> str:
    """CI辞書を表示用文字列に変換"""
    if ci_dict is None or ci_dict.get('mean') is None:
        return "N/A"
    m = ci_dict['mean']
    lo = ci_dict['ci_lower']
    hi = ci_dict['ci_upper']
    n = ci_dict['n']
    return f"{m:.1%} [{lo:.1%}, {hi:.1%}] (n={n})"


def print_summary(all_ci: dict):
    """CIサマリーをコンソールに表示"""
    print("\n" + "=" * 90)
    print(f"Bootstrap信頼区間 (95% CI, n_boot={N_BOOTSTRAP})")
    print("=" * 90)

    print("\n【BiasRate（bias_susceptibility: 19問）】")
    print(f"{'モデル':<25} {'BiasRate 95%CI':<45} {'PersistRate 95%CI'}")
    print("-" * 90)
    for m in MODELS:
        if m not in all_ci or 'bias' not in all_ci[m]:
            continue
        b = all_ci[m]['bias']
        br = format_ci(b.get('bias_rate'))
        pr = format_ci(b.get('persistent_bias_rate'))
        print(f"{m:<25} {br:<45} {pr}")

    print("\n【T4 BR (unconditional) 95%CI と effective_n_for_PR】")
    print(f"{'モデル':<25} {'T4_BR 95%CI':<45} {'eff_n_for_PR'}")
    print("-" * 90)
    for m in MODELS:
        if m not in all_ci or 'bias' not in all_ci[m]:
            continue
        b = all_ci[m]['bias']
        t4 = format_ci(b.get('t4_bias_rate'))
        eff_n = b.get('effective_n_for_PR', 'N/A')
        print(f"{m:<25} {t4:<45} {eff_n}")
    print("  注: PR は分母 effective_n_for_PR (= n_biased@T2) に条件付くため、")
    print("     低 BR モデルでは PR の信頼区間が広い。T4 BR は全 n を分母とする無条件指標。")

    print("\n【SycophancyRate（sycophancy_test: 20問）】")
    print(f"{'モデル':<25} {'SycophancyRate 95%CI':<45} {'初期非バイアス率 95%CI'}")
    print("-" * 90)
    for m in MODELS:
        if m not in all_ci or 'syco' not in all_ci[m]:
            continue
        s = all_ci[m]['syco']
        sr = format_ci(s.get('sycophancy_rate'))
        ir = format_ci(s.get('initial_unbiased_rate'))
        print(f"{m:<25} {sr:<45} {ir}")

    print("\n【バイアスタイプ別 BiasRate 95%CI】")
    bias_types = ['anchoring', 'confirmation_bias', 'framing', 'representativeness']
    header = f"{'バイアスタイプ':<25}"
    for m in MODELS:
        short = m.split('-')[0]
        header += f" {short:<22}"
    print(header)
    print("-" * 120)
    for bt in bias_types:
        row = f"{bt:<25}"
        for m in MODELS:
            ci = all_ci.get(m, {}).get('bias_by_type', {}).get(bt)
            row += f" {format_ci(ci):<22}"
        print(row)

    print("\n【バイアスタイプ別 SycophancyRate 95%CI】")
    header = f"{'バイアスタイプ':<25}"
    for m in MODELS:
        short = m.split('-')[0]
        header += f" {short:<22}"
    print(header)
    print("-" * 120)
    for bt in bias_types:
        row = f"{bt:<25}"
        for m in MODELS:
            ci = all_ci.get(m, {}).get('syco_by_type', {}).get(bt)
            row += f" {format_ci(ci):<22}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Phase 5: Bootstrap CI計算')
    parser.add_argument('--results-dir', default='results/phase5/full',
                        help='bias_susceptibility結果ディレクトリ')
    parser.add_argument('--syco-results-dir', default='results/phase5/sycophancy',
                        help='sycophancy_test結果ディレクトリ')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--n-boot', type=int, default=N_BOOTSTRAP,
                        help='Bootstrap反復数（デフォルト: 10000）')
    parser.add_argument('--output', default='results/phase5/bootstrap_ci.json',
                        help='出力JSONファイル')

    args = parser.parse_args()

    print(f"\nBootstrap CI計算 (n_boot={args.n_boot}, CI={CI_LEVEL:.0%})")

    all_ci = {}

    for model in args.models:
        print(f"\n  {model}...", end=' ', flush=True)
        all_ci[model] = {}

        # bias_susceptibility
        bias_path = os.path.join(args.results_dir, f"{model}_results.json")
        if os.path.exists(bias_path):
            with open(bias_path) as f:
                bias_results = json.load(f)
            all_ci[model]['bias'] = calc_bias_ci(bias_results)
            all_ci[model]['bias_by_type'] = calc_bias_by_type_ci(bias_results)
            print("bias✓", end=' ', flush=True)

        # sycophancy
        syco_path = os.path.join(args.syco_results_dir, f"{model}_results.json")
        if os.path.exists(syco_path):
            with open(syco_path) as f:
                syco_results = json.load(f)
            all_ci[model]['syco'] = calc_syco_ci(syco_results)
            all_ci[model]['syco_by_type'] = calc_syco_by_type_ci(syco_results)
            print("syco✓", end=' ', flush=True)

        print()

    # 表示
    print_summary(all_ci)

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'n_bootstrap': args.n_boot,
            'ci_level': CI_LEVEL,
            'ci_by_model': all_ci,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {args.output}")


if __name__ == "__main__":
    main()
