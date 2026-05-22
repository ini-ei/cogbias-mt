"""
EN/JA 比較フィギュアの matplotlib テンプレート。

サーバーから EN 結果が回収されたら自動的にデータを拾って描画する。
EN データが無い場合は JA のみで描画 (placeholder 状態)。

生成図:
1. fig_multilingual_br.png — 6 モデル × {EN, JA} BR 比較棒グラフ
2. fig_multilingual_t4_br.png — 6 モデル × {EN, JA} T4 BR 比較棒グラフ
3. fig_multilingual_8020.png — 80/20 方向非対称性の EN vs JA 比較
4. fig_multilingual_evaluator_hierarchy.png — 評価者厳格性ヒエラルキー EN vs JA
5. fig_multilingual_ranking.png — モデルランキング EN vs JA (Spearman 注釈付き)

使用:
    python scripts/plot_multilingual_figures.py
    # → docs/figures/multilingual/*.png に出力
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("results/phase5_v2")
EN_BASE = Path("results/phase5_v2_en")
OUT_DIR = Path("docs/figures/multilingual")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_MODELS = ["gpt-5", "claude-4-sonnet", "gemini-3.1-pro", "llama-4-maverick"]
NEW_MODELS = ["gpt-5.5", "claude-opus-4.7"]
OPEN_MODELS = ["qwen3.5-27b", "gemma-4-31b", "qwen3.5-122b"]
FRONTIER_MODELS = ORIGINAL_MODELS + NEW_MODELS  # back-compat
ALL_MODELS = FRONTIER_MODELS + OPEN_MODELS
EVALUATORS = ["gpt-5", "claude-4-sonnet", "gemini-3.1-pro"]

# Visual config
COLOR_JA = "#1f77b4"
COLOR_EN = "#ff7f0e"
PLACEHOLDER_ALPHA = 0.3  # transparent for placeholder bars


def load(p: Path) -> list:
    return json.load(open(p)) if p.exists() else []


def metrics(results: list) -> dict:
    valid = [r for r in results if "error" not in r]
    n = len(valid)
    if n == 0:
        return {"n": 0, "BR": None, "T4_BR": None, "PR": None, "n_biased": 0}
    biased = [r for r in valid if r.get("turn2_bias_detected")]
    persistent = [r for r in biased if r.get("turn4_bias_detected") or r.get("persistent_bias")]
    t4 = [r for r in valid if r.get("turn4_bias_detected") or r.get("persistent_bias")]
    return {
        "n": n,
        "BR": len(biased) / n,
        "T4_BR": len(t4) / n,
        "PR": (len(persistent) / len(biased)) if biased else None,
        "n_biased": len(biased),
    }


def ja_self(m):
    if m in OPEN_MODELS:
        return load(BASE / "full_n80_open" / f"{m}_results.json")
    return load(BASE / "full_n80" / f"{m}_results.json")


def en_self(m):
    if m in OPEN_MODELS:
        return load(EN_BASE / "full_n80_en_open" / f"{m}_results.json")
    return load(EN_BASE / "full_n80_en" / f"{m}_results.json")


def ja_cross(ev, m):
    is_new = m in NEW_MODELS
    is_open = m in OPEN_MODELS
    if ev == "gpt-5": base = "cross_eval_gpt5"
    elif ev == "claude-4-sonnet": base = "cross_eval_claude"
    elif ev == "gemini-3.1-pro": base = "cross_eval_gemini"
    else: return []
    if is_open: d = f"{base}_open"
    elif is_new: d = f"{base}_v3"
    else: d = base
    return load(BASE / d / f"{m}_results.json")


def en_cross(ev, m):
    is_open = m in OPEN_MODELS
    if ev == "gpt-5": base = "cross_eval_gpt5_en"
    elif ev == "claude-4-sonnet": base = "cross_eval_claude_en"
    elif ev == "gemini-3.1-pro": base = "cross_eval_gemini_en"
    else: return []
    d = f"{base}_open" if is_open else base
    return load(EN_BASE / d / f"{m}_results.json")


# ----------------------------------------------------------------
# Fig 1, 2: BR / T4 BR 比較棒グラフ
# ----------------------------------------------------------------

def plot_br_comparison(metric_key: str, title: str, filename: str):
    """Side-by-side bar chart: each model has JA and EN bar."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ja_vals = [metrics(ja_self(m)).get(metric_key) for m in ALL_MODELS]
    en_vals = [metrics(en_self(m)).get(metric_key) for m in ALL_MODELS]

    en_available = any(v is not None for v in en_vals)

    x = np.arange(len(ALL_MODELS))
    width = 0.4

    ja_plot = [v if v is not None else 0 for v in ja_vals]
    en_plot = [v if v is not None else 0 for v in en_vals]

    bars_ja = ax.bar(x - width/2, ja_plot, width, label="JA (n=80)", color=COLOR_JA)
    alpha = 1.0 if en_available else PLACEHOLDER_ALPHA
    bars_en = ax.bar(x + width/2, en_plot, width, label="EN (n=80)" if en_available else "EN (TBD)",
                     color=COLOR_EN, alpha=alpha,
                     hatch="" if en_available else "//")

    # Annotate values on bars
    for bar, v in zip(bars_ja, ja_vals):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", fontsize=8)
    if en_available:
        for bar, v in zip(bars_en, en_vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_MODELS, rotation=15, ha="right")
    ax.set_ylabel(title)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{title}: JA vs EN (n=80, all clear CE)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    if not en_available:
        ax.text(0.5, 0.95, "EN data pending — placeholder bars (hatched)",
                transform=ax.transAxes, ha="center", fontsize=10, color="gray", style="italic")

    plt.tight_layout()
    out = OUT_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")


# ----------------------------------------------------------------
# Fig 3: 80/20 方向非対称性 (3 評価者ペア × {EN, JA})
# ----------------------------------------------------------------

def asymmetry_for(loader, ev_a, ev_b):
    """Returns (n_disagree, ratio_a_only)"""
    a_map = {}
    b_map = {}
    for m in ALL_MODELS:
        for r in loader(ev_a, m):
            if "error" in r: continue
            a_map[(m, r.get("problem_id"))] = (r.get("turn4_bias_detected") or r.get("persistent_bias"))
        for r in loader(ev_b, m):
            if "error" in r: continue
            b_map[(m, r.get("problem_id"))] = (r.get("turn4_bias_detected") or r.get("persistent_bias"))
    common = set(a_map) & set(b_map)
    a_only = b_only = 0
    for k in common:
        av, bv = a_map[k], b_map[k]
        if av is None or bv is None: continue
        if av != bv:
            if av and not bv: a_only += 1
            elif bv and not av: b_only += 1
    total = a_only + b_only
    return total, (a_only / total if total else None)


def plot_8020_asymmetry():
    pairs = [
        ("gpt-5", "claude-4-sonnet"),
        ("gpt-5", "gemini-3.1-pro"),
        ("claude-4-sonnet", "gemini-3.1-pro"),
    ]
    pair_labels = [f"{a}\nvs\n{b}" for a, b in pairs]

    ja_ratios = []
    en_ratios = []
    ja_disagree = []
    en_disagree = []
    for a, b in pairs:
        n_ja, r_ja = asymmetry_for(ja_cross, a, b)
        n_en, r_en = asymmetry_for(en_cross, a, b)
        ja_ratios.append(r_ja or 0)
        en_ratios.append(r_en or 0)
        ja_disagree.append(n_ja)
        en_disagree.append(n_en)

    en_available = any(n > 0 for n in en_disagree)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(pairs))
    width = 0.4

    bars_ja = ax.bar(x - width/2, ja_ratios, width, label=f"JA", color=COLOR_JA)
    alpha = 1.0 if en_available else PLACEHOLDER_ALPHA
    bars_en = ax.bar(x + width/2, en_ratios, width,
                     label="EN" if en_available else "EN (TBD)",
                     color=COLOR_EN, alpha=alpha,
                     hatch="" if en_available else "//")

    # Reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="symmetric (0.5)")

    # Annotate disagreement counts
    for bar, n in zip(bars_ja, ja_disagree):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"n={n}", ha="center", fontsize=8)
    if en_available:
        for bar, n in zip(bars_en, en_disagree):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"n={n}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=9)
    ax.set_ylabel("Asymmetry ratio (a-side only / total disagreement)")
    ax.set_ylim(0, 1.05)
    ax.set_title("80/20 directional asymmetry (T4): JA vs EN")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    if not en_available:
        ax.text(0.5, 0.95, "EN data pending — placeholder bars (hatched)",
                transform=ax.transAxes, ha="center", fontsize=10, color="gray", style="italic")

    plt.tight_layout()
    out = OUT_DIR / "fig_multilingual_8020.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")


# ----------------------------------------------------------------
# Fig 4: 評価者厳格性ヒエラルキー (mean BR per evaluator)
# ----------------------------------------------------------------

def plot_evaluator_hierarchy():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(EVALUATORS))
    width = 0.4

    ja_means = []
    en_means = []
    for ev in EVALUATORS:
        ja_brs = [metrics(ja_cross(ev, m)).get("BR") for m in ALL_MODELS]
        en_brs = [metrics(en_cross(ev, m)).get("BR") for m in ALL_MODELS]
        ja_brs = [b for b in ja_brs if b is not None]
        en_brs = [b for b in en_brs if b is not None]
        ja_means.append(sum(ja_brs)/len(ja_brs) if ja_brs else 0)
        en_means.append(sum(en_brs)/len(en_brs) if en_brs else 0)

    en_available = any(v > 0 for v in en_means)

    bars_ja = ax.bar(x - width/2, ja_means, width, label="JA", color=COLOR_JA)
    alpha = 1.0 if en_available else PLACEHOLDER_ALPHA
    bars_en = ax.bar(x + width/2, en_means, width,
                     label="EN" if en_available else "EN (TBD)",
                     color=COLOR_EN, alpha=alpha,
                     hatch="" if en_available else "//")

    for bar, v in zip(bars_ja, ja_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=8)
    if en_available:
        for bar, v in zip(bars_en, en_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["GPT-5.4", "Claude 4.6", "Gemini 3.1"], fontsize=10)
    ax.set_ylabel("Mean BR (across 6 subject models)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Evaluator stringency hierarchy: JA vs EN")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    if not en_available:
        ax.text(0.5, 0.95, "EN data pending — placeholder bars (hatched)",
                transform=ax.transAxes, ha="center", fontsize=10, color="gray", style="italic")

    plt.tight_layout()
    out = OUT_DIR / "fig_multilingual_evaluator_hierarchy.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")


# ----------------------------------------------------------------
# Fig 5: ranking parallel coordinates
# ----------------------------------------------------------------

def plot_ranking_parallel():
    fig, ax = plt.subplots(figsize=(10, 6))
    ja_brs = [metrics(ja_self(m)).get("BR") for m in ALL_MODELS]
    en_brs = [metrics(en_self(m)).get("BR") for m in ALL_MODELS]

    en_available = any(v is not None for v in en_brs)

    # Sort by JA BR ascending
    sorted_idx = sorted(range(len(ALL_MODELS)), key=lambda i: ja_brs[i])
    sorted_models = [ALL_MODELS[i] for i in sorted_idx]
    sorted_ja = [ja_brs[i] for i in sorted_idx]
    sorted_en = [en_brs[i] for i in sorted_idx]

    y = np.arange(len(sorted_models))
    ax.scatter([0]*len(y), y, c=COLOR_JA, s=80, label="JA")
    if en_available:
        ax.scatter([1]*len(y), y, c=COLOR_EN, s=80, label="EN")

    # Connecting lines
    for i in range(len(y)):
        if sorted_en[i] is not None:
            # We use rank position as y-axis. So for fairness, also rank EN.
            pass

    # Ranking labels on left and right
    for i, m in enumerate(sorted_models):
        ja_v = sorted_ja[i]
        ax.text(-0.15, i, f"{m}\n{ja_v:.3f}" if ja_v is not None else m,
                ha="right", va="center", fontsize=9)

    if en_available:
        en_sorted_idx = sorted(range(len(ALL_MODELS)), key=lambda j: en_brs[j])
        en_sorted_models = [ALL_MODELS[j] for j in en_sorted_idx]
        en_sorted_brs = [en_brs[j] for j in en_sorted_idx]
        for i, m in enumerate(en_sorted_models):
            ax.text(1.15, i, f"{m}\n{en_sorted_brs[i]:.3f}",
                    ha="left", va="center", fontsize=9)
        # Connect by model identity
        for i, m in enumerate(sorted_models):
            j = en_sorted_models.index(m)
            ax.plot([0, 1], [i, j], color="gray", alpha=0.5, linewidth=0.8)

        # Compute Spearman
        ja_ranks = {m: sorted_models.index(m) for m in ALL_MODELS}
        en_ranks = {m: en_sorted_models.index(m) for m in ALL_MODELS}
        n = len(ALL_MODELS)
        d2 = sum((ja_ranks[m] - en_ranks[m])**2 for m in ALL_MODELS)
        rho = 1 - 6*d2 / (n*(n*n - 1))
        ax.set_title(f"Ranking preservation: JA vs EN BR (Spearman ρ = {rho:.3f})")
    else:
        ax.set_title("Ranking by JA BR (EN pending)")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, len(sorted_models) - 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["JA rank", "EN rank"])
    ax.set_yticks([])
    if not en_available:
        ax.text(0.5, 0.5, "EN data pending",
                transform=ax.transAxes, ha="center", fontsize=12, color="gray", style="italic")

    plt.tight_layout()
    out = OUT_DIR / "fig_multilingual_ranking.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")


if __name__ == "__main__":
    print(f"Generating figures to {OUT_DIR}/ ...")
    plot_br_comparison("BR", "BiasRate (T2)", "fig_multilingual_br.png")
    plot_br_comparison("T4_BR", "T4 BR (unconditional)", "fig_multilingual_t4_br.png")
    plot_8020_asymmetry()
    plot_evaluator_hierarchy()
    plot_ranking_parallel()
    print("Done.")
