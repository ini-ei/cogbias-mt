"""
Phase 3: Multi-turn問題の品質チェック

人間レビュアーによる品質管理
"""

import json
import sys


def load_problems(input_path):
    """生成された問題を読み込む"""
    problems = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def create_quality_check_template(problems, output_path):
    """品質チェック用テンプレートを作成"""
    template = {
        "reviewer_name": "<レビュアー名を入力>",
        "date": "<レビュー日を入力>",
        "instructions": """
以下の各問題について品質をチェックしてください。

【チェック項目】
1. multiturn_structure_ok: Multi-turn構造が適切か（true/false）
   - 4ターンが論理的に繋がっているか
   - Turn 2でバイアスが誘発されるか
   - Turn 3で反証情報が提示されるか
   - Turn 4で再評価が求められるか

2. japanese_natural: 日本語として自然か（true/false）
   - 不自然な表現がないか
   - 日本語の規範に従っているか

3. culturally_appropriate: 文化的に適切か（true/false）
   - 日本社会の文脈に馴染むか
   - ステレオタイプを含まないか

4. bias_effective: バイアス誘発が効果的か（true/false）
   - Turn 2で適切にバイアスが誘発されるか
   - 評価基準が明確か

5. no_stereotypes: ステレオタイプを含まないか（true/false）
   - 性別、年齢、地域などへの偏見がないか

6. no_harm_risk: 実害リスクがないか（true/false）
   - 実際に使用しても危害を及ぼさないか

7. overall_quality: 総合評価（1-5）
   - 1: 不合格（大幅な修正が必要）
   - 2: 要改善（部分的な修正が必要）
   - 3: 合格（軽微な修正推奨）
   - 4: 良好
   - 5: 優秀

8. comments: コメント（任意）
   - 問題点や改善提案を記述
        """,
        "problems": []
    }

    for problem in problems:
        template["problems"].append({
            "id": problem.get("id", "unknown"),
            "bias_type": problem.get("bias_type", "unknown"),
            "domain": problem.get("domain", "unknown"),
            "content": problem,
            "quality_check": {
                "multiturn_structure_ok": None,
                "japanese_natural": None,
                "culturally_appropriate": None,
                "bias_effective": None,
                "no_stereotypes": None,
                "no_harm_risk": None,
                "overall_quality": None,
                "comments": ""
            }
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"品質チェック用テンプレートを保存: {output_path}")


def analyze_quality_checks(review_paths, output_path):
    """複数のレビュー結果を集約・分析"""
    reviews = []
    for path in review_paths:
        with open(path, 'r', encoding='utf-8') as f:
            reviews.append(json.load(f))

    # 問題IDごとに集約
    problem_scores = {}

    for review in reviews:
        for problem in review['problems']:
            prob_id = problem['id']
            if prob_id not in problem_scores:
                problem_scores[prob_id] = {
                    'overall_quality_scores': [],
                    'checks': {
                        'multiturn_structure_ok': [],
                        'japanese_natural': [],
                        'culturally_appropriate': [],
                        'bias_effective': [],
                        'no_stereotypes': [],
                        'no_harm_risk': []
                    },
                    'comments': []
                }

            qc = problem['quality_check']
            if qc['overall_quality'] is not None:
                problem_scores[prob_id]['overall_quality_scores'].append(qc['overall_quality'])

            for check_key in problem_scores[prob_id]['checks']:
                if qc.get(check_key) is not None:
                    problem_scores[prob_id]['checks'][check_key].append(qc[check_key])

            if qc.get('comments'):
                problem_scores[prob_id]['comments'].append({
                    'reviewer': review['reviewer_name'],
                    'comment': qc['comments']
                })

    # 統計を計算
    results = {
        'total_problems': len(problem_scores),
        'passed': 0,
        'needs_revision': 0,
        'failed': 0,
        'problem_details': []
    }

    for prob_id, scores in problem_scores.items():
        if scores['overall_quality_scores']:
            avg_score = sum(scores['overall_quality_scores']) / len(scores['overall_quality_scores'])

            if avg_score >= 3.0:
                results['passed'] += 1
                status = 'passed'
            elif avg_score >= 2.0:
                results['needs_revision'] += 1
                status = 'needs_revision'
            else:
                results['failed'] += 1
                status = 'failed'

            results['problem_details'].append({
                'id': prob_id,
                'average_score': avg_score,
                'status': status,
                'scores': scores['overall_quality_scores'],
                'checks': {
                    key: {
                        'pass_rate': sum(vals) / len(vals) if vals else 0,
                        'n_reviewers': len(vals)
                    }
                    for key, vals in scores['checks'].items()
                },
                'comments': scores['comments']
            })

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n品質チェック分析結果を保存: {output_path}")

    # サマリー表示
    print("\n" + "=" * 80)
    print("品質チェック結果サマリー")
    print("=" * 80)
    print(f"\n  総問題数: {results['total_problems']}")
    print(f"  合格: {results['passed']}問 ({results['passed']/results['total_problems']*100:.1f}%)")
    print(f"  要改善: {results['needs_revision']}問 ({results['needs_revision']/results['total_problems']*100:.1f}%)")
    print(f"  不合格: {results['failed']}問 ({results['failed']/results['total_problems']*100:.1f}%)")

    # 不合格・要改善の問題をリスト
    if results['failed'] > 0:
        print(f"\n【不合格の問題】")
        for detail in results['problem_details']:
            if detail['status'] == 'failed':
                print(f"  - {detail['id']}: 平均スコア {detail['average_score']:.2f}")

    if results['needs_revision'] > 0:
        print(f"\n【要改善の問題】")
        for detail in results['problem_details']:
            if detail['status'] == 'needs_revision':
                print(f"  - {detail['id']}: 平均スコア {detail['average_score']:.2f}")


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 3品質チェック')
    parser.add_argument('--mode', choices=['create_template', 'analyze'],
                        required=True, help='実行モード')
    parser.add_argument('--problems', help='生成された問題のパス（create_templateモード）')
    parser.add_argument('--reviews', nargs='+', help='レビュー結果のパス（analyzeモード）')
    parser.add_argument('--output', required=True, help='出力ファイルパス')

    args = parser.parse_args()

    if args.mode == 'create_template':
        if not args.problems:
            print("Error: --problems is required for create_template mode")
            sys.exit(1)

        print("\n品質チェック用テンプレートを作成")
        problems = load_problems(args.problems)
        print(f"  問題数: {len(problems)}")
        create_quality_check_template(problems, args.output)

    elif args.mode == 'analyze':
        if not args.reviews:
            print("Error: --reviews is required for analyze mode")
            sys.exit(1)

        print("\nレビュー結果を分析")
        analyze_quality_checks(args.reviews, args.output)


if __name__ == "__main__":
    main()
