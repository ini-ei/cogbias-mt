"""
Phase 6 multilingual: English version of exp_c_clear_counter_evidence.py.

Rewrites Turn 3 of English-language benchmark items from subtle/ambiguous
counter-evidence to explicit "clear" counter-evidence (matches the JA pipeline).

Usage:
    python scripts/exp_c_clear_counter_evidence_en.py \\
        --benchmark results/phase5_v2_en/hard_all_40_en_filtered.jsonl \\
        --output results/phase5_v2_en/hard_all_40_en_clear_ce.jsonl \\
        --step rewrite \\
        --rewrite-model gpt-5
"""

import json
import os
import sys
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.openrouter_client import create_openrouter_client, get_model_name


CLEAR_CE_REWRITE_PROMPT = """\
Rewrite Turn 3 (counter-evidence) of the following cognitive-bias benchmark item into a more EXPLICIT, DIRECTLY CONTRADICTING form.

[Problem]
Bias type: {bias_type}
Domain: {domain}

[Turn 2 (bias-inducing information)]
{turn2_content}

[Original Turn 3 (subtle / ambiguous counter-evidence — to be rewritten)]
{turn3_content}

[Rewrite policy]
The original Turn 3 is indirect / suggestive. Rewrite it to be directly contradictory:
- Make the counter-evidence DIRECTLY CONTRADICT the information presented in Turn 2.
- Use formulations like "Actually, the data shows…" or "The diagnostic results disprove…",
  so that the disconfirmation is explicit with little interpretive flexibility.
- Keep the information volume similar to the original Turn 3 (do not radically change length).
- Preserve scenario naturalness (use realistic information sources, data formats, study citations).
- Make the new content explicitly negate / overturn the bias-inducing claims of Turn 2.

[Important]
- DO NOT modify Turn 1, Turn 2, or Turn 4.
- ONLY rewrite the content field of Turn 3.
- Tailor the direct contradiction to the bias type ({bias_type}).
- Output natural, idiomatic English.

Output ONLY the rewritten Turn 3 content (no JSON, no explanation, plain text only).
"""


def rewrite_turn3_to_clear(client, model: str, problem: dict) -> str:
    """Rewrite Turn 3 to a 'clear' (explicit) counter-evidence style."""
    turns = {t["turn_number"]: t for t in problem["turns"]}

    turn2_content = turns.get(2, {}).get("content", "")
    turn2_q = turns.get(2, {}).get("question", "")
    if turn2_q:
        turn2_content += f"\nQuestion: {turn2_q}"
    turn3_content = turns.get(3, {}).get("content", "")

    prompt = CLEAR_CE_REWRITE_PROMPT.format(
        bias_type=problem.get("bias_type", ""),
        domain=problem.get("domain", ""),
        turn2_content=turn2_content,
        turn3_content=turn3_content,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in cognitive-bias research. "
                    "Rewrite the Turn 3 counter-evidence as instructed. "
                    "Output plain text only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()


def step_rewrite(args):
    print("\n" + "=" * 70)
    print("Experiment C Step 1 (EN): Rewrite Turn 3 to clear counter-evidence")
    print("=" * 70)
    print(f"  Input:  {args.benchmark}")
    print(f"  Output: {args.output}")

    problems = []
    with open(args.benchmark, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    print(f"  Items:  {len(problems)}")

    client = create_openrouter_client(args.rewrite_model)
    model_name = get_model_name(args.rewrite_model)

    existing = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    existing[p["id"]] = p
        print(f"  Resume: {len(existing)} existing items")

    results = []
    for i, problem in enumerate(problems):
        pid = problem["id"]
        print(f"\n[{i+1}/{len(problems)}] {pid}", end="", flush=True)

        if pid in existing:
            print(" (skip: existing)")
            results.append(existing[pid])
            continue

        try:
            new_turn3_content = rewrite_turn3_to_clear(client, model_name, problem)

            new_problem = dict(problem)
            new_problem["counter_evidence_condition"] = "clear"
            new_problem["original_difficulty"] = problem.get("difficulty", "hard")

            new_turns = []
            for turn in problem["turns"]:
                if turn["turn_number"] == 3:
                    new_turn = dict(turn)
                    new_turn["original_content"] = turn["content"]
                    new_turn["content"] = new_turn3_content
                    new_turns.append(new_turn)
                else:
                    new_turns.append(dict(turn))
            new_problem["turns"] = new_turns

            print(" -> rewritten")

        except Exception as e:
            print(f" (error: {e})")
            new_problem = dict(problem)
            new_problem["counter_evidence_condition"] = "clear"
            new_problem["rewrite_error"] = str(e)

        results.append(new_problem)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for p in results:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        time.sleep(0.5)

    print(f"\n\nDone: {args.output} ({len(results)} items)")
    errors = [r for r in results if r.get("rewrite_error")]
    print(f"Errors: {len(errors)}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 multilingual: EN clear-CE rewrite (Exp C)")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", default="rewrite", choices=["rewrite"])
    parser.add_argument("--rewrite-model", default="gpt-5")
    args = parser.parse_args()

    if args.step == "rewrite":
        step_rewrite(args)


if __name__ == "__main__":
    main()
