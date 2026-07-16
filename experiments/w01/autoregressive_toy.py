"""
这是一个固定 transition 的 deterministic greedy autoregressive toy
它记录逐步 trace 并输出 JSON
它不加载真实模型、Transformers 或 PyTorch
"""

import argparse
import json
from pathlib import Path


VOCABULARY = (
    "<bos>",
    "I",
    " like",
    " cats",
    "<eos>",
)
EOS_TOKEN_ID = 4
MAX_NEW_TOKENS = 5
PROMPT_IDS = (0, 1)
EOS_TRANSITIONS = {
    1: (-2.0, 0.1, 3.2, 1.0, -1.0),
    2: (-2.0, 0.0, 0.2, 2.8, -0.5),
    3: (-2.0, -1.0, 0.0, 0.1, 4.0),
}
LENGTH_TRANSITIONS = {
    1: (-2.0, 0.1, 3.2, 1.0, -1.0),
    2: (-2.0, 0.0, 0.2, 2.8, -0.5),
    3: (-2.0, 0.0, 3.0, 0.2, -1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic greedy autoregressive toy and write JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="JSON output path",
    )
    return parser.parse_args()


def argmax(values: tuple[float, ...]) -> int:
    max_idx, _ = max(enumerate(values), key=lambda x: x[1])
    return max_idx


def transition(
    sequence: list[int],
    table: dict[int, tuple[float, ...]],
) -> tuple[float, ...]:
    current_id = sequence[-1]
    logits = table[current_id]
    if len(logits) != len(VOCABULARY):
        raise ValueError(f"logits length: {len(logits)}")
    return logits


def greedy_generate(
    name: str,
    prompt_ids: tuple[int, ...],
    table: dict[int, tuple[float, ...]],
    eos_token_id: int,
    max_new_tokens: int,
) -> dict[str, object]:
    sequence = list(prompt_ids)
    generated_ids = list()
    steps = list()
    scenario_stop_reason = None
    for step_number in range(1, max_new_tokens + 1):
        ids_before = sequence.copy()
        logits = transition(sequence, table)
        next_id = argmax(logits)
        if next_id < 0 or next_id >= len(VOCABULARY):
            raise ValueError("logits index wrong")
        sequence.append(next_id)
        generated_ids.append(next_id)
        ids_after = sequence.copy()

        step_stop_reason = None
        if next_id == eos_token_id:
            step_stop_reason = "eos"
        elif len(generated_ids) == max_new_tokens:
            step_stop_reason = "max_new_tokens"
        stopped = step_stop_reason is not None
        step_record = {
            "step ": step_number,
            "ids_before": ids_before,
            "logits": logits,
            "argmax_id": next_id,
            "ids_after": ids_after,
            "stopped": stopped,
            "stop_reason": step_stop_reason,
        }
        steps.append(step_record)
        if stopped:
            scenario_stop_reason = step_stop_reason
            break
    return {
        "name": name,
        "prompt_ids": list(prompt_ids),
        "steps": steps,
        "generated_ids": generated_ids,
        "final_ids": sequence,
        "stop_reason": scenario_stop_reason,
    }


def build_payload() -> dict[str, object]:
    vocabulary = [{"id": idx, "token": token} for idx, token in enumerate(VOCABULARY)]
    scenario_eos_path = greedy_generate(
        "eos_path",
        PROMPT_IDS,
        EOS_TRANSITIONS,
        EOS_TOKEN_ID,
        MAX_NEW_TOKENS,
    )
    scenario_length_path = greedy_generate(
        "length_path",
        PROMPT_IDS,
        LENGTH_TRANSITIONS,
        EOS_TOKEN_ID,
        MAX_NEW_TOKENS,
    )
    return {
        "vocabulary": vocabulary,
        "eos_token_id": EOS_TOKEN_ID,
        "max_new_tokens": MAX_NEW_TOKENS,
        "scenarios": [
            scenario_eos_path,
            scenario_length_path,
        ],
    }


def write_json(output: Path, payload: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> None:
    args = parse_args()
    payload = build_payload()
    write_json(args.output, payload)
    scenarios = payload["scenarios"]
    print(f"scenario number: {len(scenarios)}, output path: {args.output}")


if __name__ == "__main__":
    main()
