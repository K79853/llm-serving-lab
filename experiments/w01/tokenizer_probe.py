"""
Probe a tokenizer on fixed texts and write deterministic JSON.
Tokenizer assets only; no model weights are loaded.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROBE_CASES: tuple[tuple[str, str], ...] = (
    ("english_sentence", "LLM serving latency depends on token count."),
    ("chinese_sentence", "大模型推理延迟取决于输入和输出 token 数量。"),
    ("python_signature", "def stream_tokens(n: int) -> list[str]:"),
    ("mixed_unicode", "Hello，世界! 🚀"),
)
SPECIAL_TOKEN_SETTINGS: tuple[bool, bool] = (False, True)


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)


def probe_one(
    tokenizer, case_name: str, text: str, add_special_tokens: bool
) -> dict[str, object]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    token_ids = list(encoded["input_ids"])
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
    )
    return {
        "case_name": case_name,
        "text": text,
        "character_count": len(text),
        "add_special_tokens": add_special_tokens,
        "tokens": tokens,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "decoded_text": decoded_text,
    }


def build_samples(tokenizer) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for case_name, text in PROBE_CASES:
        for add_special_tokens in SPECIAL_TOKEN_SETTINGS:
            samples.append(
                probe_one(
                    tokenizer=tokenizer,
                    case_name=case_name,
                    text=text,
                    add_special_tokens=add_special_tokens,
                )
            )
    return samples


def build_payload(
    model_id: str,
    samples: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a tokenizer on fixed Week 1 texts and write JSON results."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face tokenizer/model repository ID",
    )
    parser.add_argument(
        "--output", type=Path, required=True, metavar="PATH", help="JSON output path"
    )
    return parser.parse_args()


def write_json(
    output: Path,
    payload: dict[str, object],
) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with output.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model_id)
    samples = build_samples(tokenizer)
    payload = build_payload(args.model_id, samples)
    write_json(args.output, payload)
    print(f"wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
