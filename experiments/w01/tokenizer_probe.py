'''
Probe a tokenizer on fixed texts and write deterministic JSON.
Tokenizer assets only; no model weights are loaded.
'''
import argparse
from pathlib import Path

DEFALUT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROBE_CASES:tuple[tuple[str, str], ...] = (
    ("english_sentence", "LLM serving latency depends on token count."),
    ("chinese_sentence", "大模型推理延迟取决于输入和输出 token 数量。"),
    ("python_signature", "def stream_tokens(n: int) -> list[str]:"),
    ("mixed_unicode", "Hello，世界! 🚀"),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a tokenizer on fixed Week 1 texts and write JSON results.")
    parser.add_argument(
        "--model-id",
        default=DEFALUT_MODEL_ID,
        help="Hugging Face tokenizer/model repository ID"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="JSON output path"
    )
    return parser.parse_args()

def main() -> None:
    parse_args()

if __name__ == "__main__":
    main()
