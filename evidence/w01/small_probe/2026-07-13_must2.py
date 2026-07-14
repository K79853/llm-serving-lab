from transformers import AutoTokenizer

text = "Give me a short introduction to large language model."
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. get ids without special tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
with_special = tokenizer(text, add_special_tokens=True)["input_ids"]

# 2. get tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids)
tokens = tokenizer.convert_ids_to_tokens(with_special)

# 3. get decode text
decode_text = tokenizer.decode(input_ids, skip_special_tokens=False)
decode_text = tokenizer.decode(with_special, skip_special_tokens=False)

# 4. print
print("without_special_ids:", input_ids)
print("with_special_ids:", with_special)

print("without_special_tokens:", tokenizer.convert_ids_to_tokens(input_ids))
print("with_special_tokens:", tokenizer.convert_ids_to_tokens(with_special))

print(
    "without_special_decoded:",
    tokenizer.decode(input_ids, skip_special_tokens=False),
)
print(
    "with_special_decoded:",
    tokenizer.decode(with_special, skip_special_tokens=False),
)

print("ids_equal:", input_ids == with_special)
print("token_count_without:", len(input_ids))
print("token_count_with:", len(with_special))

'''
output:
kai@LAPTOP-796L8S4G:~/code/codex_project/llm-serving-lab$ uv run evidence/w01/small_probe/2026-07-13_must2.py
[transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
without_special_ids: [35127, 752, 264, 2805, 16800, 311, 3460, 4128, 1614, 13]
with_special_ids: [35127, 752, 264, 2805, 16800, 311, 3460, 4128, 1614, 13]
without_special_tokens: ['Give', 'Ġme', 'Ġa', 'Ġshort', 'Ġintroduction', 'Ġto', 'Ġlarge', 'Ġlanguage', 'Ġmodel', '.']
with_special_tokens: ['Give', 'Ġme', 'Ġa', 'Ġshort', 'Ġintroduction', 'Ġto', 'Ġlarge', 'Ġlanguage', 'Ġmodel', '.']
without_special_decoded: Give me a short introduction to large language model.
with_special_decoded: Give me a short introduction to large language model.
ids_equal: True
token_count_without: 10
token_count_with: 10
'''