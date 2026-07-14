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
