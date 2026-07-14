# Must 2 tokenizer API micro-probe observations

1. add_special_tokens=False 的 token 数：10
2. add_special_tokens=True 的 token 数：10
3. 两组 IDs 是否相同：是
4. decode 是否与原始文本一致：是
5. 本 probe 证明了什么：证明：指定 tokenizer API 能把一条文本转换为 IDs/tokens，并能 decode。
6. 本 probe 没有证明什么：add_special_tokens=True时tokens可能会有特殊token