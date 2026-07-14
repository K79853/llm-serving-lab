# Must 2 tokenizer API micro-probe observations

1. add_special_tokens=False 的 token 数：10
2. add_special_tokens=True 的 token 数：10
3. 两组 IDs 是否相同：是
4. decode 是否与原始文本一致：是
5. 本 probe 证明了什么：证明：指定 tokenizer API 能把一条文本转换为 IDs/tokens，并能 decode。
6. 本 probe 没有证明什么：add_special_tokens=True时tokens可能会有特殊token
修改：
6. 本 probe 没有证明什么：

本 probe 只说明在当前 Qwen tokenizer、当前普通文本和当前调用方式下，
add_special_tokens=False/True 得到了相同 IDs。

它不能证明：
- 其他 tokenizer 的 add_special_tokens=True 也不会改变序列；
- chat template 不会插入控制 token；
- 真实模型的 logits 或生成结果；
- prefill/decode 性能或推理速度。
