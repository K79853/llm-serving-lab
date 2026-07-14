1. token由分词器按照特定规则划分的，不一定等于单词和字符。
2. vocabulary是token与id的映射，id是token在词表的整数索引
3. 文本转token，再查表转ids
4. logits是模型的输出，并不是概率，softmax之后才是概率。
5. 因为softmax操作不改变最大值在输出中的索引，两者一样。
6. 如果输出EOS token，则停止，或者当前生成token数大于max_new_toke也会停止。
7. tokenizer.decode()将ids直接转为可阅读文本，后者不知道是啥。
## 评审后订正

### 6. EOS 与 max_new_tokens

- 如果新生成的 token ID 等于 eos_token_id，则因 EOS 停止。
- 如果已经生成的新增 token 数达到 max_new_tokens，则因长度限制停止。
- 条件是“达到”，不是“大于”；否则会多生成一个 token。

### 7. 两种 decode

- tokenizer.decode(...)：把 token ID 序列转换回可读文本。
- Serving 的 decode phase：模型在 prefill 之后逐步执行推理，每一步预测并生成一个新的 token。
- 前者是 tokenizer 的格式转换；后者是模型推理阶段。