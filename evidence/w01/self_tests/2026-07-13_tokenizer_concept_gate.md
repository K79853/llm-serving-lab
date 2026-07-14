1. token由分词器按照特定规则划分的，不一定等于单词和字符。
2. vocabulary是token与id的映射，id是token在词表的整数索引
3. 文本转token，再查表转ids
4. logits是模型的输出，并不是概率，softmax之后才是概率。
5. 因为softmax操作不改变最大值在输出中的索引，两者一样。
6. 如果输出EOS token，则停止，或者当前生成token数大于max_new_toke也会停止。
7. tokenizer.decode()将ids直接转为可阅读文本，后者不知道是啥。