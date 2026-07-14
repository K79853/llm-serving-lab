1. token是根据tokenizer策略得到的，token id是token在vocabulary的索引id，构成大模型的输入数据，vocabulary是包含所有token到token id映射的词表集合。
2. 2.1
3. 读到特殊tokenEOS或者生成的token达到最大
## 学后订正

2. `[1.2, -0.5, 2.1]` 的最大值是 `2.1`，但 greedy 选择的是其索引，因此 `argmax = 2`。