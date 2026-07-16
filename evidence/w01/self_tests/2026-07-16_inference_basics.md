# Inference Basics Self-Test

- date: 2026-07-16
- mode: closed-book first attempt
- first_score: X/5
- final_score: X/5

## Question 1
tokenizer、token、token ID 和 vocabulary 分别是什么，它们如何组成 text→ID 链条？
### First answer
tokenizer是应用一定的规则来讲原始文本转化为token，token是经过tokenizer划分得到的最小单位，token id是每个token对应的整数索引，vocabulary包含了token到id的一一映射，每个tokenizer有自己独特的分词规则和vocabulary，raw text--tokenizer-->token--vocabulary lookup-->token ID
### Review
- result: correct / partial / incorrect
- issue: ...
### Corrected answer
...

## Question 2
为什么不能只用字符数或请求数比较两个 LLM Serving workload？必须引用当前英文和中文 probe 数据。
### First answer
字符数不代表token数，probe中的中英文的字符数和token数都不同，请求数也太片面，长短请求的workload明显不同。
### Review
- result: correct / partial / incorrect
- issue: ...
### Corrected answer
...

## Question 3
add_special_tokens=True 可能改变什么？为什么当前 False/True IDs 相同，benchmark 仍必须固定该设置？
### First answer
可能改变tokenizer.encode时，输出的token序列不一样。当前ids相同代表对于这个tokenizer和这段输入来说是相同的，发生改变后可能就会输出不同的ids，作为benchmark必须固定这个设置。
### Review
- result: correct / partial / incorrect
- issue: ...
### Corrected answer
...

## Question 4
写出 greedy autoregressive generation 的核心循环，并说明 EOS 与 max_new_tokens 的停止语义和优先级。
### First answer
1.根据当前sequence产生logits，
2.取argmax(logits)作为下一个新token，
3.将token放到sequence之后，
4.如果这个token是eos或者输出token数达到了max_new_tokens，则退出循环，否则，返回第一步。
eos停止表示生成了eos token，max_new_tokens停止表示一直生成了max_new_tokens个新token后停止，优先级eos>max_new_tokens。
### Review
- result: correct / partial / incorrect
- issue: ...
### Corrected answer
...

## Question 5
两个系统都报告 50 output tokens/s，用户看到的文本速度和端到端体验为什么仍可能不同？至少写出三个原因。
### First answer
模型参数和架构，prefix caching，KV cache，精度和量化方式。
### Review
- result: correct / partial / incorrect
- issue: ...
### Corrected answer
...
