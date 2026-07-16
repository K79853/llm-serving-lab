# Inference Basics Self-Test

- date: 2026-07-16
- mode: closed-book first attempt
- first_score: 2/5
- final_score: pending targeted closed-book retest
- reviewer: project lead

## Question 1
tokenizer、token、token ID 和 vocabulary 分别是什么，它们如何组成 text→ID 链条？

### First answer
tokenizer是应用一定的规则来讲原始文本转化为token，token是经过tokenizer划分得到的最小单位，token id是每个token对应的整数索引，vocabulary包含了token到id的一一映射，每个tokenizer有自己独特的分词规则和vocabulary，raw text--tokenizer-->token--vocabulary lookup-->token ID

### Review
- result: partial
- issue: 主链条正确，但“token 是最小单位”不准确。token 是特定 tokenizer 按其规则产生的离散单位，可能是词、子词、字符、标点、带空格片段或字节片段，不存在跨 tokenizer 固定的“最小单位”。token ID 也只在对应 vocabulary 中有意义。

### Corrected answer
tokenizer 是把原始文本按固定规则转换为模型输入表示的组件。它先把 text 转换成 token 序列，再通过自己的 vocabulary 把每个 token 映射为整数 token ID。token 是该 tokenizer 产生的离散单位，可能是词、子词、字符、标点、空格片段或字节片段；vocabulary 保存 token 与 ID 的映射；token ID 是 token 在该 vocabulary 中的整数标识。因此链条是：`raw text → tokenizer/tokenization → tokens → vocabulary lookup → token IDs`。

## Question 2
为什么不能只用字符数或请求数比较两个 LLM Serving workload？必须引用当前英文和中文 probe 数据。

### First answer
字符数不代表token数，probe中的中英文的字符数和token数都不同，请求数也太片面，长短请求的workload明显不同。

### Review
- result: partial
- issue: 判断方向正确，但没有按题目要求引用具体 probe 数字，也没有明确区分 input tokens 对 prefill 工作量和 output tokens 对 decode 迭代次数的影响。

### Corrected answer
字符数不能直接代表 token 数：当前 probe 中，英文输入有 43 个 Python 字符但只有 9 个 token，中文输入有 25 个字符却有 12 个 token。请求数也不足以描述 workload，因为两个请求可能具有完全不同的 input/output token 长度。input tokens 是 prefill 工作量的重要 proxy，output tokens 是标准自回归 decode 迭代次数的重要 proxy。因此容量规划至少要记录请求数、input tokens 和 output tokens，而不能只看字符串长度或请求数。

## Question 3
add_special_tokens=True 可能改变什么？为什么当前 False/True IDs 相同，benchmark 仍必须固定该设置？

### First answer
可能改变tokenizer.encode时，输出的token序列不一样。当前ids相同代表对于这个tokenizer和这段输入来说是相同的，发生改变后可能就会输出不同的ids，作为benchmark必须固定这个设置。

### Review
- result: correct
- issue: 无。答案已经说明该设置可能改变 token/ID 序列，当前相同只适用于当前 tokenizer 与输入，benchmark 必须固定变量以保证可比性。

### Corrected answer

## Question 4
写出 greedy autoregressive generation 的核心循环，并说明 EOS 与 max_new_tokens 的停止语义和优先级。

### First answer
1.根据当前sequence产生logits，
2.取argmax(logits)作为下一个新token，
3.将token放到sequence之后，
4.如果这个token是eos或者输出token数达到了max_new_tokens，则退出循环，否则，返回第一步。
eos停止表示生成了eos token，max_new_tokens停止表示一直生成了max_new_tokens个新token后停止，优先级eos>max_new_tokens。

### Review
- result: correct
- issue: 无。循环、两个停止语义和 EOS 优先级均正确。

### Corrected answer

## Question 5
两个系统都报告 50 output tokens/s，用户看到的文本速度和端到端体验为什么仍可能不同？至少写出三个原因。

### First answer
模型参数和架构，prefix caching，KV cache，精度和量化方式。

### Review
- result: incorrect
- issue: 这些因素可能影响系统如何达到吞吐，但题目已假定两个系统测得的 output rate 都是 50 tokens/s。答案没有解释在该速率相同的前提下，用户可见文本速度和端到端体验为什么仍不同。

### Corrected answer
即使两个系统都报告 50 output tokens/s，体验仍可能不同。第一，不同 tokenizer 的一个 token 可能对应不同数量的字符或词，因此相同 tokens/s 不等于相同的可见文本速度。第二，TTFT 可能不同：input tokens、prefill、排队和 prefix-cache 命中情况会影响用户多久看到第一个 token。第三，端到端延迟还包括排队、网络传输、序列化和流式刷新开销。第四，请求最终生成的 output token 总数可能不同，因此完成时间仍不同。比较体验时至少要同时观察 TTFT、输出速率、端到端延迟、输出 token 数和 tokenizer。

## Review summary

- 完全正确：Q3、Q4
- 部分正确：Q1、Q2
- 不正确：Q5
- 当前掌握状态：核心生成循环已掌握；token 定义、Serving workload 证据引用以及“相同 tokens/s 不等于相同体验”仍需一次针对性闭卷复测。
