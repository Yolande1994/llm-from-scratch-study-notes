# 第 7 章：指令微调（Finetuning to Follow Instructions）

本文件夹包含用于 **模型评估** 的工具代码。

&nbsp;

## 使用 OpenAI API 评估指令响应

- [llm-instruction-eval-openai.ipynb](llm-instruction-eval-openai.ipynb) 笔记本使用 **OpenAI 的 GPT-4** 来评估 **指令微调模型** 生成的响应。  
- 该笔记本适用于以下格式的 **JSON 文件**：

```python
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- The target given in the test set
    "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- Response by an LLM
    "model 2 response": "\nThe atomic number of helium is 3."    # <-- Response by a 2nd LLM
},
```

&nbsp;

## 本地使用 Ollama 评估指令响应

- [llm-instruction-eval-ollama.ipynb](llm-instruction-eval-ollama.ipynb) 笔记本提供了 **另一种评估方法**，使用 **Ollama 本地下载的 LLaMA 3 模型** 进行评估。  
