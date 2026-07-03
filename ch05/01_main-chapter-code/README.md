# 第五章：在无标签数据上进行预训练

### 章节主要代码

- [ch05.ipynb](ch05.ipynb) 包含了章节中出现的所有代码。
- [previous_chapters.py](previous_chapters.py) 是一个 Python 模块，包含了上一章的 `MultiHeadAttention` 模块和 `GPTModel` 类，我们在 [ch05.ipynb](ch05.ipynb) 中导入该模块以进行 GPT 模型的预训练。
- [gpt_download.py](gpt_download.py) 包含了用于下载预训练 GPT 模型权重的工具函数。
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含了本章习题的解答。

### 可选代码

- [gpt_train.py](gpt_train.py) 是一个独立的 Python 脚本文件，包含了我们在 [ch05.ipynb](ch05.ipynb) 中实现的训练 GPT 模型的代码（你可以将其看作是本章代码的总结）。
- [gpt_generate.py](gpt_generate.py) 是一个独立的 Python 脚本文件，包含了我们在 [ch05.ipynb](ch05.ipynb) 中实现的加载并使用 OpenAI 提供的预训练模型权重的代码。