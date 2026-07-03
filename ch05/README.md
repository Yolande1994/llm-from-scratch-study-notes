# 第五章：在无标签数据上进行预训练

&nbsp;
## 章节主要代码

- [01_main-chapter-code](01_main-chapter-code) 包含了章节的主要代码。

&nbsp;
## 额外材料

- [02_alternative_weight_loading](02_alternative_weight_loading) 包含了从替代来源加载 GPT 模型权重的代码，以防 OpenAI 提供的模型权重不可用。
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) 包含了使用 Project Gutenberg 的全部书籍语料库上对大语言模型进行更长时间预训练的代码。
- [04_learning_rate_schedulers](04_learning_rate_schedulers) 包含了实现更复杂训练功能的代码，包括学习率调度器和梯度裁剪。
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) 包含了一个可选的超参数调优脚本。
- [06_user_interface](06_user_interface) 实现了一个交互式用户界面，用于与预训练的大语言模型进行互动。
- [07_gpt_to_llama](07_gpt_to_llama) 包含了将 GPT 架构实现转换为 Llama 3.2 的逐步指南，并加载来自 Meta AI 的预训练权重。
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) 包含了一个额外的笔记本，展示如何通过 PyTorch 的 `load_state_dict` 方法更高效地加载模型权重。