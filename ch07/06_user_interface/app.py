# 版权所有 (c) Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt 文件）
# 《从零构建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

from previous_chapters import (
    generate,
    GPTModel,
    text_to_token_ids,
    token_ids_to_text,
)

# 自动判断设备：有CUDA显卡则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    加载第7章微调完成权重的GPT-2模型代码
    需要先运行第7章代码，生成所需的 gpt2-medium355M-sft.pth 权重文件
    """

    # 355M参数量GPT模型超参配置
    GPT_CONFIG_355M = {
        "vocab_size": 50257,     # 词表总大小
        "context_length": 1024,  # 上下文窗口长度（原始值：1024）
        "emb_dim": 1024,         # 词嵌入维度
        "n_heads": 16,           # 注意力头数量
        "n_layers": 24,          # Transformer层数
        "drop_rate": 0.0,        # Dropout丢弃概率
        "qkv_bias": True         # Query-Key-Value线性层是否启用偏置项
    }

    # 加载GPT2原生分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 拼接微调模型权重文件路径
    model_path = Path("..") / "01_main-chapter-code" / "gpt2-medium355M-sft.pth"
    # 判断权重文件是否存在
    if not model_path.exists():
        print(
            f"未找到权重文件 {model_path}。请先运行第7章代码 "
            "（ch07.ipynb）生成 gpt2-medium355M-sft.pth 文件。"
        )
        # 终止程序运行
        sys.exit()

    # 加载模型权重断点，仅读取权重参数
    checkpoint = torch.load(model_path, weights_only=True)
    # 根据配置初始化GPT模型
    model = GPTModel(GPT_CONFIG_355M)
    # 将加载的权重参数赋值给模型
    model.load_state_dict(checkpoint)
    # 将模型迁移至指定设备（GPU/CPU）
    model.to(device)

    # 返回分词器、模型、模型超参配置
    return tokenizer, model, GPT_CONFIG_355M


def extract_response(response_text, input_text):
    # 截取模型输出部分，移除指令前缀与标记，去除首尾空白字符
    return response_text[len(input_text):].replace("### Response:", "").strip()


# 提前加载推理所需的分词器与模型，供下方Chainlit交互接口调用
tokenizer, model, model_config = get_model_and_tokenizer()


# Chainlit消息接收装饰器，监听前端输入
@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit 交互主逻辑函数
    """

    # 固定随机种子保证生成结果可复现
    torch.manual_seed(123)

    # 构造SFT微调标准指令模板
    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """

    # 调用文本生成函数（函数内部已封装 torch.no_grad() 禁用梯度计算）
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),  # 将前端输入文本转为token id并迁移至运算设备
        max_new_tokens=35,
        context_size=model_config["context_length"],
        eos_id=50256
    )

    # 将生成的token id序列转回可读文本
    text = token_ids_to_text(token_ids, tokenizer)
    # 提取模型纯回复内容
    response = extract_response(text, prompt)

    # 将模型回复发送至前端交互界面
    await chainlit.Message(
        content=f"{response}",  # 向前端返回模型生成的回答内容
    ).send()