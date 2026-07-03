# 版权所有 © Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt 文件）
# 配套书籍《从零构建大模型》(Build a Large Language Model From Scratch) 源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 官方代码仓库：https://github.com/rasbt/LLMs-from-scratch

import tiktoken
import torch
import chainlit

from previous_chapters import (
    download_and_load_gpt2,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
)

# 自动选择设备：有NVIDIA显卡则使用CUDA，无则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    加载搭载OpenAI预训练权重的GPT-2模型
    实现逻辑与第5章代码基本一致
    若当前目录不存在模型文件，程序会自动下载权重
    """

    CHOOSE_MODEL = "gpt2-small (124M)"  # 可替换下方model_configs内任意一款模型

    BASE_CONFIG = {
        "vocab_size": 50257,     # 词表总规模
        "context_length": 1024,  # 上下文窗口长度
        "drop_rate": 0.0,        # Dropout丢弃概率
        "qkv_bias": True         # Query/Key/Value线性层是否启用偏置项
    }

    # 四种GPT2模型尺寸对应的超参配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # 提取模型参数字符串，如 "124M"
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    # 将选定模型的专属参数合并到基础配置中
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 下载并读取GPT2预训练权重与配置
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # 初始化GPT模型结构，并载入预训练权重
    gpt = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()  # 切换至推理模式，关闭Dropout等训练专用层

    # 加载GPT2官方分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    return tokenizer, gpt, BASE_CONFIG


# 提前加载分词器与模型，供下方对话接口调用
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit对话交互主逻辑函数
    """
    token_ids = generate(  # generate函数内部已封装torch.no_grad()，无需额外关闭梯度
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # message.content 为用户输入文本
        max_new_tokens=50,
        context_size=model_config["context_length"],
        top_k=1,
        temperature=0.0
    )

    # 将模型输出token序列转回可读文本
    text = token_ids_to_text(token_ids, tokenizer)

    # 将生成结果发送至前端对话界面
    await chainlit.Message(
        content=f"{text}",
    ).send()