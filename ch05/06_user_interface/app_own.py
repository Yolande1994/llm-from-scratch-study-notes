# 版权所有 © Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt 文件）
# 配套书籍《从零构建大模型》(Build a Large Language Model From Scratch) 源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 官方代码仓库：https://github.com/rasbt/LLMs-from-scratch

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

# 自动选择运行设备：有CUDA显卡则使用GPU，无则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    加载第5章训练产出、带有本地预训练权重的GPT-2模型
    必须先运行第5章代码生成 model.pth 权重文件，否则本脚本无法正常执行
    """

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # 词表总容量
        "context_length": 256,  # 缩短后的上下文窗口长度（原版为1024）
        "emb_dim": 768,         # 词嵌入向量维度
        "n_heads": 12,          # 多头注意力头数量
        "n_layers": 12,         # Transformer 层数
        "drop_rate": 0.1,       # Dropout 随机失活概率
        "qkv_bias": False       # QKV线性层是否开启偏置项
    }

    # 加载GPT2官方分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 拼接本地模型权重文件路径
    model_path = Path("..") / "01_main-chapter-code" / "model.pth"
    # 校验权重文件是否存在
    if not model_path.exists():
        print(f"未找到权重文件 {model_path}，请先运行第5章代码 ch05.ipynb 生成 model.pth 文件。")
        sys.exit()

    # 加载模型权重，仅读取权重数据
    checkpoint = torch.load(model_path, weights_only=True)
    # 初始化124M参数量GPT模型结构
    model = GPTModel(GPT_CONFIG_124M)
    # 将权重载入模型
    model.load_state_dict(checkpoint)
    # 将模型迁移至指定设备（GPU/CPU）
    model.to(device)

    return tokenizer, model, GPT_CONFIG_124M


# 预加载分词器与模型，供下方对话接口调用
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit 对话交互主处理函数
    """
    token_ids = generate(  # generate 函数内部已自带 torch.no_grad()，无需额外关闭梯度计算
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # message.content 为用户输入的对话文本
        max_new_tokens=50,
        context_size=model_config["context_length"],
        top_k=1,
        temperature=0.0
    )

    # 将模型输出的token ID序列还原为可读文本
    text = token_ids_to_text(token_ids, tokenizer)

    # 将模型生成的回复推送至前端对话界面展示
    await chainlit.Message(
        content=f"{text}",
    ).send()