# 版权所有 (c) Sebastian Raschka，遵循 Apache 2.0 开源协议（详见 LICENSE.txt）。
# 《从零搭建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

from previous_chapters import (
    classify_review,
    GPTModel
)

# 自动检测设备：优先使用CUDA显卡，无显卡则切换CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    加载第6章微调完成的GPT-2模型
    运行本代码前必须先执行第6章代码，生成所需的模型权重文件model.pth
    """

    # 124M参数量GPT-2基础配置
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词表总大小
        "context_length": 1024,  # 上下文窗口长度
        "emb_dim": 768,          # 词嵌入向量维度
        "n_heads": 12,           # 多头注意力头数量
        "n_layers": 12,          # Transformer堆叠层数
        "drop_rate": 0.1,        # Dropout随机失活概率
        "qkv_bias": True         # 查询/键/值线性层是否启用偏置项
    }

    # 加载GPT2原生分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 拼接微调分类器权重文件路径
    model_path = Path("..") / "01_main-chapter-code" / "review_classifier.pth"
    # 判断权重文件是否存在
    if not model_path.exists():
        print(
            f"未找到 {model_path} 文件，请先运行第6章代码"
            "（ch06.ipynb）生成 review_classifier.pth 权重文件。"
        )
        # 找不到权重直接终止程序
        sys.exit()

    # 实例化基础GPT模型
    model = GPTModel(GPT_CONFIG_124M)

    # 按照第6章6.5小节的方式，将基础GPT改造为文本二分类模型
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)

    # 加载训练好的模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    # 将模型迁移至指定设备（GPU/CPU）
    model.to(device)
    # 切换为推理评估模式（关闭Dropout、BatchNorm训练行为）
    model.eval()

    return tokenizer, model


# 提前加载分词器与模型，供下方Chainlit交互函数调用
tokenizer, model = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit网页交互主逻辑函数
    """
    # 获取用户输入文本
    user_input = message.content

    # 调用分类函数，输出文本情感正负标签
    label = classify_review(user_input, model, tokenizer, device, max_length=120)

    # 将分类结果返回网页前端界面
    await chainlit.Message(
        content=f"{label}",  # 向交互页面返回模型推理结果
    ).send()