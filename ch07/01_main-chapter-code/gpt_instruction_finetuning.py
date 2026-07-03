# 版权所有 (c) Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt）
# 《从零构建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch
#
# 基于第7章代码实现的极简指令微调脚本

from functools import partial
from importlib.metadata import version
import json
import os
import re
import time
import urllib

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 从当前目录本地文件导入依赖函数
from gpt_download import download_and_load_gpt2
from previous_chapters import (
    calc_loss_loader,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text
)


class InstructionDataset(Dataset):
    # 指令微调数据集封装类
    def __init__(self, data, tokenizer):
        self.data = data

        # 预编码所有文本
        self.encoded_texts = []
        for entry in data:
            # 拼接指令与输入上下文
            instruction_plus_input = format_input(entry)
            # 拼接标准答案回复
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            # 将完整文本编码为token id存入列表
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        # 根据索引返回单条编码后的样本
        return self.encoded_texts[index]

    def __len__(self):
        # 返回数据集总样本数量
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 找出当前批次内最长序列的长度
    batch_max_length = max(len(item)+1 for item in batch)

    # 初始化输入、目标标签列表，用于存放批次张量
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 在序列末尾添加结束符 <|endoftext|>
        new_item += [pad_token_id]
        # 使用padding token补齐至批次最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # 输入序列：截取padding后全部token，去掉最后一位
        inputs = torch.tensor(padded[:-1])
        # 目标标签：整体右移一位，作为自回归预测目标
        targets = torch.tensor(padded[1:])

        # 新增逻辑：除第一个padding token外，其余padding标签全部置为ignore_index不参与loss计算
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 可选：截断序列至设定的最大长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将列表堆叠为批量张量，并迁移至指定运算设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    # 若本地无数据集文件，则从远程链接下载
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    # 本地已存在文件则直接读取
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # 加载JSON格式数据集
    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def format_input(entry):
    # 构造SFT标准指令模板
    instruction_text = (
        f"下面是一条描述任务的指令，请写出能恰当完成该任务的回答。"
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 若样本存在输入上下文，则拼接Input字段；无则为空字符串
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    # 创建画布，尺寸12×6英寸
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制训练损失与验证损失曲线（横轴为训练轮数）
    ax1.plot(epochs_seen, train_losses, label="训练损失")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="验证损失")
    ax1.set_xlabel("训练轮数 Epochs")
    ax1.set_ylabel("损失 Loss")
    ax1.legend(loc="upper right")

    # 创建共享Y轴的第二条X轴（横轴为已处理token总数）
    ax2 = ax1.twiny()
    # 绘制不可见曲线，用于对齐坐标轴刻度
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("已处理Token数量")

    fig.tight_layout()  # 自动调整布局，避免标签重叠
    plot_name = "loss-plot-standalone.pdf"
    print(f"损失曲线已保存为 {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


def main(test_mode=False):
    #######################################
    # 打印依赖库版本信息
    #######################################
    print()
    pkgs = [
        "matplotlib",  # 绘图可视化库
        "tiktoken",    # OpenAI分词器
        "torch",       # 深度学习框架
        "tqdm",        # 命令行进度条工具
        "tensorflow",  # 用于读取OpenAI官方预训练权重
    ]
    for p in pkgs:
        print(f"{p} 版本: {version(p)}")
    print(50*"-")

    #######################################
    # 下载并预处理数据集
    #######################################
    file_path = "instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)  # 85%数据划分为训练集
    test_portion = int(len(data) * 0.1)    # 10%数据划分为测试集

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # 测试模式：仅使用极小子集快速调试
    if args.test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    print("训练集样本数量:", len(train_data))
    print("验证集样本数量:", len(val_data))
    print("测试集样本数量:", len(test_data))
    print(50*"-")

    tokenizer = tiktoken.get_encoding("gpt2")
    # 自动选择运算设备：有CUDA则用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("运算设备:", device)
    print(50*"-")

    # 固定collate_fn参数：设备、最大序列长度1024
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)  # 固定随机种子保证实验可复现

    # 构建训练集DataLoader
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,        # 训练集开启随机打乱
        drop_last=True,      # 丢弃不足一个batch的末尾样本
        num_workers=num_workers
    )

    # 构建验证集DataLoader
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,       # 验证集不打乱
        drop_last=False,     # 保留不足一个batch的样本
        num_workers=num_workers
    )

    #######################################
    # 加载预训练GPT模型
    #######################################

    # 测试模式：使用超小型GPT快速调试
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,    # 词表大小
            "context_length": 120,  # 上下文窗口长度
            "drop_rate": 0.0,       # Dropout概率
            "qkv_bias": False,      # QKV线性层不使用偏置
            "emb_dim": 12,          # 词嵌入维度
            "n_layers": 1,          # Transformer层数
            "n_heads": 2            # 注意力头数量
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"
        CHOOSE_MODEL = "小型测试模型"

    # 正式章节训练逻辑
    else:
        BASE_CONFIG = {
            "vocab_size": 50257,     # 词表总大小
            "context_length": 1024,  # 上下文窗口长度
            "drop_rate": 0.0,        # Dropout丢弃概率
            "qkv_bias": True         # QKV线性层启用偏置项
        }

        # 各尺寸GPT2模型超参
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        # 选定使用gpt2-medium 355M模型
        CHOOSE_MODEL = "gpt2-medium (355M)"

        # 合并基础配置与选定模型专属参数
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        # 提取模型尺寸名称，下载对应预训练权重
        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        # 初始化GPT模型并加载官方预训练权重
        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        model.eval()
        model.to(device)

    print("已加载模型:", CHOOSE_MODEL)
    print(50*"-")

    #######################################
    # 执行模型指令微调
    #######################################
    print("微调前初始损失值")
    with torch.no_grad():
        # 仅取5个batch计算初始损失，加快执行速度
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   训练损失:", train_loss)
    print("   验证损失:", val_loss)

    start_time = time.time()
    # 优化器：AdamW，学习率5e-5，权重衰减0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2  # 训练总轮数

    torch.manual_seed(123)
    # 执行完整训练流程，返回损失记录与已处理token数量
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"训练完成，总耗时 {execution_time_minutes:.2f} 分钟。")

    # 生成训练损失曲线图
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    #######################################
    # 保存微调结果与模型权重
    #######################################
    print("生成测试集模型回复")
    # 遍历全部测试样本生成模型输出
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        # 截取模型生成的回复部分，去除标记并清理首尾空白
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    # 写入带模型输出的完整测试集JSON
    test_data_path = "instruction-data-with-response-standalone.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # indent=4 格式化输出，方便阅读
    print(f"带模型回复的数据集已保存为 {test_data_path}")

    # 清洗模型名称，拼接权重文件名
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft-standalone.pth"
    # 保存微调完成的模型权重
    torch.save(model.state_dict(), file_name)
    print(f"微调后模型权重已保存为 {file_name}")


if __name__ == "__main__":

    import argparse

    # 命令行参数解析器
    parser = argparse.ArgumentParser(
        description="对GPT模型执行指令微调（SFT）"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("开启测试模式，仅使用极小数据集快速调试代码；"
              "不添加该参数则使用完整数据集正式训练（推荐）")
    )
    args = parser.parse_args()

    main(args.test_mode)