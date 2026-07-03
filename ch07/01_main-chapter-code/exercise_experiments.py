# 版权所有 (c) Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt）
# 《从零构建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch
#
# 课后习题运行脚本；详细说明参考 exercise-solutions.ipynb

from functools import partial
from importlib.metadata import version
import json
import math
import os
import re
import time
import urllib

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 导入当前目录下的本地依赖文件
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
    # 基础指令微调数据集类
    def __init__(self, data, tokenizer):
        self.data = data

        # 提前将全部文本编码为token序列
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        # 根据索引返回单条编码后的样本
        return self.encoded_texts[index]

    def __len__(self):
        # 返回数据集总样本数
        return len(self.data)


class InstructionDatasetWithMasking(Dataset):
    # 支持指令部分损失掩码的数据集类
    def __init__(self, data, tokenizer):
        self.data = data

        # 新增：单独存储每条样本指令部分的token长度
        self.instruction_lengths = []
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

            # 新增：记录指令部分编码后的token长度
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)

    def __getitem__(self, index):
        # 新增：同时返回指令长度与编码文本
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionDatasetPhi(Dataset):
    # 适配Phi系列模型对话模板的数据集类
    def __init__(self, data, tokenizer):
        self.data = data

        # 预编码所有文本
        self.encoded_texts = []
        for entry in data:

            ###################################################################
            # 新增：使用Phi专属输入格式化函数，并替换回复标记模板
            instruction_plus_input = format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            ###################################################################
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class LinearWithLoRA(torch.nn.Module):
    # 封装LoRA低秩适配的线性层包装类
    def __init__(self, linear, rank, alpha):
        super().__init__()
        # 原始预训练线性层（冻结权重）
        self.linear = linear
        # 初始化LoRA适配层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # 原始输出 + LoRA微调增量
        return self.linear(x) + self.lora(x)


class LoRALayer(torch.nn.Module):
    # LoRA低秩矩阵层实现
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # 低秩矩阵A，输入维度 -> 秩维度
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        # 采用标准 kaiming 均匀初始化
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # 低秩矩阵B，秩维度 -> 输出维度，初始化为0
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        # 缩放超参alpha
        self.alpha = alpha

    def forward(self, x):
        # 计算LoRA增量：alpha * x @ A @ B
        x = self.alpha * (x @ self.A @ self.B)
        return x


def replace_linear_with_lora(model, rank, alpha):
    # 递归遍历模型，将所有原生Linear层替换为LoRA包装层
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 替换原生线性层为带LoRA的线性层
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归对子模块执行相同替换逻辑
            replace_linear_with_lora(module, rank, alpha)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 找出当前批次最长序列长度
    batch_max_length = max(len(item)+1 for item in batch)

    # 存储填充后的输入与标签序列
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 序列末尾添加结束符 <|endoftext|>
        new_item += [pad_token_id]
        # 填充至批次统一最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 输入序列：去掉最后一位token
        targets = torch.tensor(padded[1:])  # 标签序列：整体右移一位

        # 新增：标签中除第一个padding token外，其余padding全部置为ignore_index不参与loss计算
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 新增：可选截断至设定最大序列长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 列表堆叠为批量张量，并迁移至指定运算设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def custom_collate_with_masking_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 获取批次内最长序列长度（批次元素现在是(指令长度, token序列)元组）
    batch_max_length = max(len(item)+1 for instruction_length, item in batch)

    # 存储填充后的输入与标签序列
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:
        new_item = item.copy()
        # 序列末尾添加结束符 <|endoftext|>
        new_item += [pad_token_id]
        # 填充至批次统一最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # 将标签中除第一个padding外的填充位设为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 新增：掩码指令与输入部分token，仅计算回复部分loss
        targets[:instruction_length-1] = -100

        # 可选截断至最大序列长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 堆叠为批量张量并迁移设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    # 本地无文件则从远程链接下载数据集
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    # 本地已有文件直接读取
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # 加载JSON数据集
    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def format_input_phi(entry):
    # Phi模型专属对话提示词模板
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )

    # 存在输入上下文则拼接，无则为空
    input_text = f"\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def format_input(entry):
    # 标准GPT指令微调提示词模板
    instruction_text = (
        f"下面是一条描述任务的指令，请写出能恰当完成该任务的回答。"
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, plot_name):
    # 绘制训练、验证损失曲线
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 横轴为训练轮数，绘制损失曲线
    ax1.plot(epochs_seen, train_losses, label="训练损失")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="验证损失")
    ax1.set_xlabel("训练轮 Epochs")
    ax1.set_ylabel("损失 Loss")
    ax1.legend(loc="upper right")
    # X轴仅显示整数刻度
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 创建共享Y轴的第二条X轴（横轴为已处理Token总数）
    ax2 = ax1.twiny()
    # 绘制不可见曲线用于对齐坐标轴刻度
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("已处理Token数量")

    fig.tight_layout()  # 自动适配布局防止文字重叠
    print(f"损失曲线图已保存为 {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


def main(mask_instructions=False, alpaca52k=False, phi3_prompt=False, lora=False):
    #######################################
    # 打印依赖库版本
    #######################################
    print()
    pkgs = [
        "matplotlib",  # 绘图可视化库
        "tiktoken",    # OpenAI分词器
        "torch",       # 深度学习框架
        "tqdm",        # 命令行进度条
        "tensorflow",  # 用于读取OpenAI官方TF权重
    ]
    for p in pkgs:
        print(f"{p} 版本: {version(p)}")
    print(50*"-")

    #######################################
    # 下载并划分数据集
    #######################################
    file_path = "instruction-data.json"

    # 选择Alpaca52k数据集或本书内置数据集
    if alpaca52k:
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    else:
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)  # 85% 训练集
    test_portion = int(len(data) * 0.1)    # 10% 测试集

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("训练集样本量:", len(train_data))
    print("验证集样本量:", len(val_data))
    print("测试集样本量:", len(test_data))
    print(50*"-")

    tokenizer = tiktoken.get_encoding("gpt2")
    # 自动选择GPU/CPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("运算设备:", device)
    print(50*"-")

    # Alpaca数据集限制最大序列长度512，原生数据集为1024
    if alpaca52k:
        allowed_max_length = 512
    else:
        allowed_max_length = 1024

    # 指令掩码与Phi对话模板模式不可同时启用
    if mask_instructions and phi3_prompt:
        raise ValueError("暂未实现同时开启指令掩码与Phi3对话模板的逻辑")

    # 根据参数选择数据集类与批处理函数
    if mask_instructions:
        customized_collate_fn = partial(custom_collate_with_masking_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetWithMasking
    elif phi3_prompt:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetPhi
    else:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDataset

    num_workers = 0

    # Alpaca数据集使用更小batch_size节省显存
    if alpaca52k:
        batch_size = 4
    else:
        batch_size = 8

    torch.manual_seed(123)  # 固定随机种子保证复现

    # 构建训练集DataLoader
    train_dataset = CustomDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,        # 训练集打乱
        drop_last=True,      # 丢弃不足一个batch的样本
        num_workers=num_workers
    )

    # 构建验证集DataLoader
    val_dataset = CustomDataset(val_data, tokenizer)
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
    BASE_CONFIG = {
        "vocab_size": 50257,     # 词表大小
        "context_length": 1024,  # 上下文窗口长度
        "drop_rate": 0.0,        # Dropout概率
        "qkv_bias": True         # QKV线性层启用偏置
    }

    # 各规格GPT2模型超参
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    # 合并基础配置与选定模型参数
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 提取模型尺寸，下载对应权重
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)

    print("已加载模型:", CHOOSE_MODEL)
    print(50*"-")

    # LoRA微调分支逻辑
    if lora:
        # 替换前可训练参数总量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"开启LoRA前，可训练参数总量：{total_params:,}")

        # 冻结模型全部原生参数
        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"冻结全部主干后，可训练参数总量：{total_params:,}")
        # 替换所有线性层为LoRA包装层，秩16、缩放系数16
        replace_linear_with_lora(model, rank=16, alpha=16)

        # 仅LoRA矩阵参与训练
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA可训练参数总量：{total_params:,}")
        model.to(device)

    #######################################
    # 执行指令微调
    #######################################
    print("微调前初始损失")
    with torch.no_grad():
        # 仅取5个batch快速评估初始损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   训练损失:", train_loss)
    print("   验证损失:", val_loss)

    start_time = time.time()

    num_epochs = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    torch.manual_seed(123)

    # 根据是否启用Phi模板选择起始提示词
    start_context = format_input_phi(val_data[0]) if phi3_prompt else format_input(val_data[0])

    # 执行训练循环，返回损失记录与已处理token计数
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"训练完成，总耗时 {execution_time_minutes:.2f} 分钟。")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

    # 根据实验分支区分曲线图文件名
    plot_name = "loss-plot.pdf"
    if mask_instructions:
        plot_name = plot_name.replace(".pdf", "-mask-instructions.pdf")
    if alpaca52k:
        plot_name = plot_name.replace(".pdf", "-alpaca52k.pdf")
    if phi3_prompt:
        plot_name = plot_name.replace(".pdf", "-phi3-prompt.pdf")
    if lora:
        plot_name = plot_name.replace(".pdf", "-lora.pdf")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        plot_name = plot_name.replace(".pdf", "-baseline.pdf")

    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_name)
    print(50*"-")

    #######################################
    # 生成测试集输出并保存结果
    #######################################
    print("生成测试集模型回复")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input_phi(entry) if phi3_prompt else format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        # 根据模板清理输出文本，截取模型回复
        if phi3_prompt:
            response_text = generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
        else:
            response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    # 根据实验分支区分输出JSON与权重文件名
    test_data_path = "instruction-data-with-response.json"
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"

    if mask_instructions:
        test_data_path = test_data_path.replace(".json", "-mask-instructions.json")
        file_name = file_name.replace(".pth", "-mask-instructions.pth")
    if alpaca52k:
        test_data_path = test_data_path.replace(".json", "-alpaca52k.json")
        file_name = file_name.replace(".pth", "-alpaca52k.pth")
    if phi3_prompt:
        test_data_path = test_data_path.replace(".json", "-phi3-prompt.json")
        file_name = file_name.replace(".pth", "-phi3-prompt.pth")
    if lora:
        test_data_path = test_data_path.replace(".json", "-lora.json")
        file_name = file_name.replace(".pth", "-lora.pth")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        test_data_path = test_data_path.replace(".json", "-baseline.json")
        file_name = file_name.replace(".pth", "-baseline.pth")

    # 写入带模型输出的数据集JSON
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # indent格式化输出便于阅读
    print(f"带模型回复的数据集已保存为 {test_data_path}")

    # 保存微调后模型权重
    torch.save(model.state_dict(), file_name)
    print(f"模型权重已保存为 {file_name}")


if __name__ == "__main__":

    import argparse

    # 命令行参数解析器
    parser = argparse.ArgumentParser(
        description="对GPT模型执行指令微调（多实验分支）"
    )
    options = {"baseline", "mask_instructions", "alpaca_52k", "phi3_prompt", "lora"}
    parser.add_argument(
        "--exercise_solution",
        type=str,
        default="last_block",
        help=(
            f"选择要运行的实验分支，可选值：{options}"
        )
    )
    args = parser.parse_args()

    # 根据传入参数启动对应实验
    if args.exercise_solution == "baseline":
        main()
    elif args.exercise_solution == "mask_instructions":
        main(mask_instructions=True)
    elif args.exercise_solution == "alpaca_52k":
        main(alpaca52k=True)
    elif args.exercise_solution == "phi3_prompt":
        main(phi3_prompt=True)
    elif args.exercise_solution == "lora":
        main(lora=True)
    else:
        raise ValueError(f"参数 {args.exercise_solution} 不合法，可选实验：{options}")