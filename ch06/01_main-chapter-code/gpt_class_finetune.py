# 版权所有 (c) Sebastian Raschka，遵循 Apache 2.0 开源协议（详见 LICENSE.txt）。
# 《从零搭建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch

# 本文件为第6章核心知识点汇总代码

import urllib.request
import zipfile
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=False):
    # 如果目标数据集文件已存在，跳过下载和解压流程
    if data_file_path.exists():
        print(f"{data_file_path} 已存在，跳过下载与解压步骤。")
        return

    if test_mode:  # CI自动化测试环境网络不稳定，开启多次重试机制
        max_retries = 5
        delay = 5  # 每次重试间隔秒数
        for attempt in range(max_retries):
            try:
                # 下载压缩包文件
                with urllib.request.urlopen(url, timeout=10) as response:
                    with open(zip_path, "wb") as out_file:
                        out_file.write(response.read())
                break  # 下载成功则跳出重试循环
            except urllib.error.URLError as e:
                print(f"第{attempt + 1}次下载失败：{e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)  # 失败后等待一段时间再重试
                else:
                    print("多次尝试后仍下载失败。")
                    return  # 全部重试次数耗尽，直接退出函数

    else:  # 书本正文标准下载逻辑（无重试）
        # 下载压缩包文件
        with urllib.request.urlopen(url) as response:
            with open(zip_path, "wb") as out_file:
                out_file.write(response.read())

    # 解压压缩包到指定目录
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # 为原始文件添加.tsv后缀标识文件格式
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"文件下载完成，保存至 {data_file_path}")


def create_balanced_dataset(df):
    # 统计垃圾短信(spam)样本数量
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 随机采样与垃圾短信数量相同的正常短信(ham)样本，平衡正负样本
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # 合并采样后的正常短信与全部垃圾短信，构建均衡数据集
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    # 随机打乱全量数据集，固定随机种子保证实验可复现
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算训练集、验证集分割下标
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 根据下标切分训练集、验证集、测试集
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        # 读取数据集csv文件
        self.data = pd.read_csv(csv_file)

        # 预对所有文本执行分词编码
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            # 未指定最大序列长度时，自动取所有文本编码后的最长长度
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 若文本编码后长度超过上限，执行截断操作
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 对所有短文本填充padding占位符，统一所有序列长度至max_length
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        # 根据索引取出编码文本与对应标签
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        # 返回数据集总样本数量
        return len(self.data)

    def _longest_encoded_length(self):
        # 遍历全部编码文本，计算最长序列长度
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()  # 切换模型为评估推理模式
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # 限制评估批次不超过加载器总批次数量
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 将输入、标签迁移至指定计算设备（GPU/CPU）
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # 仅取序列最后一个token的输出分类分数
            predicted_labels = torch.argmax(logits, dim=-1)

            # 累计总样本数与预测正确样本数
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    # 返回整体预测准确率
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    # 将输入、标签迁移至指定计算设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # 仅取序列最后一个token的输出分类分数
    # 计算分类交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    # 返回该批次范围内的平均损失
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 切换为评估模式
    with torch.no_grad():  # 关闭梯度计算，节省显存、加速推理
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 切回训练模式继续训练
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, tokenizer):
    # 初始化列表，用于记录训练损失、验证损失、训练准确率、验证准确率
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环，遍历全部训练轮次
    for epoch in range(num_epochs):
        model.train()  # 切换模型至训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清空上一批次累积的梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 根据梯度更新模型权重参数
            examples_seen += input_batch.shape[0]  # 累计已训练样本总数
            global_step += 1

            # 每间隔指定步数执行一次损失评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"轮次 {epoch+1} (迭代步数 {global_step:06d}): "
                      f"训练损失 {train_loss:.3f}, 验证损失 {val_loss:.3f}")

        # 每轮训练结束后，计算训练集与验证集准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"训练集准确率: {train_accuracy*100:.2f}% | ", end="")
        print(f"验证集准确率: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练、验证指标随训练轮次变化曲线
    ax1.plot(epochs_seen, train_values, label=f"训练{label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"验证{label}")
    ax1.set_xlabel("训练轮次 Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # 创建共享Y轴的第二条X轴，展示已训练样本数量
    ax2 = ax1.twiny()
    # 绘制透明曲线，对齐双坐标轴刻度
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("已训练样本总数")

    fig.tight_layout()  # 自动调整布局，避免文字重叠
    plt.savefig(f"{label}-plot.pdf")
    # plt.show()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="对GPT模型进行微调，实现文本二分类任务"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("该参数用于内部自动化测试；"
              "不添加该参数则运行书本标准训练流程（推荐）。")
    )
    args = parser.parse_args()

    ########################################
    # 下载并预处理数据集
    ########################################

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=args.test_mode)
    # 读取原始数据集，制表符分隔，手动指定列名
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    # 标签映射：正常短信0，垃圾短信1
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # 划分训练、验证、测试集并保存为csv文件
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    ########################################
    # 构建数据集加载器DataLoader
    ########################################
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    # 训练集加载器：打乱样本，丢弃最后不足batch的批次
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # 验证集加载器：不打乱，保留不足batch的样本
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # 测试集加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ########################################
    # 加载预训练GPT模型
    ########################################

    # 测试模式：使用极小参数量模型，快速验证代码逻辑
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"

    # 书本标准训练流程，使用124M GPT2预训练权重
    else:
        CHOOSE_MODEL = "gpt2-small (124M)"
        INPUT_PROMPT = "Every effort moves"

        BASE_CONFIG = {
            "vocab_size": 50257,     # 词表总大小
            "context_length": 1024,  # 上下文窗口长度
            "drop_rate": 0.0,        # Dropout失活概率
            "qkv_bias": True         # QKV线性层是否启用偏置项
        }

        # 各尺寸GPT2模型超参配置
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        # 合并基础配置与选中模型的专属参数
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        # 校验数据集序列长度不超过模型上下文上限
        assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
            f"数据集序列长度 {train_dataset.max_length} 超出模型上下文上限 "
            f"{BASE_CONFIG['context_length']}。初始化数据集时请设置 "
            f"`max_length={BASE_CONFIG['context_length']}`"
        )

        # 提取模型尺寸标识用于下载权重
        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        # 实例化GPT模型并载入预训练权重
        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        # 自动选择GPU/CPU设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################################
    # 修改预训练模型，适配分类任务
    ########################################

    # 默认冻结全部预训练参数
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)

    num_classes = 2
    # 替换输出头为二分类线性层
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    model.to(device)

    # 解冻最后一层Transformer块参数参与微调
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    # 解冻最终归一化层参数参与微调
    for param in model.final_norm.parameters():
        param.requires_grad = True

    ########################################
    # 微调修改后的分类模型
    ########################################

    start_time = time.time()
    torch.manual_seed(123)

    # 定义AdamW优化器，设置学习率与权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
        tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"模型训练完成，总耗时 {execution_time_minutes:.2f} 分钟。")

    ########################################
    # 绘制训练结果图表
    ########################################

    # 绘制损失曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    # 绘制准确率曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")