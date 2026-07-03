# 版权所有 (c) Sebastian Raschka，遵循 Apache 2.0 开源协议（详见 LICENSE.txt）。
# 《从零搭建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch

import argparse
import math
import os
from pathlib import Path
import time
import urllib.request
import zipfile

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # LoRA低秩矩阵A，输入维度×秩
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        # 凯明均匀初始化矩阵A
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # LoRA低秩矩阵B，秩×输出维度，初始化为全0
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        # LoRA缩放超参alpha
        self.alpha = alpha

    def forward(self, x):
        # LoRA分支前向计算：alpha * X @ A @ B
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        # 保存原始全连接层（冻结权重）
        self.linear = linear
        # 实例化LoRA低秩分支
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # 原始线性输出 + LoRA分支输出
        return self.linear(x) + self.lora(x)


# 该LoRA实现逻辑与LinearWithLoRA等价，只是将权重提前融合计算
class LinearWithLoRAMerged(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # 预融合A、B矩阵得到低秩增量权重
        lora = self.lora.A @ self.lora.B
        # 原始权重叠加LoRA增量权重
        combined_weight = self.linear.weight + self.lora.alpha*lora.T
        # 一次性执行线性计算
        return torch.nn.functional.linear(x, combined_weight, self.linear.bias)


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, no_padding=False):
        # 读取数据集csv文件
        self.data = pd.read_csv(csv_file)
        # 未指定最大序列长度时，自动计算文本编码后的最长长度
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # 预编码所有文本，截断至最大长度
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["Text"]
        ]

        if not no_padding:
            # 对短文本补齐padding，统一序列长度
            self.encoded_texts = [
                et + [pad_token_id] * (self.max_length - len(et))
                for et in self.encoded_texts
            ]

    def __getitem__(self, index):
        # 根据索引返回编码文本与对应标签
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        # 返回数据集样本总数
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        # 遍历全部文本，计算编码后最长序列长度
        max_length = 0
        for text in self.data["Text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def download_and_unzip(url, zip_path, extract_to, new_file_path):
    # 目标文件已存在则跳过下载解压流程
    if new_file_path.exists():
        print(f"{new_file_path} already exists. Skipping download and extraction.")
        return

    # 下载压缩包
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 解压压缩包到指定目录
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # 重命名原始文件，添加后缀标识文件格式
    original_file = Path(extract_to) / "SMSSpamCollection"
    os.rename(original_file, new_file_path)
    print(f"File downloaded and saved as {new_file_path}")


def random_split(df, train_frac, validation_frac):
    # 随机打乱全量数据集，固定随机种子保证实验可复现
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算训练集、验证集切分下标
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 按下标切分训练、验证、测试集
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def create_dataset_csvs(new_file_path):
    # 读取原始短信数据集，制表符分隔，手动指定列名
    df = pd.read_csv(new_file_path, sep="\t", header=None, names=["Label", "Text"])

    # 构建正负样本均衡数据集
    n_spam = df[df["Label"] == "spam"].shape[0]
    # 采样与垃圾短信数量相等的正常短信样本
    ham_sampled = df[df["Label"] == "ham"].sample(n_spam, random_state=123)
    balanced_df = pd.concat([ham_sampled, df[df["Label"] == "spam"]])
    # 再次打乱均衡数据集
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    # 标签映射：正常短信0，垃圾短信1
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # 划分数据集并保存为csv文件
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)


def instantiate_model(choose_model, load_weights):
    # GPT基础通用配置
    BASE_CONFIG = {
        "vocab_size": 50257,     # 词表总大小
        "context_length": 1024,  # 上下文窗口长度
        "drop_rate": 0.0,        # Dropout失活概率
        "qkv_bias": True         # QKV线性层是否启用偏置项
    }

    # 各尺寸GPT2专属超参
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # 合并基础配置与所选模型专属参数
    BASE_CONFIG.update(model_configs[choose_model])

    # 不加载预训练权重时固定随机种子初始化参数
    if not load_weights:
        torch.manual_seed(123)
    # 实例化GPT模型，可选关闭因果掩码
    model = GPTModel(BASE_CONFIG, disable_causal_mask=args.disable_causal_mask)

    if load_weights:
        # 提取模型尺寸标识，下载并加载官方GPT2权重
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        load_weights_into_gpt(model, params)

    # 切换模型至评估模式
    model.eval()
    return model


def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, ignore_index=-100, average_embeddings=False):
    # 将输入、标签迁移至指定设备（GPU/CPU）
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    if trainable_token_pos == "flexible":  # 自动选取每条文本填充符前最后一个有效token
        # 参考讨论：https://github.com/rasbt/LLMs-from-scratch/discussions/434
        pad_token_id = 50256  # 填充符token <|endoftext|>
        mask = input_batch != pad_token_id
        last_token_pos = mask.sum(dim=1) - 1  # 计算每条序列最后有效token下标

        # 模型前向推理得到完整序列输出logits
        logits = model(input_batch)  # shape: [batch_size, seq_len, num_classes]

        # 取出每条样本最后有效token对应的输出logits
        batch_size = logits.size(0)
        selected_logits = logits[torch.arange(batch_size), last_token_pos]

        # 计算交叉熵损失
        loss = torch.nn.functional.cross_entropy(selected_logits, target_batch)
        return loss

    else:
        model_output = model(input_batch)
        if average_embeddings:
            # 对整条序列维度做平均池化，得到全局表征
            logits = model_output.mean(dim=1)
        else:
            # 选取指定位置token的输出表征用于分类
            logits = model_output[:, trainable_token_pos, :]

        loss = torch.nn.functional.cross_entropy(logits, target_batch, ignore_index=ignore_index)
        return loss


def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     ignore_index=-100, average_embeddings=False):
    total_loss = 0.
    # 数据集加载器为空返回NaN
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 限制评估批次不超过加载器总批次
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )
            total_loss += loss.item()
        else:
            break
    # 返回平均损失
    return total_loss / num_batches


@torch.no_grad()  # 禁用梯度计算，节省显存、提升推理速度
def calc_accuracy_loader(data_loader, model, device, num_batches=None,
                         trainable_token_pos=-1, average_embeddings=False):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    if trainable_token_pos == "flexible":
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                # 计算每条序列最后一个非填充token下标
                pad_token_id = 50256  # <|endoftext|> 填充token
                mask = input_batch != pad_token_id
                last_token_pos = mask.sum(dim=1) - 1

                with torch.no_grad():
                    logits = model(input_batch)
                    # 提取每条样本末尾有效token的输出
                    batch_size = logits.size(0)
                    selected_logits = logits[torch.arange(batch_size), last_token_pos]
                    predicted_labels = torch.argmax(selected_logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break

    else:
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                model_output = model(input_batch)
                if average_embeddings:
                    # 序列维度全局平均池化
                    logits = model_output.mean(dim=1)
                else:
                    # 取指定位置token表征
                    logits = model_output[:, trainable_token_pos, :]

                predicted_labels = torch.argmax(logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
    # 返回整体准确率
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device,
                   eval_iter, trainable_token_pos=-1,
                   ignore_index=-100, average_embeddings=False):
    # 切换模型评估模式
    model.eval()
    with torch.no_grad():
        # 计算训练集、验证集平均损失
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
    # 切回训练模式继续迭代
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            accumulation_steps=1, ignore_index=-100, average_embeddings=False):
    # 初始化列表记录训练损失、验证损失、训练准确率、验证准确率
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环，遍历所有训练轮次
    for epoch in range(num_epochs):
        model.train()  # 切换模型为训练模式

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )

            # 梯度累积：批次大于1时缩放损失
            # 原理讲解参考：https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html
            loss /= accumulation_steps

            loss.backward()  # 反向传播计算梯度

            # 判断是否到达梯度更新节点
            is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if is_update_step:
                optimizer.step()  # 利用累积梯度更新模型权重
                optimizer.zero_grad()  # 清空梯度缓存

            examples_seen += input_batch.shape[0]  # 累计已训练样本总数
            global_step += 1

            # 每隔指定步数执行一次评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                    average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # 达到最大训练步数则提前终止训练
            if max_steps is not None and global_step > max_steps:
                break

        # 每轮训练结束后计算训练、验证集准确率
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def replace_linear_with_lora(model, rank, alpha, alternative=False):
    # 递归遍历模型所有子模块
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 将原生Linear层替换为带LoRA分支的自定义层
            if alternative:
                setattr(model, name, LinearWithLoRAMerged(module, rank, alpha))
            else:
                setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归处理嵌套子模块
            replace_linear_with_lora(module, rank, alpha)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2-small (124M)",
        help=(
            "选择使用的GPT模型尺寸。可选值：'gpt2-small (124M)', 'gpt2-medium (355M)',"
            " 'gpt2-large (774M)', 'gpt2-xl (1558M)'."
        )
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained",
        help=(
            "权重初始化方式：'pretrained'加载官方预训练权重，'random'随机初始化权重"
        )
    )
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="last_block",
        help=(
            "指定需要参与微调的网络层。可选值：'all'(全部层), 'last_block'(最后一个Transformer块), "
            "'last_two_blocks'(最后两层Transformer), 'last_layer'(仅输出分类层), "
            "'lora'(LoRA低秩微调), 'lora_alternative'(权重融合版LoRA)"
        )
    )
    parser.add_argument(
        "--trainable_token_pos",
        type=str,
        default="last",
        help=(
            "用于分类的token位置。可选值：'first'(首token), 'last'(末尾token), 'flexible'(每条文本最后有效token)"
        )
    )
    parser.add_argument(
        "--average_embeddings",
        action='store_true',
        default=False,
        help=(
            "开启后对整条序列所有token表征做平均池化，而非仅使用指定位置的token表征"
        )
    )
    parser.add_argument(
        "--context_length",
        type=str,
        default="longest_training_example",
        help=(
            "输入文本上下文长度设置。可选值：'longest_training_example'(训练集最长文本长度), "
            "'model_context_length'(模型原生最大上下文长度)，或直接传入整数数值"
        )
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help=(
            "当--trainable_layers设置为lora时，指定LoRA低秩矩阵的秩大小"
        )
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help=(
            "当--trainable_layers设置为lora时，指定LoRA缩放超参alpha"
        )
    )
    parser.add_argument(
        "--no_padding",
        action='store_true',
        default=False,
        help=(
            "关闭序列填充操作，每条样本长度不统一；开启后必须设置--batch_size 1"
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help=(
            "训练总轮次"
        )
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "训练批次大小"
        )
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help=(
            "梯度累积步数，模拟更大批次训练。原理参考：https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html"
            "例如 batch_size=8 + accumulation_steps=1 与 batch_size=1 + accumulation_steps=8 等效，后者迭代次数更多"
        )
    )
    parser.add_argument(
        "--disable_causal_mask",
        action='store_true',
        default=False,
        help=(
            "关闭Transformer因果注意力掩码（允许token看到后文信息，适用于分类任务）"
        )
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help=(
            "交叉熵损失中忽略的标签下标，对应无需计算损失的样本"
        )
    )

    args = parser.parse_args()

    # 解析token位置参数
    if args.trainable_token_pos == "first":
        args.trainable_token_pos = 0
    elif args.trainable_token_pos == "last":
        args.trainable_token_pos = -1
    # flexible模式自动取填充前最后一个有效token
    # 参考讨论：https://github.com/rasbt/LLMs-from-scratch/discussions/434
    elif args.trainable_token_pos == "flexible":
        args.trainable_token_pos = "flexible"
    else:
        raise ValueError("无效的 --trainable_token_pos 参数")

    ###############################
    # 加载模型
    ###############################

    if args.weights == "pretrained":
        load_weights = True
    elif args.weights == "random":
        load_weights = False
    else:
        raise ValueError("无效的 --weights 参数")

    model = instantiate_model(args.model_size, load_weights)
    # 默认冻结全部参数
    for param in model.parameters():
        param.requires_grad = False

    # 根据模型尺寸获取嵌入维度
    if args.model_size == "gpt2-small (124M)":
        in_features = 768
    elif args.model_size == "gpt2-medium (355M)":
        in_features = 1024
    elif args.model_size == "gpt2-large (774M)":
        in_features = 1280
    elif args.model_size == "gpt2-xl (1558M)":
        in_features = 1600
    else:
        raise ValueError("无效的 --model_size 参数")

    torch.manual_seed(123)
    # 替换输出头为二分类线性层
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)

    # 根据微调策略解冻对应网络层
    if args.trainable_layers == "last_layer":
        pass
    elif args.trainable_layers == "last_block" or args.trainable_layers == "last_two_blocks":
        # 解冻最后一层Transformer + 最终归一化层
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
        # 若选择两层则额外解冻倒数第二层Transformer
        if args.trainable_layers == "last_two_blocks":
            for param in model.trf_blocks[-2].parameters():
                param.requires_grad = True
    elif args.trainable_layers == "all":
        # 解冻全部网络层，全量微调
        for param in model.parameters():
            param.requires_grad = True
    elif args.trainable_layers in ("lora", "lora_alternative"):
        # 替换所有Linear层为LoRA分支结构
        alternative = True if args.trainable_layers == "lora_alternative" else False
        replace_linear_with_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, alternative=alternative)
    else:
        raise ValueError("无效的 --trainable_layers 参数")

    # 自动选择GPU/CPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ###############################
    # 构建数据集加载器DataLoader
    ###############################

    # 短信垃圾数据集下载地址
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extract_to = "sms_spam_collection"
    new_file_path = Path(extract_to) / "SMSSpamCollection.tsv"

    base_path = Path(".")
    file_names = ["train.csv", "validation.csv", "test.csv"]
    # 判断训练/验证/测试集文件是否全部存在
    all_exist = all((base_path / file_name).exists() for file_name in file_names)

    # 数据集缺失则自动下载、解压、生成均衡csv文件
    if not all_exist:
        download_and_unzip(url, zip_path, extract_to, new_file_path)
        create_dataset_csvs(new_file_path)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = None

    if args.no_padding:
        max_length = None

    else:
        if args.context_length == "model_context_length":
            # 使用模型原生最大上下文长度
            max_length = model.pos_emb.weight.shape[0]
        elif args.context_length == "longest_training_example":
            # 先读取训练集，自动获取最长文本长度作为上下文上限
            train_dataset = SpamDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer, no_padding=args.no_padding)
            max_length = train_dataset.max_length
        else:
            # 转换用户传入的整数上下文长度
            try:
                max_length = int(args.context_length)
            except ValueError:
                raise ValueError("无效的 --context_length 参数")

    # 实例化训练、验证、测试数据集
    if train_dataset is None:
        train_dataset = SpamDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)
    val_dataset = SpamDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)
    test_dataset = SpamDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)

    tokenizer = tiktoken.get_encoding("gpt2")

    num_workers = 0

    # 训练集加载器：打乱样本，丢弃最后不足batch的批次
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # 验证集加载器：不打乱，保留不足batch的样本
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # 测试集加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # 校验文本最大长度不超过模型原生上下文窗口
    assert train_dataset.max_length <= model.pos_emb.weight.shape[0], (
        f"数据集文本最大长度 {train_dataset.max_length} 超出模型上下文上限 "
        f"{model.pos_emb.weight.shape[0]}。重新初始化数据集时设置 "
        f"`max_length={model.pos_emb.weight.shape[0]}`"
    )

    ###############################
    # 执行模型微调训练
    ###############################

    start_time = time.time()
    torch.manual_seed(123)
    # 定义AdamW优化器，设置权重衰减正则
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=5,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        accumulation_steps=args.accumulation_steps, average_embeddings=args.average_embeddings
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"训练完成，总耗时 {execution_time_minutes:.2f} 分钟。")

    ###############################
    # 在完整数据集上评估模型效果
    ###############################

    train_accuracy = calc_accuracy_loader(
        train_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )

    print(f"训练集准确率: {train_accuracy*100:.2f}%")
    print(f"验证集准确率: {val_accuracy*100:.2f}%")
    print(f"测试集准确率: {test_accuracy*100:.2f}%")