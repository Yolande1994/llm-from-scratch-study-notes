# 版权所有 (c) Sebastian Raschka，遵循 Apache 2.0 开源协议（详见 LICENSE.txt）。
# 《从零搭建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch

import argparse
import os
from pathlib import Path
import time
import urllib
import zipfile

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, no_padding=False):
        # 读取数据集csv文件
        self.data = pd.read_csv(csv_file)
        # 若未指定最大长度，则自动计算文本编码后的最长长度
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # 预先对所有文本进行分词编码，截断至设定最大长度
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["Text"]
        ]

        if not no_padding:
            # 对短文本补齐padding，统一序列长度至最长文本长度
            self.encoded_texts = [
                et + [pad_token_id] * (self.max_length - len(et))
                for et in self.encoded_texts
            ]

    def __getitem__(self, index):
        # 根据索引取出编码文本与对应标签
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        # 转为long类型张量返回
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        # 返回数据集总样本数量
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
    # 若目标文件已存在，跳过下载和解压流程
    if new_file_path.exists():
        print(f"{new_file_path} already exists. Skipping download and extraction.")
        return

    # 下载数据集压缩包
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 解压压缩包到指定目录
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # 重命名原始文件，添加文件后缀标识格式
    original_file = Path(extract_to) / "SMSSpamCollection"
    os.rename(original_file, new_file_path)
    print(f"File downloaded and saved as {new_file_path}")


def random_split(df, train_frac, validation_frac):
    # 随机打乱全量数据集，固定随机种子保证复现
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算训练集、验证集分割下标
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
    # 采样与垃圾短信数量相同的正常短信样本
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
    test_df.to_csv("test.csv")


class SPAMDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, use_attention_mask=False):
        self.data = pd.read_csv(csv_file)
        # 设定序列最大长度，未传参则自动计算最长文本
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)
        self.pad_token_id = pad_token_id
        # 是否启用注意力掩码区分padding
        self.use_attention_mask = use_attention_mask

        # 预编码文本，超长自动截断
        self.encoded_texts = [
            tokenizer.encode(text, truncation=True, max_length=self.max_length)
            for text in self.data["Text"]
        ]
        # 统一补齐padding至max_length
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

        if self.use_attention_mask:
            # 生成每条文本对应的注意力掩码
            self.attention_masks = [
                self._create_attention_mask(et)
                for et in self.encoded_texts
            ]
        else:
            self.attention_masks = None

    def _create_attention_mask(self, encoded_text):
        # 生成掩码：真实token为1，padding占位符为0
        return [1 if token_id != self.pad_token_id else 0 for token_id in encoded_text]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]

        if self.use_attention_mask:
            attention_mask = self.attention_masks[index]
        else:
            # 不使用掩码时，全部token视为有效
            attention_mask = torch.ones(self.max_length, dtype=torch.long)

        # 返回编码文本、注意力掩码、分类标签
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        # 返回数据集样本总数
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        # 遍历所有文本，获取编码后最长序列长度
        max_length = 0
        for text in self.data["Text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device):
    # 将掩码、输入、标签迁移至指定设备GPU/CPU
    attention_mask_batch = attention_mask_batch.to(device)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # logits = model(input_batch)[:, -1, :]  # 仅取最后一个token的输出logits
    # 传入注意力掩码获取模型分类输出
    logits = model(input_batch, attention_mask=attention_mask_batch).logits
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# 与第5章损失计算逻辑一致
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    # 未指定批次数量则遍历全部批次
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # 限制评估批次不超过数据集总批次
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    # 返回平均损失
    return total_loss / num_batches


@torch.no_grad()  # 禁用梯度计算，提升推理速度、节省显存
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            attention_mask_batch = attention_mask_batch.to(device)
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # logits = model(input_batch)[:, -1, :]  # 仅取最后一个token输出logits
            logits = model(input_batch, attention_mask=attention_mask_batch).logits
            # 取概率最大类别作为预测标签
            predicted_labels = torch.argmax(logits, dim=1)
            num_examples += predicted_labels.shape[0]
            # 统计预测正确样本数量
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    # 返回整体准确率
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 切换模型至评估模式
    model.eval()
    with torch.no_grad():
        # 分别计算训练集、验证集平均损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    # 切回训练模式继续训练
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None):
    # 初始化列表，用于记录训练损失、验证损失、训练准确率、验证准确率
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环，遍历所有训练轮次
    for epoch in range(num_epochs):
        model.train()  # 切换模型为训练模式

        for input_batch, attention_mask_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清空上一批次累积梯度
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 根据梯度更新模型参数权重
            examples_seen += input_batch.shape[0]  # 累计已训练样本总数
            global_step += 1

            # 每隔指定步数执行一次评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # 达到最大训练步数则提前终止训练
            if max_steps is not None and global_step > max_steps:
                break

        # 每轮训练结束后计算训练、验证集准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


if __name__ == "__main__":

    # 命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="all",
        help=(
            "指定需要参与微调的网络层。可选参数：'all'(全部层), 'last_block'(最后一个transformer块), 'last_layer'(仅输出层)"
        )
    )
    parser.add_argument(
        "--use_attention_mask",
        type=str,
        default="true",
        help=(
            "是否使用注意力掩码屏蔽padding占位符。可选参数：'true'启用，'false'关闭"
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert",
        help=(
            "选择待训练预训练模型。可选参数：'distilbert', 'bert', 'roberta'"
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help=(
            "训练轮次总数"
        )
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help=(
            "模型微调学习率"
        )
    )
    args = parser.parse_args()

    ###############################
    # 加载预训练模型与分词器
    ###############################

    torch.manual_seed(123)
    if args.model == "distilbert":
        # 加载二分类DistilBERT预训练权重
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        # 替换分类输出头，输出2个类别
        model.out_head = torch.nn.Linear(in_features=768, out_features=2)
        # 默认冻结全部参数
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            # 仅解冻输出分类层
            for param in model.out_head.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            # 解冻前置分类层与最后一层Transformer
            for param in model.pre_classifier.parameters():
                param.requires_grad = True
            for param in model.distilbert.transformer.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            # 解冻全部网络层，全量微调
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("无效的--trainable_layers参数输入。")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    elif args.model == "bert":
        # 加载基础BERT二分类模型
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        # 替换分类输出层
        model.classifier = torch.nn.Linear(in_features=768, out_features=2)
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            # 解冻分类层、池化层、最后一层编码器
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.bert.pooler.dense.parameters():
                param.requires_grad = True
            for param in model.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("无效的--trainable_layers参数输入。")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.model == "roberta":
        # 加载大参数量RoBERTa分类模型
        model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-large", num_labels=2
        )
        # 替换分类输出投影层
        model.classifier.out_proj = torch.nn.Linear(in_features=1024, out_features=2)
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.roberta.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("无效的--trainable_layers参数输入。")

        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
    else:
        raise ValueError(f"所选模型--model {args.model} 暂不支持。")

    # 自动选择GPU/CPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

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

    # 数据集文件缺失则自动下载、解压、生成均衡csv文件
    if not all_exist:
        download_and_unzip(url, zip_path, extract_to, new_file_path)
        create_dataset_csvs(new_file_path)

    # 解析是否启用注意力掩码
    if args.use_attention_mask.lower() == "true":
        use_attention_mask = True
    elif args.use_attention_mask.lower() == "false":
        use_attention_mask = False
    else:
        raise ValueError("`use_attention_mask` 参数输入不合法。")

    # 实例化训练、验证、测试数据集
    train_dataset = SPAMDataset(
        base_path / "train.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    val_dataset = SPAMDataset(
        base_path / "validation.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    test_dataset = SPAMDataset(
        base_path / "test.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )

    num_workers = 0
    batch_size = 8

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

    ###############################
    # 执行模型微调训练
    ###############################

    start_time = time.time()
    torch.manual_seed(123)
    # 定义AdamW优化器，设置权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"训练完成，总耗时 {execution_time_minutes:.2f} 分钟。")

    ###############################
    # 在完整数据集上评估模型效果
    ###############################

    print("\n在完整数据集上进行评估 ...\n")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"训练集准确率: {train_accuracy*100:.2f}%")
    print(f"验证集准确率: {val_accuracy*100:.2f}%")
    print(f"测试集准确率: {test_accuracy*100:.2f}%")