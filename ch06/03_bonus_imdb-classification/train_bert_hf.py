# 版权所有 (c) Sebastian Raschka，遵循 Apache 2.0 开源协议（详见 LICENSE.txt）。
# 《从零搭建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch

import argparse
from pathlib import Path
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IMDBDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, use_attention_mask=False):
        # 读取IMDB影评数据集csv文件
        self.data = pd.read_csv(csv_file)
        # 设定序列最大长度，未指定则自动计算文本编码后的最长长度
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)
        self.pad_token_id = pad_token_id
        # 是否启用注意力掩码区分填充占位符
        self.use_attention_mask = use_attention_mask

        # 预先对所有文本分词编码，超长文本自动截断，按需生成注意力掩码
        self.encoded_texts = [
            tokenizer.encode(text, truncation=True, max_length=self.max_length)
            for text in self.data["text"]
        ]
        # 统一补齐填充token，使所有文本序列长度一致
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

        if self.use_attention_mask:
            # 为每条文本生成对应的注意力掩码
            self.attention_masks = [
                self._create_attention_mask(et)
                for et in self.encoded_texts
            ]
        else:
            self.attention_masks = None

    def _create_attention_mask(self, encoded_text):
        # 生成掩码：真实文本token标记为1，填充占位符标记为0
        return [1 if token_id != self.pad_token_id else 0 for token_id in encoded_text]

    def __getitem__(self, index):
        # 根据索引取出编码文本与对应情感标签
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]

        if self.use_attention_mask:
            attention_mask = self.attention_masks[index]
        else:
            # 不使用掩码时，全部token均视为有效输入
            attention_mask = torch.ones(self.max_length, dtype=torch.long)

        # 返回编码文本、注意力掩码、分类标签张量
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        # 返回数据集总样本数量
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        # 遍历全部文本，计算编码后最长序列长度
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device):
    # 将掩码、输入、标签迁移至指定设备（GPU/CPU）
    attention_mask_batch = attention_mask_batch.to(device)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # logits = model(input_batch)[:, -1, :]  # 仅取序列最后一个token的输出logits
    # 传入注意力掩码，获取模型分类输出分值
    logits = model(input_batch, attention_mask=attention_mask_batch).logits
    # 计算交叉熵分类损失
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# 损失计算逻辑与第5章保持一致
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    # 未指定评估批次数量，则遍历加载器全部批次
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # 若指定批次超过加载器总批次，限制为实际总批次
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    # 返回批次平均损失
    return total_loss / num_batches


@torch.no_grad()  # 关闭梯度计算，节省显存、提升推理速度
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
            # logits = model(input_batch)[:, -1, :]  # 仅取序列最后一个token的输出logits
            logits = model(input_batch, attention_mask=attention_mask_batch).logits
            # 取概率最大类别作为预测标签
            predicted_labels = torch.argmax(logits, dim=1)
            num_examples += predicted_labels.shape[0]
            # 累加预测正确的样本数量
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    # 返回整体分类准确率
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 切换模型至评估模式
    model.eval()
    with torch.no_grad():
        # 分别计算训练集、验证集平均损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    # 切回训练模式继续迭代训练
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None):
    # 初始化列表，用于记录训练损失、验证损失、训练准确率、验证准确率
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环，遍历所有训练轮次
    for epoch in range(num_epochs):
        model.train()  # 将模型切换为训练模式

        for input_batch, attention_mask_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清空上一批次累积的梯度
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)
            loss.backward()  # 反向传播，计算参数梯度
            optimizer.step()  # 根据梯度更新模型权重参数
            examples_seen += input_batch.shape[0]  # 新增：统计已训练样本总数（替代原统计token）
            global_step += 1

            # 可选：定时执行模型评估
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

        # 新增：每轮训练结束后计算训练集、验证集准确率
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
            "指定参与微调的网络层。可选值：'all'(全部层), 'last_block'(最后一层Transformer块), 'last_layer'(仅输出分类层)"
        )
    )
    parser.add_argument(
        "--use_attention_mask",
        type=str,
        default="true",
        help=(
            "是否使用注意力掩码屏蔽填充token。可选值：'true'启用，'false'关闭"
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert",
        help=(
            "选择待微调的预训练模型。可选值：'distilbert', 'bert', 'roberta'"
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help=(
            "训练总轮次"
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
        # 加载二分类DistilBERT预训练模型
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        # 替换分类输出头，输出2个情感类别
        model.out_head = torch.nn.Linear(in_features=768, out_features=2)
        # 默认冻结全部网络参数
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "last_layer":
            # 仅解冻输出分类层参数
            for param in model.out_head.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            # 解冻前置分类层与最后一层Transformer
            for param in model.pre_classifier.parameters():
                param.requires_grad = True
            for param in model.distilbert.transformer.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            # 解冻全部层，全量微调
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("--trainable_layers 参数输入无效。")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    elif args.model == "bert":
        # 加载基础BERT二分类预训练模型
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
            raise ValueError("--trainable_layers 参数输入无效。")

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
            raise ValueError("--trainable_layers 参数输入无效。")

        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
    else:
        raise ValueError(f"所选模型 --model {args.model} 暂不支持。")

    # 自动选择GPU或CPU作为计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ###############################
    # 实例化数据集加载器DataLoader
    ###############################

    base_path = Path(".")

    # 解析是否启用注意力掩码
    if args.use_attention_mask.lower() == "true":
        use_attention_mask = True
    elif args.use_attention_mask.lower() == "false":
        use_attention_mask = False
    else:
        raise ValueError("`use_attention_mask` 参数输入不合法。")

    # 构建训练、验证、测试数据集实例
    train_dataset = IMDBDataset(
        base_path / "train.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    val_dataset = IMDBDataset(
        base_path / "validation.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    test_dataset = IMDBDataset(
        base_path / "test.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )

    num_workers = 0
    batch_size = 8

    # 训练集加载器：打乱样本，丢弃不足一个批次的末尾样本
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # 验证集加载器：不打乱，保留不足一个批次的样本
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
    # 定义AdamW优化器，设置权重衰减正则化
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