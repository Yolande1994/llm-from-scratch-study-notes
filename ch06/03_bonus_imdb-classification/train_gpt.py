# 版权所有 (c) Sebastian Raschka，遵循 Apache 2.0 开源协议（详见 LICENSE.txt）。
# 《从零搭建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch

import argparse
from pathlib import Path
import time

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


# IMDB电影评论数据集自定义读取类
class IMDBDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        # 若未指定最大长度，则自动计算数据集中最长文本编码后的长度
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # 提前对所有文本完成分词编码
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["text"]
        ]
        # 将所有序列填充至统一最长长度
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

    # 通过索引取单条样本
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    # 返回数据集总样本数量
    def __len__(self):
        return len(self.data)

    # 遍历全部文本，找出编码后最长序列的长度
    def _longest_encoded_length(self, tokenizer):
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


# 根据指定尺寸实例化GPT模型，可选加载预训练权重
def instantiate_model(choose_model, load_weights):

    # GPT基础通用配置
    BASE_CONFIG = {
        "vocab_size": 50257,     # 词表总量
        "context_length": 1024,  # 上下文窗口长度
        "drop_rate": 0.0,        # Dropout失活概率
        "qkv_bias": True         # 查询/键/值线性层是否启用偏置项
    }

    # 四种GPT2尺寸对应的专属参数
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # 合并基础配置与选定模型的专属参数
    BASE_CONFIG.update(model_configs[choose_model])

    # 不加载预训练权重时，固定随机种子保证初始化可复现
    if not load_weights:
        torch.manual_seed(123)
    model = GPTModel(BASE_CONFIG)

    # 加载官方GPT2预训练权重
    if load_weights:
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        load_weights_into_gpt(model, params)

    # 切换至推理模式
    model.eval()
    return model


# 计算单个批次的损失
def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, average_embeddings=False):
    # 将输入、标签迁移至指定设备(GPU/CPU)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    model_output = model(input_batch)
    if average_embeddings:
        # 在序列维度（第1维）做全局平均池化
        logits = model_output.mean(dim=1)
    else:
        # 只取指定位置token的输出向量
        logits = model_output[:, trainable_token_pos, :]

    # 交叉熵损失计算
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# 遍历数据加载器，计算平均损失
def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     average_embeddings=False):
    total_loss = 0.
    # 数据集为空时返回NaN
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 限制迭代批次不超过加载器总批次数量
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
            )
            total_loss += loss.item()
        else:
            break
    # 返回批次平均损失
    return total_loss / num_batches


# 关闭梯度计算以提升推理效率
@torch.no_grad()
def calc_accuracy_loader(data_loader, model, device,
                         num_batches=None, trainable_token_pos=-1,
                         average_embeddings=False):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            model_output = model(input_batch)
            if average_embeddings:
                # 在序列维度（第1维）做全局平均池化
                logits = model_output.mean(dim=1)
            else:
                # 只取指定位置token的输出向量
                logits = model_output[:, trainable_token_pos, :]

            # 取概率最大类别作为预测标签
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            # 统计预测正确样本数量
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    # 返回整体准确率
    return correct_predictions / num_examples


# 训练阶段评估损失
def evaluate_model(model, train_loader, val_loader, device, eval_iter,
                   trainable_token_pos=-1, average_embeddings=False):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
    # 评估完成切回训练模式
    model.train()
    return train_loss, val_loss


# 完整分类微调训练主流程
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            average_embeddings=False):
    # 初始化列表，记录训练损失、验证损失、训练准确率、验证准确率与样本量
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环，遍历全部轮次
    for epoch in range(num_epochs):
        model.train()  # 切换模型至训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清空上一批次累积梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device,
                                   trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings)
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 根据梯度更新模型权重
            examples_seen += input_batch.shape[0]  # 累计已训练样本总数
            global_step += 1

            # 每隔指定步数执行一次评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # 达到最大训练步数则提前终止
            if max_steps is not None and global_step > max_steps:
                break

        # 每轮训练结束后计算准确率
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


if __name__ == "__main__":

    # 命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2-small (124M)",
        help=(
            "选择使用的GPT模型尺寸，可选参数：'gpt2-small (124M)', 'gpt2-medium (355M)',"
            " 'gpt2-large (774M)', 'gpt2-xl (1558M)'。"
        )
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained",
        help=(
            "权重初始化方式，可选'pretrained'加载预训练权重 / 'random'随机初始化权重。"
        )
    )
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="last_block",
        help=(
            "指定可训练层范围，可选：'all'全部层、'last_block'最后一个Transformer块、'last_layer'仅输出层。"
        )
    )
    parser.add_argument(
        "--trainable_token_pos",
        type=str,
        default="last",
        help=(
            "用于分类的token位置，可选'first'首个token / 'last'末尾token。"
        )
    )
    parser.add_argument(
        "--average_embeddings",
        action='store_true',
        default=False,
        help=(
            "是否对全部token输出向量做全局平均，而非仅使用--trainable_token_pos指定位置的向量。"
        )
    )
    parser.add_argument(
        "--context_length",
        type=str,
        default="256",
        help=(
            "输入文本上下文长度，可选：'longest_training_example'取训练集最长文本、"
            "'model_context_length'使用模型原生窗口、或自定义整数数值。"
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help=(
            "训练总轮次。"
        )
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=(
            "优化器学习率。"
        )
    )
    args = parser.parse_args()

    # 将字符串参数转换为索引
    if args.trainable_token_pos == "first":
        args.trainable_token_pos = 0
    elif args.trainable_token_pos == "last":
        args.trainable_token_pos = -1
    else:
        raise ValueError("无效的 --trainable_token_pos 参数")

    ###############################
    # 加载模型
    ###############################

    # 判断是否加载预训练权重
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

    # 根据模型尺寸获取嵌入维度，用于构建分类输出头
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
    # 替换原输出层，构建二分类头
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)

    # 根据参数解冻指定层
    if args.trainable_layers == "last_layer":
        # 仅训练新增分类头，其余保持冻结
        pass
    elif args.trainable_layers == "last_block":
        # 解冻最后一个Transformer块与层归一化
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
    elif args.trainable_layers == "all":
        # 解冻模型全部参数，全量微调
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("无效的 --trainable_layers 参数")

    # 自动选择GPU/CPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ###############################
    # 构建数据加载器
    ###############################

    base_path = Path(".")

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = None
    # 按参数确定输入序列最大长度
    if args.context_length == "model_context_length":
        max_length = model.pos_emb.weight.shape[0]
    elif args.context_length == "longest_training_example":
        train_dataset = IMDBDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer)
        max_length = train_dataset.max_length
    else:
        try:
            max_length = int(args.context_length)
        except ValueError:
            raise ValueError("无效的 --context_length 参数")

    # 实例化训练、验证、测试数据集
    if train_dataset is None:
        train_dataset = IMDBDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer)
    val_dataset = IMDBDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer)
    test_dataset = IMDBDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer)

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ###############################
    # 执行模型训练
    ###############################

    start_time = time.time()
    torch.manual_seed(123)
    # 初始化AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        average_embeddings=args.average_embeddings
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"训练完成，总耗时 {execution_time_minutes:.2f} 分钟。")

    ###############################
    # 全数据集评估模型效果
    ###############################

    print("\n正在完整数据集上评估模型准确率 ...\n")

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