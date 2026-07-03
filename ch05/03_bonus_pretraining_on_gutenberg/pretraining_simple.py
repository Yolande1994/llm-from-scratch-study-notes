# 版权所有 © Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt 文件）
# 配套书籍《从零构建大模型》(Build a Large Language Model From Scratch) 源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 官方代码仓库：https://github.com/rasbt/LLMs-from-scratch

"""
本脚本用于基于古登堡项目书籍数据集，预训练124M参数量小型GPT-2模型。

运行脚本前，请按照README.md文档说明，完成数据集下载与预处理操作。
"""

import argparse
import os
from pathlib import Path
import time
import tiktoken
import torch
from previous_chapters import (
    create_dataloader_v1,
    GPTModel,
    generate_and_print_sample,
    calc_loss_batch,
    evaluate_model,
    plot_losses
)


def read_text_file(file_path):
    """读取文本文件并返回完整文本内容"""
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data


def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0):
    """
    将文本按比例划分为训练集、验证集，并生成对应数据加载器
    :param text_data: 完整文本字符串
    :param train_ratio: 训练集占总文本比例
    :param batch_size: 批次大小
    :param max_length: 上下文窗口长度
    :param stride: 滑动步长
    :param num_workers: 数据加载子进程数量
    :return: 训练集加载器、验证集加载器
    """
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader


def convert_time(seconds):
    """将总秒数换算为 时、分、秒"""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, book_start_time, index, total_files):
    """
    打印单本书耗时、总训练耗时、剩余书籍预估完成时间
    :param start_time: 整体训练起始时间
    :param book_start_time: 当前这本书处理起始时间
    :param index: 当前处理到第几本书
    :param total_files: 全部书籍总数量
    """
    book_end_time = time.time()  # 当前书籍处理完成时刻
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    books_remaining = total_files - index
    average_time_per_book = total_elapsed_time / index
    eta = average_time_per_book * books_remaining

    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"本书训练耗时 {book_h}小时 {book_m}分 {book_s}秒"
          f"\n累计总耗时 {total_h}小时 {total_m}分 {total_s}秒"
          f"\n剩余书籍预估完成时间：{eta_h}小时 {eta_m}分 {eta_s}秒")


def train_model_simple(model, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter, start_context,
                       output_dir, save_ckpt_freq, tokenizer,
                       batch_size=1024, train_ratio=0.90):
    """
    完整预训练主逻辑，遍历多书籍数据集，包含评估、采样生成、断点保存功能
    :param model: GPT模型实例
    :param optimizer: 优化器
    :param device: 运行设备（GPU/CPU）
    :param n_epochs: 训练轮数
    :param eval_freq: 每多少步执行一次损失评估
    :param eval_iter: 评估时抽取多少批次计算损失
    :param print_sample_iter: 每多少步生成一段文本样例
    :param start_context: 文本生成的起始提示词
    :param output_dir: 模型权重保存目录
    :param save_ckpt_freq: 每多少步保存一次模型断点
    :param tokenizer: GPT2分词器
    :param batch_size: 训练批次大小
    :param train_ratio: 单本书内训练集划分比例
    :return: 训练损失列表、验证损失列表、累计处理token数量列表
    """

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    start_time = time.time()

    try:
        for epoch in range(n_epochs):

            # 遍历训练语料库中的每一本图书文件
            for index, file_path in enumerate(all_files, 1):
                book_start_time = time.time()
                text_data = read_text_file(file_path) + " <|endoftext|> "
                print(f"正在处理第 {index}/{total_files} 个文本文件：{file_path}")

                # 每本书单独初始化一组数据加载器
                train_loader, val_loader = create_dataloaders(
                    text_data,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    max_length=GPT_CONFIG_124M["context_length"],
                    stride=GPT_CONFIG_124M["context_length"],
                    num_workers=0
                )
                print("开始训练...")
                model.train()
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    optimizer.step()
                    tokens_seen += input_batch.numel()
                    global_step += 1

                    # 定时执行模型评估
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        print(f"轮次 {epoch+1} (迭代 {global_step})："
                              f"训练损失 {train_loss:.3f}，验证损失 {val_loss:.3f}")

                    # 定时生成一段文本样例，直观查看模型生成效果
                    if global_step % print_sample_iter == 0:
                        generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )

                # 到达保存步数阈值，存储模型断点
                if global_step % save_ckpt_freq == 0:
                    file_name = output_dir / f"model_pg_{global_step}.pth"
                    torch.save(model.state_dict(), file_name)
                    print(f"已保存断点文件：{file_name}")

                # 打印当前训练耗时与预估剩余时间
                print_eta(start_time, book_start_time, index, total_files)

    except KeyboardInterrupt:
        # 捕获Ctrl+C中断，保存临时断点防止训练进度丢失
        file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
        torch.save(model.state_dict(), file_name)
        print(f"训练手动中断，已保存临时断点：{file_name}")

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GPT模型预训练参数配置工具')

    parser.add_argument('--data_dir', type=str, default='gutenberg/data',
                        help='存放训练文本数据集的目录')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                        help='模型断点权重保存目录')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='整体训练轮数')
    parser.add_argument('--print_sample_iter', type=int, default=1000,
                        help='每间隔多少迭代输出一段文本生成样例')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='训练过程中损失评估的迭代间隔')
    parser.add_argument('--save_ckpt_freq', type=int, default=100_000,
                        help='模型断点保存的迭代间隔')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='优化器初始学习率')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='训练批次大小')
    parser.add_argument('--debug', type=bool, default=False,
                        help='开启调试模式：使用极小模型快速验证代码流程')

    args = parser.parse_args()

    if args.debug:
        # 调试用极简GPT模型配置
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # 词表总容量
            "context_length": 10,    # 上下文窗口长度
            "emb_dim": 12,           # 词嵌入向量维度
            "n_heads": 2,            # 多头注意力头数量
            "n_layers": 2,           # Transformer层数
            "drop_rate": 0.0,        # Dropout概率，大模型一般不推荐启用，此处直接关闭
            "qkv_bias": False        # QKV线性层是否启用偏置项
        }

    else:
        # 标准124M参数量GPT2模型配置
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # 词表总容量
            "context_length": 1024,  # 上下文窗口长度
            "emb_dim": 768,          # 词嵌入向量维度
            "n_heads": 12,           # 多头注意力头数量
            "n_layers": 12,          # Transformer层数
            "drop_rate": 0.1,        # Dropout随机失活概率
            "qkv_bias": False        # QKV线性层是否启用偏置项
        }

    # 自动选择运行设备：有CUDA显卡则使用GPU，无则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")

    data_dir = args.data_dir
    # 遍历目录收集所有txt格式训练文本
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)

    if total_files == 0:
        print("未检测到训练文本文件，请确认输入目录路径是否正确")
        quit()
    print("待训练文本总数量：", total_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 启动完整预训练流程
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, optimizer, device,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        print_sample_iter=args.print_sample_iter,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    # 绘制损失变化曲线图并保存
    epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, output_dir)

    # 保存训练完成后的最终完整模型权重
    torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    print(f"训练全程GPU最大显存占用：{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")