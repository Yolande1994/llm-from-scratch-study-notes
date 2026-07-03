# 版权所有 (c) Sebastian Raschka，基于 Apache License 2.0 开源协议
# 本书配套代码：《从零构建大语言模型》
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
#   - 官方代码仓库：https://github.com/rasbt/LLMs-from-scratch

import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken

# 从本地文件导入之前章节实现的函数
from previous_chapters import GPTModel, create_dataloader_v1, generate_text_simple

def text_to_token_ids(text, tokenizer):
    """将文本转换为模型可输入的token ID张量"""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度，适配模型输入格式
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将模型输出的token ID张量转换回人类可读的文本"""
    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个批次的交叉熵损失"""
    # 将数据移动到指定设备（GPU/CPU）
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 前向传播得到logits
    logits = model(input_batch)
    # 展平logits和目标标签，计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算整个数据加载器的平均损失（用于评估）"""
    total_loss = 0.
    # 数据加载器为空时返回NaN
    if len(data_loader) == 0:
        return float("nan")
    # 如果未指定批次数量，则计算所有批次
    elif num_batches is None:
        num_batches = len(data_loader)
    # 否则取指定数量和总批次的较小值，防止越界
    else:
        num_batches = min(num_batches, len(data_loader))

    # 遍历指定数量的批次计算平均损失
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型在训练集和验证集上的平均损失"""
    # 将模型切换为评估模式（关闭Dropout等训练专用层）
    model.eval()
    # 禁用梯度计算，加速评估并节省显存
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    # 评估完成后切回训练模式
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """生成一段文本并打印，用于直观监控训练效果"""
    model.eval()
    # 获取模型支持的最大上下文长度
    context_size = model.pos_emb.weight.shape[0]
    # 将起始文本转换为token ID并移动到指定设备
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        # 调用简单生成函数生成50个新token
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        # 将生成的token ID转换回文本
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # 紧凑打印格式，将换行符替换为空格
        print(decoded_text.replace("\n", " "))

    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """简化版GPT模型训练主函数"""
    # 初始化列表，用于跟踪训练过程中的损失和已训练token数，方便后续绘图
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0  # 累计已训练的token总数
    global_step = -1  # 全局步数计数器（已完成的权重更新次数）

    # 主训练循环：遍历所有训练轮数
    for epoch in range(num_epochs):
        model.train()  # 将模型切换为训练模式

        # 遍历训练集中的每个批次
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清空上一个批次的梯度，防止梯度累加
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次损失
            loss.backward()  # 反向传播，计算所有权重的梯度
            optimizer.step()  # 根据梯度更新模型权重
            tokens_seen += input_batch.numel()  # 累加当前批次的token数量
            global_step += 1  # 全局步数加1

            # 可选评估步骤：每隔eval_freq步评估一次模型
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                # 记录损失和已训练token数
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 打印当前训练状态
                print(f"轮数 {epoch + 1} (步数 {global_step:06d}): "
                      f"训练损失 {train_loss:.3f}, 验证损失 {val_loss:.3f}")

        # 每轮训练结束后，生成一段文本样本，直观展示训练效果
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    # 返回完整的训练日志
    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制训练损失和验证损失曲线，支持双x轴（轮数和已训练token数）"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制基于轮数的损失曲线
    ax1.plot(epochs_seen, train_losses, label="训练损失", linewidth=2)
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="验证损失", linewidth=2)
    ax1.set_xlabel("训练轮数", fontsize=12)
    ax1.set_ylabel("交叉熵损失", fontsize=12)
    ax1.legend(loc="upper right", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 创建第二个x轴，显示已训练的token数
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 不可见的对齐曲线
    ax2.set_xlabel("已训练token总数", fontsize=12)

    fig.tight_layout()  # 自动调整布局，防止标签被截断


def main(gpt_config, settings):
    """训练流程主函数，整合所有步骤"""

    torch.manual_seed(123)  # 设置随机种子，保证实验可复现
    # 自动选择设备：优先使用GPU，没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # 步骤1：下载并加载训练数据
    ##############################

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    # 如果本地没有数据文件，则从GitHub下载
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    # 否则直接加载本地文件
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # 步骤2：初始化模型和优化器
    ##############################

    # 根据配置初始化GPT模型
    model = GPTModel(gpt_config)
    model.to(device)  # 将模型移动到指定设备
    # 使用AdamW优化器，是训练Transformer的标准选择
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # 步骤3：创建训练集和验证集数据加载器
    ##############################

    train_ratio = 0.90  # 训练集占总数据的90%，验证集占10%
    split_idx = int(train_ratio * len(text_data))  # 计算分割点

    # 创建训练集数据加载器
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,  # 丢弃最后一个不完整的批次
        shuffle=True,  # 打乱训练数据顺序
        num_workers=0  # Windows系统建议设为0，避免多进程问题
    )

    # 创建验证集数据加载器
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,  # 保留验证集的所有数据
        shuffle=False,  # 验证集不需要打乱
        num_workers=0
    )

    ##############################
    # 步骤4：开始训练模型
    ##############################

    # 使用GPT-2官方分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 调用训练主函数
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    # GPT-2 124M模型配置（上下文长度缩短为256，适配小数据集训练）
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # 词表大小
        "context_length": 256,  # 上下文长度（原官方为1024，这里为了小数据集训练缩短）
        "emb_dim": 768,  # 嵌入维度
        "n_heads": 12,  # 注意力头数
        "n_layers": 12,  # Transformer层数
        "drop_rate": 0.1,  # Dropout概率
        "qkv_bias": False  # QKV线性层是否使用偏置
    }

    # 训练超参数配置
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,  # 学习率
        "num_epochs": 10,  # 训练总轮数
        "batch_size": 2,  # 批次大小
        "weight_decay": 0.1  # 权重衰减系数，防止过拟合
    }

    ###########################
    # 启动训练流程
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    ###########################
    # 训练后处理
    ###########################

    # 绘制损失曲线并保存为PDF文件
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf", dpi=300, bbox_inches="tight")

    # 保存训练好的模型权重
    torch.save(model.state_dict(), "model.pth")
    # 演示如何加载保存的模型权重
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", weights_only=True))