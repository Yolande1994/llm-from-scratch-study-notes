# 版权所有 (c) Sebastian Raschka，基于 Apache License 2.0 协议（详见 LICENSE.txt）
# 出自《从零构建大模型》一书
#   - 图书主页：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 源码仓库：https://github.com/rasbt/LLMs-from-scratch

# 附录 A：PyTorch 入门（第三部分）

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 新增导入项
import os
import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# 新增：初始化分布式进程组的函数（每张 GPU 对应一个进程）
# 用于实现多进程之间的通信
def ddp_setup(rank, world_size):
    """
    参数说明：
        rank: 进程唯一标识
        world_size: 进程组内的总进程数
    """
    # 运行 rank=0 进程的主机地址
    # 本示例假设所有 GPU 都在同一台机器上
    os.environ["MASTER_ADDR"] = "localhost"
    # 本机上任意一个空闲端口
    os.environ["MASTER_PORT"] = "12345"
    if platform.system() == "Windows":
        # 禁用 libuv，Windows 版 PyTorch 不支持该组件
        os.environ["USE_LIBUV"] = "0"

    # 初始化进程组
    if platform.system() == "Windows":
        # Windows 系统需使用 gloo 作为通信后端，无法使用 nccl
        # gloo：Facebook 开源的集合通信库
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl：NVIDIA 集合通信库
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 第一个隐藏层
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 第二个隐藏层
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 输出层
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # 新增：设置为 False，后续由 DistributedSampler 负责打乱顺序
        pin_memory=True,
        drop_last=True,
        # 新增：在多张 GPU 间拆分批次，保证样本不重复
        sampler=DistributedSampler(train_ds)  # 新增
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# 新增：主函数包装器
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # 新增：初始化进程组

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # 新增：用 DDP 包装模型
    # 包装后，原始模型可通过 model.module 访问

    for epoch in range(num_epochs):
        # 新增：设置采样器的轮次，保证每个轮次打乱顺序不同
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:

            features, labels = features.to(rank), labels.to(rank)  # 新增：数据迁移到对应 GPU
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # 损失函数

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 日志输出
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)

    destroy_process_group()  # 新增：安全退出分布式模式


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())

    torch.manual_seed(123)

    # 新增：派生新进程
    # 注意：spawn 方法会自动传入 rank 参数
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size 表示每张 GPU 启动一个进程