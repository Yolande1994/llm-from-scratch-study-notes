# 版权所有 (c) Sebastian Raschka，遵循 Apache 2.0 开源协议（详见 LICENSE.txt）。
# 《从零搭建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch

import os
import sys
import tarfile
import time
import urllib.request
import pandas as pd


def reporthook(count, block_size, total_size):
    # 下载进度回调函数，实时打印下载进度、速度、耗时
    global start_time
    if count == 0:
        # 第一次调用，记录下载起始时间
        start_time = time.time()
    else:
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        # 计算当前下载百分比
        percent = count * block_size * 100 / total_size

        # 计算下载速度（单位 KB/s），避免除零
        speed = int(progress_size / (1024 * duration)) if duration else 0
        # 覆盖式打印进度信息
        sys.stdout.write(
            f"\r{int(percent)}% | {progress_size / (1024**2):.2f} MB "
            f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
        )
        # 强制刷新输出缓冲区，立即显示进度
        sys.stdout.flush()


def download_and_extract_dataset(dataset_url, target_file, directory):
    # 目标数据集文件夹不存在时，执行下载和解压
    if not os.path.exists(directory):
        # 若旧压缩包残留，先删除
        if os.path.exists(target_file):
            os.remove(target_file)
        # 带进度条下载数据集压缩包
        urllib.request.urlretrieve(dataset_url, target_file, reporthook)
        print("\nExtracting dataset ...")
        # 解压 gz 格式 tar 压缩包到当前目录
        with tarfile.open(target_file, "r:gz") as tar:
            tar.extractall()
    else:
        # 数据集目录已存在，跳过下载流程
        print(f"目录 `{directory}` 已存在，跳过下载步骤。")


def load_dataset_to_dataframe(basepath="aclImdb", labels={"pos": 1, "neg": 0}):
    data_frames = []  # 列表缓存所有单文件生成的子DataFrame
    # 遍历训练集、测试集两个子集
    for subset in ("test", "train"):
        # 遍历正向、负向情感标签文件夹
        for label in ("pos", "neg"):
            path = os.path.join(basepath, subset, label)
            # 按文件名排序遍历所有评论文本文件
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                    # 单条文本生成一行DataFrame，存入缓存列表
                    data_frames.append(pd.DataFrame({"text": [infile.read()], "label": [labels[label]]}))
    # 拼接所有子DataFrame，重置全局行索引
    df = pd.concat(data_frames, ignore_index=True)
    # 随机打乱全部样本，固定随机种子保证复现
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    return df


def partition_and_save(df, sizes=(35000, 5000, 10000)):
    # 再次打乱全量数据集
    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算数据集切分边界下标
    train_end = sizes[0]
    val_end = sizes[0] + sizes[1]

    # 按下标切分训练、验证、测试集
    train = df_shuffled.iloc[:train_end]
    val = df_shuffled.iloc[train_end:val_end]
    test = df_shuffled.iloc[val_end:]

    # 分别保存为CSV文件，不导出行索引
    train.to_csv("train.csv", index=False)
    val.to_csv("validation.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":
    # IMDB电影评论情感数据集下载地址
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    print("Downloading dataset ...")
    download_and_extract_dataset(dataset_url, "aclImdb_v1.tar.gz", "aclImdb")
    print("Creating data frames ...")
    df = load_dataset_to_dataframe()
    print("Partitioning and saving data frames ...")
    partition_and_save(df)