# 版权所有 © Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt 文件）
# 配套书籍《从零构建大模型》(Build a Large Language Model From Scratch) 源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 官方代码仓库：https://github.com/rasbt/LLMs-from-scratch

"""
本脚本用于处理古登堡项目海量零散文本，合并为少量体积更大的完整文件。
"""

import argparse
import os
import re
from tqdm import tqdm
from gutenberg.src.cleanup import strip_headers


def is_english(text, threshold=0.9):
    """判断文本是否以英文为主：统计ASCII字符占比，超过阈值则判定为英文文本"""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > threshold


def combine_files(file_paths, target_dir, max_size_mb=500, separator="<|endoftext|>", fallback_encoding="latin1"):
    """
    批量合并零散文本文件，按单文件最大容量分块存储
    :param file_paths: 待合并的文件路径列表
    :param target_dir: 合并后文件的输出目录
    :param max_size_mb: 单个合并文件最大体积，单位MB
    :param separator: 不同书籍文本之间的分隔符
    :param fallback_encoding: 读取文件失败时备用编码
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    current_content = []
    current_size = 0
    file_counter = 1

    for file_path in tqdm(file_paths):
        try:
            # 优先以utf-8编码读取文本
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # 读取编码异常时，切换备用编码重试
            tqdm.write(f"警告：文件编码解码失败，使用备用编码读取 {file_path}")
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()

        # 过滤非英文文本文件
        if not is_english(content):
            tqdm.write(f"跳过 {file_path}，该文件主要内容非英文")
            continue
        # 清除书籍头部版权、元数据等无关头部信息
        content = strip_headers(content)

        # 正则表达式：将多处连续空行统一替换为单个空行
        content = re.sub(r'\n\s*\n', '\n\n', content)
        # 估算当前文本utf-8编码后的字节大小
        estimated_size = len(content.encode("utf-8"))

        # 若新增文本后超出单文件体积上限，则先写入当前缓存内容，新建文件
        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(separator.join(current_content))
            file_counter += 1
            current_content = [content]
            current_size = estimated_size
        else:
            # 未达体积上限，将文本加入缓存等待合并
            current_content.append(content)
            current_size += estimated_size

    # 写入最后一批未保存的缓存文本
    if current_content:
        target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            target_file.write(separator.join(current_content))
    return file_counter


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="对预训练文本做预处理并批量合并文件")

    parser.add_argument("--data_dir", type=str, default="gutenberg/data/raw",
                        help="存放原始下载训练文本的目录")
    parser.add_argument("--max_size_mb", type=int, default=500,
                        help="单个合并输出文件的最大体积，单位MB")
    parser.add_argument("--output_dir", type=str, default="gutenberg_preprocessed",
                        help="预处理合并后文件的输出目录")

    args = parser.parse_args()

    # 遍历目录，收集所有txt、txt.utf8格式文本文件
    all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.data_dir)
                 for name in files if name.endswith((".txt", ".txt.utf8"))]

    print(f"待处理文件总数：{len(all_files)}")
    file_counter = combine_files(all_files, args.output_dir, max_size_mb=args.max_size_mb)
    print(f"处理完成，共生成 {file_counter} 个文件，存放路径：{os.path.abspath(args.output_dir)}")