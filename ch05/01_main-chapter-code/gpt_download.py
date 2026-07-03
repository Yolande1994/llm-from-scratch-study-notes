# 版权所有 (c) Sebastian Raschka，基于 Apache License 2.0 开源协议
# 本书配套代码：《从零构建大语言模型》
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
#   - 官方代码仓库：https://github.com/rasbt/LLMs-from-scratch


import os
import urllib.request

# 可选：使用 requests 库下载（需要额外安装 pip install requests）
# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # 验证输入的模型大小是否合法
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"模型大小必须是以下之一：{allowed_sizes}")

    # 定义文件路径和下载地址
    model_dir = os.path.join(models_dir, model_size)
    # OpenAI 官方主下载地址
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    # 作者提供的备用下载地址（国内访问更稳定）
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    # GPT-2 模型需要下载的所有文件列表
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 批量下载所有必要文件
    os.makedirs(model_dir, exist_ok=True)  # 自动创建文件夹，已存在则不报错
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # 加载模型配置和权重参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的检查点文件路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载模型超参数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 转换TensorFlow权重为PyTorch格式

    return settings, params


def download_file(url, destination, backup_url=None):
    """
    下载单个文件的工具函数，支持断点续传和备用地址
    :param url: 主下载地址
    :param destination: 本地保存路径
    :param backup_url: 备用下载地址（可选）
    """
    def _attempt_download(download_url):
        """内部函数：尝试从指定地址下载文件"""
        with urllib.request.urlopen(download_url) as response:
            # 从响应头获取文件总大小，若服务器未提供则默认0
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查本地文件是否已存在且完整（大小一致）
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"文件已存在且完整，跳过下载：{destination}")
                    return True  # 无需重新下载，直接返回成功

            block_size = 1024  # 每次读取的块大小，单位：字节（1KB）

            # 初始化进度条，显示下载进度
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break  # 读取完毕，退出循环
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # 更新进度条
            return True

    # 首先尝试从主地址下载
    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        # 主地址失败时，尝试备用地址
        if backup_url is not None:
            print(f"主地址下载失败：{url}，正在尝试备用地址：{backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # 如果执行到这里，说明两个地址都下载失败了
        error_message = (
            f"下载失败！主地址：{url}"
            f"{f'，备用地址：{backup_url}' if backup_url else ''} 均无法访问。"
            "\n请检查你的网络连接，或手动下载文件。"
            "\n获取帮助：https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"发生未知错误：{e}")


# 使用 requests 库的替代下载方法（速度更快，稳定性更好）
"""
def download_file(url, destination):
    # 发送GET请求，以流模式下载文件（避免一次性加载大文件到内存）
    response = requests.get(url, stream=True)

    # 从响应头获取文件总大小，若服务器未提供则默认0
    file_size = int(response.headers.get("content-length", 0))

    # 检查本地文件是否已存在且完整
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"文件已存在且完整，跳过下载：{destination}")
            return

    # 定义文件读取的块大小
    block_size = 1024  # 1 Kilobyte

    # 初始化进度条
    progress_bar_description = url.split("/")[-1]  # 从URL中提取文件名作为进度条描述
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 以二进制写入模式打开目标文件
        with open(destination, "wb") as file:
            # 分块迭代读取文件数据
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 将数据块写入本地文件
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    将TensorFlow格式的GPT-2检查点文件转换为原书使用的PyTorch字典格式
    :param ckpt_path: TensorFlow检查点文件路径
    :param settings: 模型超参数字典
    :return: 转换后的PyTorch权重参数字典
    """
    # 初始化参数字典，为每一层Transformer块创建空字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的所有变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量值，并移除所有单维度（压缩维度）
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名，提取有用部分，跳过最外层的'model/'前缀
        variable_name_parts = name.split("/")[1:]

        # 确定当前变量应该存入参数字典的哪个位置
        target_dict = params
        # 如果是Transformer层的变量，定位到对应的层
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 递归访问或创建嵌套字典结构
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 将变量值赋值给最后一个键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params