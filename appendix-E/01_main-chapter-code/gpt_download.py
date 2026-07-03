# 版权所有 © Sebastian Raschka，基于 Apache License 2.0 许可协议（详见 LICENSE.txt）
# 对应书籍：《从零构建大模型》(Build a Large Language Model From Scratch)
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 官方源码仓库：https://github.com/rasbt/LLMs-from-scratch


import os
import urllib.request

# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # 校验模型尺寸是否合法
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # 定义文件路径
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载模型文件
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # 加载模型配置与参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # 从响应头获取文件总大小，若头信息不存在则默认取 0
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查本地文件是否已存在且大小一致
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  # 文件完整无需重新下载，返回成功标识

            block_size = 1024  # 单次读取块大小：1KB

            # 根据文件总大小初始化进度条
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # 执行到此处说明主、备地址均下载失败
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# 使用 requests 库的备用实现方案
"""
def download_file(url, destination):
    # 发送 GET 请求，以流式方式下载文件
    response = requests.get(url, stream=True)

    # 从响应头获取文件总大小，若头信息不存在则默认取 0
    file_size = int(response.headers.get("content-length", 0))

    # 检查本地文件是否已存在且大小一致
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # 定义文件读取的块大小
    block_size = 1024  # 单次读取块大小：1KB

    # 根据文件总大小初始化进度条
    progress_bar_description = url.split("/")[-1]  # 从 URL 中提取文件名
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 以二进制写入模式打开目标文件
        with open(destination, "wb") as file:
            # 分块迭代读取文件数据
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 将数据块写入文件
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 初始化参数字典，为每一层预留空的块结构
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点文件中的所有变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除多余的单维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名，提取层级结构信息
        variable_name_parts = name.split("/")[1:]  # 跳过最外层的 'model/' 前缀

        # 定位该变量对应的目标字典位置
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 递归访问或创建嵌套字典结构
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 将变量数组赋值给最内层的键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params