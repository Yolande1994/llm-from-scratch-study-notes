# 版权所有 (c) Sebastian Raschka，遵循 Apache 2.0 开源协议（详见 LICENSE.txt）。
# 《从零搭建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch

import os
import urllib.request

# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # 校验传入的模型尺寸是否合法
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"模型尺寸不在合法列表 {allowed_sizes} 内")

    # 定义文件存储路径
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 批量下载GPT2配套权重、词表、配置文件
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # 读取模型超参配置，从TensorFlow检查点加载权重参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    # 内部函数：执行单次下载逻辑
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # 从响应头获取文件总大小，无该字段则默认0
            file_size = int(response.headers.get("Content-Length", 0))

            # 判断本地是否已存在同名完整文件，存在则跳过下载
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"文件已存在且完整，无需重新下载: {destination}")
                    return True  # 返回True代表无需下载，任务完成

            block_size = 1024  # 单次读取缓冲区大小：1KB

            # 初始化下载进度条，文件名作为进度条标题
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    # 分块循环写入本地文件
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        # 优先尝试主下载地址
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        # 主地址下载失败，切换备用镜像地址重试
        if backup_url is not None:
            print(f"主地址({url})下载失败，尝试备用镜像：{backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # 主地址、备用地址均下载失败，输出错误提示
        error_message = (
            f"主地址({url})"
            f"{' 和备用地址(' + backup_url + ')' if backup_url else ''} 全部下载失败。"
            "\n请检查网络连接或文件资源是否可用。\n"
            "如需帮助，请访问讨论区：https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"下载过程出现未知异常：{e}")


# 使用requests库实现的备选下载方案（注释备用）
"""
def download_file(url, destination):
    # 以流式请求方式下载文件，避免一次性加载全部内容至内存
    response = requests.get(url, stream=True)

    # 从响应头获取文件总大小，无该字段则默认0
    file_size = int(response.headers.get("content-length", 0))

    # 判断本地是否已存在同名完整文件，存在则跳过下载
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"文件已存在且完整，无需重新下载: {destination}")
            return

    # 单次读取缓冲区大小：1KB
    block_size = 1024

    # 从URL中截取文件名作为进度条标题
    progress_bar_description = url.split("/")[-1]
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 二进制写入模式打开本地目标文件
        with open(destination, "wb") as file:
            # 迭代读取文件分块并写入本地
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 将分块写入本地文件
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 初始化参数字典，为每一层Transformer创建空字典存储权重
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历TensorFlow检查点内所有权重变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载权重数组，并去除多余的单维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 拆分变量名，丢弃最开头的'model/'前缀
        variable_name_parts = name.split("/")[1:]

        # 定位该权重对应的存储字典
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 逐层递归创建/进入嵌套字典
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 将权重数组赋值给最终键名
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params