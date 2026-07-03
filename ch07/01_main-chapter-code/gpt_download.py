# 版权所有 (c) Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt）
# 《从零构建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 项目代码仓库：https://github.com/rasbt/LLMs-from-scratch

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
        raise ValueError(f"模型尺寸仅支持：{allowed_sizes}")

    # 定义本地存储路径
    model_dir = os.path.join(models_dir, model_size)
    # OpenAI官方GPT2权重下载主地址
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    # 备用镜像下载地址（主地址访问失败时使用）
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    # GPT2模型全套文件清单
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 批量下载所有模型文件
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # 读取模型超参配置，并将TensorFlow权重转为numpy参数字典
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    # 内部下载执行函数，接收下载链接尝试拉取文件
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # 从响应头读取文件总大小，无此字段则默认0
            file_size = int(response.headers.get("Content-Length", 0))

            # 判断本地是否已存在同大小文件，存在则跳过下载
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"文件已存在且完整，无需重复下载：{destination}")
                    return True  # 返回True代表无需下载，执行成功

            block_size = 1024  # 单次读写块大小：1KB

            # 初始化进度条，显示当前下载文件名
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    # 循环分块读取并写入本地
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    # 优先尝试主下载链接
    try:
        if _attempt_download(url):
            return
    # 主链接网络异常、404等错误时切换备用地址
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"主链接({url})下载失败，正在尝试备用镜像：{backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # 主、备链接均下载失败，打印报错信息
        error_message = (
            f"主链接 {url} 下载失败"
            f"{f'，备用链接 {backup_url} 同样失败' if backup_url else ''}。"
            "\n请检查网络连接或文件可用性\n"
            "问题求助地址：https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    # 捕获其他未知异常
    except Exception as e:
        print(f"下载过程出现未知错误：{e}")


# 下方为使用requests库实现的备用下载方案（未启用）
"""
def download_file(url, destination):
    # 流式GET请求下载文件，避免一次性加载全部内容到内存
    response = requests.get(url, stream=True)

    # 从响应头读取文件总大小，无此字段则默认0
    file_size = int(response.headers.get("content-length", 0))

    # 判断本地是否存在完整文件，存在则跳过下载
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"文件已存在且完整，无需重复下载：{destination}")
            return

    block_size = 1024  # 单次读写块大小：1KB

    # 从URL截取文件名作为进度条标题
    progress_bar_description = url.split("/")[-1]
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 二进制写入模式打开本地文件
        with open(destination, "wb") as file:
            # 分块迭代读取响应流并写入本地
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 将分块数据写入文件
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 初始化参数字典，为每一层Transformer创建独立空字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历TensorFlow断点文件里所有权重变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 读取权重并压缩去除多余单维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 分割权重名称，丢弃前缀'model/'
        variable_name_parts = name.split("/")[1:]

        # 定位该权重所属的存储字典
        target_dict = params
        # 以h开头代表某一层Transformer
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 逐层递归创建嵌套字典，直到倒数第二层
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 将权重数组赋值给最后一级键名
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params