# 版权所有 © Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt）
# 配套书籍《从零构建大模型》源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 官方代码仓库：https://github.com/rasbt/LLMs-from-scratch

# 内部使用文件：单元测试脚本

import pytest
from gpt_train import main
import http.client
from urllib.parse import urlparse


@pytest.fixture
def gpt_config():
    """测试专用轻量化GPT模型超参夹具，缩小尺寸提升测试运行速度"""
    return {
        "vocab_size": 50257,
        "context_length": 12,  # 上下文窗口缩小，加快测试
        "emb_dim": 32,         # 嵌入维度缩小，加快测试
        "n_heads": 4,          # 注意力头数量缩小，加快测试
        "n_layers": 2,         # Transformer层数缩小，加快测试
        "drop_rate": 0.1,
        "qkv_bias": False
    }


@pytest.fixture
def other_settings():
    """训练配套超参夹具，使用小轮数缩短测试耗时"""
    return {
        "learning_rate": 5e-4,
        "num_epochs": 1,    # 仅训练1轮，提升测试效率
        "batch_size": 2,
        "weight_decay": 0.1
    }


def test_main(gpt_config, other_settings):
    """完整训练流程单元测试，校验损失与token统计输出长度是否符合预期"""
    train_losses, val_losses, tokens_seen, model = main(gpt_config, other_settings)

    assert len(train_losses) == 39, "训练损失列表长度与预期不符"
    assert len(val_losses) == 39, "验证损失列表长度与预期不符"
    assert len(tokens_seen) == 39, "已处理token统计列表长度与预期不符"


def check_file_size(url, expected_size):
    """
    发送HEAD请求校验远端文件大小
    :param url: 待检测文件地址
    :param expected_size: 预期文件字节大小
    :return: (校验是否通过, 提示信息)
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme == "https":
        conn = http.client.HTTPSConnection(parsed_url.netloc)
    else:
        conn = http.client.HTTPConnection(parsed_url.netloc)

    conn.request("HEAD", parsed_url.path)
    response = conn.getresponse()
    if response.status != 200:
        return False, f"文件链接 {url} 无法访问"
    size = response.getheader("Content-Length")
    if size is None:
        return False, "响应头缺失文件长度字段 Content-Length"
    size = int(size)
    if size != expected_size:
        return False, f"{url} 文件预期大小 {expected_size} 字节，实际获取为 {size} 字节"
    return True, f"{url} 文件大小校验正确"


def test_model_files():
    """校验GPT2官方主、备用双源的各版本权重文件完整性与体积"""
    def check_model_files(base_url):
        # 124M 小参数量模型文件及对应标准字节大小
        model_size = "124M"
        files = {
            "checkpoint": 77,
            "encoder.json": 1042301,
            "hparams.json": 90,
            "model.ckpt.data-00000-of-00001": 497759232,
            "model.ckpt.index": 5215,
            "model.ckpt.meta": 471155,
            "vocab.bpe": 456318
        }

        for file_name, expected_size in files.items():
            url = f"{base_url}/{model_size}/{file_name}"
            valid, message = check_file_size(url, expected_size)
            assert valid, message

        # 355M 中参数量模型文件及对应标准字节大小
        model_size = "355M"
        files = {
            "checkpoint": 77,
            "encoder.json": 1042301,
            "hparams.json": 91,
            "model.ckpt.data-00000-of-00001": 1419292672,
            "model.ckpt.index": 10399,
            "model.ckpt.meta": 926519,
            "vocab.bpe": 456318
        }

        for file_name, expected_size in files.items():
            url = f"{base_url}/{model_size}/{file_name}"
            valid, message = check_file_size(url, expected_size)
            assert valid, message

    # 分别校验官方主下载源、备用镜像下载源
    check_model_files(base_url="https://openaipublic.blob.core.windows.net/gpt-2/models")
    check_model_files(base_url="https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2")