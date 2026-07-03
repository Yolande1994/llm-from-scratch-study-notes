# 版权所有 © Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt）
# 配套书籍《从零构建大模型》源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 官方代码仓库：https://github.com/rasbt/LLMs-from-scratch

# 内部使用文件：单元测试脚本

from pathlib import Path
import os
import subprocess


def test_pretraining():
    """预训练流程单元测试"""
    # 基础文本片段
    sequence = "a b c d"
    # 重复次数
    repetitions = 1000
    # 拼接生成测试文本
    content = sequence * repetitions

    # 测试数据存放目录
    folder_path = Path("gutenberg") / "data"
    file_name = "repeated_sequence.txt"

    # 自动创建目录，不存在则新建
    os.makedirs(folder_path, exist_ok=True)

    # 写入测试文本文件
    with open(folder_path/file_name, "w") as file:
        file.write(content)

    # 执行调试模式下的预训练脚本
    result = subprocess.run(
        ["python", "pretraining_simple.py", "--debug", "true"],
        capture_output=True, text=True
    )
    # 打印脚本标准输出日志
    print(result.stdout)
    # 断言校验：日志中必须出现显存占用输出，代表训练流程完整跑完无报错
    assert "Maximum GPU memory allocated" in result.stdout