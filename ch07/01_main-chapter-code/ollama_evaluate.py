# 版权所有 (c) Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt）
# 《从零构建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch
#
# 基于第7章代码实现的极简指令微调辅助脚本

import json
import psutil
from tqdm import tqdm
import urllib.request


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # 构造请求体字典数据
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # 下方参数用于保证模型输出结果可复现
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # 将字典转为JSON字符串并编码为字节流
    payload = json.dumps(data).encode("utf-8")

    # 构建POST请求对象，添加必要请求头
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # 发送请求并接收返回结果
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 循环读取并解码流式返回内容
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def check_if_running(process_name):
    # 检查指定进程是否正在运行
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def format_input(entry):
    # 构造标准指令微调提示词模板
    instruction_text = (
        f"下面是一条描述任务的指令，请写出能恰当完成该任务的回答。"
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 若存在输入上下文，则拼接Input字段
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def main(file_path):
    # 检测Ollama服务进程是否启动
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("未检测到Ollama运行，请先启动Ollama服务后再执行脚本。")
    print("Ollama 服务运行状态：", check_if_running("ollama"))

    # 读取测试数据集JSON文件
    with open(file_path, "r") as file:
        test_data = json.load(file)

    model = "llama3"
    # 调用打分函数，使用LLama3为模型生成结果打分
    scores = generate_model_scores(test_data, "model_response", model)
    print(f"有效打分数量：{len(scores)} / 总样本数 {len(test_data)}")
    print(f"平均分：{sum(scores)/len(scores):.2f}\n")


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    # 遍历所有样本，进度条显示打分进度
    for entry in tqdm(json_data, desc="样本打分中"):
        # 模型响应为空则直接记0分
        if entry[json_key] == "":
            scores.append(0)
        else:
            # 构造打分提示词，让大模型对输出质量进行0-100分评分
            prompt = (
                f"给定输入 `{format_input(entry)}` "
                f"以及标准答案 `{entry['output']}`，"
                f"请为模型输出 `{entry[json_key]}` 打分，"
                f"分值区间0~100，100分为最优，仅返回整数数字即可。"
            )
            score = query_model(prompt, model)
            try:
                # 将返回的分数转为整数存入列表
                scores.append(int(score))
            except ValueError:
                # 无法转为数字时打印报错并跳过该样本
                print(f"分数转换失败，原始返回内容：{score}")
                continue

    return scores


if __name__ == "__main__":

    import argparse

    # 命令行参数解析器
    parser = argparse.ArgumentParser(
        description="基于Ollama大模型自动评估模型输出结果"
    )
    parser.add_argument(
        "--file_path",
        required=True,
        help=(
            "测试数据集JSON文件路径，文件内必须包含 "
            "`output`（标准答案）和`model_response`（模型输出）字段"
        )
    )
    args = parser.parse_args()

    main(file_path=args.file_path)