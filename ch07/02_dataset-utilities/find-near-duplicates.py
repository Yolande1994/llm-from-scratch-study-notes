# 版权所有 (c) Sebastian Raschka，遵循 Apache License 2.0 开源协议（详见 LICENSE.txt）
# 《从零构建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 项目代码仓库：https://github.com/rasbt/LLMs-from-scratch

import argparse
import json
import re
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 示例JSON数据集
example_data = [
    {"instruction": "What is the capital of Italy?",
     "input": "", "output": "The capital of Italy is Rome."
     },
    {"instruction": "What's the capital city of Italy?",
     "input": "", "output": "The capital city is Rome."
     },
    {"instruction": "Identify the main verb in the sentence: 'The cat sleeps on the couch.'",
     "input": "", "output": "The verb is 'sleeps'."
     },
    {"instruction": "Identify the verb in the following sentence: The cat sleeps on the couch.",
     "input": "", "output": "The verb in the sentence is \"sleeps.\""
     },
    # ...
]


def preprocess_text(text):
    # 将文本全部转为小写
    text = text.lower()
    # 移除所有标点符号
    text = re.sub(r'[^\w\s]', '', text)
    return text


def find_near_duplicates(json_data, threshold=0.75, key="instruction"):
    """相似度阈值越高，两段文本需要达到越高相似度才会被判定为近似重复"""

    # 提取指定字段的文本内容
    text = [preprocess_text(item[key]) for item in json_data if item[key]]
    near_duplicates = []
    indices_to_remove = set()

    # 若无有效文本，直接返回空结果
    if not text:
        return {}, near_duplicates

    # 对文本做TF-IDF向量化
    vectorizer = TfidfVectorizer(stop_words=None, analyzer='char', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(text)

    # 计算所有文本两两之间的余弦相似度
    cos_sim_matrix = cosine_similarity(tfidf_matrix)

    # 根据阈值筛选近似重复的文本对

    for i in range(len(cos_sim_matrix)):
        for j in range(i+1, len(cos_sim_matrix)):
            # 相似度超过阈值则判定为近似重复
            if cos_sim_matrix[i, j] > threshold:
                # 跳过内容长度不足1的无效文本
                if len(json_data[i][key]) <= 1 or len(json_data[j][key]) <= 1:
                    continue
                # 记录重复样本对与对应相似度分数
                near_duplicates.append((json_data[i], json_data[j], cos_sim_matrix[i, j]))
                # 仅针对input/output字段标记待删除条目，不根据instruction删除样本
                if key in ("input", "output"):
                    indices_to_remove.add(j)

    # 过滤掉被标记为重复的样本
    filtered_json_data = [item for index, item in enumerate(json_data) if index not in indices_to_remove]

    return filtered_json_data, near_duplicates


def find_print_and_remove_near_duplicates(json_data, remove_duplicates=False, threshold=0.75):
    """
    遍历JSON每条数据的所有字段，查找整份数据集中的近似重复内容
    找到重复样本后打印输出
    """
    # 遍历第一条数据里所有字段名
    for key in json_data[0].keys():

        if remove_duplicates:
            # 开启去重：返回过滤后的数据集与重复样本列表
            json_data, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        else:
            # 仅检索不删除：只获取重复样本列表，数据集保持原样
            _, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        separator = 50 * '='
        print(f"\n\n{separator}\n正在检索「{key}」字段的近似重复内容...\n{separator}")
        if not near_duplicates:
            print("未找到任何近似重复样本")
        else:
            # 循环打印每一组重复样本及相似度
            for dup in near_duplicates:
                print(
                    f"找到一组近似重复文本，相似度：{dup[2]:.2f}\n"
                    f"1. {dup[0][key]}\n2. {dup[1][key]}\n"
                )
    return json_data


if __name__ == "__main__":
    print("scikit-learn 库版本：", sklearn_version)

    # 命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file",
        type=str,
        help=("数据集JSON文件路径")
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help=("相似度判定阈值，取值0~1，数值越接近1判定标准越严格")
    )
    parser.add_argument(
        "--remove_duplicates",
        action='store_true',
        default=False,
        help=(
            "开启自动去重：仅根据input/output字段删除重复样本，"
            "不会依据instruction指令字段去重；处理后的干净数据会保存至 --json_output_file 指定路径"
        )
    )
    parser.add_argument(
        "--json_output_file",
        type=str,
        help=("去重后输出的JSON文件保存路径")
    )

    args = parser.parse_args()

    # 开启去重但未指定输出文件，抛出异常
    if args.remove_duplicates and not args.json_output_file:
        raise ValueError(
            "开启去重参数时，请通过 --json_output_file 指定输出文件路径，用于保存清洗后的数据集。"
        )

    # 未传入输入文件则使用内置示例数据
    if not args.json_file:
        json_data = example_data
    else:
        # 读取外部JSON数据集文件
        with open(args.json_file, "r") as file:
            json_data = json.load(file)

    # 执行重复检索、可选去重逻辑
    json_data = find_print_and_remove_near_duplicates(
        json_data=json_data,
        remove_duplicates=args.remove_duplicates,
        threshold=args.threshold
    )

    # 开启去重时，将清洗完成的数据写入输出JSON文件
    if args.remove_duplicates:
        with open(args.json_output_file, "w") as file:
            json.dump(json_data, file, indent=4)