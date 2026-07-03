# 版权所有 (c) Sebastian Raschka，基于 Apache License 2.0 开源协议
# 本书配套代码：《从零构建大语言模型》
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
#   - 官方代码仓库：https://github.com/rasbt/LLMs-from-scratch

import json
import numpy as np
import os
import urllib.request

# 可选：使用 requests 库下载（需要额外安装 pip install requests）
# import requests
import tensorflow as tf
import tiktoken
import torch
from tqdm import tqdm

# 从本地文件导入之前章节实现的GPT模型
from previous_chapters import GPTModel


def text_to_token_ids(text, tokenizer):
    """将文本转换为模型可输入的token ID张量"""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度，适配模型输入格式
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将模型输出的token ID张量转换回人类可读的文本"""
    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())


def download_and_load_gpt2(model_size, models_dir):
    """
    下载GPT-2官方预训练权重并转换为PyTorch可用的参数字典
    :param model_size: 模型大小，可选值："124M", "355M", "774M", "1558M"
    :param models_dir: 模型文件保存的根目录
    :return: (settings, params) 模型超参数字典和权重参数字典
    """
    # 验证输入的模型大小是否合法
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"模型大小必须是以下之一：{allowed_sizes}")

    # 定义文件路径和下载地址
    model_dir = os.path.join(models_dir, model_size)
    # OpenAI 官方主下载地址（国内访问可能不稳定）
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
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
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # 加载模型配置和权重参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的TensorFlow检查点文件路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载模型超参数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 转换TensorFlow权重为PyTorch格式

    return settings, params


# 使用 requests 库的替代下载方法（速度更快，稳定性更好）
"""
def download_file(url, destination):
    # 发送GET请求，以流模式下载文件（避免一次性加载大文件到内存）
    response = requests.get(url, stream=True)

    # 从响应头获取文件总大小，若服务器未提供则默认0
    file_size = int(response.headers.get("content-length", 0))

    # 检查本地文件是否已存在且完整（大小一致）
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


def download_file(url, destination):
    """
    使用标准库urllib下载单个文件，支持断点续传和进度条
    :param url: 下载地址
    :param destination: 本地保存路径
    """
    # 发送GET请求下载文件
    with urllib.request.urlopen(url) as response:
        # 从响应头获取文件总大小，若服务器未提供则默认0
        file_size = int(response.headers.get("Content-Length", 0))

        # 检查本地文件是否已存在且完整（大小一致）
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"文件已存在且完整，跳过下载：{destination}")
                return

        block_size = 1024  # 每次读取的块大小，单位：字节（1KB）

        # 初始化进度条，显示下载进度
        progress_bar_description = os.path.basename(url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # 以二进制写入模式打开目标文件
            with open(destination, "wb") as file:
                # 分块读取并写入文件
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break  # 读取完毕，退出循环
                    file.write(chunk)
                    progress_bar.update(len(chunk))  # 更新进度条


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


def assign(left, right):
    """
    权重赋值辅助函数，带严格的形状校验
    :param left: 目标模型层的权重
    :param right: 要赋值的权重值
    :return: 包装好的PyTorch可训练参数
    """
    # 严格的形状校验：如果形状不匹配，直接抛出错误
    if left.shape != right.shape:
        raise ValueError(f"形状不匹配！目标层形状: {left.shape}, 权重形状: {right.shape}")

    # 将numpy数组转换为PyTorch可训练的Parameter类型
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    将转换好的GPT-2预训练权重，100%精确地映射到我们自己实现的GPTModel中
    :param gpt: 我们自己实现的GPT模型实例
    :param params: 转换好的权重参数字典
    """
    # 加载最顶层的全局权重
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])  # 位置嵌入层
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])  # 词嵌入层

    # 循环加载每一个Transformer块
    for b in range(len(params["blocks"])):
        # 把OpenAI合并的QKV大矩阵，拆分成三个独立的Q、K、V权重矩阵
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        # 赋值给我们自己模型的三个独立线性层，注意必须转置（TensorFlow与PyTorch格式差异）
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 加载自注意力层的QKV偏置项（偏置是一维向量，不需要转置）
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 加载自注意力层的输出投影层
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)  # 线性层权重，必须转置
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])  # 偏置，不需要转置

        # 加载前馈网络(FFN)层
        # 第一个线性层：c_fc (升维)
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)  # 转置
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        # 第二个线性层：c_proj (降维)
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)  # 转置
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 加载层归一化(LayerNorm)层
        # 第一个层归一化：ln_1 (注意力前)
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])  # g = gamma = 缩放参数
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])  # b = beta = 偏移参数
        # 第二个层归一化：ln_2 (前馈网络前)
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 加载最终的全局层归一化
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 输出头权重复用词嵌入层权重（GPT-2经典设计）
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    增强版文本生成函数，支持top-k采样和温度缩放
    :param model: GPT模型实例
    :param idx: 起始文本的token ID张量
    :param max_new_tokens: 最多生成多少个新token
    :param context_size: 模型支持的最大上下文长度
    :param temperature: 温度系数，控制生成的随机性，0=确定性输出，越大越随机
    :param top_k: 只从概率最高的k个token中采样，None表示不限制
    :param eos_id: 结束符ID，遇到该token时提前停止生成
    :return: 生成的完整token ID张量
    """
    # 自回归生成循环：一次生成一个token
    for _ in range(max_new_tokens):
        # 裁剪上下文，只保留最近的context_size个token
        idx_cond = idx[:, -context_size:]
        # 前向传播得到logits，禁用梯度计算加速
        with torch.no_grad():
            logits = model(idx_cond)
        # 只取最后一个位置的logits用于预测下一个token
        logits = logits[:, -1, :]

        # 应用top-k采样：只保留概率最高的k个token
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            # 将低于第k个token的logits设为负无穷，使其概率为0
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 应用温度缩放：控制生成的随机性
        if temperature > 0.0:
            logits = logits / temperature
            # 计算概率分布
            probs = torch.softmax(logits, dim=-1)
            # 从概率分布中采样一个token
            idx_next = torch.multinomial(probs, num_samples=1)
        # 温度为0时，直接选择概率最高的token（贪心解码）
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 如果遇到结束符，提前停止生成
        if idx_next == eos_id:
            break

        # 将生成的token拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main(gpt_config, input_prompt, model_size):
    """
    主函数：下载权重、加载模型、生成文本
    :param gpt_config: GPT模型配置字典
    :param input_prompt: 输入提示文本
    :param model_size: 模型大小（如"124M"）
    """
    # 自动选择设备：优先使用GPU，没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 下载并加载GPT-2预训练权重
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # 初始化我们自己实现的GPT模型
    gpt = GPTModel(gpt_config)
    # 将预训练权重加载到模型中
    load_weights_into_gpt(gpt, params)
    # 将模型移动到指定设备
    gpt.to(device)
    # 将模型切换为评估模式（关闭Dropout等训练专用层）
    gpt.eval()

    # 使用GPT-2官方分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 设置随机种子，保证生成结果可复现
    torch.manual_seed(123)

    # 生成文本
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=25,
        context_size=gpt_config["context_length"],
        top_k=50,
        temperature=1.0
    )

    # 打印生成结果
    print("输出文本:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    # 设置随机种子，保证全局可复现
    torch.manual_seed(123)

    # 选择要加载的模型和输入提示
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves you"

    # GPT模型基础配置
    BASE_CONFIG = {
        "vocab_size": 50257,  # 词表大小
        "context_length": 1024,  # 最大上下文长度
        "drop_rate": 0.0,  # Dropout概率（推理时设为0）
        "qkv_bias": True  # QKV线性层是否使用偏置（GPT-2官方开启）
    }

    # GPT-2各版本模型的核心配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # 提取模型大小字符串（如"124M"）
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    # 更新基础配置，得到完整的模型配置
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 启动主流程
    main(BASE_CONFIG, INPUT_PROMPT, model_size)