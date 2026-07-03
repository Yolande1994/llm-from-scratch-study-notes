# 版权所有 © Sebastian Raschka，基于 Apache License 2.0 许可协议（详见 LICENSE.txt）
# 对应书籍：《从零构建大模型》(Build a Large Language Model From Scratch)
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 官方源码仓库：https://github.com/rasbt/LLMs-from-scratch

# 本文件为内部单元测试专用

import io
import os
import sys
import types
import nbformat
from packaging import version
from typing import Optional, Tuple
import torch
import pytest
import transformers
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


transformers_version = transformers.__version__

# 下方 litgpt_build_rope_cache 函数取自 LitGPT 开源项目：https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
# LitGPT 项目采用 Apache v2 开源协议：https://github.com/Lightning-AI/litgpt/blob/main/LICENSE


def litgpt_build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    带旋转位置编码的增强型 Transformer（RoPE）

    参数:
        seq_len (int): 序列长度
        n_elem (int): 单头维度大小
        device (torch.device, 可选): 张量分配使用的设备
        base (int, 可选): 计算逆频率的基数
        condense_ratio (int, 可选): 位置索引压缩倍率
        extra_config (dict, 可选): 频率调整配套参数（用于 Llama 3.1 / 3.2 扩展缩放）

    返回:
        Tuple[torch.Tensor, torch.Tensor]: RoPE 所需余弦、正弦缓存张量
    """

    # 计算逆频率 theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        wavelen = 2 * torch.pi / theta
        ratio = orig_context_len / wavelen
        smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

        # 计算修正后的 theta，无需掩码索引
        adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
        theta = adjusted_theta

    # 生成位置索引序列 `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # 计算位置索引与 θ_i 的外积
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


# 以下代码取自 LitGPT 项目：https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
# LitGPT 项目采用 Apache v2 开源协议：https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
def litgpt_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (批次, 头数, 序列长度, 单头维度/2)
    x2 = x[..., head_size // 2:]  # (批次, 头数, 序列长度, 单头维度/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (批次, 头数, 序列长度, 单头维度)
    if cos.dim() > 1:
        # 批次维度需要对齐
        # sin/cos 形状为 (批次, 序列长度, 单头维度)，需要在头数维度扩维
        # 从后往前数维度是 RoPE 通用处理逻辑
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


@pytest.fixture(scope="module")
def notebook():
    def import_definitions_from_notebook(notebooks):
        imported_modules = {}

        for fullname, names in notebooks.items():
            # 获取当前测试文件所在目录
            current_dir = os.path.dirname(__file__)
            path = os.path.join(current_dir, "..", fullname + ".ipynb")
            path = os.path.normpath(path)

            # 加载 Notebook 文件
            if not os.path.exists(path):
                raise FileNotFoundError(f"Notebook 文件未找到，路径：{path}")

            with io.open(path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # 创建独立模块，存放从 Notebook 导入的函数、类
            mod = types.ModuleType(fullname)
            sys.modules[fullname] = mod

            # 遍历 Notebook 所有代码单元格，仅执行函数/类定义代码
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell_code = cell.source
                    for name in names:
                        # 匹配函数定义或类定义
                        if f"def {name}" in cell_code or f"class {name}" in cell_code:
                            exec(cell_code, mod.__dict__)

            imported_modules[fullname] = mod

        return imported_modules

    notebooks = {
        "converting-gpt-to-llama2": ["SiLU", "RMSNorm", "precompute_rope_params", "compute_rope"],
        "converting-llama2-to-llama3": ["precompute_rope_params"]
    }

    return import_definitions_from_notebook(notebooks)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(123)


def test_rope_llama2(notebook):

    this_nb = notebook["converting-gpt-to-llama2"]

    # 测试超参配置
    batch_size = 1
    context_len = 4096
    num_heads = 4
    head_dim = 16
    theta_base = 10_000

    # 预生成 RoPE 余弦、正弦缓存
    cos, sin = this_nb.precompute_rope_params(head_dim=head_dim, context_length=context_len)

    # 构造测试用 Query、Key 张量
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # 执行旋转位置编码
    queries_rot = this_nb.compute_rope(queries, cos, sin)
    keys_rot = this_nb.compute_rope(keys, cos, sin)

    # 基于 HuggingFace Transformers 生成标准参考结果

    if version.parse(transformers_version) < version.parse("4.48"):
        rot_emb = LlamaRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=context_len,
            base=theta_base
        )
    else:
        class RoPEConfig:
            dim: int = head_dim
            rope_theta = theta_base
            max_position_embeddings: int = 8192
            hidden_size = head_dim * num_heads
            num_attention_heads = num_heads

        config = RoPEConfig()
        rot_emb = LlamaRotaryEmbedding(config=config)

    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)
    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)

    # 基于 LitGPT 实现生成标准参考结果
    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(context_len, n_elem=head_dim, base=10_000)
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    torch.testing.assert_close(sin, litgpt_sin)
    torch.testing.assert_close(cos, litgpt_cos)
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)


def test_rope_llama3(notebook):

    nb1 = notebook["converting-gpt-to-llama2"]
    nb2 = notebook["converting-llama2-to-llama3"]

    # 测试超参配置
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    theta_base = 500_000

    # 预生成 RoPE 余弦、正弦缓存
    cos, sin = nb2.precompute_rope_params(
        head_dim=head_dim,
        context_length=context_len,
        theta_base=theta_base
    )

    # 构造测试用 Query、Key 张量
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # 执行旋转位置编码
    queries_rot = nb1.compute_rope(queries, cos, sin)
    keys_rot = nb1.compute_rope(keys, cos, sin)

    # 基于 HuggingFace Transformers 生成标准参考结果
    if version.parse(transformers_version) < version.parse("4.48"):
        rot_emb = LlamaRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=context_len,
            base=theta_base
        )
    else:
        class RoPEConfig:
            dim: int = head_dim
            rope_theta = theta_base
            max_position_embeddings: int = 8192
            hidden_size = head_dim * num_heads
            num_attention_heads = num_heads

        config = RoPEConfig()
        rot_emb = LlamaRotaryEmbedding(config=config)

    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)

    # 基于 LitGPT 实现生成标准参考结果
    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(context_len, n_elem=head_dim, base=theta_base)
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    torch.testing.assert_close(sin, litgpt_sin)
    torch.testing.assert_close(cos, litgpt_cos)
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)


def test_rope_llama3_12(notebook):

    nb1 = notebook["converting-gpt-to-llama2"]
    nb2 = notebook["converting-llama2-to-llama3"]

    # 测试超参配置
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    rope_theta = 500_000

    rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }

    # 预生成带缩放参数的 RoPE 余弦、正弦缓存
    cos, sin = nb2.precompute_rope_params(
        head_dim=head_dim,
        theta_base=rope_theta,
        context_length=context_len,
        freq_config=rope_config,
    )

    # 构造测试用 Query、Key 张量
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # 执行旋转位置编码
    queries_rot = nb1.compute_rope(queries, cos, sin)
    keys_rot = nb1.compute_rope(keys, cos, sin)

    # 基于 HuggingFace Transformers 生成标准参考结果
    hf_rope_params = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    }

    class RoPEConfig:
        rope_type = "llama3"
        rope_scaling = hf_rope_params
        factor = 1.0
        dim: int = head_dim
        rope_theta = 500_000
        max_position_embeddings: int = 8192
        hidden_size = head_dim * num_heads
        num_attention_heads = num_heads

    config = RoPEConfig()

    rot_emb = LlamaRotaryEmbedding(config=config)
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)

    # 基于 LitGPT 实现生成标准参考结果
    litgpt_rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_seq_len": 8192
    }

    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(
        context_len,
        n_elem=head_dim,
        base=rope_theta,
        extra_config=litgpt_rope_config
    )
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    torch.testing.assert_close(sin, litgpt_sin)
    torch.testing.assert_close(cos, litgpt_cos)
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)


def test_silu(notebook):
    example_batch = torch.randn(2, 3, 4)
    silu = notebook["converting-gpt-to-llama2"].SiLU()
    assert torch.allclose(silu(example_batch), torch.nn.functional.silu(example_batch))


@pytest.mark.skipif(torch.__version__ < "2.4", reason="需要 PyTorch 2.4 或更高版本")
def test_rmsnorm(notebook):
    example_batch = torch.randn(2, 3, 4)
    rms_norm = notebook["converting-gpt-to-llama2"].RMSNorm(emb_dim=example_batch.shape[-1], eps=1e-5)
    rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)

    assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))