"""
基于原书GPT-2代码的基础版KV Cache推理扩展
- 保持原模型架构完全不变，仅新增增量推理逻辑
- 提供带缓存生成函数，可与原版生成结果做一致性校验
- 新增标准性能基准测试：覆盖256/512/1024上下文，输出延迟、显存对比表格
"""

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import time

#####################################
# Chapter 3 - 改造：支持KV Cache的多头注意力
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, kv_cache=None):
        """
        Args:
            x: 输入张量，形状 (batch, num_tokens, d_in)
            kv_cache: 历史KV缓存，元组 (keys, values)，形状均为 (batch, num_heads, past_len, head_dim)，None表示无缓存
        Returns:
            context_vec: 注意力输出，形状 (batch, num_tokens, d_out)
            new_kv_cache: 更新后的KV缓存，元组 (keys, values)
        """
        b, num_tokens, d_in = x.shape

        # 计算当前输入的Q、K、V
        queries = self.w_q(x)
        keys = self.w_k(x)
        values = self.w_v(x)

        # 拆分为多头格式 (b, nh, num_tokens, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # ---------- KV Cache 核心逻辑 ----------
        if kv_cache is not None:
            past_keys, past_values = kv_cache
            # 将当前token的K/V拼接到历史缓存之后
            keys = torch.cat([past_keys, keys], dim=2)
            values = torch.cat([past_values, values], dim=2)
        new_kv_cache = (keys, values)
        # ----------------------------------------

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)

        # 掩码处理：仅在预填充阶段（多token输入）需要因果掩码
        # 增量解码阶段query只有1个token，所有key都是历史，天然满足因果，跳过掩码
        if num_tokens > 1:
            mask_bool = self.mask.bool()[:num_tokens, :keys.shape[2]]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放 + softmax + dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量并合并多头
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec, new_kv_cache


#####################################
# Chapter 4 - 改造：Transformer块透传缓存
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, kv_cache=None):
        # 注意力残差分支
        shortcut = x
        x = self.norm1(x)
        x, new_kv_cache = self.att(x, kv_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈网络残差分支
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x, new_kv_cache


#####################################
# Chapter 4 - 改造：GPT主模型，支持全层缓存 + 位置编码修正
#####################################
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, past_kv_cache=None):
        """
        Args:
            in_idx: 输入token id，形状 (batch, seq_len)
            past_kv_cache: 所有层的历史缓存列表，第i层对应 (keys, values)，None表示无缓存
        Returns:
            logits: 模型输出logits，形状 (batch, seq_len, vocab_size)
            new_kv_cache: 更新后的全层缓存列表
        """
        batch_size, seq_len = in_idx.shape

        # --------------- 增量推理的位置编码偏移 ----------------
        past_len = past_kv_cache[0][0].shape[2] if past_kv_cache is not None else 0
        pos_ids = torch.arange(past_len, past_len + seq_len, device=in_idx.device)
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(pos_ids)
        # ---------------------------------------------------

        x = tok_emb + pos_emb
        x = self.drop_emb(x)

        # 逐层前向，逐层更新缓存
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = past_kv_cache[i] if past_kv_cache is not None else None
            x, layer_new_cache = block(x, layer_cache)
            new_kv_cache.append(layer_new_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits, new_kv_cache


#####################################
# 新增：带KV Cache的自回归生成函数
#####################################
def generate_kv_cache(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    基础版KV Cache生成：预填充阶段计算完整prompt的KV，后续每步仅输入单个token
    生成结果与原版generate完全一致，可用于正确性校验
    """
    model.eval()
    kv_cache = None
    generated = idx.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # 上下文截断：缓存总长度不能超过context_size
            if kv_cache is not None and kv_cache[0][0].shape[2] >= context_size:
                # 超出时丢弃最早的token（简单滑动窗口，基础版实现）
                kv_cache = [(k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in kv_cache]

            # 当前输入：第一步是完整prompt，后续是上一步生成的单个token
            if step == 0:
                input_ids = generated
            else:
                input_ids = generated[:, -1:]

            # 前向推理
            logits, kv_cache = model(input_ids, past_kv_cache=kv_cache)
            # 取最后一个位置的logits用于预测
            logits = logits[:, -1, :]

            # Top-K 采样
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf'), device=logits.device), logits)

            # 温度缩放与采样
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # EOS提前终止
            if eos_id is not None and idx_next.item() == eos_id:
                break

            generated = torch.cat((generated, idx_next), dim=1)

    return generated


#####################################
# 原版无缓存生成函数（保留，用于对比）
#####################################
def generate_original(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf'), device=logits.device), logits)

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_id is not None and idx_next.item() == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)
    return idx


#####################################
# 工具函数：文本与ID互转
#####################################
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


#####################################
# 权重加载函数
#####################################
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    """
    加载官方GPT-2 124M预训练权重
    params: 从官方权重文件解析得到的嵌套字典
    权重需自行从OpenAI官方或huggingface获取，本仓库不分发权重文件
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.blocks[b].att.w_q.weight = assign(gpt.blocks[b].att.w_q.weight, q_w.T)
        gpt.blocks[b].att.w_k.weight = assign(gpt.blocks[b].att.w_k.weight, k_w.T)
        gpt.blocks[b].att.w_v.weight = assign(gpt.blocks[b].att.w_v.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.blocks[b].att.w_q.bias = assign(gpt.blocks[b].att.w_q.bias, q_b)
        gpt.blocks[b].att.w_k.bias = assign(gpt.blocks[b].att.w_k.bias, k_b)
        gpt.blocks[b].att.w_v.bias = assign(gpt.blocks[b].att.w_v.bias, v_b)

        gpt.blocks[b].att.out_proj.weight = assign(gpt.blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.blocks[b].att.out_proj.bias   = assign(gpt.blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.blocks[b].ff.layers[0].weight = assign(gpt.blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.blocks[b].ff.layers[0].bias   = assign(gpt.blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.blocks[b].ff.layers[2].weight = assign(gpt.blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.blocks[b].ff.layers[2].bias   = assign(gpt.blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.blocks[b].norm1.scale = assign(gpt.blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.blocks[b].norm1.shift = assign(gpt.blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.blocks[b].norm2.scale = assign(gpt.blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.blocks[b].norm2.shift = assign(gpt.blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight  = assign(gpt.out_head.weight, params["wte"])


#####################################
# 新增：性能基准测试工具集
#####################################
def get_peak_memory_mb(device):
    """获取当前CUDA峰值显存占用，单位MB；CPU环境返回N/A"""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return float("nan")


def benchmark_generate(model, prompt_ids, max_new_tokens, context_size, mode="original", warmup=1, repeats=2):
    """
    单组性能测试：返回首token延迟、后续token平均延迟、峰值显存
    Args:
        mode: "original" 原版无缓存 / "kv_cache" 带缓存
    """
    device = next(model.parameters()).device
    generate_fn = generate_original if mode == "original" else generate_kv_cache

    # 预热
    for _ in range(warmup):
        generate_fn(model, prompt_ids, max_new_tokens, context_size)

    # 重置显存统计
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    first_token_times = []
    total_times = []

    for _ in range(repeats):
        # 单独测量首token时间
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        if mode == "original":
            # 原版：第一次前向就是完整prompt
            with torch.no_grad():
                logits, _ = model(prompt_ids)
                _ = logits[:, -1, :]
        else:
            # KV版：预填充阶段
            with torch.no_grad():
                _, kv_cache = model(prompt_ids)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_first = time.perf_counter() - t_start
        first_token_times.append(t_first)

        # 测量完整生成总时间
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total_start = time.perf_counter()
        generate_fn(model, prompt_ids, max_new_tokens, context_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total = time.perf_counter() - t_total_start
        total_times.append(t_total)

    # 取平均
    avg_first_token = sum(first_token_times) / len(first_token_times)
    avg_total = sum(total_times) / len(total_times)
    # 后续token平均延迟 = (总时间 - 首token时间) / (生成token数 - 1)
    avg_rest_token = (avg_total - avg_first_token) / (max_new_tokens - 1) if max_new_tokens > 1 else 0.0
    peak_mem = get_peak_memory_mb(device)

    return {
        "first_token_ms": avg_first_token * 1000,
        "rest_token_avg_ms": avg_rest_token * 1000,
        "peak_memory_mb": peak_mem
    }


def run_full_benchmark(model, config, prompt_lengths=[256, 512, 1024], max_new_tokens=32):
    """运行完整基准测试，输出对比表格"""
    device = next(model.parameters()).device
    context_size = config["context_length"]
    vocab_size = config["vocab_size"]

    print("\n" + "=" * 80)
    print(f"标准性能基准测试  设备：{device} | 生成token数：{max_new_tokens} | 测试轮次：2次取平均")
    print("=" * 80)

    table_rows = []

    for plen in prompt_lengths:
        # 构造指定长度的随机prompt
        prompt = torch.randint(0, vocab_size, (1, plen), device=device)

        # 原版测试
        res_ori = benchmark_generate(model, prompt, max_new_tokens, context_size, mode="original")
        # KV Cache测试
        res_kv = benchmark_generate(model, prompt, max_new_tokens, context_size, mode="kv_cache")

        # 计算加速比
        speedup = res_ori["rest_token_avg_ms"] / res_kv["rest_token_avg_ms"] if res_kv["rest_token_avg_ms"] > 0 else 0

        table_rows.append({
            "prompt_len": plen,
            "ori_first": res_ori["first_token_ms"],
            "kv_first": res_kv["first_token_ms"],
            "ori_rest": res_ori["rest_token_avg_ms"],
            "kv_rest": res_kv["rest_token_avg_ms"],
            "ori_mem": res_ori["peak_memory_mb"],
            "kv_mem": res_kv["peak_memory_mb"],
            "speedup": speedup
        })

    # 对齐格式输出
    total_width = 98
    print("\n" + "=" * total_width)
    print(f"{'KV Cache 性能对比表（GPT-2 124M）':^{total_width}}")
    print(f"{'延迟单位: ms | 显存单位: MB':^{total_width}}")
    print("=" * total_width)
    print(
        f"| {'上下文':^4} | {'首token(原版)':^13} | {'首token(缓存)':^12} | {'后续延迟(原版)':^10} | {'后续延迟(缓存)':^11} | {'显存(原版)':^8} | {'显存(缓存)':^8} | {'加速比':^6} |"
    )
    print("-" * total_width)
    for row in table_rows:
        mem_ori = f'{row["ori_mem"]:.1f}' if not np.isnan(row["ori_mem"]) else "N/A"
        mem_kv = f'{row["kv_mem"]:.1f}' if not np.isnan(row["kv_mem"]) else "N/A"
        print(
            f"| {row['prompt_len']:^6} | {row['ori_first']:>14.2f} | {row['kv_first']:>14.2f} | {row['ori_rest']:>14.2f} | {row['kv_rest']:>14.2f} | {mem_ori:>10} | {mem_kv:>10} | {row['speedup']:>7.2f}x |"
        )
    print("=" * total_width)

    print("\n说明：")
    print("- 首token为预填充阶段，两者计算量相当，缓存版无显著加速")
    print("- 上下文越长，后续token的加速效果越显著")
    print("- CPU环境下显存项显示为 N/A，使用GPU运行可查看显存数据")


#####################################
# 主程序：正确性验证 + 完整基准测试
#####################################
def main():
    USE_GPU = True  # 设为False则使用CPU

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.0,  # 推理模式关闭dropout
        "qkv_bias": False
    }

    # 设备选择：开关控制 + 可用性兜底
    if USE_GPU and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M).to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    max_new_tokens = 20

    print("=" * 60)
    print("正确性验证：两种生成方式结果是否一致")
    print("=" * 60)

    # 原版无缓存生成
    out_original = generate_original(model, encoded, max_new_tokens, GPT_CONFIG_124M["context_length"])
    text_original = token_ids_to_text(out_original, tokenizer)

    # KV Cache生成
    out_kv = generate_kv_cache(model, encoded, max_new_tokens, GPT_CONFIG_124M["context_length"])
    text_kv = token_ids_to_text(out_kv, tokenizer)

    print(f"原版生成: {text_original}")
    print(f"缓存生成: {text_kv}")
    print(f"结果一致: {torch.equal(out_original, out_kv)}")

    # 运行完整基准测试
    run_full_benchmark(model, GPT_CONFIG_124M, prompt_lengths=[256, 512, 1024], max_new_tokens=32)


if __name__ == "__main__":
    main()