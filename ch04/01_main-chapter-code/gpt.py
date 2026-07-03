# 本文件汇总了第2-4章涉及的所有相关代码，可作为独立脚本运行
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # 对整个文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # 使用滑动窗口将文本切分为长度为max_length的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 合并多头输出的线性层 (d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))  # 形状和句子长度绑定，掩码未来词（attn_scores/attn_weights 也是）

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 一次性计算所有头的QKV
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)  # 形状: (b, num_tokens, d_out) （num_tokens一固定 → Mask、QKV、注意力分数与权重矩阵，全部维度都被锁死）
        # 拆分最后一个维度并转置         形状: (b, num_heads, num_tokens, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.   view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values. view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # 注意力得分
        attn_scores = queries @ keys.transpose(2, 3)     # 计算点积注意力分数 （num_tokens，num_tokens）
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 将原始掩码截断到token数量并转换为布尔值
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # 使用掩码填充注意力分数
        # 注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # 缩放 + softmax
        attn_weights = self.dropout(attn_weights)
        # 上下文向量 (b, num_heads, num_tokens, head_dim) → (b, num_tokens, num_heads, head_dim) → (b, num_tokens, d_out)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 最终线性映射，融合多头信息 (b, num_tokens, d_out)
        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # 可学习的平移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                # 均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 方差（有偏估计=False）
        norm_x = (x - mean) / torch.sqrt(var + self.eps)   # 标准化
        return self.scale * norm_x + self.shift  # 可学习的缩放和平移（wx+b）


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一层：升维线性层 (768,3072)
                                    GELU(),
                                    nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),) # 第三层：降维线性层 (3072,768)

    def forward(self, x):  # 输入x形状(batch_size, seq_len, emb_dim)
        return self.layers(x)  # 输出(batch_size, seq_len, emb_dim) → 和输入完全一致


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],               # 输入特征维度：768
            d_out=cfg["emb_dim"],              # 输出特征维度：768
            context_length=cfg["context_length"], # 上下文长度：1024
            num_heads=cfg["n_heads"],          # 注意力头的数量：12
            dropout=cfg["drop_rate"],          # Dropout率：0.1
            qkv_bias=cfg["qkv_bias"]           # QKV偏置：False
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 注意力层前归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 前馈网络前归一化
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 残差连接前 Dropout

    def forward(self, x):  # x形状：(batch_size, num_tokens, emb_dim)，输出形状=输入
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)    # 形状 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # （context_length：位置表最大容量）
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最终层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 线性输出层

    def forward(self, in_idx):  # 形状 [batch_size, num_tokens]
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 形状 [batch_size, num_tokens, emb_size] （代入索引查表）
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # [num_tokens, emb_size] （seq_len：当前这批句子真实长度，不是固定1024！）
        x = tok_embeds + pos_embeds        # 形状 [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)  # [batch_size, num_tokens, 50257]
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):  # idx是输入的 token ID 张量，形状 [batch_size, seq_len]
    for _ in range(max_new_tokens):  # 自回归：每次生成1个词，达到设置的新词上限
        idx_cond = idx[:, -context_size:]  # 上下文截断：只取序列的最后 context_size 个 token （‘ ：’倒数第N个开始，一直取到最后）
        with torch.no_grad():
            logits = model(idx_cond)  # [batch_size, seq_len, 50257]
        logits = logits[:, -1, :]     # [batch_size, 50257]
        # 贪心选择：选概率最大的索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 形状 [batch, 1]
        # 新采样的索引拼接到序列中
        idx = torch.cat((idx, idx_next), dim=1)  # 形状 [batch, n_tokens+1]
    return idx


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词表大小
        "context_length": 1024,  # 上下文长度
        "emb_dim": 768,          # 嵌入维度
        "n_heads": 12,           # 注意力头数
        "n_layers": 12,          # 层数
        "drop_rate": 0.1,        # Dropout率
        "qkv_bias": False        # Query-Key-Value偏置
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # 禁用dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)            # tokenizer.encode() → 返回 列表
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 模型需要 → Tensor，增加一个批次维度

    print(f"{50*'='}\n{22*' '}输入\n{50*'='}")
    print("输入文本:", start_context)
    print("编码后文本:", encoded)
    print("编码后形状:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())  # 模型输出是[1, seq_len]的张量，tokenizer.decode() 只接收列表

    print(f"\n{50*'='}\n{22*' '}输出\n{50*'='}")
    print("输出:", out)
    print("输出长度:", len(out[0]))  # 4+10
    print("输出文本:", decoded_text)


if __name__ == "__main__":
    main()
