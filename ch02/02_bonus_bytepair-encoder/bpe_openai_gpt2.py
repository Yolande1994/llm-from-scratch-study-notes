# 源码来源：https://github.com/openai/gpt-2/blob/master/src/encoder.py
# 许可证：
# 改进版 MIT 许可证

# 软件版权所有 © 2019 OpenAI

# 我们不对您使用 GPT-2 生成的内容主张所有权，您可自行处置。
# 仅要求您负责任地使用 GPT-2，并明确标注内容由 GPT-2 生成。

# 特此免费授予任何获得本软件及相关文档文件（以下简称“软件”）副本的人员
# 不受限制地处理本软件的权利，包括但不限于使用、复制、修改、合并、出版、
# 分发、再许可和/或销售软件副本的权利，并允许向其提供软件的人员这样做，
# 但须符合以下条件：

# 上述版权声明和本许可声明应包含在
# 软件的所有副本或主要部分中。
# 上述版权声明和本许可声明无需包含在
# 由本软件生成的内容中。

# 本软件按“原样”提供，不提供任何明示或暗示的保证，
# 包括但不限于对适销性、
# 特定用途适用性和非侵权性的保证。在任何情况下，作者或版权持有人
# 均不对任何索赔、损害或其他责任负责，无论是合同诉讼、
# 侵权行为还是其他形式，由软件或软件使用或其他
# 相关行为引起、衍生或与之相关。

import os
import json
import regex as re
import requests
from tqdm import tqdm
from functools import lru_cache


@lru_cache()
def bytes_to_unicode():
    """
    返回 UTF-8 字节列表与对应 Unicode 字符串的映射表。
    可逆的 BPE 编码基于 Unicode 字符串运行。
    这意味着，如果想要避免出现未登录词（UNK），词表中需要包含大量 Unicode 字符。
    当处理 100 亿词元规模的数据集时，大约需要 5000 个字符才能保证较好的覆盖度。
    这在通常 32000 大小的 BPE 词表中占了相当大的比例。
    为避免这一问题，我们需要建立 UTF-8 字节与 Unicode 字符串的查找表。
    同时也避免映射到 BPE 代码无法处理的空白/控制字符。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    返回单词中所有相邻符号对的集合。
    单词以符号元组的形式表示（符号为可变长度的字符串）。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # 解码时的字符错误处理策略
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # 其实应该加上 re.IGNORECASE，这样缩写的大写形式也能触发 BPE 合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    # 代码修改自以下来源
    subdir = 'gpt2_model'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # Windows 系统兼容处理

    for filename in ['encoder.json', 'vocab.bpe']:
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 块大小设为 1000，因为以太网数据包大小约为 1500 字节
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)