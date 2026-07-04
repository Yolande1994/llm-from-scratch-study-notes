import torch
import torch.nn as nn
import torch.optim as optim

# ===================== 1. 数据预处理（和之前完全一样）=====================
sentences = [["我", "喜欢", "吃", "苹果"],
             ["我", "喜欢", "吃", "香蕉"],
             ["我", "喜欢", "吃", "橙子"],
             ["小猫", "喜欢", "玩", "球"],
             ["小狗", "喜欢", "玩", "玩具"],]

# 构建词汇表
words = list({w for s in sentences for w in s})
word_to_id = {w: i for i, w in enumerate(words)}
id_to_word = {i: w for w, i in word_to_id.items()}

vocab_size = len(words)  # 词汇表大小
vec_size = 5     # 词向量维度
window_size = 1  # 上下文窗口
epochs = 1000    # 训练轮数
lr = 0.01        # 学习率

# ===================== 2. PyTorch 模型定义（核心！）=====================
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, vec_size):
        super(Word2Vec, self).__init__()
        # 可训练权重：自动支持梯度计算
        self.W1 = nn.Parameter(torch.randn(vocab_size, vec_size) * 0.1)  # 输入→隐藏（词向量）  (11, 5)
        self.W2 = nn.Parameter(torch.randn(vec_size, vocab_size) * 0.1)  # 隐藏→输出          (5, 11)
        '''   为什么可以不构建显式网络层？（3 个核心原因）
        1. 网络结构极度简单
        Word2Vec 只有两层线性变换，无隐藏层激活函数、无复杂结构
        → 根本没必要用 nn.Sequential 封装，手动写更直观
        
        2. One-Hot 特性带来的极致优化
        输入是 One-Hot 向量，nn.Linear 会做冗余的矩阵乘法，直接索引 W1[target_id] 等价且高效
        → 直接省略了第一层显式线性层
        
        3. PyTorch 的自由度极高
        只要把权重定义为 nn.Parameter，无论你怎么计算（手动乘法/调用层），PyTorch 都会自动追踪梯度、支持训练
        → 不需要拘泥于「必须写层」的形式
        
        注意:
        手动写矩阵运算（隐去线性层）→ 必须手动定义权重 W （用 nn.Parameter 包装，让 PyTorch 识别为可训练参数）
        使用 PyTorch nn.Linear 线性层 → 权重自动生成，无需手动定义
        
        Word2Vec 不加偏置 b ➜ 精度不受影响，是标准写法.  加了 b ➜ 不会更准，反而容易过拟合
        Word2Vec 的核心目标：学习词语的相对语义关系（苹果↔香蕉近，苹果↔小猫远） 
        偏置 b 的作用：给向量做整体平移（比如所有维度+0.1）, 而余弦相似度、向量方向、语义相对关系，完全不受平移影响！ → 少了b，对词向量质量 零损失
        
        只有复杂神经网络才需要偏置：图像分类、大模型、多层带激活函数的网络.  需要拟合复杂偏移、非线性关系的场景
        简单的词嵌入训练 → 永远不加 b
        '''
    # 前向传播
    def forward(self, target_id):
        # 1. 查词向量：直接取 W1 的行
        h = self.W1[target_id]  # shape: (5,)
        # 2. 矩阵乘法 + Softmax 预测概率
        u = torch.matmul(h, self.W2)  # shape: (11,)
        y_pred = torch.softmax(u, dim=0)
        return y_pred, h

# ===================== 3. 初始化模型、优化器 =====================
model = Word2Vec(vocab_size, vec_size)
optimizer = optim.SGD(model.parameters(), lr=lr)  # 随机梯度下降

# ===================== 4. 训练（PyTorch 标准流程）=====================
print("开始训练（PyTorch 版）...")
for epoch in range(epochs):
    total_loss = 0
    for sentence in sentences:
        for i, target_word in enumerate(sentence):
            target_id = torch.tensor(word_to_id[target_word], dtype=torch.long)
            optimizer.zero_grad()  # 每个目标词开始,梯度清零（关键）
            # 获取上下文词
            context_words = []
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if j != i:
                    context_words.append(sentence[j])
            loss_sum = 0  # 累加损失（不再单独backward）
            # 对每个上下文词计算损失
            for context_word in context_words:
                context_id = word_to_id[context_word]
                y_pred, _ = model(target_id)  # 前向传播
                # 交叉熵损失（标准NLP损失）
                loss = -torch.log(y_pred[context_id] + 1e-10)
                loss_sum += loss  # 累加所有上下文词损失
            loss_sum.backward()  # 只在最后调用一次 backward（修复核心）
            optimizer.step()     # 更新权重
            total_loss += loss.item()
    # 打印日志
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {total_loss:.4f}")

# ===================== 5. 查看结果 =====================
print("\n=== PyTorch 训练完成，词向量如下 ===")
with torch.no_grad():  # 推理阶段关闭梯度
    for word in words:
        vec = model.W1[word_to_id[word]].numpy().round(4)
        print(f"{word:4s}: {vec}")

# 余弦相似度
def cos_sim(a, b):
    a = model.W1[word_to_id[a]]
    b = model.W1[word_to_id[b]]
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()

print("\n=== 语义相似度 ===")
print(f"苹果 ↔ 香蕉: {cos_sim('苹果', '香蕉'):.4f}")
print(f"苹果 ↔ 小猫: {cos_sim('苹果', '小猫'):.4f}")