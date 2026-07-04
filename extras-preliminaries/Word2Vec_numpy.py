import numpy as np
# ===================== Skip-Gram 模型（NumPy 手动实现）=====================
# Skip-Gram（跳字模型）是 Word2Vec 的两大核心训练模式之一，核心思想是用中心词预测上下文词，通过这个预测任务学习高质量的词向量表示。
# 核心目标：通过「目标词预测上下文词」任务，学习语义词向量 W1
# 训练完成后，W2 直接丢弃，仅保留 W1 作为词嵌入表

# ===================== 1. 准备数据 =====================
sentences = [["我", "喜欢", "吃", "苹果"],
             ["我", "喜欢", "吃", "香蕉"],
             ["我", "喜欢", "吃", "橙子"],
             ["小猫", "喜欢", "玩", "球"],
             ["小狗", "喜欢", "玩", "玩具"],
             ["苹果", "是", "水果"],
             ["香蕉", "是", "水果"],
             ["小狗", "是", "动物"],
             ["小猫", "是", "动物"],]

# 构建词汇表与映射
words = list({w for s in sentences for w in s})
word_to_id = {w: i for i, w in enumerate(words)}
id_to_word = {i: w for w, i in word_to_id.items()}

vocab_size = len(words)
vec_size = 5       # 词向量维度
window_size = 1    # 上下文窗口大小

# ===================== 2. 权重初始化（NumPy 矩阵）=====================
W1 = np.random.uniform(-0.5, 0.5, (vocab_size, vec_size))  # W1: (11, 5) → 词向量矩阵 (实际工程中会用更规范的初始化（如Xavier初始化）)
W2 = np.random.uniform(-0.5, 0.5, (vec_size, vocab_size))  # W2: (5, 11) → 分类器权重，仅用于训练，完成后丢弃

# ===================== 3. 辅助函数 =====================
def softmax(x):
    exps = np.exp(x - np.max(x))  # 防溢出
    return exps / np.sum(exps)

# ===================== 4. 训练 =====================
def train(epochs=1000, lr=0.01):
    global W1, W2  # 声明全局变量，允许在函数内修改权重 / 或直接在这里初始化权重
    for epoch in range(epochs):
        total_loss = 0.0
        for sentence in sentences:
            for i, target_word in enumerate(sentence):
                target_id = word_to_id[target_word]
                # -------- 前向传播 --------
                h = W1[target_id]   # 隐藏层, 直接取词向量,形状: (5,)                 h=One-Hot向量×W1 → 结果：直接取出 W1 中对应行的词向量  (没有显式写成一堆0/1 然后 h=np.dot(one_hot,W1), 但用「词ID索引取值」完美等价实现)
                u = h @ W2          # 输出层, 矩阵乘法，得到输出层原始分数 (11,)
                y_pred = softmax(u) # Softmax归一化，得到上下文词概率分布

                # 取上下文词
                '''
                这一步的核心目的，是为 Skip-Gram 模型准备训练的 “标签”（Ground Truth）：
                Skip-Gram 的任务是：给定目标词（输入），预测它周围的上下文词
                模型的输出是预测的上下文概率分布（y_pred），而训练需要 “标准答案” 来计算损失，再通过反向传播更新权重
                context_words 就是这个 “标准答案”，它记录了目标词在真实句子中周围出现过哪些词，是模型要学习模仿的目标
                简单说：没有context_words，模型就不知道 “预测什么才是对的”，根本无法训练

                context_words 最终的形状:
                假设 window_size=1：
                句中的目标词：前后各 1 个词 → context_words 长度为 2
                句首的目标词：后面 1 个词 → context_words 长度为 1
                句尾的目标词：前面 1 个词 → context_words 长度为 1

                举个例子：句子是 ["我", "喜欢", "吃", "苹果"]（长度 4），目标词在索引i=2（词 “吃”），window_size=1：
                左边界：max(0, 2-1) = 1
                右边界：min(4, 2+1+1) = 4
                循环范围：range(1,4) → 索引1,2,3（对应词 “喜欢”“吃”“苹果”）
                排除目标词本身，只保留周围的词
                context_words = ["喜欢", "苹果"]
                '''
                context_words = []
                for j in range(max(0, i-window_size), min(len(sentence), i+window_size+1)): # 窗口范围：[i-window_size, i+window_size]，排除目标词本身
                    if j != i:  # 排除目标词本身，只保留周围的词
                        context_words.append(sentence[j])  # 模型要预测的 “真实上下文词”，即训练标签，用于计算损失和更新权重
                # -------- 反向传播（对每个上下文词）--------
                for context_word in context_words:  # 每个上下文词都会单独作为一次 “标签”，驱动模型更新一次权重
                    context_id = word_to_id[context_word]

                    # dLoss/du = y_pred - y_true（损失对输出层输入u的梯度，反向传播的起点）
                    e = y_pred.copy()
                    e[context_id] -= 1.0  # 只计算正确的上下文的预测概率与真实概率1的误差，y_true的其他位置都是 0
                    '''
                    初始e:    [0.9, 0.9, 0.9,...,0.9]   11类,每类概率值约等于0.9
                    减y_true: [0.9, -0.91, 0.9,...,0.9]  只有正确位置是负数（-0.91），代表模型预测不足. 其他位置都是正数（0.09），代表模型预测过度
                    
                    梯度含义：
                    - 正确位置：e[context_id] = y_pred[context_id] - 1（负数，代表预测不足）
                    - 其他位置：e[k] = y_pred[k] - 0（正数，代表预测过度）
                    后续更新会驱动模型：提高正确词概率，降低错误词概率
                    最终让y_pred的分布越来越接近真实的 One-Hot 分布。
                    '''
                    # 更新 W2：dLoss/dW2 = h^T @ e（外积等价于列向量×行向量）
                    dW2 = np.outer(h, e)
                    W2 -= lr * dW2

                    # 更新 W1：dLoss/dW1[target_id] = W2 @ e（仅更新目标词对应的那一行词向量，其他词保持不变）
                    dW1 = W2 @ e  # (5, 11) @ (11,) → 结果形状为 (5,)    e是(11,)的向量 → 相当于(11, 1)的列向量
                    W1[target_id] -= lr * dW1  # W1整体形状：(11, 5)   W1[target_id]的形状：(5,)    W1的每一行对应一个词的词向量，更新时只修改目标词对应的那一行，形状和dW1完全匹配

                    # 累加损失
                    total_loss += -np.log(y_pred[context_id] + 1e-10)

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:4d} | Loss: {total_loss:.4f}")

# ===================== 5. 开始训练 =====================
print("开始训练（NumPy 版）...")
train()

# ===================== 6. 查看词向量 =====================
'''
词向量绝对数值 = 没用，每次变是正常现象
词向量相对关系（相似度）= 唯一有用的东西
随机初始化 + 随机梯度下降 → 数值必然变动
空间旋转不变性 → 相似度永远不变
'''
print("\n=== 词向量 === (词向量的「绝对数值」毫无意义，「相对关系（相似度/距离）」才是唯一有价值的东西)")
for word in words:
    vec = W1[word_to_id[word]]
    print(f"{word:4s}: {vec.round(4)}")

# ===================== 7. 相似度 =====================
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("\n=== 相似度 ===")
print("苹果 ↔ 香蕉:", cos_sim(W1[word_to_id["苹果"]], W1[word_to_id["香蕉"]]).round(4))
print("苹果 ↔ 小猫:", cos_sim(W1[word_to_id["苹果"]], W1[word_to_id["小猫"]]).round(4))
print("苹果 ↔ 橙子:", cos_sim(W1[word_to_id["苹果"]], W1[word_to_id["橙子"]]).round(4))
print("小狗 ↔ 小猫:", cos_sim(W1[word_to_id["小狗"]], W1[word_to_id["小猫"]]).round(4))