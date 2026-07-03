# 版权所有 (c) Sebastian Raschka，遵循 Apache 2.0 开源协议（详见 LICENSE.txt）。
# 《从零搭建大语言模型》配套源码
#   - 书籍官网：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.metrics import balanced_accuracy_score
from sklearn.dummy import DummyClassifier


def load_dataframes():
    # 读取训练集、验证集、测试集CSV文件并返回
    df_train = pd.read_csv("train.csv")
    df_val = pd.read_csv("validation.csv")
    df_test = pd.read_csv("test.csv")

    return df_train, df_val, df_test


def eval(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # 执行模型预测
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # 计算标准准确率与均衡准确率（均衡准确率代码已注释）
    accuracy_train = accuracy_score(y_train, y_pred_train)
    # balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)

    accuracy_val = accuracy_score(y_val, y_pred_val)
    # balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    # balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)

    # 打印各数据集准确率结果
    print(f"Training Accuracy: {accuracy_train*100:.2f}%")
    print(f"Validation Accuracy: {accuracy_val*100:.2f}%")
    print(f"Test Accuracy: {accuracy_test*100:.2f}%")

    # print(f"\nTraining Balanced Accuracy: {balanced_accuracy_train*100:.2f}%")
    # print(f"Validation Balanced Accuracy: {balanced_accuracy_val*100:.2f}%")
    # print(f"Test Balanced Accuracy: {balanced_accuracy_test*100:.2f}%")


if __name__ == "__main__":
    df_train, df_val, df_test = load_dataframes()

    #########################################
    # 将文本转换为词袋（Bag-of-Words）向量表示
    vectorizer = CountVectorizer()
    #########################################

    # 训练集拟合词袋转换器并完成向量化，验证/测试集仅执行转换
    X_train = vectorizer.fit_transform(df_train["text"])
    X_val = vectorizer.transform(df_val["text"])
    X_test = vectorizer.transform(df_test["text"])
    y_train, y_val, y_test = df_train["label"], df_val["label"], df_test["label"]

    #####################################
    # 模型训练与效果评估流程
    #####################################

    # 构建基准哑分类器：预测训练集中占比最高的类别（简单基线对照）
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)

    print("Dummy classifier:")
    eval(dummy_clf, X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n\nLogistic regression classifier:")
    # 逻辑回归分类模型，增大迭代次数保证收敛
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    eval(model, X_train, y_train, X_val, y_val, X_test, y_test)