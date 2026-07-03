# 构建用户界面以交互指令微调的 GPT 模型

本 **附加文件夹（bonus folder）** 包含 **运行 ChatGPT 风格用户界面** 的代码，以便与 **第 7 章** 训练的 **指令微调 GPT** 进行交互，如下所示：

![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-sft.webp?2)

为了实现该用户界面，我们使用了 **开源 Python 库** [Chainlit](https://github.com/Chainlit/chainlit)。

&nbsp;

## 步骤 1：安装依赖项

首先，我们通过以下命令安装 `chainlit` 库：


```bash
pip install chainlit
```

（或者，执行 `pip install -r requirements-extra.txt`。）

&nbsp;

## 步骤 2：运行 `app` 代码

[`app.py`](app.py) 文件包含 **基于 UI 的代码**，您可以打开并检查这些文件，以了解其具体实现。  

该文件 **加载并使用** 我们在 **第 7 章** 生成的 **GPT-2 权重**，因此需要 **先执行** [`../01_main-chapter-code/ch07.ipynb`](../01_main-chapter-code/ch07.ipynb)。  

在终端中执行以下命令，以启动 **UI 服务器**：


```bash
chainlit run app.py
```

- 运行上述命令后，应会 **自动打开一个新的浏览器标签页**，您可以在其中与模型进行交互。  
- 如果浏览器未自动打开，请检查 **终端输出**，然后 **手动复制本地地址** 到浏览器地址栏（通常是 `http://localhost:8000`）。  
