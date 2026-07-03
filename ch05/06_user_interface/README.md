# **构建用户界面，与预训练 LLM 进行交互**  

本目录提供 **ChatGPT 风格的用户界面**，用于与 **第 5 章 训练的 LLM 进行交互**，示例如下：  

![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-orig.webp)  

为了实现此用户界面，我们使用了 **开源的 [Chainlit Python 库](https://github.com/Chainlit/chainlit)**。  

---

## **步骤 1：安装依赖**  

首先，通过以下命令安装 `chainlit` 包：

```bash
pip install chainlit
```

（或者执行 `pip install -r requirements-extra.txt` 进行安装。）  

---

## **步骤 2：运行 `app` 代码**  

本目录包含以下两个文件：  

1. **[`app_orig.py`](app_orig.py)**：  
   - 该文件 **加载并使用 OpenAI 提供的原始 GPT-2 权重**。  

2. **[`app_own.py`](app_own.py)**：  
   - 该文件 **加载并使用我们在第 5 章生成的 GPT-2 权重**。  
   - 运行该文件前，需 **先执行** [`../01_main-chapter-code/ch05.ipynb`](../01_main-chapter-code/ch05.ipynb)。  

（建议打开并阅读这些文件，以了解其具体实现。）  

---

从终端运行以下命令之一，即可启动 UI 服务器：

```bash
chainlit run app_orig.py
```

or

```bash
chainlit run app_own.py
```

运行上述命令后，系统应 **自动在浏览器中打开新标签页**，让你直接与模型交互。  

如果浏览器 **未自动打开**，请检查终端输出，并 **复制本地访问地址**（通常为 **`http://localhost:8000`**）到浏览器地址栏手动访问。  