# 安装本书所需的 Python 包和库

本文档提供了关于如何检查 Python 版本和已安装包的详细信息。（如需了解如何安装 Python 和 Python 包，请参阅 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 目录。）

本书使用了[此处](https://github.com/rasbt/LLMs-from-scratch/blob/main/requirements.txt)列出的库。这些库的较新版本通常也是兼容的，但如果您在运行代码时遇到问题，可以尝试使用这些指定版本作为备用方案。

为了更方便地安装这些依赖项，您可以使用代码库根目录中的 `requirements.txt` 文件，并运行以下命令：

```bash
pip install -r requirements.txt
```

或者，您也可以通过 GitHub URL 安装，命令如下：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/requirements.txt
```

安装完成后，请使用以下命令检查所有包是否已安装并更新至最新版本：

```bash
python python_environment_check.py
```
![检查安装结果](https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_1.jpg)

建议在 JupyterLab 中运行本目录下的 `python_environment_check.ipynb` 文件来检查版本，结果应与上图一致。

![JupyterLab 检查结果](https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_2.jpg)

如果您看到以下问题，可能是您的 JupyterLab 实例连接到了错误的 conda 环境：

![JupyterLab 环境问题](https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/jupyter-issues.jpg)

此时，您可以使用 `watermark` 工具，并通过 `--conda` 标志检查是否在正确的 conda 环境中打开了 JupyterLab 实例：

![使用 watermark 检查环境](https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/watermark.jpg)

<br>
<br>

## 安装 PyTorch

PyTorch 可以像其他 Python 库一样使用 pip 安装。例如：

```bash
pip install torch
```

由于 PyTorch 是一个功能全面的库，支持 CPU 和 GPU 兼容的代码，安装时可能需要额外的设置和说明（更多信息请参阅本书的 *A.1.3 安装 PyTorch* 部分）。

强烈建议参考 PyTorch 官方网站 [https://pytorch.org](https://pytorch.org) 上的安装指南。

![PyTorch 安装界面](https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/pytorch-installer.jpg)

<br>

---

如有任何问题，欢迎在 [讨论区](https://github.com/rasbt/LLMs-from-scratch/discussions) 中提问。