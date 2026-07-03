# Python 环境设置指南

以下介绍了几种安装 Python 并设置计算环境的方法。这是我的个人推荐方法。

（我使用的是运行 macOS 的计算机，但此工作流程与 Linux 机器类似，可能也适用于其他操作系统。）

<br>
<br>

## 1. 下载并安装 Miniforge

从 [GitHub 仓库](https://github.com/conda-forge/miniforge) 下载 Miniforge。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/download.png" alt="download" width="600px">

根据操作系统的不同，这会下载一个 `.sh` 文件（适用于 macOS 和 Linux）或 `.exe` 文件（适用于 Windows）。

对于 `.sh` 文件，打开命令行终端并执行以下命令：

```bash
sh ~/Desktop/Miniforge3-MacOSX-arm64.sh
```

其中，`Desktop/` 是下载 Miniforge 安装程序的文件夹。在你的计算机上，可能需要将其替换为 `Downloads/`。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/miniforge-install.png" alt="miniforge-install" width="600px">

接下来，按照下载说明逐步操作，按 "Enter" 键确认。


<br>
<br>


## 2. 创建新的虚拟环境

安装成功后，我建议创建一个名为 `LLMs` 的新虚拟环境，你可以通过执行以下命令完成：

```bash
conda create -n LLMs python=3.10
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/new-env.png" alt="new-env" width="600px">

> 许多科学计算库并不立即支持最新版本的 Python。因此，在安装 PyTorch 时，建议使用比最新版本低一到两个版本的 Python。例如，如果最新版本是 Python 3.13，推荐使用 Python 3.10 或 3.11。

接下来，激活你新创建的虚拟环境（每次打开新的终端窗口或标签页时都需要激活）： 

```bash
conda activate LLMs
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/activate-env.png" alt="activate-env" width="600px">

<br>
<br>

## 可选：美化你的terminal

如果你希望将终端样式设置得与我的类似，以便显示当前激活的虚拟环境，可以查看 [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh) 项目。

<br>
<br>

## 3. 安装新的 Python 库

现在，你可以使用 `conda` 包管理器来安装新的 Python 库。例如，可以通过以下命令安装 [JupyterLab](https://jupyter.org/install) 和 [watermark](https://github.com/rasbt/watermark)：

```bash
conda install jupyterlab watermark
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/conda-install.png" alt="conda-install" width="600px">



你也可以继续使用 `pip` 安装库。默认情况下，`pip` 应已链接到你的新 `LLMs` conda 环境：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/check-pip.png" alt="check-pip" width="600px">

<br>
<br>

## 4. 安装 PyTorch

PyTorch 可以像其他 Python 库或包一样通过 pip 安装。例如：

```bash
pip install torch
```

然而，由于 PyTorch 是一个功能全面的库，支持 CPU 和 GPU 代码，其安装可能需要额外的设置和说明（更多信息请参阅书中 *A.1.3 Installing PyTorch* 部分）。

同时，强烈建议参考 PyTorch 官方网站的安装指南：[https://pytorch.org](https://pytorch.org)。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/pytorch-installer.jpg" width="600px">

## 5. 安装本书使用的 Python 包和库

请参考 [Installing Python packages and libraries used in this book](../02_installing-python-libraries/README.md) 文档，了解如何安装所需库。

<br>

---

有任何问题？欢迎在 [讨论论坛](https://github.com/rasbt/LLMs-from-scratch/discussions) 中提问。