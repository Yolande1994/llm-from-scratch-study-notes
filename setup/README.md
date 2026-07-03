# 可选的安装设置说明

本文档列出了设置机器并使用本仓库中代码的不同方法。我建议从上到下浏览各个部分，然后决定哪种方法最适合您的需求。

&nbsp;

## 快速开始

如果您的机器上已经安装了Python，最快的开始方式是通过执行以下pip安装命令，从本代码仓库的根目录安装[../requirements.txt](../requirements.txt)文件中的包依赖：
```bash
pip install -r requirements.txt
```

&nbsp;

# 本地设置

本节提供了在本地运行本书代码的建议。请注意，本书主章节中的代码设计为能够在常规笔记本电脑上运行，并且在合理的时间内完成，不需要专用硬件。我在M3款MacBook Air笔记本电脑上测试了所有主章节的代码。此外，如果您的笔记本或台式电脑配备了NVIDIA GPU，代码将自动利用该硬件。

&nbsp;

## 设置Python

如果您的机器上还没有设置Python，我在以下目录中写了我的个人Python设置偏好：

- [01_optional-python-setup-preferences](./01_optional-python-setup-preferences)
- [02_installing-python-libraries](./02_installing-python-libraries)

下面的*使用DevContainers*部分介绍了在您的机器上安装项目依赖的另一种方法。

&nbsp;

## 使用Docker DevContainers

作为上述*设置Python*部分的替代方案，如果您更倾向于一种隔离项目依赖和配置的开发环境，使用Docker是一个非常有效的解决方案。这种方法避免了手动安装软件包和库，并确保开发环境的一致性。您可以在以下链接中找到有关设置Docker和使用DevContainer的更多说明：

- [03_optional-docker-environment](03_optional-docker-environment)

&nbsp;

## Visual Studio Code 编辑器

有许多优秀的代码编辑器可供选择。我的首选是流行的开源编辑器[Visual Studio Code (VSCode)](https://code.visualstudio.com)，它可以通过许多实用的插件和扩展进行轻松增强（更多信息请见下面的*VSCode扩展*部分）。有关macOS、Linux和Windows的下载说明，请访问[VSCode官网](https://code.visualstudio.com)。

&nbsp;

## VSCode 扩展

如果您使用Visual Studio Code (VSCode)作为主要代码编辑器，您可以在`.vscode`子文件夹中找到推荐的扩展。这些扩展提供了增强的功能和对本仓库有帮助的工具。

要安装这些扩展，请在VSCode中打开此“setup”文件夹（文件 -> 打开文件夹...），然后点击右下角弹出菜单中的“安装”按钮。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/vs-code-extensions.webp?1" alt="1" width="700">

另外，您也可以将`.vscode`扩展文件夹移动到此GitHub仓库的根目录中：

```bash
mv setup/.vscode ./
```

然后，每次您打开`LLMs-from-scratch`主文件夹时，VSCode会自动检查推荐的扩展是否已经安装在您的系统中。

&nbsp;

# 云资源

本节介绍了运行本书中代码的云平台替代方案。

虽然代码可以在没有专用GPU的常规笔记本电脑和台式机上运行，但使用配备NVIDIA GPU的云平台可以显著提高代码的运行速度，特别是在第5到第7章中。

&nbsp;

## 使用Lightning Studio

为了在云端获得顺畅的开发体验，我推荐使用[Lightning AI Studio](https://lightning.ai/)平台，该平台允许用户设置持久化的环境，并在云端的CPU和GPU上使用VSCode和Jupyter Lab。

一旦您启动了新的Studio，可以打开终端并执行以下设置步骤来克隆仓库并安装依赖：

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -r requirements.txt
```

（与Google Colab不同，这些步骤只需执行一次，因为Lightning AI Studio环境是持久化的，即使在CPU和GPU机器之间切换，也不需要重新执行。）

接下来，导航到您想要运行的Python脚本或Jupyter笔记本。可选地，您还可以轻松地连接GPU来加速代码的运行速度，例如，在第5章进行LLM预训练或在第6和第7章进行微调时。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

&nbsp;

## 使用Google Colab

要在云端使用Google Colab环境，请访问[https://colab.research.google.com/](https://colab.research.google.com/)，并通过GitHub菜单或将笔记本拖入*上传*字段来打开相应章节的笔记本，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


同时，请确保将相关文件（数据集文件和笔记本所导入的.py文件）上传到Colab环境中，如下所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


您还可以通过更改*运行时*来选择性地在GPU上运行代码，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">

&nbsp;

# 问题的交流

如果您有任何问题，请随时通过本GitHub仓库中的[讨论区](https://github.com/rasbt/LLMs-from-scratch/discussions)与他们联系。