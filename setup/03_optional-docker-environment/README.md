# Docker 环境设置指南

如果你更喜欢隔离项目的依赖项和配置，使用 Docker 是一种非常有效的解决方案。通过这种方法，可以避免手动安装软件包和库，同时确保开发环境的一致性。

本指南将引导你为本书设置一个可选的 Docker 环境，如果你不想使用 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 和 [../02_installing-python-libraries](../02_installing-python-libraries) 中解释的 conda 方法，可以参考此方法。

<br>

## 下载并安装 Docker

使用 Docker 的最简单方法是为你的平台安装 [Docker Desktop](https://docs.docker.com/desktop/)。

Linux（Ubuntu）用户可以选择安装 [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) 并按照 [安装后配置步骤](https://docs.docker.com/engine/install/linux-postinstall/) 进行操作。

<br>

## 在 Visual Studio Code 中使用 Docker DevContainer

Docker DevContainer（开发容器）是一种工具，它允许开发者将 Docker 容器作为功能完整的开发环境使用。这种方法可以确保用户快速搭建一个一致的开发环境，而无需依赖本地机器的配置。

虽然 DevContainer 也可以与其他 IDE 一起使用，但一个常用的 IDE/编辑器是 Visual Studio Code（VS Code）。以下指南说明了如何在 VS Code 环境中使用本书的 DevContainer，但类似的过程也适用于 PyCharm。如果你尚未安装 VS Code，并且想使用它，请 [下载](https://code.visualstudio.com/download)。

1. 克隆此 GitHub 仓库并使用 `cd` 命令进入项目根目录。

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
```

2. 将 `.devcontainer` 文件夹从 `setup/03_optional-docker-environment/` 移动到当前目录（项目根目录）。

```bash
mv setup/03_optional-docker-environment/.devcontainer ./
```

3. 在 Docker Desktop 中，确保 **_desktop-linux_ builder** 正在运行并将用于构建 Docker 容器（路径：_Docker Desktop_ -> _Change settings_ -> _Builders_ -> _desktop-linux_ -> _..._ -> _Use_）。

4. 
   1. 安装 NVIDIA Container Toolkit

    按照 [此处](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) 的说明安装 **NVIDIA Container Toolkit**。根据 [此处](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2) 的说明，NVIDIA Container Toolkit 在 WSL 2 上受支持。

    2. 在 Docker Engine 配置中添加 NVIDIA 运行时

        在 Docker Engine 守护程序（daemon）配置中添加 `_nvidia_` 作为运行时（路径：_Docker Desktop_ -> _更改设置_ -> _Docker Engine_）。在配置文件中添加以下内容：

    ```json
    "runtimes": {
        "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
    ```

    例如，完整的 Docker Engine 守护程序配置 JSON 代码应如下所示：

    ```json
    {
      "builder": {
        "gc": {
          "defaultKeepStorage": "20GB",
          "enabled": true
        }
      },
      "experimental": false,
      "runtimes": {
        "nvidia": {
          "path": "nvidia-container-runtime",
          "runtimeArgs": []
        }
      }
    }
    ```

    然后重启 Docker Desktop。


5. 在终端中输入 `code .` 以在 VS Code 中打开项目。或者，你也可以启动 VS Code，并从界面中选择项目打开。

6. 从 VS Code 左侧的 _Extensions_ 菜单中安装 **Remote Development** 扩展。

7. 打开 DevContainer。

由于 `.devcontainer` 文件夹位于主目录 `LLMs-from-scratch` 中（根据系统设置，可能无法直接看到以 `.` 开头的文件夹），VS Code 应自动检测到它，并询问你是否希望在 DevContainer 中打开项目。如果没有弹出提示，可以按 `Ctrl + Shift + P` 打开命令面板，输入 `dev containers` 查看所有 DevContainer 相关选项。

8. 选择 **Reopen in Container**。

如果尚未构建 Docker 镜像，Docker 将开始根据 `.devcontainer` 配置文件构建镜像；如果镜像已在注册表中，则会拉取镜像。

整个过程是自动化的，可能需要几分钟，具体取决于你的系统和网络速度。可以选择点击 VS Code 右下角的 "Starting Dev Container (show log)" 查看当前构建进度。

完成后，VS Code 将自动连接到容器并在新创建的 Docker 开发环境中重新打开项目。你可以像在本地机器上一样编写、运行和调试代码，同时享受 Docker 带来的隔离性和一致性。

> [!警告]
> 如果在构建过程中遇到错误，这可能是由于你的机器不支持 NVIDIA 容器工具包，因为你的 GPU 不兼容。在这种情况下，编辑 `devcontainer.json` 文件，删除 `"runArgs": ["--runtime=nvidia", "--gpus=all"],` 行，然后重新运行 "Reopen Dev Container" 步骤。

9. 完成。

镜像拉取和构建完成后，你的项目将挂载到容器内，并安装好所有必要的包，随时可以进行开发。

<br>

## 卸载 Docker 镜像

以下是卸载或移除 Docker 容器和镜像的说明，如果你不再计划使用它们。这一过程不会从系统中移除 Docker 本身，而是清理与项目相关的 Docker 文件。

1. 列出所有 Docker 镜像，以找到与 DevContainer 相关的镜像：
```bash
docker image ls
```

2. 使用镜像 ID 或名称删除 Docker 镜像：

```bash
docker image rm [IMAGE_ID_OR_NAME]
```

<br>

## 卸载 Docker

如果你决定不再使用 Docker 并希望卸载它，请参阅官方文档 [这里](https://docs.docker.com/desktop/uninstall/)，其中详细说明了针对不同操作系统的卸载步骤。
