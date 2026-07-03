# AWS CloudFormation 模板：包含 LLMs-from-scratch 仓库的 Jupyter Notebook

此 CloudFormation 模板在 Amazon SageMaker 中创建一个启用 GPU 的 Jupyter Notebook，并配置执行角色和 LLMs-from-scratch GitHub 仓库。

## 功能说明：

1. 创建一个具有必要权限的 IAM 角色，用于 SageMaker Notebook 实例。
2. 创建一个 KMS 密钥和别名，用于加密 Notebook 实例。
3. 配置 Notebook 实例生命周期配置脚本，执行以下操作：
   - 在用户主目录下安装独立的 Miniconda。
   - 创建一个自定义 Python 环境，安装 TensorFlow 2.15.0 和 PyTorch 2.1.0，并支持 CUDA。
   - 安装其他常用库，如 Jupyter Lab、Matplotlib 等。
   - 将自定义环境注册为 Jupyter 内核。
4. 使用指定配置创建 SageMaker Notebook 实例，包括启用 GPU 的实例类型、执行角色和默认代码仓库。

## 使用步骤：

1. 下载 CloudFormation 模板文件（`cloudformation-template.yml`）。
2. 在 AWS 管理控制台中，导航至 CloudFormation 服务。
3. 创建新堆栈并上传模板文件。
4. 为 Notebook 实例命名（例如：“LLMsFromScratchNotebook”）（默认使用 LLMs-from-scratch GitHub 仓库）。
5. 审核并接受模板参数后创建堆栈。
6. 堆栈创建完成后，SageMaker Notebook 实例将在 SageMaker 控制台中可用。
7. 打开 Notebook 实例，开始使用预配置环境进行 LLMs-from-scratch 项目开发。

## 关键要点：

- 模板创建启用 GPU 的 Notebook 实例（`ml.g4dn.xlarge`），并配备 50GB 存储。
- 配置一个自定义 Miniconda 环境，安装 TensorFlow 2.15.0 和 PyTorch 2.1.0，支持 CUDA。
- 自定义环境注册为 Jupyter 内核，供 Notebook 使用。
- 模板还创建了一个 KMS 密钥以加密 Notebook 实例，并配置了所需权限的 IAM 角色。
