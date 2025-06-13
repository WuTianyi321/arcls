# 用于可解释性的自回归分类模型

本项目旨在验证一个想法：将混合类型数据（不定长动态序列和静态表格特征）及预测标签统一Token化为单一序列，利用自回归模型（如Transformer解码器）学习从输入特征序列到目标类别标签Token的生成式预测。

## 项目结构
- `src/`: 包含项目核心的Python模块。
    - `数据处理.py`: 数据加载和Token化逻辑。
    - `模型.py`: 内置Transformer模型（Qwen2）的定义、训练和评估。
    - `基线模型.py`: CatBoost基线模型。
    - `主程序.py`: 驱动内部实验的核心逻辑。
- `external_pipeline/`: 包含用于对接外部训练流水线的脚本和产出物。
    - `导出数据.py`: 将数据处理并导出为外部流水线所需的格式。
    - `评估外部模型.py`: 用于评估在外部训练好的模型。
- `main.py`: 项目主入口点，用于运行内部实验。
- `pyproject.toml`: 项目依赖和配置。
- `.gitignore`: Git忽略文件配置。

## 使用说明

本项目使用 `uv`进行环境管理。

### 1. 环境设置

创建虚拟环境并安装所有依赖（包括GPU版本的PyTorch）：
```bash
uv venv
uv sync
```

### 2. 运行内置实验

这将运行 `main.py`，它会依次训练和评估CatBoost基线模型和项目内定义的Qwen2模型。
```bash
uv run python main.py
```

### 3. 对接外部训练流水线

如果你想使用自己的外部训练流水线，可以按以下步骤操作：

**a. 导出数据**

运行以下命令，它会在 `external_pipeline/` 目录下生成 `train_data.jsonl`, `test_data.jsonl`, `vocab.txt`, 和 `bins.json` 文件。
```bash
uv run python external_pipeline/导出数据.py
```

**b. 外部训练**

使用上一步生成的 `train_data.jsonl` 和 `vocab.txt` 在你的流水线中训练模型。

**c. 评估外部模型**

在你训练好模型并将其保存后（例如，保存为 `my_model`），运行以下命令进行评估。该脚本会默认使用同目录下的 `test_data.jsonl` 和 `vocab.txt`。
```bash
# 评估保存在本地路径的模型
uv run python external_pipeline/评估外部模型.py ./path/to/my_model

# 评估来自Hugging Face Hub的模型
uv run python external_pipeline/评估外部模型.py username/my_model_on_hub
```
