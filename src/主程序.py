import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from loguru import logger
from functools import partial

from .数据处理 import 获取训练和测试数据
from .模型 import 分类序列数据集, 整理批次, 创建模型, 训练模型, 评估模型
from .基线模型 import 训练和评估CatBoost

def 主函数():
    # --- 配置 ---
    测试集比例 = 0.2
    随机种子 = 42
    批次大小 = 16
    学习率 = 5e-5
    训练周期 = 10
    数值分箱数量 = 50
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {设备}")

    # --- 1. 加载和预处理数据 ---
    logger.info("步骤 1: 加载和预处理数据...")
    训练序列, 测试序列, 词表, raw_data = 获取训练和测试数据(
        测试集比例=测试集比例, 随机种子=随机种子, 分箱数量=数值分箱数量
    )
    训练数据, 测试数据, 数值型列, 类别型列 = raw_data
    pad_token_id = 词表['[PAD]']
    logger.info("数据加载完成。")

    # --- 2. 训练和评估 CatBoost 基线模型 ---
    # logger.info("\n步骤 2: 运行 CatBoost 基线模型...")
    # catboost_accuracy = 训练和评估CatBoost(训练数据, 测试数据, 数值型列, 类别型列)
    # logger.info("CatBoost 基线模型运行完毕。")

    # --- 3. 训练和评估 Transformer 模型 ---
    logger.info("\n步骤 3: 运行 Transformer 自回归模型...")
    # a. 创建数据加载器
    训练集 = 分类序列数据集(训练序列)
    整理函数 = partial(整理批次, pad_token_id=pad_token_id)
    训练加载器 = DataLoader(训练集, batch_size=批次大小, shuffle=True, collate_fn=整理函数)

    # b. 创建模型和优化器
    模型 = 创建模型(len(词表))
    优化器 = AdamW(模型.parameters(), lr=学习率)

    # c. 创建学习率调度器
    总训练步数 = len(训练加载器) * 训练周期
    调度器 = get_linear_schedule_with_warmup(
        优化器,
        num_warmup_steps=0,
        num_training_steps=总训练步数
    )

    # d. 训练模型
    训练模型(模型, 训练加载器, 优化器, 调度器, 设备, 周期=训练周期)

    # e. 评估模型
    transformer_accuracy = 评估模型(模型, 测试序列, 词表, 设备)
    logger.info("Transformer 模型运行完毕。")

    # --- 4. 结果汇总 ---
    logger.info("\n--- 实验结果汇总 ---")
    # logger.info(f"CatBoost 基线模型准确率: {catboost_accuracy:.4f}")
    logger.info(f"Transformer 自回归模型准确率: {transformer_accuracy:.4f}")
    logger.info("--------------------")

if __name__ == "__main__":
    主函数() 