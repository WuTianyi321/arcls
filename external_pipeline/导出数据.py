import json
import numpy as np
import pandas as pd
from loguru import logger
import sys
import os
from sklearn.model_selection import train_test_split

# 将项目根目录添加到Python路径中，以便能找到`src`模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 为了让此脚本独立，我们从主项目中复制必要的函数
from src.数据处理 import 获取数据, 创建词表和分箱

def 数据集转为文本行(数据集, 分箱信息, 数值型列, 类别型列):
    """一个辅助函数，用于将给定的数据集（训练集或测试集）转换为文本行列表。"""
    文本行 = []
    最长序列长度 = 0
    for _, 行 in 数据集.iterrows():
        raw_tokens = ["[CLS]"]
        for 列 in 数值型列:
            if 分箱信息.get(列) is not None:
                bins = 分箱信息[列]
                箱体索引 = np.digitize(行[列], bins=bins) - 1
                箱体索引 = np.clip(箱体索引, 0, len(bins) - 2)
                raw_tokens.append(f"{列}_bin_{箱体索引}")

        for 列 in 类别型列:
            raw_tokens.append(f"{列}_{行[列]}")

        raw_tokens.append("[SEP]")
        raw_tokens.append(f"[TARGET_{行['收入']}]")

        if len(raw_tokens) > 最长序列长度:
            最长序列长度 = len(raw_tokens)
        
        final_tokens = [t + " " if not t.startswith("[TARGET_") else t for t in raw_tokens]
        文本行.append("".join(final_tokens))
    return 文本行, 最长序列长度

def 主函数():
    """
    该脚本负责准备数据，以适配外部的Transformer训练流水线。
    核心修正：
    1. 将数据预先划分为训练集和测试集。
    2. 词表和分箱信息严格只从训练集上构建，防止信息泄露。
    3. 分别导出 train_data.jsonl 和 test_data.jsonl。
    """
    output_dir = os.path.dirname(os.path.abspath(__file__))
    分箱数量 = 50
    随机种子 = 42

    logger.info("1. 加载并划分数据...")
    数据, 数值型列, 类别型列 = 获取数据()
    训练数据, 测试数据 = train_test_split(数据, test_size=0.2, random_state=随机种子, stratify=数据['收入'])
    logger.info(f"数据划分完成。训练集: {len(训练数据)} 条, 测试集: {len(测试数据)} 条。")

    logger.info("2. 从训练集创建词表和分箱信息...")
    词表, 分箱信息 = 创建词表和分箱(训练数据, 数值型列, 类别型列, 分箱数量=分箱数量)

    logger.info("3. 处理并导出训练集...")
    训练文本行, 训练最长序列 = 数据集转为文本行(训练数据, 分箱信息, 数值型列, 类别型列)
    训练输出文件 = os.path.join(output_dir, "train_data.jsonl")
    with open(训练输出文件, 'w', encoding='utf-8') as f:
        for text in 训练文本行:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    logger.info(f"训练数据已写入到 {训练输出文件}")

    logger.info("4. 处理并导出测试集...")
    测试文本行, 测试最长序列 = 数据集转为文本行(测试数据, 分箱信息, 数值型列, 类别型列)
    测试输出文件 = os.path.join(output_dir, "test_data.jsonl")
    with open(测试输出文件, 'w', encoding='utf-8') as f:
        for text in 测试文本行:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    logger.info(f"测试数据已写入到 {测试输出文件}")
    
    logger.info("5. 写入元数据文件...")
    # 写入词表
    输出文件_词表 = os.path.join(output_dir, "vocab.txt")
    反转词表 = {v: k for k, v in 词表.items()}
    with open(输出文件_词表, 'w', encoding='utf-8') as f:
        for token_id in sorted(反转词表.keys()):
            token_str = 反转词表[token_id]
            final_token_str = token_str + " " if not token_str.startswith("[TARGET_") else token_str
            f.write(f"{token_id + 1} '{final_token_str}' {len(final_token_str.encode('utf-8'))}\n")
    logger.info(f"词表已写入到 {输出文件_词表}")

    # 写入分箱信息
    输出文件_分箱 = os.path.join(output_dir, "bins.json")
    分箱信息_可序列化 = {k: v.tolist() if v is not None else None for k, v in 分箱信息.items()}
    with open(输出文件_分箱, 'w', encoding='utf-8') as f:
        json.dump(分箱信息_可序列化, f)
    logger.info(f"分箱信息已写入到 {输出文件_分箱}")

    logger.info("-" * 30)
    logger.info(f"训练集最长序列: {训练最长序列}")
    logger.info(f"测试集最长序列: {测试最长序列}")
    logger.info(f"用于配置模型的最大序列长度: {max(训练最长序列, 测试最长序列)}")
    logger.info("数据导出完成！")


if __name__ == "__main__":
    主函数() 