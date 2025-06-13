import json
import numpy as np
import pandas as pd
from loguru import logger
import sys
import os

# 将项目根目录添加到Python路径中，以便能找到`src`模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 为了让此脚本独立，我们从主项目中复制必要的函数
from src.数据处理 import 获取数据, 创建词表和分箱

def 主函数():
    """
    该脚本负责准备数据，以适配外部的Transformer训练流水线。
    主要改动：为所有非目标Token添加尾随空格，以简化外部流水线的分词。
    """
    # 获取脚本所在目录，确保输出文件在该目录下
    output_dir = os.path.dirname(os.path.abspath(__file__))

    logger.info("开始导出数据用于外部流水线训练...")
    分箱数量 = 50

    数据, 数值型列, 类别型列 = 获取数据()
    logger.info(f"完整数据集加载完成，共 {len(数据)} 条记录。")

    词表, 分箱信息 = 创建词表和分箱(数据, 数值型列, 类别型列, 分箱数量=分箱数量)

    logger.info("正在将数据转换为带空格的text格式...")
    文本行 = []
    最长序列长度 = 0
    for _, 行 in 数据.iterrows():
        # 步骤 1: 构建原始的、无空格的token列表
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
        
        # 更新最长序列长度
        if len(raw_tokens) > 最长序列长度:
            最长序列长度 = len(raw_tokens)

        # 步骤 2: 为非目标token添加尾随空格，然后直接拼接
        final_tokens = []
        for token in raw_tokens:
            if token.startswith("[TARGET_"):
                final_tokens.append(token)
            else:
                final_tokens.append(token + " ")
        
        文本行.append("".join(final_tokens))

    # 写入 jsonl 文件
    输出文件_jsonl = os.path.join(output_dir, "train_data.jsonl")
    with open(输出文件_jsonl, 'w', encoding='utf-8') as f:
        for text in 文本行:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    logger.info(f"数据已写入到 {输出文件_jsonl}")

    # 写入词表文件 (ID从1开始)
    输出文件_词表 = os.path.join(output_dir, "vocab.txt")
    反转词表 = {v: k for k, v in 词表.items()}
    
    with open(输出文件_词表, 'w', encoding='utf-8') as f:
        for token_id in range(len(反转词表)):
            token_str = 反转词表[token_id]
            
            if token_str.startswith("[TARGET_"):
                final_token_str = token_str
            else:
                final_token_str = token_str + " "
            
            f.write(f"{token_id + 1} '{final_token_str}' {len(final_token_str)}\n")
            
    logger.info(f"词表已写入到 {输出文件_词表}")

    # 保存分箱信息，供评估脚本使用
    输出文件_分箱 = os.path.join(output_dir, "bins.json")
    分箱信息_可序列化 = {k: v.tolist() if v is not None else None for k, v in 分箱信息.items()}
    with open(输出文件_分箱, 'w', encoding='utf-8') as f:
        json.dump(分箱信息_可序列化, f)
    logger.info(f"分箱信息已写入到 {输出文件_分箱}")

    logger.info("-" * 30)
    logger.info(f"最长数字序列长度 (max_sequence_length): {最长序列长度}")
    logger.info("数据导出完成！")


if __name__ == "__main__":
    主函数() 