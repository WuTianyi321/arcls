import torch
import json
import argparse
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from sklearn.model_selection import train_test_split
import os

# 注意：此脚本不依赖src内部的逻辑，使其可以独立分发和运行
# 因此，我们在此处重新定义或复制必要的功能

def 加载原始数据(url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"):
    列名 = [
        "年龄", "工作类型", "fnlwgt", "教育", "教育年数", "婚姻状况",
        "职业", "亲属关系", "种族", "性别", "资本利得", "资本损失",
        "周工作小时数", "原籍国", "收入"
    ]
    数据 = pd.read_csv(url, header=None, names=列名, sep=r",\s*", engine="python", na_values="?")
    数据.dropna(inplace=True)
    数据["收入"] = 数据["收入"].apply(lambda x: 1 if x == ">50K" else 0)
    数值型列 = [col for col in 数据.select_dtypes(include=np.number).columns if col != '教育年数']
    类别型列 = [col for col in 数据.select_dtypes(include="object").columns if col != '收入']
    return 数据, 数值型列, 类别型列

def 加载词表(路径):
    词表 = {}
    with open(路径, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 格式: ID 'token' length
                parts = line.strip().split(' ', 2)
                token_id = int(parts[0])
                token_str = parts[1].strip("'")
                词表[token_str] = token_id
            except (ValueError, IndexError):
                logger.warning(f"无法解析词表行: {line.strip()}")
    return 词表

def 主函数(args):
    """
    该脚本负责评估一个在外部训练好的、兼容transformers库的模型。
    1. 加载评估所需的资源：模型、词表、分箱信息。
    2. 加载原始数据并划分出固定的测试集。
    3. 使用加载的词表和分箱信息，将测试数据Token化。
    4. 逐条评估，计算准确率。
    """
    logger.info(f"开始评估模型: {args.model_name}")
    设备 = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载资源
    logger.info("加载模型、词表和分箱信息...")
    模型 = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.dtype).to(设备)
    模型.eval()

    词表 = 加载词表(args.vocab_path)
    with open(args.bins_path, 'r') as f:
        分箱信息 = json.load(f)
        for k, v in 分箱信息.items():
            if v is not None:
                分箱信息[k] = np.array(v)

    # 2. 获取并处理测试数据
    logger.info("加载并处理测试数据...")
    数据, 数值型列, 类别型列 = 加载原始数据()
    _, 测试数据 = train_test_split(数据, test_size=0.2, random_state=42, stratify=数据['收入'])
    
    测试序列 = []
    for _, 行 in tqdm(测试数据.iterrows(), total=len(测试数据), desc="Tokenizing测试集"):
        # 与新版导出脚本完全相同的Token化逻辑
        # 1. 构建原始token列表
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
        
        # 2. 将原始token转换为带空格的token
        特征部分_spaced = [t + " " for t in raw_tokens]
        目标_str = f"[TARGET_{行['收入']}]"
        
        # 3. 使用加载的词表将字符串转换为ID
        try:
            特征部分_ids = [词表[t] for t in 特征部分_spaced]
            目标_id = 词表[目标_str] # 目标token在词表中没有空格
            测试序列.append(特征部分_ids + [目标_id])
        except KeyError as e:
            logger.warning(f"在词表中找不到token: {e}，跳过此行。")

    # 3. 执行评估
    logger.info("开始执行模型评估...")
    正确预测数 = 0
    with torch.no_grad():
        for 序列 in tqdm(测试序列, desc="评估中"):
            真实目标_id = 序列[-1]
            输入_ids = torch.tensor([序列[:-1]], device=设备)

            输出 = 模型(input_ids=输入_ids)
            logits = 输出.logits
            
            # 获取序列最后一个位置的logits，并找到ID最大的token
            预测_logits = logits[0, -1, :]
            预测_id = torch.argmax(预测_logits).item()
            
            if 预测_id == 真实目标_id:
                正确预测数 += 1

    准确率 = 正确预测数 / len(测试序列) if 测试序列 else 0
    logger.info(f"模型 '{args.model_name}' 评估完成。")
    logger.info(f"准确率: {准确率:.4f} ({正确预测数}/{len(测试序列)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估一个外部训练的自回归分类模型")
    
    # 获取脚本所在目录，确保默认文件路径正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("model_name", type=str, help="要评估的模型的名称或路径 (例如 'myttt196')")
    parser.add_argument("--vocab_path", type=str, default=os.path.join(script_dir, "vocab.txt"), help="词表文件路径")
    parser.add_argument("--bins_path", type=str, default=os.path.join(script_dir, "bins.json"), help="分箱信息文件路径")
    parser.add_argument("--dtype", type=lambda x: getattr(torch, x), default=torch.float16, help="模型的数据类型 (例如 'float16', 'float32')")
    
    args = parser.parse_args()
    主函数(args) 