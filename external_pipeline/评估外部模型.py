import fla
import torch
import json
import argparse
import pandas as pd
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import os

# 注意：此脚本不依赖src内部的逻辑，使其可以独立分发和运行
# 因此，我们在此处重新定义或复制必要的功能

def 加载词表(路径):
    """加载词表文件，使用eval正确解析带空格的token。"""
    词表 = {}
    with open(路径, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 采用更健壮的解析方式
                idx_part = line[:line.index(' ')]
                len_part = line[line.rindex(' '):]
                str_part = line[line.index(' '):line.rindex(' ')].strip()
                
                token_id = int(idx_part)
                token_str = eval(str_part)  # eval能正确处理如"'token '"这样的字符串
                
                词表[token_str] = token_id
            except (ValueError, IndexError, SyntaxError):
                logger.warning(f"无法解析词表行: {line.strip()}")
    return 词表

def 文本转ID序列(text, 词表):
    """使用贪心最长匹配算法将文本字符串转换回ID序列。"""
    ids = []
    idx = 0
    while idx < len(text):
        longest_match = ""
        # 寻找从当前位置开始的最长匹配token
        for token_str in 词表.keys():
            if text.startswith(token_str, idx):
                if len(token_str) > len(longest_match):
                    longest_match = token_str
        
        if longest_match:
            ids.append(词表[longest_match])
            idx += len(longest_match)
        else:
            # 正常情况下不应发生，因为我们的文本就是由词表token构成的
            logger.error(f"致命错误：在位置 {idx} 找不到匹配的token: '{text[idx:idx+20]}...'")
            # 为避免无限循环，跳过一个字符
            idx += 1
    return ids

def 加载测试数据(路径, 词表):
    """直接加载预处理好的jsonl测试数据，并使用最长匹配算法转换为ID序列。"""
    序列列表 = []
    with open(路径, 'r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line)['text']
            sequence_ids = 文本转ID序列(text, 词表)
            if sequence_ids:
                序列列表.append(sequence_ids)
    return 序列列表

def 主函数(args):
    """
    该脚本负责评估一个在外部训练好的、兼容transformers库的模型。
    新版：直接读取预处理和划分好的测试集。
    """
    logger.info(f"开始评估模型: {args.model_name}")
    设备 = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("1. 加载模型和词表...")
    模型 = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.dtype).to(设备)
    模型.eval()
    词表 = 加载词表(args.vocab_path)

    logger.info("2. 加载并解析预处理的测试数据...")
    测试序列 = 加载测试数据(args.test_data_path, 词表)
    if not 测试序列:
        logger.error("未能加载任何测试数据，程序退出。")
        return

    logger.info("3. 执行模型评估...")
    正确预测数 = 0
    with torch.no_grad():
        for 序列 in tqdm(测试序列, desc="评估中"):
            真实目标_id = 序列[-1]
            输入_ids = torch.tensor([序列[:-1]], device=设备)

            输出 = 模型(input_ids=输入_ids)
            logits = 输出.logits
            
            预测_logits = logits[0, -1, :]
            预测_id = torch.argmax(预测_logits).item()
            
            if 预测_id == 真实目标_id:
                正确预测数 += 1

    准确率 = 正确预测数 / len(测试序列) if 测试序列 else 0
    logger.info(f"模型 '{args.model_name}' 评估完成。")
    logger.info(f"准确率: {准确率:.4f} ({正确预测数}/{len(测试序列)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估一个外部训练的自回归分类模型")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("model_name", type=str, help="要评估的模型的名称或路径")
    parser.add_argument("--test_data_path", type=str, default=os.path.join(script_dir, "test_data.jsonl"), help="预处理后的测试集文件路径 (.jsonl)")
    parser.add_argument("--vocab_path", type=str, default=os.path.join(script_dir, "vocab.txt"), help="词表文件路径")
    parser.add_argument("--dtype", type=lambda x: getattr(torch, x), default=torch.float16, help="模型的数据类型 (例如 'float16', 'float32')")
    
    args = parser.parse_args()
    主函数(args) 