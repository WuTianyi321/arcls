import json
import os
import sys
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from tokenizers import ByteLevelBPETokenizer

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.数据处理 import 获取数据

def dataframe_to_string(df, 数值型列, 类别型列):
    """将DataFrame的每一行转换为结构化的字符串，并在末尾添加EOS token。"""
    string_rows = []
    for _, row in df.iterrows():
        parts = ["[CLS]"]
        for col in 数值型列 + 类别型列:
            parts.append(f"{col}={row[col]}")
        parts.append("[SEP]")
        parts.append(f"收入={row['收入']}")
        parts.append("</s>")
        string_rows.append(" | ".join(parts))
    return string_rows

def 主函数():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    随机种子 = 42
    词表大小 = 2000

    logger.info("1. 加载并划分数据...")
    数据, 数值型列, 类别型列 = 获取数据()
    训练数据, 测试数据 = train_test_split(数据, test_size=0.2, random_state=随机种子, stratify=数据['收入'])
    logger.info(f"数据划分完成。训练集: {len(训练数据)} 条, 测试集: {len(测试数据)} 条。")

    logger.info("2. 将训练数据序列化为字符串，并创建语料库文件...")
    训练字符串 = dataframe_to_string(训练数据, 数值型列, 类别型列)
    corpus_path = os.path.join(output_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in 训练字符串:
            f.write(line + "\n")
    logger.info(f"语料库已保存到 {corpus_path}")

    logger.info("3. 训练Hugging Face分词器...")
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "</s>"]
    
    # 使用封装好的ByteLevelBPETokenizer，它的API更直接
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[corpus_path], vocab_size=词表大小, min_frequency=2, special_tokens=special_tokens)

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logger.info(f"分词器已训练并保存到 {tokenizer_path}")

    logger.info("4. 使用新分词器编码数据...")
    # 编码训练集并写入jsonl
    train_output_path = os.path.join(output_dir, "train_data.jsonl")
    with open(train_output_path, "w", encoding="utf-8") as f:
        for text in 训练字符串:
            encoded = tokenizer.encode(text)
            f.write(json.dumps({"text": text, "token_ids": encoded.ids}, ensure_ascii=False) + "\n")
    logger.info(f"编码后的训练数据已写入到 {train_output_path}")

    # 编码测试集并写入jsonl
    测试字符串 = dataframe_to_string(测试数据, 数值型列, 类别型列)
    test_output_path = os.path.join(output_dir, "test_data.jsonl")
    with open(test_output_path, "w", encoding="utf-8") as f:
        for text in 测试字符串:
            encoded = tokenizer.encode(text)
            f.write(json.dumps({"text": text, "token_ids": encoded.ids}, ensure_ascii=False) + "\n")
    logger.info(f"编码后的测试数据已写入到 {test_output_path}")

    logger.info("5. 转换并保存为外部流水线所需的vocab.txt格式...")
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
    
    vocab_txt_path = os.path.join(output_dir, "vocab.txt")
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        for token, idx in sorted_vocab:
            line_idx = idx + 1
            token_for_eval = f"'{token}'"
            byte_length = len(token.encode('utf-8'))
            f.write(f"{line_idx} {token_for_eval} {byte_length}\n")
    logger.info(f"兼容格式的词表已写入到 {vocab_txt_path}")
    
    os.remove(corpus_path)
    logger.info("临时语料库文件已删除。")
    logger.info("数据准备完成！")

if __name__ == "__main__":
    主函数() 