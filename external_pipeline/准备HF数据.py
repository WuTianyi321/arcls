import json
import os
import sys
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
# 核心修正：切换回最终正确的ByteLevelBPETokenizer
from tokenizers import ByteLevelBPETokenizer

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.数据处理 import 获取数据

def 数据帧转为字符串列表(数据帧, 数值型列, 类别型列):
    """将DataFrame的每一行转换为结构化的字符串，保留空格作为BPE学习的边界。"""
    字符串行列表 = []
    for _, 单行数据 in 数据帧.iterrows():
        特征部分列表 = []
        for 列名 in 数值型列 + 类别型列:
            特征部分列表.append(f"{列名}={单行数据[列名]}")
        特征字符串 = " ".join(特征部分列表)
        最终字符串 = f"[CLS] {特征字符串} [SEP] [TARGET_{单行数据['收入']}] </s>"
        字符串行列表.append(最终字符串)
    return 字符串行列表

def 主函数():
    输出目录 = os.path.dirname(os.path.abspath(__file__))
    随机种子 = 42
    词表大小 = 8000 # BPE词表可以适当增大，以学习更丰富的子词
    分箱数 = 100

    logger.info(f"1. 加载并进行分箱处理 (分箱数={分箱数})...")
    完整数据, 数值型列, 类别型列 = 获取数据(分箱数=分箱数)
    if 完整数据 is None:
        logger.error("数据加载失败，程序终止。")
        return
    训练数据, 测试数据 = train_test_split(完整数据, test_size=0.2, random_state=随机种子, stratify=完整数据['收入'])

    logger.info("2. 创建临时语料库文件用于BPE训练...")
    训练字符串列表 = 数据帧转为字符串列表(训练数据, 数值型列, 类别型列)
    语料库路径 = os.path.join(输出目录, "corpus.txt")
    with open(语料库路径, "w", encoding="utf-8") as 文件:
        for 行 in 训练字符串列表:
            文件.write(行 + "\n")

    logger.info("3. 训练Byte-Level BPE分词器...")
    特殊Token列表 = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "</s>", "[TARGET_0]", "[TARGET_1]"]
    
    分词器 = ByteLevelBPETokenizer()
    分词器.train(
        files=[语料库路径], 
        vocab_size=词表大小, 
        min_frequency=2, 
        special_tokens=特殊Token列表
    )
    
    # ---------- GPT-2 byte <-> unicode 映射 ----------
    def _bytes_to_unicode():
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))

    _byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}

    def _token_to_bytes(token_str: str) -> bytes:
        return bytes([_byte_decoder[ch] for ch in token_str])

    分词器路径 = os.path.join(输出目录, "tokenizer.json")
    分词器.save(分词器路径)
    logger.info(f"分词器已训练并保存到 {分词器路径}")

    logger.info("4. 使用新分词器编码数据...")
    训练输出路径 = os.path.join(输出目录, "train_data.jsonl")
    with open(训练输出路径, "w", encoding="utf-8") as 文件:
        for 文本 in 训练字符串列表:
            编码结果 = 分词器.encode(文本)
            文件.write(json.dumps({"text": 文本, "token_ids": 编码结果.ids}, ensure_ascii=False) + "\n")

    测试字符串列表 = 数据帧转为字符串列表(测试数据, 数值型列, 类别型列)
    测试输出路径 = os.path.join(输出目录, "test_data.jsonl")
    with open(测试输出路径, "w", encoding="utf-8") as 文件:
        for 文本 in 测试字符串列表:
            编码结果 = 分词器.encode(文本)
            文件.write(json.dumps({"text": 文本, "token_ids": 编码结果.ids}, ensure_ascii=False) + "\n")

    logger.info("5. 生成与外部流水线完全兼容的vocab.txt...")
    最终词表 = 分词器.get_vocab()
    排序后词表 = sorted(最终词表.items(), key=lambda item: item[1])
    
    词表文本路径 = os.path.join(输出目录, "vocab.txt")
    with open(词表文本路径, "w", encoding="utf-8") as 文件:
        for 词元_字符串, 索引 in 排序后词表:
            try:
                # ByteLevelBPETokenizer 将每个字节映射到 Unicode 码点，我们需要先还原原始字节
                原始字节: bytes = _token_to_bytes(词元_字符串)

                # Helper: 将 bytes 转为 TRIE_TOKENIZER 友好的 Python 字面量
                def bytes_to_literal(b: bytes) -> str:
                    special_chars = {'\\', "'", '"'}
                    is_printable_ascii = all(32 <= x <= 126 for x in b) and not any(chr(x) in special_chars for x in b)

                    if is_printable_ascii:
                        return f"'{b.decode('ascii')}'"  # 直接用字符串字面量

                    # 如果所有字节 < 0x80，则用 \xNN 转义放入字符串字面量
                    if all(x < 128 for x in b):
                        escaped = ''.join(chr(x) if 32 <= x <= 126 and chr(x) not in "\\'\"" else f"\\x{x:02x}" for x in b)
                        return f"'{escaped}'"

                    # 否则使用 bytes 字面量
                    escaped_bytes = ''.join(f"\\x{x:02x}" for x in b)
                    return f"b'{escaped_bytes}'"

                用于评估的词元 = bytes_to_literal(原始字节)
                字节长度 = len(原始字节)
            except Exception as e:
                # 如果转换失败，跳过这个词元
                logger.warning(f"跳过无法处理的词元索引 {索引}: {e}")
                continue
            
            行索引 = 索引 + 1
            # 使用空格分隔，但确保格式正确：索引 词元表示 字节长度
            文件.write(f"{行索引} {用于评估的词元} {字节长度}\n")
    
    # 清理临时文件
    os.remove(语料库路径)
    logger.info(f"临时语料库文件 '{语料库路径}' 已删除。")
    logger.info("最终数据准备流程完成！")

if __name__ == "__main__":
    主函数() 