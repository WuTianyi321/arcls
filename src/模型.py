import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2Config, Qwen2ForCausalLM
from loguru import logger
from tqdm import tqdm
import random

class 分类序列数据集(Dataset):
    def __init__(self, 序列列表):
        self.序列列表 = 序列列表

    def __len__(self):
        return len(self.序列列表)

    def __getitem__(self, idx):
        return torch.tensor(self.序列列表[idx], dtype=torch.long)

def 整理批次(批次, pad_token_id):
    序列 = pad_sequence(批次, batch_first=True, padding_value=pad_token_id)
    注意力掩码 = (序列 != pad_token_id).long()
    return 序列, 注意力掩码

def 创建模型(词表大小):
    配置 = Qwen2Config(
        vocab_size=词表大小,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=2048,
        max_position_embeddings=512,
        bos_token_id=2, # [CLS]
        eos_token_id=3, # [SEP]
        pad_token_id=0, # [PAD]
    )
    模型 = Qwen2ForCausalLM(配置)
    logger.info(f"Qwen2 模型创建完成，参数量: {模型.num_parameters() / 1e6:.2f}M")
    return 模型

def 训练模型(模型, 训练加载器, 优化器, 设备, 周期=3):
    模型.train()
    模型.to(设备)
    for i in range(周期):
        logger.info(f"开始第 {i+1}/{周期} 轮训练")
        总损失 = 0
        for 批次 in tqdm(训练加载器, desc=f"训练周期 {i+1}"):
            输入序列, 注意力掩码 = 批次
            输入序列, 注意力掩码 = 输入序列.to(设备), 注意力掩码.to(设备)

            优化器.zero_grad()

            输出 = 模型(input_ids=输入序列, attention_mask=注意力掩码, labels=输入序列)
            损失 = 输出.loss
            损失.backward()
            优化器.step()
            总损失 += 损失.item()

        平均损失 = 总损失 / len(训练加载器)
        logger.info(f"周期 {i+1} 完成，平均损失: {平均损失:.4f}")

def 评估模型(模型, 测试序列, 词表, 设备):
    模型.eval()
    模型.to(设备)

    正确预测数 = 0
    sep_token_id = 词表['[SEP]']
    target_0_id = 词表['[TARGET_0]']
    target_1_id = 词表['[TARGET_1]']

    with torch.no_grad():
        for 序列 in tqdm(测试序列, desc="模型评估中"):
            真实目标_id = 序列[-1]
            输入序列 = torch.tensor(序列[:-1], dtype=torch.long).unsqueeze(0).to(设备)

            # 找到 [SEP] 的位置，确保我们只生成一个token
            sep_index = (输入序列[0] == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_index) == 0: continue

            max_len = 输入序列.shape[1] + 1

            # 使用 generate 方法自回归地生成下一个 token
            生成输出 = 模型.generate(
                输入序列,
                max_length=max_len,
                num_return_sequences=1,
                pad_token_id=词表['[PAD]'],
                eos_token_id=真实目标_id # For early stopping, not strictly needed but good practice
            )

            预测目标_id = 生成输出[0][-1].item()

            if 预测目标_id == 真实目标_id:
                正确预测数 += 1

    准确率 = 正确预测数 / len(测试序列)
    logger.info(f"模型评估完成，准确率: {准确率:.4f}")
    return 准确率 