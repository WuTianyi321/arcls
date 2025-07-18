import json
import torch
from transformers import AutoModelForCausalLM
from external_pipeline.rwkv_tokenizer import TRIE_TOKENIZER  # 假设分词器实现放在此处

模型名称 = 'myttt196'
词表路径 = 'external_pipeline/vocab.txt'
测试数据路径 = 'external_pipeline/test_data.jsonl'

模型 = AutoModelForCausalLM.from_pretrained(模型名称).to(torch.float16).cuda()
模型.eval()
分词器 = TRIE_TOKENIZER(词表路径)

# 获取目标 token 的 id
目标0_id = 分词器.encode("[TARGET_0]")[0]
目标1_id = 分词器.encode("[TARGET_1]")[0]

正确 = 0
总数 = 0

with open(测试数据路径, 'r', encoding='utf-8') as 文件:
    for 行 in 文件:
        数据 = json.loads(行)
        序列 = torch.tensor(数据['token_ids'][:-1]).unsqueeze(0).cuda()
        真实 = 数据['token_ids'][-1]
        with torch.no_grad():
            输出 = 模型(input_ids=序列)
        最后步_logits = 输出.logits[0, -1]
        # 仅比较两类目标 token
        预测 = 目标1_id if 最后步_logits[目标1_id] > 最后步_logits[目标0_id] else 目标0_id
        if 预测 == 真实:
            正确 += 1
        总数 += 1

准确率 = 正确 / 总数 if 总数 else 0
print(f'样本数 {总数} 准确 {正确} 准确率 {准确率:.4f}') 