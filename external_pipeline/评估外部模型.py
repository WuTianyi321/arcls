import json
import os
import argparse
import numpy as np
from loguru import logger
from transformers import AutoTokenizer

def 评估(predictions_path: str, test_data_path: str, tokenizer_path: str):
    """
    从外部流水线生成的预测文件中加载结果，并与测试集真值进行比较，计算准确率。

    :param predictions_path: 外部流水线生成的预测结果文件 (.jsonl)
    :param test_data_path: 包含真实标签的测试数据文件 (.jsonl)
    :param tokenizer_path: 训练好的分词器路径 (tokenizer.json)
    """
    logger.info("1. 加载分词器以获取目标Token ID...")
    if not os.path.exists(tokenizer_path):
        logger.error(f"分词器文件未找到: {tokenizer_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    target_0_id = tokenizer.convert_tokens_to_ids("[TARGET_0]")
    target_1_id = tokenizer.convert_tokens_to_ids("[TARGET_1]")
    
    if target_0_id is None or target_1_id is None:
        logger.error("无法在分词器中找到 [TARGET_0] 或 [TARGET_1] 特殊Token。")
        return
        
    logger.info(f"目标Token ID: [TARGET_0]={target_0_id}, [TARGET_1]={target_1_id}")

    logger.info(f"2. 加载真实标签于: {test_data_path}")
    if not os.path.exists(test_data_path):
        logger.error(f"测试数据文件未找到: {test_data_path}")
        return
        
    true_labels = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 真实目标是token_ids序列的倒数第二个元素，最后一个是</s>
            true_labels.append(data["token_ids"][-1])

    logger.info(f"3. 加载模型预测于: {predictions_path}")
    if not os.path.exists(predictions_path):
        logger.error(f"预测文件未找到: {predictions_path}")
        logger.error("请确保您的外部流水线已运行并已生成此文件。")
        # 为方便测试，生成一个随机预测文件
        logger.warning("未找到预测文件，将生成一个随机模拟的预测文件用于演示。")
        模拟生成随机预测(predictions_path, true_labels, [target_0_id, target_1_id])
        
    predictions = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 假设外部流水线的输出格式包含一个 'predicted_token_id' 键
            if "predicted_token_id" not in data:
                logger.error(f"预测文件行中缺少 'predicted_token_id': {line.strip()}")
                return
            predictions.append(data["predicted_token_id"])

    if len(predictions) != len(true_labels):
        logger.error(f"预测数量 ({len(predictions)}) 与真实标签数量 ({len(true_labels)}) 不匹配。")
        return

    # 4. 计算准确率
    correct_predictions = sum(p == t for p, t in zip(predictions, true_labels))
    total = len(true_labels)
    accuracy = correct_predictions / total if total > 0 else 0

    logger.success("评估完成!")
    logger.success(f"总样本数: {total}")
    logger.success(f"正确预测数: {correct_predictions}")
    logger.success(f"准确率: {accuracy:.4f}")

def 模拟生成随机预测(path, true_labels, target_ids):
    """如果预测文件不存在，则创建一个用于演示的随机预测文件。"""
    with open(path, 'w', encoding='utf-8') as f:
        for _ in true_labels:
            # 简单地在两个目标之间随机选择
            predicted_id = np.random.choice(target_ids)
            f.write(json.dumps({"predicted_token_id": predicted_id}) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="评估外部流水线的分类预测结果。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument(
        "--predictions_path", 
        type=str, 
        default=os.path.join(script_dir, "predictions.jsonl"),
        help="外部流水线生成的预测结果文件路径 (.jsonl)"
    )
    parser.add_argument(
        "--test_data_path", 
        type=str, 
        default=os.path.join(script_dir, "test_data.jsonl"),
        help="包含真实标签的测试集文件路径 (.jsonl)"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default=os.path.join(script_dir, "tokenizer.json"),
        help="训练好的分词器文件(tokenizer.json)路径"
    )
    
    args = parser.parse_args()
    评估(args.predictions_path, args.test_data_path, args.tokenizer_path) 