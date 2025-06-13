import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger
import numpy as np

pd.options.mode.copy_on_write = True

def 获取数据():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    列名 = [
        "年龄", "工作类型", "fnlwgt", "教育", "教育年数", "婚姻状况",
        "职业", "亲属关系", "种族", "性别", "资本利得", "资本损失",
        "周工作小时数", "原籍国", "收入"
    ]
    数据 = pd.read_csv(url, header=None, names=列名, sep=r",\s*", engine="python", na_values="?")
    数据.dropna(inplace=True)

    数值型列 = 数据.select_dtypes(include=np.number).columns.tolist()
    数值型列.remove("教育年数") # 这个和"教育"高度相关，移除一个

    类别型列 = 数据.select_dtypes(include="object").columns.tolist()
    类别型列.remove("收入")

    数据["收入"] = 数据["收入"].apply(lambda x: 1 if x == ">50K" else 0)

    logger.info(f"数据集加载完成，共 {len(数据)} 条记录")
    logger.info(f"数值型特征: {数值型列}")
    logger.info(f"类别型特征: {类别型列}")

    return 数据, 数值型列, 类别型列

def 创建词表和分箱(数据, 数值型列, 类别型列, 分箱数量=10):
    词表 = {
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
        "[TARGET_0]": 4, "[TARGET_1]": 5
    }
    当前索引 = len(词表)
    分箱信息 = {}

    for 列 in 数值型列:
        try:
            数据[f"{列}_分箱"], 分箱信息[列] = pd.qcut(数据[列], q=分箱数量, retbins=True, duplicates='drop')
            for i in range(len(分箱信息[列]) - 1):
                token = f"{列}_bin_{i}"
                if token not in 词表:
                    词表[token] = 当前索引
                    当前索引 += 1
        except ValueError as e:
            logger.warning(f"列 '{列}' 分箱失败: {e}, 使用原始值")
            分箱信息[列] = None


    for 列 in 类别型列:
        for 值 in 数据[列].unique():
            token = f"{列}_{值}"
            if token not in 词表:
                词表[token] = 当前索引
                当前索引 += 1

    logger.info(f"词表创建完成，共 {len(词表)} 个token")
    return 词表, 分箱信息

def _tokenize_row(行, 数值型列, 类别型列, 词表, 分箱信息):
    tokens = ["[CLS]"]
    for 列 in 数值型列:
        if 分箱信息.get(列) is not None:
            箱体索引 = np.digitize(行[列], bins=分箱信息[列]) - 1
            # np.digitize returns 1-based index, so we subtract 1. It might go to -1 or len, so clip
            箱体索引 = np.clip(箱体索引, 0, len(分箱信息[列]) - 2)
            tokens.append(f"{列}_bin_{箱体索引}")
        else: # 如果分箱失败
            pass

    for 列 in 类别型列:
        tokens.append(f"{列}_{行[列]}")

    tokens.append("[SEP]")
    return [词表.get(t, 词表["[UNK]"]) for t in tokens]


def 序列化数据(数据, 数值型列, 类别型列, 词表, 分箱信息):
    序列列表 = []
    标签列表 = []

    for _, 行 in 数据.iterrows():
        token_ids = _tokenize_row(行, 数值型列, 类别型列, 词表, 分箱信息)

        目标token_id = 词表[f"[TARGET_{行['收入']}]"]
        token_ids.append(目标token_id)

        序列列表.append(token_ids)
        标签列表.append(行['收入'])

    logger.info(f"数据序列化完成，共 {len(序列列表)} 条序列")
    return 序列列表, 标签列表

def 获取训练和测试数据(测试集比例=0.2, 随机种子=42):
    数据, 数值型列, 类别型列 = 获取数据()
    训练数据, 测试数据 = train_test_split(数据, test_size=测试集比例, random_state=随机种子, stratify=数据['收入'])

    词表, 分箱信息 = 创建词表和分箱(训练数据, 数值型列, 类别型列)

    训练序列, _ = 序列化数据(训练数据, 数值型列, 类别型列, 词表, 分箱信息)
    测试序列, _ = 序列化数据(测试数据, 数值型列, 类别型列, 词表, 分箱信息)

    return 训练序列, 测试序列, 词表, (训练数据, 测试数据, 数值型列, 类别型列) 