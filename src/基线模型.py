from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from loguru import logger

def 训练和评估CatBoost(训练数据, 测试数据, 数值型列, 类别型列):
    logger.info("开始训练和评估 CatBoost 基线模型...")

    X_train = 训练数据[数值型列 + 类别型列]
    y_train = 训练数据['收入']

    X_test = 测试数据[数值型列 + 类别型列]
    y_test = 测试数据['收入']

    模型 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        verbose=100,
        cat_features=类别型列,
        random_seed=42
    )

    模型.fit(X_train, y_train)

    预测值 = 模型.predict(X_test)
    准确率 = accuracy_score(y_test, 预测值)

    logger.info(f"CatBoost 模型评估完成，准确率: {准确率:.4f}")
    return 准确率 