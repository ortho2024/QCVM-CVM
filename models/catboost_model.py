import os
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from tabulate import tabulate


def custom_classification_report(y_true, y_pred, decimal_places=6):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import numpy as np

    classes = np.unique(y_true)
    report = []
    for cls in classes:
        precision = precision_score(y_true, y_pred, labels=[cls], average=None)[0]
        recall = recall_score(y_true, y_pred, labels=[cls], average=None)[0]
        f1 = f1_score(y_true, y_pred, labels=[cls], average=None)[0]
        # 单类别准确率的计算
        mask = (y_true == cls)  # 找到 cls 类别的样本
        acc = accuracy_score(y_true[mask], y_pred[mask])  # 对该类别样本的预测准确性

        report.append({
            "Class": cls,
            "Accuracy": round(acc, decimal_places),
            "Precision": round(precision, decimal_places),
            "Recall": round(recall, decimal_places),
            "F1 Score": round(f1, decimal_places)
        })

    return report


# 定义函数：分组划分训练集和测试集
def split_data_by_group(data, target_col, train_ratio=0.8):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    grouped = data.groupby(target_col)
    for _, group in grouped:
        train_group, test_group = train_test_split(
            group, train_size=train_ratio, random_state=42
        )
        train_df = pd.concat([train_df, train_group])
        test_df = pd.concat([test_df, test_group])
    return train_df, test_df


# 加载数据
file_path = r'sorted_example.xlsx'
data = pd.read_excel(file_path)
data = data.drop(columns=["name"])

# 分组划分数据
train_df, test_df = split_data_by_group(data, target_col='target', train_ratio=0.8)

# 提取特征和标签
X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

print(f"训练集大小: {X_train.shape}")
print(f"训练标签: {y_train.shape}")

# # 初始化模型
# model = CatBoostClassifier(
#     iterations=1000,
#     learning_rate=0.05,
#     depth=6,
#     l2_leaf_reg=5,
#     random_strength=0.5,
#     loss_function='MultiClass',
#     eval_metric='Accuracy',
#     # early_stopping_rounds=50,
#     # cat_features=[0, 2, 5],  # 假设第0、2、5列是类别特征
#     # task_type='GPU',         # 如果有 GPU 可用
#     # devices='0:1',           # 使用 GPU 设备 0 和 1
#     verbose=100,
#     class_weights=[0.2, 0.3, 0.2, 0.1, 0.1, 0.1]
# )

# 初始化模型
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=5,
    random_strength=0.5,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    # early_stopping_rounds=50,
    # cat_features=[0, 2, 5],  # 假设第0、2、5列是类别特征
    # task_type='GPU',         # 如果有 GPU 可用
    # devices='0:1',           # 使用 GPU 设备 0 和 1
    verbose=100,
    class_weights=[0.2, 0.3, 0.2, 0.1, 0.1, 0.1]
)

# 训练模型
model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    # early_stopping_rounds=50
)

# 评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy}")
print("分类报告:")
# 使用该函数
result = custom_classification_report(y_test, y_pred)
# for r in result:
#     print(r)
# 转换为表格
headers = result[0].keys()  # 表头
rows = [r.values() for r in result]  # 表格行
# 使用 tabulate 格式化输出
print(tabulate(rows, headers=headers, tablefmt="grid"))
# print(classification_report(y_test, y_pred, output_dict=True))

# # 保存模型
# model_dir = r'D:\D_project\C_measure\models'
# os.makedirs(model_dir, exist_ok=True)  # 如果目录不存在则创建
# model_path = os.path.join(model_dir, 'catboost_model.cbm')
# model.save_model(model_path)
#
# print(f"模型已保存到 {model_path}")
