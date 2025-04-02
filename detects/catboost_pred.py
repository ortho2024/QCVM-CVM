import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
file_path = r'measure.xls'
data = pd.read_excel(file_path)
# 保存原始第一列 "name"
names = data["name"]
# 删除无关列并分组划分数据
data = data.drop(columns=["name"])


model = CatBoostClassifier()
model.load_model(r'catboost_model.cbm')
y_pred = model.predict(data)

# 将测试集的预测结果保存回表格
data["name"] = data.index.map(names)  # 复原测试集的 "name"
data["pred"] = y_pred  # 添加预测结果列

# 调整列顺序为：name、数据列、target 和 pred
output_df = data[["name"] + list(data.columns) + ["pred"]]
# 保存到 Excel 文件
output_file = r'reult.xlsx'
output_df.to_excel(output_file, index=False)
print(f"预测结果已保存到 {output_file}")
